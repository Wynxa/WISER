import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    HazardTrainDataset,
    PrototypeInventory,
    collate_train_samples,
    validate_dataset_compatibility,
)
from src.modeling import DualEncoderModel, move_token_batch_to_device, predict_major_labels

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc evaluation for the hazard reasoning dual encoder."
    )
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--prototype-inventory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("/root/autodl-tmp/hf_models"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--share-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Optional training config JSON. If provided, model hyperparameters are loaded from it.",
    )
    return parser.parse_args()


def build_tokenizer(pretrained_model_name: str, cache_dir: str | None = None, local_files_only: bool = False):
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for the hazard reasoning dual encoder. "
            "Please install transformers before evaluation."
        )
    return AutoTokenizer.from_pretrained(
        pretrained_model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def tokenize_texts(tokenizer, texts, max_length: int) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def maybe_load_train_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.train_config is None:
        return args

    with args.train_config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.pretrained_model = cfg.get("pretrained_model", args.pretrained_model)
    args.max_length = int(cfg.get("max_length", args.max_length))
    args.projection_hidden_dim = int(cfg.get("projection_hidden_dim", args.projection_hidden_dim))
    args.projection_dim = int(cfg.get("projection_dim", args.projection_dim))
    args.dropout = float(cfg.get("dropout", args.dropout))
    args.share_backbone = bool(cfg.get("share_backbone", args.share_backbone))
    args.local_files_only = bool(cfg.get("local_files_only", args.local_files_only))
    model_cache_dir = cfg.get("model_cache_dir")
    if model_cache_dir:
        args.model_cache_dir = Path(model_cache_dir)
    return args


def build_label_names(prototype_inventory: PrototypeInventory) -> List[str]:
    num_major_labels = len(prototype_inventory.major_label_to_id)
    label_names = ["unknown"] * num_major_labels
    for label_name, label_id in prototype_inventory.major_label_to_id.items():
        label_names[int(label_id)] = label_name
    return label_names


def build_confusion_matrix(y_true: List[int], y_pred: List[int], num_labels: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for gt, pred in zip(y_true, y_pred):
        matrix[int(gt)][int(pred)] += 1
    return matrix


def compute_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    per_class = {}
    for label_id, label_name in enumerate(label_names):
        tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt == label_id and pred == label_id)
        fp = sum(1 for gt, pred in zip(y_true, y_pred) if gt != label_id and pred == label_id)
        fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt == label_id and pred != label_id)
        support = sum(1 for gt in y_true if gt == label_id)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return per_class


def compute_aggregate_metrics(
    per_class: Dict[str, Dict[str, float]],
    total_samples: int,
    correct_samples: int,
) -> Dict[str, float]:
    metric_rows = list(per_class.values())
    macro_precision = sum(row["precision"] for row in metric_rows) / max(len(metric_rows), 1)
    macro_recall = sum(row["recall"] for row in metric_rows) / max(len(metric_rows), 1)
    macro_f1 = sum(row["f1"] for row in metric_rows) / max(len(metric_rows), 1)

    weighted_precision = sum(row["precision"] * row["support"] for row in metric_rows) / max(total_samples, 1)
    weighted_recall = sum(row["recall"] * row["support"] for row in metric_rows) / max(total_samples, 1)
    weighted_f1 = sum(row["f1"] * row["support"] for row in metric_rows) / max(total_samples, 1)

    return {
        "accuracy": correct_samples / max(total_samples, 1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def main() -> None:
    args = maybe_load_train_config(parse_args())

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.model_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.model_cache_dir))

    dataset = HazardTrainDataset(args.test_jsonl)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_train_samples,
    )
    prototype_inventory = PrototypeInventory(args.prototype_inventory)
    validate_dataset_compatibility(dataset, prototype_inventory)
    label_names = build_label_names(prototype_inventory)

    tokenizer = build_tokenizer(
        args.pretrained_model,
        cache_dir=str(args.model_cache_dir),
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device)
    model = DualEncoderModel(
        pretrained_model_name=args.pretrained_model,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        share_backbone=args.share_backbone,
        cache_dir=str(args.model_cache_dir),
        local_files_only=args.local_files_only,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    prototype_tokens = tokenize_texts(
        tokenizer,
        prototype_inventory.prototype_texts,
        max_length=args.max_length,
    )
    prototype_tokens = move_token_batch_to_device(prototype_tokens, device)
    prototype_major_label_ids = prototype_inventory.prototype_major_label_ids.to(device)
    num_major_labels = len(prototype_inventory.major_label_to_id)

    y_true: List[int] = []
    y_pred: List[int] = []
    prediction_rows = []

    with torch.no_grad():
        for batch in data_loader:
            scene_tokens = tokenize_texts(tokenizer, batch["scene_texts"], max_length=args.max_length)
            scene_tokens = move_token_batch_to_device(scene_tokens, device)
            outputs = predict_major_labels(
                model=model,
                scene_tokens=scene_tokens,
                prototype_tokens=prototype_tokens,
                prototype_major_label_ids=prototype_major_label_ids,
                num_major_labels=num_major_labels,
            )

            gt_ids = batch["major_label_ids"].tolist()
            pred_ids = outputs["pred_major_label_ids"].detach().cpu().tolist()
            major_logits = outputs["major_logits"].detach().cpu().tolist()

            y_true.extend(gt_ids)
            y_pred.extend(pred_ids)

            for sample_id, gt_id, pred_id, logits in zip(
                batch["sample_ids"],
                gt_ids,
                pred_ids,
                major_logits,
            ):
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "true_label_id": int(gt_id),
                        "true_label": label_names[int(gt_id)],
                        "pred_label_id": int(pred_id),
                        "pred_label": label_names[int(pred_id)],
                        "major_logits": logits,
                    }
                )

    confusion_matrix = build_confusion_matrix(y_true, y_pred, len(label_names))
    per_class = compute_per_class_metrics(y_true, y_pred, label_names)
    overall = compute_aggregate_metrics(
        per_class=per_class,
        total_samples=len(y_true),
        correct_samples=sum(int(gt == pred) for gt, pred in zip(y_true, y_pred)),
    )

    output_payload = {
        "meta": {
            "task": "hazard_reasoning_major_label_classification",
            "checkpoint": str(args.checkpoint),
            "test_jsonl": str(args.test_jsonl),
            "prototype_inventory": str(args.prototype_inventory),
            "pretrained_model": args.pretrained_model,
            "num_samples": len(y_true),
            "label_order": label_names,
            "confusion_matrix_definition": "rows=true labels, cols=predicted labels",
        },
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "predictions": prediction_rows,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(output_payload["overall"], ensure_ascii=False, indent=2))
    print(f"Saved evaluation JSON to {args.output_json}")


if __name__ == "__main__":
    main()
