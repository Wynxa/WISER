import argparse
import json
import os
import random
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
from src.modeling import (
    DualEncoderModel,
    build_optimizer,
    compute_training_losses,
    move_token_batch_to_device,
    predict_major_labels,
)

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dual-encoder hazard reasoning model."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "trainable_worker_samples_train_v1.jsonl",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "trainable_worker_samples_val_v1.jsonl",
    )
    parser.add_argument(
        "--prototype-inventory",
        type=Path,
        default=PROJECT_ROOT / "config" / "rule_prototypes_v1.json",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/hf_models"),
        help="Local Hugging Face cache directory on the training machine.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model only from local cache without network access.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dual_encoder_v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--share-backbone", action="store_true")
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--alignment-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--val-is-test",
        action="store_true",
        help="Record that the validation JSONL is intentionally the same as the test JSONL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(pretrained_model_name: str, cache_dir: str | None = None, local_files_only: bool = False):
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for the hazard reasoning dual encoder. "
            "Please install transformers before training."
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


def build_label_names(prototype_inventory: PrototypeInventory) -> List[str]:
    num_major_labels = len(prototype_inventory.major_label_to_id)
    label_names = ["unknown"] * num_major_labels
    for label_name, label_id in prototype_inventory.major_label_to_id.items():
        label_names[int(label_id)] = label_name
    return label_names


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


def evaluate(
    model: DualEncoderModel,
    data_loader: DataLoader,
    tokenizer,
    prototype_inventory: PrototypeInventory,
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    total_alignment_loss = 0.0
    total_classification_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []
    label_names = build_label_names(prototype_inventory)

    prototype_tokens = tokenize_texts(
        tokenizer,
        prototype_inventory.prototype_texts,
        max_length=max_length,
    )
    prototype_tokens = move_token_batch_to_device(prototype_tokens, device)
    prototype_major_label_ids = prototype_inventory.prototype_major_label_ids.to(device)
    num_major_labels = len(prototype_inventory.major_label_to_id)

    for batch in data_loader:
        scene_tokens = tokenize_texts(tokenizer, batch["scene_texts"], max_length=max_length)
        scene_tokens = move_token_batch_to_device(scene_tokens, device)
        positive_mask = prototype_inventory.build_positive_mask(batch["positive_prototype_ids"]).to(device)
        major_label_ids = batch["major_label_ids"].to(device)

        scene_embeddings = model.encode_scene(scene_tokens)
        rule_embeddings = model.encode_rule(prototype_tokens)
        prototype_logits = model.compute_similarity(scene_embeddings, rule_embeddings)
        losses = compute_training_losses(
            prototype_logits=prototype_logits,
            positive_mask=positive_mask,
            major_label_ids=major_label_ids,
            prototype_major_label_ids=prototype_major_label_ids,
            num_major_labels=num_major_labels,
        )
        predictions = losses["major_logits"].argmax(dim=1)
        correct += (predictions == major_label_ids).sum().item()
        total += major_label_ids.size(0)
        all_targets.extend(major_label_ids.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        batch_size = major_label_ids.size(0)
        total_loss += losses["loss"].item() * batch_size
        total_alignment_loss += losses["alignment_loss"].item() * batch_size
        total_classification_loss += losses["classification_loss"].item() * batch_size

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    avg_alignment_loss = total_alignment_loss / max(total, 1)
    avg_classification_loss = total_classification_loss / max(total, 1)
    return {
        "loss": avg_loss,
        "alignment_loss": avg_alignment_loss,
        "classification_loss": avg_classification_loss,
        "accuracy": accuracy,
        "per_class": compute_per_class_metrics(all_targets, all_predictions, label_names),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.model_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.model_cache_dir))

    train_dataset = HazardTrainDataset(args.train_jsonl)
    val_dataset = HazardTrainDataset(args.val_jsonl)
    prototype_inventory = PrototypeInventory(args.prototype_inventory)
    validate_dataset_compatibility(train_dataset, prototype_inventory)
    validate_dataset_compatibility(val_dataset, prototype_inventory)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_train_samples,
    )

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
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    prototype_tokens = tokenize_texts(
        tokenizer,
        prototype_inventory.prototype_texts,
        max_length=args.max_length,
    )
    prototype_tokens = move_token_batch_to_device(prototype_tokens, device)
    prototype_major_label_ids = prototype_inventory.prototype_major_label_ids.to(device)
    num_major_labels = len(prototype_inventory.major_label_to_id)

    best_val_accuracy = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_alignment_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            scene_tokens = tokenize_texts(
                tokenizer,
                batch["scene_texts"],
                max_length=args.max_length,
            )
            scene_tokens = move_token_batch_to_device(scene_tokens, device)
            positive_mask = prototype_inventory.build_positive_mask(
                batch["positive_prototype_ids"]
            ).to(device)
            major_label_ids = batch["major_label_ids"].to(device)

            scene_embeddings = model.encode_scene(scene_tokens)
            rule_embeddings = model.encode_rule(prototype_tokens)
            prototype_logits = model.compute_similarity(scene_embeddings, rule_embeddings)

            losses = compute_training_losses(
                prototype_logits=prototype_logits,
                positive_mask=positive_mask,
                major_label_ids=major_label_ids,
                prototype_major_label_ids=prototype_major_label_ids,
                num_major_labels=num_major_labels,
                classification_loss_weight=args.classification_loss_weight,
                alignment_loss_weight=args.alignment_loss_weight,
            )

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            predictions = losses["major_logits"].argmax(dim=1)
            batch_size = major_label_ids.size(0)
            epoch_correct += (predictions == major_label_ids).sum().item()
            epoch_total += batch_size
            epoch_loss += losses["loss"].item() * batch_size
            epoch_alignment_loss += losses["alignment_loss"].item() * batch_size
            epoch_classification_loss += losses["classification_loss"].item() * batch_size

        train_metrics = {
            "loss": epoch_loss / max(epoch_total, 1),
            "alignment_loss": epoch_alignment_loss / max(epoch_total, 1),
            "classification_loss": epoch_classification_loss / max(epoch_total, 1),
            "accuracy": epoch_correct / max(epoch_total, 1),
        }
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            tokenizer=tokenizer,
            prototype_inventory=prototype_inventory,
            max_length=args.max_length,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    with (output_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    summary = {
        "best_val_accuracy": best_val_accuracy,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_prototypes": prototype_inventory.summary["num_prototypes"],
        "val_alias_of": "test" if args.val_is_test else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
