"""Prepare, train and evaluate the four decoupled-knowledge research questions."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AiC_resubmission.knowledge_decoupling_experiment import data
from AiC_resubmission.knowledge_decoupling_experiment.data import (
    HAZARD, LABELS, corrupt, digest, file_hash, load_cases, read_json, read_jsonl, write_json, write_jsonl)
from AiC_resubmission.knowledge_decoupling_experiment.metrics import (
    calibrate_threshold, paired_bootstrap, retrieval, summarize)
from AiC_resubmission.knowledge_decoupling_experiment.models import (
    NEURAL, RETRIEVERS, NeuralScorer, load_scorer, training_loss)


def library(args):
    path = args.external_library or args.suite / "libraries" / args.family / (args.library_name + ".json")
    rows = read_json(path)
    manifest = read_json(args.suite / "manifest.json")
    if not args.external_library and file_hash(path) != manifest["libraries_sha256"][args.family + "/" + args.library_name]:
        raise ValueError("Prepared library changed after protocol freeze")
    ids = [r["candidate_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Library must contain unique candidate IDs")
    for p in rows:
        if p["major_label"] not in LABELS or not p["text"].strip() or not p.get("semantic_id"):
            raise ValueError("Library requires text, semantic_id and existing six-class major_label")
    return rows, path


def train_reference(args):
    conditions = args.baseline_train_conditions
    return [r for condition in conditions for r in load_cases(args.suite, "train", condition)]


def validate_checkpoint_protocol(args):
    if args.method not in NEURAL:
        return
    if not args.train_config or not args.checkpoint:
        raise ValueError("Neural methods require checkpoint and training config")
    config = read_json(args.train_config)
    if not config.get("suite_manifest_sha256"):
        if not args.allow_legacy_checkpoint:
            raise ValueError("Historical checkpoint: explicitly pass --allow-legacy-checkpoint for diagnostic-only evaluation")
    elif config["suite_manifest_sha256"] != file_hash(args.suite / "manifest.json"):
        raise ValueError("Checkpoint training split/protocol differs from suite")


def scorer_for(args, candidates):
    validate_checkpoint_protocol(args)
    policy = read_json(args.suite / "policy.json")
    manifest = read_json(args.suite / "manifest.json")
    if file_hash(args.suite / "policy.json") != manifest["policy_sha256"]:
        raise ValueError("Prepared policy changed after protocol freeze")
    base = read_json(args.suite / "libraries" / args.family / "base.json")
    if file_hash(args.suite / "libraries" / args.family / "base.json") != manifest["libraries_sha256"][args.family + "/base"]:
        raise ValueError("Prepared base library changed after protocol freeze")
    return load_scorer(args, train_reference(args), base, policy)


def predict_rows(scorer, cases, candidates, family, threshold=None):
    matched = [r for r in cases if r.get("matched", True)]
    predictions = scorer.score(matched, candidates) if matched else []
    lookup = dict(zip((r["sample_id"] for r in matched), predictions))
    active_labels = {p["major_label"] for p in candidates}
    results = []
    for r in cases:
        pred = lookup.get(r["sample_id"], {"pred_label": None, "ranking": None,
                                          "confidence": None, "margin": None})
        positives = set(r["relevant"][family])
        results.append({"sample_id": r["sample_id"], "image_id": r["image_id"],
                        "worker_id": r.get("worker_id"), "true_label": r["true_label"],
                        "matched": r.get("matched", True), "scene_sha256": digest(r["scene_text"]),
                        "target_in_library": r["true_label"] in active_labels,
                        "relevant_ids": [p["candidate_id"] for p in candidates if p["semantic_id"] in positives],
                        **pred, "abstained": r["sample_id"] not in lookup or (
                            threshold is not None and pred["margin"] < threshold)})
    return results


def evaluate(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty result directory")
    candidates, path = library(args)
    scorer = scorer_for(args, candidates)
    manifest = read_json(args.suite / "manifest.json")
    chosen = args.conditions or list(manifest["conditions"])
    if args.predicted_dir and args.conditions is None:
        chosen.append("predicted")
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate conditions")
    calibration = read_json(args.calibration) if args.calibration else None
    if calibration and (calibration["suite_manifest_sha256"] != file_hash(args.suite / "manifest.json") or
                        calibration["method"] != args.method or calibration["family"] != args.family or
                        calibration["model_identity"] != scorer_identity(scorer)):
        raise ValueError("Calibration belongs to another model, family or protocol")
    threshold = calibration["threshold"] if calibration else None
    independent = {}
    if args.reviewed_gold:
        for r in read_jsonl(args.reviewed_gold):
            if not r.get("reviewed"):
                continue
            if not isinstance(r.get("reviewer"), str) or not r["reviewer"].strip():
                raise ValueError("Reviewed gold must identify its reviewer")
            if not isinstance(r.get("relevant_ids"), list):
                raise ValueError("Reviewed gold needs a relevant_ids list, including [] for no applicable candidate")
            key = (r["condition"], r["sample_id"])
            if key in independent:
                raise ValueError("Duplicate independently reviewed reference")
            if not set(r["relevant_ids"]) <= {p["candidate_id"] for p in candidates}:
                raise ValueError("Reviewed candidate IDs not in active library; review each library separately")
            independent[key] = r
    metrics = {}
    used_review = set()
    train_texts = {r["scene_text"] for r in load_cases(args.suite, "train", "clean")}
    source_texts = {r["sample_id"]: r["scene_text"] for r in load_cases(args.suite, args.split, "clean")}
    for condition in chosen:
        cases = load_cases(args.suite, args.split, condition, args.predicted_dir)
        rows = predict_rows(scorer, cases, candidates, args.family, threshold)
        metrics[condition] = summarize(rows)
        metrics[condition]["by_clean_source_text_overlap"] = {
            name: summarize([r for r in rows if (source_texts[r["sample_id"]] in train_texts) == seen])
            for name, seen in (("seen_in_clean_train", True), ("unseen_in_clean_train", False))}
        reviewed_rows = []
        for r in rows:
            key = (condition, r["sample_id"])
            if key in independent:
                gold = independent[key]
                if gold.get("scene_sha256") != r["scene_sha256"]:
                    raise ValueError("Review gold was produced from different observed evidence")
                reviewed_rows.append({**r, "relevant_ids": gold["relevant_ids"]})
                used_review.add(key)
        if reviewed_rows:
            metrics[condition]["independent_retrieval"] = retrieval([r for r in reviewed_rows if r["matched"]], True)
        metrics[condition]["input_information_loss"] = data.information_loss(cases)
        saved = copy.deepcopy(rows)
        if args.save_top_k > 0:
            for r in saved:
                if r["ranking"] is not None:
                    r["ranking"] = r["ranking"][:args.save_top_k]
        write_jsonl(args.output_dir / condition / "predictions.jsonl", saved)
        write_json(args.output_dir / condition / "metrics.json", metrics[condition])
    if independent and set(independent) != used_review:
        raise ValueError("Some reviewed gold IDs/conditions were not evaluated")
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "run.json", {
        "method": args.method, "family": args.family, "library": args.library_name,
        "library_sha256": file_hash(path), "suite_manifest_sha256": file_hash(args.suite / "manifest.json"),
        "split": args.split, "conditions": chosen, "scorer": scorer.metadata,
        "token_truncations": getattr(scorer, "truncations", None),
        "encoded_texts": getattr(scorer, "encoded_texts", None),
        "calibration": calibration, "test_status": manifest["test_status"],
        "predicted_manifest": read_json(args.predicted_dir / "manifest.json") if args.predicted_dir else None,
        "reviewed_gold_sha256": file_hash(args.reviewed_gold) if args.reviewed_gold else None,
        "saved_ranking_depth": args.save_top_k or "all", "retrieval_metrics_use_full_ranking": True,
        "augmentation": getattr(scorer, "config", {}).get("augmentation", "mixed" if len(args.baseline_train_conditions) > 1 else "clean"),
        "training_seed": getattr(scorer, "config", {}).get("seed"),
        "model_variant": "anonymous_rules" if getattr(scorer, "config", {}).get("anonymous_rules") else "standard",
    })
    return metrics


def scorer_identity(scorer):
    return digest(scorer.metadata)


def calibrate(args):
    candidates, path = library(args)
    scorer = scorer_for(args, candidates)
    rows = predict_rows(scorer, load_cases(args.suite, "dev", args.condition), candidates, args.family)
    chosen = calibrate_threshold(rows, args.max_risk, args.minimum_accepted)
    result = {**chosen, "method": args.method, "family": args.family, "library": args.library_name,
              "library_sha256": file_hash(path), "suite_manifest_sha256": file_hash(args.suite / "manifest.json"),
              "model_identity": scorer_identity(scorer), "split": "dev", "condition": args.condition,
              "requested_empirical_max_risk": args.max_risk,
              "note": "dev empirical threshold, not a statistical safety guarantee; library/noise shift may invalidate calibration"}
    write_json(args.output, result)
    return result


def training_case(row, epoch, args, manifest):
    if args.augmentation == "clean":
        return row
    choices = args.train_conditions
    index = int(digest([args.seed, epoch, row["sample_id"], "augmentation"])[:16], 16) % len(choices)
    condition = choices[index]
    return corrupt(row, manifest["conditions"][condition], manifest["noise_seed"])


def train(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty checkpoint directory")
    manifest = read_json(args.suite / "manifest.json")
    candidates = read_json(args.suite / "libraries" / args.family / "base.json")
    if file_hash(args.suite / "libraries" / args.family / "base.json") != manifest["libraries_sha256"][args.family + "/base"] or file_hash(args.suite / "policy.json") != manifest["policy_sha256"]:
        raise ValueError("Training policy/library changed after protocol freeze")
    training = load_cases(args.suite, "train")
    for condition in args.train_conditions + args.validation_conditions:
        if condition not in manifest["conditions"]:
            raise ValueError("Training/validation condition not in prepared protocol: " + condition)
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs, batch-size and gradient-accumulation must be positive")
    if args.anonymous_rules and args.method != "dual":
        raise ValueError("--anonymous-rules is a same-architecture dual control only")
    config = {"method": args.method, "labels": LABELS, "candidate_family": args.family,
              "candidate_ids": [p["semantic_id"] for p in candidates],
              "candidate_library_sha256": digest(candidates),
              "suite_manifest_sha256": file_hash(args.suite / "manifest.json"),
              "pretrained_model": args.pretrained_model, "model_cache_dir": str(args.model_cache_dir) if args.model_cache_dir else None,
              "projection_hidden_dim": args.projection_hidden_dim, "projection_dim": args.projection_dim,
              "share_backbone": args.share_backbone, "dropout": args.dropout,
              "max_length": args.max_length, "pair_max_length": args.pair_max_length,
              "seed": args.seed, "augmentation": args.augmentation, "train_conditions": args.train_conditions,
              "validation_conditions": args.validation_conditions, "epochs": args.epochs,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "batch_size": args.batch_size, "pair_batch_size": args.pair_batch_size,
              "gradient_accumulation": args.gradient_accumulation,
              "anonymous_rules": args.anonymous_rules,
              "effective_batch_size": args.batch_size * args.gradient_accumulation,
              "alignment_loss": args.alignment_loss, "model_selection": "mean_dev_macro_f1_predeclared_conditions",
              "historical_test_status": manifest["test_status"], "validation_is_test": False}
    if args.dry_run:
        return {"dry_run": True, "workers": len(training), "configuration": config,
                "note": "No neural model loaded or trained"}
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policy = read_json(args.suite / "policy.json")
    scorer = NeuralScorer(config, args, policy, load=False)
    config["tokenizer_vocab_sha256"] = digest(scorer.tokenizer.get_vocab())
    config["torch_version"] = torch.__version__
    config["training_device"] = args.device
    optimizer = torch.optim.AdamW(scorer.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    config["parameters"] = sum(p.numel() for p in scorer.model.parameters())
    best, history = -1.0, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        scorer.clear_cache()
        scorer.model.train()
        indices = list(range(len(training)))
        random.Random(args.seed + epoch).shuffle(indices)
        losses = []
        effective_batch = args.batch_size * args.gradient_accumulation
        for offset in range(0, len(indices), effective_batch):
            group = indices[offset:offset + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for chunk in range(0, len(group), args.batch_size):
                rows = [training_case(training[i], epoch, args, manifest) for i in group[chunk:chunk + args.batch_size]]
                loss = training_loss(scorer, rows, candidates, args.family, args.alignment_loss)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                weighted = loss * len(rows) / len(group)
                weighted.backward()
                group_loss += float(weighted.detach().cpu())
            torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1.0)
            optimizer.step()
            losses.append(group_loss)
        scorer.model.eval()
        scorer.clear_cache()
        dev = {}
        for condition in args.validation_conditions:
            rows = predict_rows(scorer, load_cases(args.suite, "dev", condition), candidates, args.family)
            dev[condition] = summarize(rows)
        selection = statistics.mean(m["forced_matched"]["macro_f1"] for m in dev.values())
        record = {"epoch": epoch + 1, "train_loss": statistics.mean(losses), "selection_score": selection, "dev": dev}
        history.append(record)
        print(json.dumps({"epoch": epoch + 1, "loss": record["train_loss"], "dev_macro_f1": selection}), flush=True)
        if selection > best:
            best = selection
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": scorer.model.state_dict(), "epoch": epoch + 1}, args.output_dir / "best_model.pt")
            config["best_epoch"] = epoch + 1
        write_json(args.output_dir / "history.json", history)
        write_json(args.output_dir / "train_config.json", config)
    config.update({"training_seconds": time.perf_counter() - start, "best_dev_macro_f1": best,
                   "token_truncations": scorer.truncations, "encoded_texts": scorer.encoded_texts})
    write_json(args.output_dir / "train_config.json", config)
    return {"best_dev_macro_f1": best, "best_epoch": config["best_epoch"], "output_dir": str(args.output_dir)}


def prepare_predicted(args):
    path = ROOT / "AiC_resubmission/rule_vs_model_experiment/experiment.py"
    spec = importlib.util.spec_from_file_location("shared_predicted_graph_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = load_cases(args.suite, "test")
    templates = [{"sample_id": r["sample_id"], "image_id": r["image_id"], "split": "test",
                  "major_label": r["true_label"], "major_label_id": r["true_label_id"],
                  "scene_text": r["scene_text"], "rule_text": "", "context_id": "", "risk_level": "",
                  "positive_prototype_ids": r["relevant"]["coarse"]}
                 for r in cases]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty predicted evidence directory")
    template = args.output_dir / "reference_template.jsonl"
    write_jsonl(template, templates)
    module.prepare_predicted(SimpleNamespace(reference=args.reference, model_template=template,
        graph_dir=args.graph_dir, provenance=args.provenance, min_worker_iou=args.min_worker_iou,
        allow_missing_images=args.allow_missing_images, output_dir=args.output_dir))
    load_cases(args.suite, "test", "predicted", args.output_dir)
    return read_json(args.output_dir / "manifest.json")


def compare(args):
    if len(args.results) < 2:
        raise ValueError("At least two run directories required")
    if args.bootstrap_repeats < 10:
        raise ValueError("At least 10 bootstrap repeats required")
    runs = [(path, read_json(path / "run.json")) for path in args.results]
    first = runs[0][1]
    for _, run in runs:
        if (run["suite_manifest_sha256"], run["split"]) != (first["suite_manifest_sha256"], first["split"]):
            raise ValueError("Cannot compare different suites or splits")
    comparisons = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            pa, ma = runs[a]
            pb, mb = runs[b]
            common = sorted(set(ma["conditions"]) & set(mb["conditions"]))
            for condition in common:
                left = read_jsonl(pa / condition / "predictions.jsonl")
                right = read_jsonl(pb / condition / "predictions.jsonl")
                ci = paired_bootstrap(left, right, args.bootstrap_repeats, args.seed)
                matched = [(x, y) for x, y in zip(sorted(left, key=lambda r: r["sample_id"]),
                                                sorted(right, key=lambda r: r["sample_id"])) if x["matched"]]
                comparisons.append({"left": str(pa), "right": str(pb), "condition": condition,
                    "same_library": ma["library_sha256"] == mb["library_sha256"], "paired_bootstrap": ci,
                    "changed_prediction_rate": sum(x["pred_label"] != y["pred_label"] for x, y in matched) / len(matched) if matched else None})
    write_json(args.output_dir / "comparisons.json", comparisons)
    lines = ["# Paired Experiment Comparisons", "", "Reference: internal policy states, not independent normative truth.", "",
             "| Left | Right | Condition | Accuracy Delta | Image Bootstrap 95% CI | Same Library |",
             "|---|---|---|---:|---|---|"]
    for r in comparisons:
        ci = r["paired_bootstrap"]
        lines.append(f"| {Path(r['left']).name} | {Path(r['right']).name} | {r['condition']} | {ci['point']:.4f} | {ci['ci95']} | {r['same_library']} |")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparisons


def benchmark(args):
    if args.repeats < 2 or args.warmup < 1 or args.max_workers < 1:
        raise ValueError("Require repeats >= 2, warmup >= 1 and positive max-workers")
    candidates, _ = library(args)
    scorer = scorer_for(args, candidates)
    cases = [r for r in load_cases(args.suite, "test", "clean") if r["matched"]][:args.max_workers]
    neural = args.method in NEURAL
    if neural:
        scorer.audit_lengths = False

    def sync():
        if neural and str(args.device).startswith("cuda"):
            scorer.torch.cuda.synchronize(args.device)

    reports = []
    for name in args.libraries:
        active_path = args.suite / "libraries" / args.family / (name + ".json")
        if not args.external_library and file_hash(active_path) != read_json(args.suite / "manifest.json")["libraries_sha256"][args.family + "/" + name]:
            raise ValueError("Benchmark library changed after protocol freeze")
        active = read_json(active_path)
        if args.external_library:
            active = read_json(args.external_library)
        if neural:
            scorer.clear_cache()
        counts_before = copy.deepcopy(getattr(scorer, "encoded_texts", {}))
        sync()
        start = time.perf_counter()
        if args.method in ("dual", "learned_prototypes"):
            with scorer.torch.no_grad():
                scorer.cached_vectors = scorer.candidate_vectors(active)
                scorer.cached_key = digest(active)
        sync()
        precompute = time.perf_counter() - start
        for _ in range(args.warmup):
            scorer.score(cases, active)
        sync()
        if neural and str(args.device).startswith("cuda"):
            scorer.torch.cuda.reset_peak_memory_stats(args.device)
        online_counts_before = copy.deepcopy(getattr(scorer, "encoded_texts", {}))
        times = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            scorer.score(cases, active)
            sync()
            times.append(time.perf_counter() - start)
        ordered = sorted(times)
        report = {"library": name, "candidates": len(active), "workers": len(cases),
            "unique_candidate_texts": len({p['text'] for p in active}),
            "unique_semantic_states": len({p['semantic_id'] for p in active}),
            "offline_candidate_precompute_seconds": precompute if args.method in ("dual", "learned_prototypes") else None,
            "online_seconds": {"mean": statistics.mean(times), "std": statistics.stdev(times),
                               "median": statistics.median(times), "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]},
            "workers_per_second": len(cases) / statistics.mean(times),
            "peak_cuda_allocated_bytes": scorer.torch.cuda.max_memory_allocated(args.device) if neural and str(args.device).startswith("cuda") else None,
            "online_encoded_texts": {k: v - online_counts_before[k] for k, v in getattr(scorer, "encoded_texts", {}).items()},
            "measurement": "tokenization + model forward + transfer + ranking; excludes model load; no query-text deduplication",
            "scope": "existing states and paraphrases, not a large independently curated normative library"}
        if args.method == "dual" and not args.external_library:
            added = data.make_libraries(active, read_json(args.suite / "policy.json"), "rebar_work_unsafe")["add_paraphrases"]
            for i, p in enumerate(added[len(active):]):
                p["candidate_id"] = active[i]["candidate_id"] + "::benchmark_added::" + str(i)
            before = scorer.encoded_texts["rule"]
            sync()
            start = time.perf_counter()
            with scorer.torch.no_grad():
                scorer.candidate_vectors(added)
            sync()
            report["incremental_update"] = {"seconds": time.perf_counter() - start,
                                             "newly_encoded_rule_texts": scorer.encoded_texts["rule"] - before,
                                             "added_candidate_ids": len(added) - len(active)}
        report["encoded_texts_total_this_library"] = {k: v - counts_before[k] for k, v in getattr(scorer, "encoded_texts", {}).items()}
        reports.append(report)
        if args.external_library:
            break
    write_json(args.output, {"method": args.method, "family": args.family, "device": args.device,
        "batch_size": args.batch_size, "pair_batch_size": args.pair_batch_size,
        "repeats": args.repeats, "warmup": args.warmup, "scorer": scorer.metadata, "libraries": reports})
    return reports


def review_queue(args):
    candidates, _ = library(args)
    rows = load_cases(args.suite, args.split, args.condition, args.predicted_dir)
    groups = {}
    for r in rows:
        if r["matched"]:
            groups.setdefault(r["true_label"], []).append(r)
    chosen = []
    for label, group in sorted(groups.items()):
        random.Random(args.seed + LABELS.index(label)).shuffle(group)
        chosen.extend(group[:args.max_per_label])
    write_jsonl(args.output_dir / "review_queue.jsonl", [{"sample_id": r["sample_id"], "image_id": r["image_id"],
        "worker_id": r.get("worker_id"), "condition": args.condition, "scene_text": r["scene_text"],
        "scene_sha256": digest(r["scene_text"]), "reviewed": False, "relevant_ids": None,
        "reviewer": None, "source_clause_ids": [], "notes": ""} for r in chosen])
    write_json(args.output_dir / "candidate_catalog.json", candidates)
    return {"queued_workers": len(chosen), "note": "Teacher labels/clean context hidden; stratified sample is not the full-test population"}


def report(args):
    groups = {}
    for path in args.results:
        run = read_json(path / "run.json")
        metrics = read_json(path / "metrics.json")
        for condition, value in metrics.items():
            method = run["method"] + ("_anonymous" if run.get("model_variant") == "anonymous_rules" else "")
            key = (method, run["family"], run["library"], run.get("augmentation", "clean"), condition)
            groups.setdefault(key, []).append((run.get("training_seed"), value, str(path)))
    result = []
    for key, entries in sorted(groups.items()):
        seeds = [seed for seed, _, _ in entries]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate training seed in a report group; do not treat repeated evaluation as seed variance")
        values = {}
        for field in ("accuracy", "macro_f1", "unsafe_to_safe_rate", "safe_to_unsafe_rate"):
            numbers = [m["forced_matched"][field] for _, m, _ in entries if m["forced_matched"].get(field) is not None]
            values[field] = {"mean": statistics.mean(numbers) if numbers else None,
                             "std": statistics.stdev(numbers) if len(numbers) > 1 else None}
        result.append({"method": key[0], "family": key[1], "library": key[2], "augmentation": key[3],
                       "condition": key[4], "training_seeds": seeds, "runs": [p for _, _, p in entries], "metrics": values})
    write_json(args.output_dir / "summary.json", result)
    lines = ["# Four-RQ Results", "", "All automatic targets are derived from the internal six-class policy.",
             "Independent normative reasoning and unseen-class generalization remain unverified.", "",
             "| Method | Family | Library | Augmentation | Condition | Seeds | Accuracy | Macro-F1 | Unsafe-to-Safe |",
             "|---|---|---|---|---|---:|---:|---:|---:|"]
    def cell(value):
        if value["mean"] is None:
            return "NA"
        return f"{value['mean']:.4f}" + (f" +/- {value['std']:.4f}" if value["std"] is not None else "")
    for r in result:
        lines.append(f"| {r['method']} | {r['family']} | {r['library']} | {r['augmentation']} | {r['condition']} | {len(r['training_seeds'])} | " +
                     " | ".join(cell(r["metrics"][f]) for f in ("accuracy", "macro_f1", "unsafe_to_safe_rate")) + " |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("prepare")
    q.add_argument("--train", type=Path, default=HAZARD / "outputs/six_class_v1/train.jsonl")
    q.add_argument("--test", type=Path, default=HAZARD / "outputs/six_class_v1/test.jsonl")
    q.add_argument("--policy", type=Path, default=HAZARD / "config/hazard_context_policies_six_class_v1.json")
    q.add_argument("--inventory", type=Path, default=HAZARD / "config/rule_prototypes_six_class_v1.json")
    q.add_argument("--dev-fraction", type=float, default=0.15)
    q.add_argument("--split-seed", type=int, default=20261013)
    q.add_argument("--noise-seed", type=int, default=42)
    q.add_argument("--drop-rates", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    q.add_argument("--drop-label", choices=LABELS, default="rebar_work_unsafe")
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=data.prepare)

    def common(q):
        q.add_argument("--suite", type=Path, required=True)
        q.add_argument("--family", choices=["coarse", "atomic"], default="coarse")

    def neural(q):
        q.add_argument("--hazard-project-root", type=Path, default=HAZARD)
        q.add_argument("--model-cache-dir", type=Path)
        q.add_argument("--device", default="cuda")
        q.add_argument("--batch-size", type=int, default=16)
        q.add_argument("--pair-batch-size", type=int, default=32)

    def scoring(q):
        common(q)
        neural(q)
        q.add_argument("--method", choices=["rule", "lookup", "tfidf", *NEURAL], required=True)
        q.add_argument("--checkpoint", type=Path)
        q.add_argument("--train-config", type=Path)
        q.add_argument("--allow-legacy-checkpoint", action="store_true")
        q.add_argument("--library-name", default="base")
        q.add_argument("--external-library", type=Path)
        q.add_argument("--baseline-train-conditions", nargs="+", default=["clean"])

    q = commands.add_parser("train")
    common(q)
    neural(q)
    q.add_argument("--method", choices=NEURAL, required=True)
    q.add_argument("--augmentation", choices=["clean", "mixed"], default="clean")
    q.add_argument("--train-conditions", nargs="+", default=["clean", "d1_0.10", "d1_0.30", "d1_0.50", "d5_1.00"])
    q.add_argument("--validation-conditions", nargs="+", default=["clean", "d1_0.30"])
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--epochs", type=int, default=3)
    q.add_argument("--gradient-accumulation", type=int, default=1)
    q.add_argument("--pretrained-model", default="bert-base-uncased")
    q.add_argument("--learning-rate", type=float, default=2e-5)
    q.add_argument("--weight-decay", type=float, default=0.01)
    q.add_argument("--max-length", type=int, default=96)
    q.add_argument("--pair-max-length", type=int, default=192)
    q.add_argument("--projection-hidden-dim", type=int, default=512)
    q.add_argument("--projection-dim", type=int, default=256)
    q.add_argument("--dropout", type=float, default=0.1)
    q.add_argument("--share-backbone", action="store_true")
    q.add_argument("--anonymous-rules", action="store_true")
    q.add_argument("--alignment-loss", choices=["multi_positive", "balanced_bce"], default="multi_positive")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=train)

    q = commands.add_parser("evaluate")
    scoring(q)
    q.add_argument("--split", choices=["dev", "test"], default="test")
    q.add_argument("--conditions", nargs="+")
    q.add_argument("--predicted-dir", type=Path)
    q.add_argument("--calibration", type=Path)
    q.add_argument("--reviewed-gold", type=Path)
    q.add_argument("--save-top-k", type=int, default=5, help="0 saves full ranking; metrics always use full ranking")
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=evaluate)

    q = commands.add_parser("calibrate")
    scoring(q)
    q.add_argument("--condition", default="clean")
    q.add_argument("--max-risk", type=float, default=0.05)
    q.add_argument("--minimum-accepted", type=int, default=30)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=calibrate)

    q = commands.add_parser("prepare-predicted")
    common(q)
    q.add_argument("--reference", type=Path, default=HAZARD / "rule_graph_baseline/data/six_class_v1/test.json")
    q.add_argument("--graph-dir", type=Path, required=True)
    q.add_argument("--provenance", type=Path, required=True)
    q.add_argument("--min-worker-iou", type=float, default=0.5)
    q.add_argument("--allow-missing-images", action="store_true")
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=prepare_predicted)

    q = commands.add_parser("compare")
    q.add_argument("--results", nargs="+", type=Path, required=True)
    q.add_argument("--bootstrap-repeats", type=int, default=1000)
    q.add_argument("--seed", type=int, default=123)
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=compare)

    q = commands.add_parser("report")
    q.add_argument("--results", nargs="+", type=Path, required=True)
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=report)

    q = commands.add_parser("benchmark")
    scoring(q)
    q.add_argument("--libraries", nargs="+", default=["no_generic", "base", "add_paraphrases"])
    q.add_argument("--max-workers", type=int, default=128)
    q.add_argument("--repeats", type=int, default=5)
    q.add_argument("--warmup", type=int, default=1)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=benchmark)

    q = commands.add_parser("review-queue")
    common(q)
    q.add_argument("--library-name", default="base")
    q.add_argument("--external-library", type=Path)
    q.add_argument("--split", choices=["dev", "test"], default="test")
    q.add_argument("--condition", default="clean")
    q.add_argument("--predicted-dir", type=Path)
    q.add_argument("--max-per-label", type=int, default=20)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--output-dir", type=Path, required=True)
    q.set_defaults(func=review_queue)
    return p


def main():
    args = parser().parse_args()
    result = args.func(args)
    if args.command == "evaluate":
        result = {condition: {"accuracy": m["forced_matched"]["accuracy"],
                              "macro_f1": m["forced_matched"]["macro_f1"],
                              "unsafe_to_safe_rate": m["forced_matched"].get("unsafe_to_safe_rate"),
                              "mrr": m["retrieval"]["mrr"],
                              "accepted_coverage": m["accepted_coverage_all_workers"]}
                  for condition, m in result.items()}
    elif args.command in ("compare", "report"):
        result = {"groups": len(result), "output_dir": str(args.output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
