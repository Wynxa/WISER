"""Dependency-free protocol preparation; target fields never become model inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAZARD = ROOT / "projects" / "hazard_reasoning"
LABELS = ["dangerous_approach_to_excavator_unsafe", "general_work_unsafe",
          "mechanical_operation_unsafe", "rebar_work_unsafe", "safe", "scaffold_work_unsafe"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_rows(rows):
    ids = [row["sample_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty data or duplicate sample_id")
    return rows


def parse_scene(text):
    marker = "worker [OBSERVED_RELATIONS] "
    if not text.startswith(marker):
        raise ValueError("Expected observed-only compact scene_text: " + text[:100])
    tail = text[len(marker):].strip()
    if tail in ("none", ""):
        return []
    relations = []
    for fact in tail.split(";"):
        parts = fact.strip().split(" ", 1)
        if len(parts) != 2:
            raise ValueError("Malformed relation: " + fact)
        relations.append({"predicate": parts[0], "object_name": parts[1]})
    return relations


def scene_text(relations):
    facts = " ; ".join(f"{r['predicate']} {r['object_name']}" for r in relations)
    return "worker [OBSERVED_RELATIONS] " + (facts or "none")


def context_for(relations, policy):
    pairs = {(r["predicate"], r["object_name"]) for r in relations}
    for ctx in sorted(policy["contexts"], key=lambda r: int(r["priority"]), reverse=True):
        triggers = {(r["predicate"], r["object_name"]) for r in ctx["trigger_relations"]}
        mode = ctx["match_mode"]
        if mode not in ("default", "any", "all"):
            raise ValueError("Unknown context match mode")
        if mode == "default" or (mode == "any" and triggers & pairs) or (
                mode == "all" and triggers <= pairs):
            return ctx
    raise ValueError("No matching/default context")


def policy_state(relations, policy):
    ctx = context_for(relations, policy)
    if ctx.get("ppe_match", "all") != "all":
        raise ValueError("This protocol implements the existing all-required-PPE policy only")
    worn = {r["object_name"] for r in relations if r["predicate"] == "wearing"}
    missing = [ppe for ppe in ctx["required_ppe"] if ppe not in worn]
    label = (ctx.get("unsafe_label") or ctx["context_id"] + "_unsafe") if missing else "safe"
    if label not in LABELS:
        raise ValueError("Policy escaped the frozen six-class label space: " + label)
    return label, ctx, missing


def observed_rule_text(relations, policy):
    # Select from OBSERVED evidence, never from the clean/target context_id.
    ctx = context_for(relations, policy)
    return f"Workers engaged in {ctx['context_text']} shall wear " + ", ".join(ctx["required_ppe"]) + "."


def split_images(train, test, dev_fraction, seed):
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between 0 and 1")
    train_images = {r["image_id"] for r in train}
    if train_images & {r["image_id"] for r in test}:
        raise ValueError("Original train/test image overlap")
    if len(train_images) < 2:
        raise ValueError("At least two training images required")
    images = sorted(train_images)
    random.Random(seed).shuffle(images)
    count = min(len(images) - 1, max(1, round(len(images) * dev_fraction)))
    dev_images = set(images[:count])
    return ([r for r in train if r["image_id"] not in dev_images],
            [r for r in train if r["image_id"] in dev_images], test)


def canonical_candidates(inventory):
    protos = inventory["prototypes"]
    if len({p["prototype_id"] for p in protos}) != len(protos):
        raise ValueError("Duplicate prototype_id")
    candidates = []
    for p in protos:
        if p["major_label"] not in LABELS or int(p["major_label_id"]) != LABELS.index(p["major_label"]):
            raise ValueError("Inventory uses another label map")
        candidates.append({**p, "candidate_id": p["prototype_id"],
                           "semantic_id": p["prototype_id"], "origin": "existing_state_prototype",
                           "generic": "violates the required ppe rule" in p["text"]})
    return candidates


def atomic_candidates(policy):
    candidates = []
    for ctx in policy["contexts"]:
        name, cid = ctx["context_text"], ctx["context_id"]
        unsafe = ctx.get("unsafe_label") or cid + "_unsafe"
        base = {"context_id": cid, "origin": "internal_policy_not_independent_norm",
                "generic": False}
        if unsafe == "safe":
            # Retained auxiliary classes are collapsed to Safe by the original task.
            candidates.append({**base, "candidate_id": cid + "::retained_context",
                               "text": f"worker in {name}", "major_label": "safe",
                               "kind": "auxiliary_label_projection", "ppe": None})
        else:
            for ppe in ctx["required_ppe"]:
                candidates.append({**base, "candidate_id": cid + "::missing::" + ppe,
                                   "text": f"worker in {name} is not wearing required {ppe}",
                                   "major_label": unsafe, "kind": "missing_ppe", "ppe": ppe})
            candidates.append({**base, "candidate_id": cid + "::compliant",
                               "text": f"worker in {name} wears all required items: " + ", ".join(ctx["required_ppe"]),
                               "major_label": "safe", "kind": "compliant", "ppe": None})
    for p in candidates:
        p["semantic_id"] = p["candidate_id"]
        p["major_label_id"] = LABELS.index(p["major_label"])
    return candidates


def paraphrase_candidates(candidates, policy):
    contexts = {c["context_id"]: c for c in policy["contexts"]}
    result = copy.deepcopy(candidates)
    for i, p in enumerate(result):
        ctx = contexts[p["context_id"]]
        name, items = ctx["context_text"], ", ".join(ctx["required_ppe"])
        if p.get("kind") == "missing_ppe":
            p["text"] = f"during {name}, the worker lacks required {p['ppe']}"
        elif p.get("kind") == "auxiliary_label_projection":
            p["text"] = f"a person engaged in {name}"
        elif p["major_label"] == "safe":
            p["text"] = (f"during {name}, every mandatory item is worn by the worker: {items}" if i % 2 else
                         f"a worker performing {name} has all required equipment on: {items}")
        else:
            variants = [f"during {name}, at least one mandatory item is absent: {items}",
                        f"a worker doing {name} lacks one or more required items: {items}",
                        f"not all required items are worn during {name}: {items}"]
            p["text"] = variants[i % 3]
    return result


def make_libraries(candidates, policy, drop_label):
    para = paraphrase_candidates(candidates, policy)
    extra = copy.deepcopy(para)
    for p in extra:
        p["candidate_id"] += "::added_paraphrase"
    return {"base": candidates, "paraphrase": para,
            "no_generic": [p for p in candidates if not p["generic"]],
            "paraphrase_no_generic": [p for p in para if not p["generic"]],
            "add_paraphrases": candidates + extra,
            "drop_label": [p for p in candidates if p["major_label"] != drop_label],
            "restore": copy.deepcopy(candidates)}


def prepare_case(row, policy, coarse, atomic, split):
    relations = parse_scene(row["scene_text"])
    predicted, ctx, missing = policy_state(relations, policy)
    if row["major_label"] not in LABELS or int(row["major_label_id"]) != LABELS.index(row["major_label"]):
        raise ValueError("Source labels differ from frozen six-class map")
    positives = list(row["positive_prototype_ids"])
    known = {p["candidate_id"]: p for p in coarse}
    if not positives or any(pid not in known or known[pid]["major_label"] != row["major_label"] for pid in positives):
        raise ValueError("Invalid source positives")
    fine = [p["semantic_id"] for p in atomic if p["context_id"] == ctx["context_id"] and (
        p["kind"] == "auxiliary_label_projection" or
        (p["kind"] == "compliant" and not missing) or
        (p["kind"] == "missing_ppe" and p["ppe"] in missing))]
    return {"sample_id": row["sample_id"], "image_id": row["image_id"], "worker_id": row.get("worker_id"),
            "split": split, "true_label": row["major_label"], "true_label_id": int(row["major_label_id"]),
            "relations": relations, "scene_text": scene_text(relations), "matched": True,
            "clean_context_id": ctx["context_id"], "clean_missing_ppe": missing,
            "policy_label": predicted, "policy_agrees": predicted == row["major_label"],
            "relevant": {"coarse": positives, "atomic": fine}}


def conditions(rates):
    for rate in rates:
        if not 0 < rate < 1:
            raise ValueError("D1 rates must be strictly between 0 and 1")
    return {"clean": {"kind": "clean", "rate": 0.0},
            **{f"d1_{r:.2f}": {"kind": "edge_drop", "rate": r} for r in rates},
            "d5_1.00": {"kind": "context_drop", "rate": 1.0}}


def corrupt(case, spec, seed):
    result = copy.deepcopy(case)
    if not case.get("matched", True):
        return result
    original = case["relations"]
    h = digest([seed, case["sample_id"], spec["kind"]])
    rng = random.Random(int(h[:16], 16))
    if spec["kind"] == "edge_drop":
        kept = [r for r in original if rng.random() >= spec["rate"]]
    elif spec["kind"] == "context_drop":
        kept = [r for r in original if r["predicate"] == "wearing"]
    elif spec["kind"] == "clean":
        kept = original
    else:
        raise ValueError("Unsupported corruption kind")
    result["relations"] = copy.deepcopy(kept)
    result["scene_text"] = scene_text(kept)
    result["deleted_edges"] = len(original) - len(kept)
    result["original_edges"] = len(original)
    return result


def information_loss(rows):
    grouped = defaultdict(Counter)
    for row in rows:
        if row.get("matched", True):
            grouped[row["scene_text"]][row["true_label"]] += 1
    count = sum(sum(c.values()) for c in grouped.values())
    return {"matched_samples": count, "unique_observed_texts": len(grouped),
            "conflicting_texts": sum(len(c) > 1 for c in grouped.values()),
            "samples_in_conflicting_texts": sum(sum(c.values()) for c in grouped.values() if len(c) > 1),
            "retrospective_exact_text_accuracy_ceiling": sum(max(c.values()) for c in grouped.values()) / count if count else None,
            "ceiling_is_test_label_diagnostic_not_trainable_baseline": True}


def load_cases(suite, split, condition="clean", predicted=None):
    suite = Path(suite)
    manifest = read_json(suite / "manifest.json")
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    if file_hash(suite / (split + ".jsonl")) != manifest["split_sha256"][split]:
        raise ValueError("Prepared cases changed after protocol freeze")
    if condition == "predicted":
        if split != "test" or predicted is None:
            raise ValueError("Predicted evidence is supported only on test with --predicted-dir")
        base = unique_rows(read_jsonl(suite / "test.jsonl"))
        evidence = unique_rows(read_jsonl(Path(predicted) / "evidence.jsonl"))
        lookup = {r["sample_id"]: r for r in evidence}
        if set(lookup) != {r["sample_id"] for r in base}:
            raise ValueError("Predicted evidence IDs differ from suite test IDs")
        result = []
        for case in base:
            e = lookup[case["sample_id"]]
            if e["true_label"] != case["true_label"] or e["image_id"] != case["image_id"]:
                raise ValueError("Predicted reference differs from suite reference")
            row = copy.deepcopy(case)
            row.update({"matched": e["matched"], "relations": e["relations"],
                        "scene_text": scene_text(e["relations"]) if e["matched"] else None,
                        "matched_iou": e.get("matched_iou"), "source": e["source"]})
            result.append(row)
        return result
    if split not in ("train", "dev", "test") or condition not in manifest["conditions"]:
        raise ValueError("Unknown split or condition")
    return [corrupt(r, manifest["conditions"][condition], manifest["noise_seed"])
            for r in unique_rows(read_jsonl(suite / (split + ".jsonl")))]


def prepare(args):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; protocols must not be silently overwritten")
    source_train = unique_rows(read_jsonl(args.train))
    source_test = unique_rows(read_jsonl(args.test))
    if {r["sample_id"] for r in source_train} & {r["sample_id"] for r in source_test}:
        raise ValueError("Original train/test sample overlap")
    policy = read_json(args.policy)
    coarse = canonical_candidates(read_json(args.inventory))
    atomic = atomic_candidates(policy)
    splits = split_images(source_train, source_test, args.dev_fraction, args.split_seed)
    case_splits = {name: [prepare_case(r, policy, coarse, atomic, name) for r in rows]
                   for name, rows in zip(("train", "dev", "test"), splits)}
    if any(not r["policy_agrees"] for rows in case_splits.values() for r in rows):
        raise ValueError("Clean policy/source labels disagree; investigate rather than silently changing targets")
    specs = conditions(args.drop_rates)
    for name, rows in case_splits.items():
        if set(LABELS) != {r["true_label"] for r in rows}:
            raise ValueError(f"{name} does not contain all six classes; choose another predeclared split seed")
        write_jsonl(output / (name + ".jsonl"), rows)
    write_json(output / "policy.json", policy)
    for family, candidates in (("coarse", coarse), ("atomic", atomic)):
        for name, rows in make_libraries(candidates, policy, args.drop_label).items():
            write_json(output / "libraries" / family / (name + ".json"), rows)
    write_json(output / "manifest.json", {
        "protocol_version": 1, "labels": LABELS, "split_seed": args.split_seed,
        "noise_seed": args.noise_seed, "dev_fraction": args.dev_fraction, "conditions": specs,
        "source_sha256": {k: file_hash(v) for k, v in {"train": args.train, "test": args.test,
                               "policy": args.policy, "inventory": args.inventory}.items()},
        "source_paths": {"train": str(args.train.resolve()), "test": str(args.test.resolve())},
        "drop_label": args.drop_label,
        "split_summary": {name: {"workers": len(rows), "images": len({r['image_id'] for r in rows}),
                                 "label_counts": dict(Counter(r['true_label'] for r in rows))}
                          for name, rows in case_splits.items()},
        "test_status": "original_test_previously_used_in_historical_model_selection_not_fresh_holdout",
        "reference": "internal_policy_teacher_not_independent_safety_or_norm_truth",
        "library_sizes": {"coarse": len(coarse), "atomic": len(atomic)},
        "source_code_sha256": file_hash(__file__),
        "split_sha256": {name: file_hash(output / (name + ".jsonl")) for name in case_splits},
        "policy_sha256": file_hash(output / "policy.json"),
        "libraries_sha256": {f"{family}/{name}": file_hash(output / "libraries" / family / (name + ".json"))
                              for family in ("coarse", "atomic")
                              for name in make_libraries(coarse if family == "coarse" else atomic, policy, args.drop_label)},
    })
    diagnostics = {}
    for condition, spec in specs.items():
        rows = [corrupt(r, spec, args.noise_seed) for r in case_splits["test"]]
        diagnostics[condition] = {**information_loss(rows),
            "actual_deleted_edges": sum(r["deleted_edges"] for r in rows),
            "original_edges": sum(r["original_edges"] for r in rows),
            "corrupted_rule_disagreement_rate": sum(policy_state(r["relations"], policy)[0] != r["true_label"]
                                                     for r in rows) / len(rows)}
    write_json(output / "input_diagnostics.json", diagnostics)
    return read_json(output / "manifest.json")
