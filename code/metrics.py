"""Safety, retrieval and selective-prediction metrics with explicit denominators."""

import math
import random
from collections import Counter, defaultdict

from .data import LABELS


def softmax(values):
    maximum = max(values)
    exps = [math.exp(v - maximum) if math.isfinite(v) else 0.0 for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def rank_prediction(scores, candidates):
    if len(scores) != len(candidates) or not scores or any(not math.isfinite(s) for s in scores):
        raise ValueError("Candidate scores must be finite and match the nonempty library")
    ranked = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]["candidate_id"]))
    logits = [max((s for p, s in ranked if p["major_label"] == label), default=-math.inf)
              for label in LABELS]
    return class_prediction(logits), [{"candidate_id": p["candidate_id"], "semantic_id": p["semantic_id"],
                                      "text": p["text"], "major_label": p["major_label"], "score": s}
                                     for p, s in ranked]


def class_prediction(logits):
    probs = softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return {"pred_label": LABELS[order[0]], "confidence": probs[order[0]],
            "margin": probs[order[0]] - probs[order[1]], "probabilities": dict(zip(LABELS, probs))}


def classification(rows):
    if not rows:
        return {"samples": 0, "accuracy": None, "macro_f1": None}
    cm = {label: Counter() for label in LABELS}
    for r in rows:
        cm[r["true_label"]][r["pred_label"]] += 1
    per_class = {}
    for label in LABELS:
        tp = cm[label][label]
        support = sum(cm[label].values())
        fp = sum(cm[other][label] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    unsafe = [r for r in rows if r["true_label"] != "safe"]
    safe = [r for r in rows if r["true_label"] == "safe"]
    return {"samples": len(rows),
            "accuracy": sum(r["true_label"] == r["pred_label"] for r in rows) / len(rows),
            "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
            "unsafe_to_safe_rate": sum(r["pred_label"] == "safe" for r in unsafe) / len(unsafe) if unsafe else None,
            "safe_to_unsafe_rate": sum(r["pred_label"] != "safe" for r in safe) / len(safe) if safe else None,
            "per_class": per_class, "confusion": {k: dict(v) for k, v in cm.items()}}


def retrieval(rows, independent=False):
    # Skip classifiers and references absent from the active library, report both counts.
    scored = [r for r in rows if r.get("ranking") is not None]
    eligible = [r for r in scored if r.get("relevant_ids")]
    result = {"scored_samples": len(scored), "eligible_samples": len(eligible),
              "no_active_positive_samples": len(scored) - len(eligible),
              "reference_kind": "independent_review" if independent else "policy_or_class_derived_not_norm_gold"}
    for k in (1, 3, 5):
        result[f"hit_at_{k}"] = sum(bool(set(r["relevant_ids"]) & {
            p["candidate_id"] for p in r["ranking"][:k]}) for r in eligible) / len(eligible) if eligible else None
        result[f"recall_at_{k}"] = sum(len(set(r["relevant_ids"]) & {
            p["candidate_id"] for p in r["ranking"][:k]}) / len(set(r["relevant_ids"]))
            for r in eligible) / len(eligible) if eligible else None
    rr = []
    for r in eligible:
        relevant = set(r["relevant_ids"])
        rank = next((i for i, p in enumerate(r["ranking"], 1) if p["candidate_id"] in relevant), None)
        rr.append(1 / rank if rank else 0.0)
    result["mrr"] = sum(rr) / len(rr) if rr else None
    empty = [r for r in scored if not r.get("relevant_ids")]
    result["no_positive_abstention_rate"] = sum(r.get("abstained", False) for r in empty) / len(empty) if empty else None
    return result


def summarize(rows, independent=False):
    matched = [r for r in rows if r["matched"]]
    accepted = [r for r in matched if not r.get("abstained", False)]
    unsafe = [r for r in rows if r["true_label"] != "safe"]
    unavailable = [r for r in matched if not r["target_in_library"]]
    return {"reference_workers": len(rows), "matched_workers": len(matched),
            "matched_coverage": len(matched) / len(rows) if rows else None,
            "accepted_coverage_all_workers": len(accepted) / len(rows) if rows else None,
            "forced_matched": classification(matched), "selective_accepted": classification(accepted),
            "retrieval": retrieval(matched, independent),
            "out_of_library_workers": len(unavailable),
            "out_of_library_false_acceptance_rate": sum(not r.get("abstained", False) for r in unavailable) / len(unavailable) if unavailable else None,
            "operational_unsafe_not_flagged_rate": sum(not r["matched"] or r.get("abstained", False) or
                r["pred_label"] == "safe" for r in unsafe) / len(unsafe) if unsafe else None,
            "operational_note": "missing workers and abstentions are not safe predictions, but do not produce an automatic unsafe flag"}


def calibrate_threshold(rows, max_risk, minimum_accepted):
    if not 0 <= max_risk <= 1 or minimum_accepted < 1:
        raise ValueError("Invalid risk or minimum accepted count")
    matched = [r for r in rows if r["matched"]]
    best = None
    for threshold in sorted({0.0, 1.000001, *(r["margin"] for r in matched)}):
        accepted = [r for r in matched if r["margin"] >= threshold]
        risk = sum(r["pred_label"] != r["true_label"] for r in accepted) / len(accepted) if accepted else 0.0
        if len(accepted) >= minimum_accepted and risk <= max_risk:
            candidate = {"threshold": threshold, "accepted": len(accepted), "empirical_risk": risk}
            if best is None or candidate["accepted"] > best["accepted"]:
                best = candidate
    return best or {"threshold": 1.000001, "accepted": 0, "empirical_risk": None,
                    "reason": "No dev threshold meets requested empirical risk and minimum sample count"}


def paired_bootstrap(left, right, repeats=1000, seed=123):
    a, b = {r["sample_id"]: r for r in left}, {r["sample_id"]: r for r in right}
    if set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired runs require unique identical IDs")
    images = defaultdict(list)
    for sid, r in a.items():
        other = b[sid]
        for key in ("image_id", "true_label", "matched", "scene_sha256"):
            if r.get(key) != other.get(key):
                raise ValueError("Different paired evidence/reference: " + key)
        images[r["image_id"]].append(sid)
    keys = sorted(images)
    rng = random.Random(seed)
    deltas = []
    groups = {image: (sum(a[s]["matched"] and a[s]["pred_label"] == a[s]["true_label"] for s in ids),
                      sum(b[s]["matched"] and b[s]["pred_label"] == b[s]["true_label"] for s in ids),
                      len(ids)) for image, ids in images.items()}
    for _ in range(repeats):
        values = [groups[rng.choice(keys)] for _ in keys]
        deltas.append(sum(v[0] - v[1] for v in values) / sum(v[2] for v in values))
    deltas.sort()
    return {"metric": "all_reference_worker_correct_rate_left_minus_right",
            "point": (sum(v[0] - v[1] for v in groups.values()) / len(left)),
            "ci95": [deltas[int(0.025 * (repeats - 1))], deltas[int(0.975 * (repeats - 1))]],
            "bootstrap_unit": "image", "repeats": repeats, "seed": seed}
