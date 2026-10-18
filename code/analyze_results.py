"""Audit synchronized results and derive paired evidence without model weights."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from AiC_resubmission.knowledge_decoupling_experiment import data


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def averages(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values), "maximum": max(values)}


def confusion_metrics(truth, pred):
    matrix = np.bincount(truth * 7 + pred, minlength=42).reshape(6, 7)
    diagonal = np.diag(matrix[:, :6])
    recall = np.divide(diagonal, matrix.sum(1), out=np.zeros(6), where=matrix.sum(1) > 0)
    precision = np.divide(diagonal, matrix[:, :6].sum(0), out=np.zeros(6), where=matrix[:, :6].sum(0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=precision + recall > 0)
    safe = data.LABELS.index("safe")
    return {"accuracy": float(diagonal.sum() / len(truth)), "macro_f1": float(f1.mean()),
            "unsafe_to_safe_rate": float(((truth != safe) & (pred == safe)).sum() / (truth != safe).sum()),
            "safe_to_unsafe_rate": float(((truth == safe) & (pred != safe)).sum() / (truth == safe).sum())}


class Analysis:
    def __init__(self, root, output, repeats, seed):
        self.root, self.output = root, output
        self.suite = root / "suite_v2"
        self.artifacts = root / "aic_knowledge_decoupling"
        self.manifest = data.read_json(self.suite / "manifest.json")
        self.suite_hash = data.file_hash(self.suite / "manifest.json")
        reference = data.load_cases(self.suite, "test")
        self.ids = sorted(r["sample_id"] for r in reference)
        self.index = {sid: i for i, sid in enumerate(self.ids)}
        by_id = {r["sample_id"]: r for r in reference}
        self.truth = np.array([by_id[sid]["true_label_id"] for sid in self.ids])
        images = sorted({r["image_id"] for r in reference})
        image_map = {image: i for i, image in enumerate(images)}
        self.image_index = np.array([image_map[by_id[sid]["image_id"]] for sid in self.ids])
        self.image_counts = np.bincount(self.image_index, minlength=len(images))
        self.unsafe = self.truth != data.LABELS.index("safe")
        self.unsafe_image_counts = np.bincount(self.image_index, weights=self.unsafe, minlength=len(images))
        self.bootstrap = np.random.default_rng(seed).multinomial(len(images), np.full(len(images), 1 / len(images)), size=repeats).astype(float)
        self.bootstrap_n = self.bootstrap @ self.image_counts
        self.bootstrap_unsafe_n = self.bootstrap @ self.unsafe_image_counts
        self.references = {}
        for condition in self.manifest["conditions"]:
            rows = data.load_cases(self.suite, "test", condition)
            self.references[condition] = {r["sample_id"]: (r["image_id"], r["true_label"], data.digest(r["scene_text"]), r)
                                          for r in rows}
        train_texts = {r["scene_text"] for r in data.load_cases(self.suite, "train")}
        self.unseen = np.array([by_id[sid]["scene_text"] not in train_texts for sid in self.ids])
        self.runs, self.predictions, self.audit, self.metrics_rows = {}, {}, [], []
        self.repeats, self.seed = repeats, seed

    def read_predictions(self, key, path, condition, family, library):
        n = len(self.ids)
        pred = np.full(n, 6, dtype=np.int8)
        hit = np.full(n, np.nan)
        recall5 = np.full(n, np.nan)
        accepted = np.zeros(n, dtype=bool)
        seen, active = set(), {p["candidate_id"]: p for p in library}
        reference = self.references[condition]
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                r = json.loads(line)
                sid = r["sample_id"]
                if sid in seen or sid not in reference:
                    raise ValueError("Duplicate/unknown prediction ID: " + sid)
                seen.add(sid)
                image, label, scene_hash, case = reference[sid]
                if (r["image_id"], r["true_label"], r["scene_sha256"]) != (image, label, scene_hash):
                    raise ValueError("Predictions differ from frozen evidence: " + str(path))
                if not r["matched"]:
                    raise ValueError("Unexpected unmatched worker in clean/synthetic GT evaluation")
                i = self.index[sid]
                pred[i] = data.LABELS.index(r["pred_label"])
                accepted[i] = not r["abstained"]
                expected = {pid for pid, p in active.items() if p["semantic_id"] in case["relevant"][family]}
                if set(r["relevant_ids"]) != expected:
                    raise ValueError("Wrong relevance reference: " + str(path))
                if r["ranking"] is not None and expected:
                    ranks = [p["candidate_id"] for p in r["ranking"]]
                    if len(ranks) < min(5, len(library)):
                        raise ValueError("Saved ranking insufficient for Recall@5 audit")
                    hit[i] = float(ranks[0] in expected)
                    recall5[i] = len(set(ranks[:5]) & expected) / len(expected)
        if seen != set(self.ids):
            raise ValueError("Truncated/missing prediction rows: " + str(path))
        arrays = {"pred": pred, "correct": (pred == self.truth).astype(float), "hit_at_1": hit,
                  "recall_at_5": recall5, "accepted": accepted,
                  "unsafe_to_safe": (self.unsafe & (pred == data.LABELS.index("safe"))).astype(float)}
        self.predictions[(key, condition)] = arrays
        return arrays

    def load(self):
        for section in ("main", "anonymous", "atomic"):
            for run_file in sorted((self.artifacts / section / "results").glob("*/run.json")):
                self.load_run(section, run_file)
        for run_file in sorted((self.artifacts / "main/updates").glob("*/*/run.json")):
            self.load_run("updates", run_file)

    def load_run(self, section, run_file):
        run = data.read_json(run_file)
        if run["suite_manifest_sha256"] != self.suite_hash or run["split"] != "test":
            raise ValueError("Run uses another suite/split")
        library_path = self.suite / "libraries" / run["family"] / (run["library"] + ".json")
        if data.file_hash(library_path) != run["library_sha256"]:
            raise ValueError("Run library does not match synchronized suite")
        library = data.read_json(library_path)
        metric = data.read_json(run_file.parent / "metrics.json")
        method = run["method"] + ("_anonymous" if run.get("model_variant") == "anonymous_rules" else "")
        seed = run["training_seed"]
        key = f"{section}/{run_file.parent.parent.name}/{run_file.parent.name}" if section == "updates" else f"{section}/{run_file.parent.name}"
        self.runs[key] = {"metadata": run, "metrics": metric, "method": method, "section": section, "path": str(run_file.parent)}
        if set(metric) != set(run["conditions"]):
            raise ValueError("Run condition metadata/metrics differ")
        checkpoint_name = run_file.parent.parent.name if section == "updates" else run_file.parent.name
        checkpoint = self.artifacts / ("main" if section == "updates" else section) / "checkpoints" / checkpoint_name
        config = data.read_json(checkpoint / "train_config.json") if (checkpoint / "train_config.json").exists() else None
        history = data.read_json(checkpoint / "history.json") if (checkpoint / "history.json").exists() else None
        history_complete = config is not None and history is not None and len(history) == config["epochs"]
        if seed is not None and (not history_complete or config["seed"] != seed or config["suite_manifest_sha256"] != self.suite_hash):
            raise ValueError("Evaluation refers to incomplete/inconsistent training: " + key)
        for condition, m in metric.items():
            arrays = self.read_predictions(key, run_file.parent / condition / "predictions.jsonl", condition, run["family"], library)
            recalculated = confusion_metrics(self.truth, arrays["pred"])
            for field, value in recalculated.items():
                if abs(value - m["forced_matched"][field]) > 1e-9:
                    raise ValueError("Classification metric does not reproduce: " + key + "/" + field)
            for field in ("hit_at_1", "recall_at_5"):
                values = arrays[field]
                if np.isfinite(values).any() and abs(float(np.nanmean(values)) - m["retrieval"][field]) > 1e-9:
                    raise ValueError("Retrieval metric does not reproduce: " + key + "/" + field)
            row = {"section": section, "run": key, "method": method, "family": run["family"],
                   "augmentation": run["augmentation"], "seed": seed, "library": run["library"], "condition": condition,
                   **recalculated, "hit_at_1": m["retrieval"]["hit_at_1"], "recall_at_5": m["retrieval"]["recall_at_5"],
                   "mrr": m["retrieval"]["mrr"], "accepted_coverage": m["accepted_coverage_all_workers"],
                   "out_of_library_false_acceptance_rate": m["out_of_library_false_acceptance_rate"]}
            self.metrics_rows.append(row)
        self.audit.append({"run": key, "conditions": len(metric), "prediction_rows_per_condition": len(self.ids),
                           "suite_and_library_verified": True, "classification_and_topk_metrics_reproduced": True,
                           "history_complete": history_complete, "training_seed": seed,
                           "parameters": config.get("parameters") if config else None,
                           "effective_batch_size": config.get("effective_batch_size") if config else None,
                           "epochs": config.get("epochs") if config else None,
                           "token_truncations": run.get("token_truncations"),
                           "checkpoint_sha256": run["scorer"].get("checkpoint_sha256")})

    def bootstrap_delta(self, left, right, condition, metric="correct"):
        values = []
        for a, b in zip(left, right):
            va = self.predictions[(a, condition)][metric]
            vb = self.predictions[(b, condition)][metric]
            if not np.isfinite(va).all() or not np.isfinite(vb).all():
                raise ValueError("Pair metric requires the same eligible reference workers")
            values.append(va - vb)
        delta = np.mean(values, axis=0)
        image_delta = np.bincount(self.image_index, weights=delta, minlength=len(self.image_counts))
        denominator = self.bootstrap_unsafe_n if metric == "unsafe_to_safe" else self.bootstrap_n
        boot = (self.bootstrap @ image_delta) / denominator
        divisor = self.unsafe.sum() if metric == "unsafe_to_safe" else len(self.ids)
        return {"difference": float(delta.sum() / divisor), "image_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "per_seed_differences": [float(v.sum() / divisor) for v in values],
                "conditioning": "image resampling conditional on the available trained seeds; not independent training-seed inference"}

    def paired(self):
        results = []
        for augmentation in ("clean", "mixed"):
            left = [f"main/coarse_dual_{augmentation}_s{s}" for s in (42, 43, 44)]
            for other, section, tag in (("scene", "main", "scene"), ("learned_prototypes", "main", "learned_prototypes"),
                                        ("dual_anonymous", "anonymous", "dual_anonymous"), ("cross", "main", "cross")):
                right = [f"{section}/coarse_{tag}_{augmentation}_s{s}" for s in (42, 43, 44)]
                for condition in self.manifest["conditions"]:
                    results.append({"comparison": "dual_minus_" + other, "augmentation": augmentation, "condition": condition,
                                    "metric": "accuracy", **self.bootstrap_delta(left, right, condition)})
        for condition in self.manifest["conditions"]:
            a = [f"main/coarse_dual_mixed_s{s}" for s in (42, 43, 44)]
            b = [f"main/coarse_dual_clean_s{s}" for s in (42, 43, 44)]
            for name, field in (("accuracy", "correct"), ("unsafe_to_safe_rate", "unsafe_to_safe")):
                results.append({"comparison": "dual_mixed_minus_clean_training", "augmentation": "mixed_vs_clean",
                                "condition": condition, "metric": name, **self.bootstrap_delta(a, b, condition, field)})
        for variant in ("base", "paraphrase", "paraphrase_no_generic"):
            a = [f"updates/coarse_dual_mixed_s42/{variant}"]
            b = [f"updates/coarse_cross_mixed_s42/{variant}"]
            results.append({"comparison": "dual_minus_cross_library_update", "augmentation": variant,
                            "condition": "clean", "metric": "accuracy", **self.bootstrap_delta(a, b, "clean")})
        for augmentation in ("clean", "mixed"):
            for other in ("learned_prototypes", "cross"):
                for condition in ("clean", "d1_0.30", "d1_0.50"):
                    a = [f"atomic/atomic_dual_{augmentation}_s{s}" for s in (42, 43, 44)]
                    b = [f"atomic/atomic_{other}_{augmentation}_s{s}" for s in (42, 43, 44)]
                    for field in ("hit_at_1", "recall_at_5"):
                        results.append({"comparison": "atomic_dual_minus_" + other,
                                        "augmentation": augmentation + "_three_seed",
                                        "condition": condition, "metric": field,
                                        **self.bootstrap_delta(a, b, condition, field)})
        return results

    def group_summary(self):
        groups = defaultdict(list)
        for row in self.metrics_rows:
            groups[tuple(row[k] for k in ("section", "method", "family", "augmentation", "library", "condition"))].append(row)
        results = []
        for key, rows in sorted(groups.items()):
            r = dict(zip(("section", "method", "family", "augmentation", "library", "condition"), key))
            r["n"] = len(rows)
            r["seeds"] = [v["seed"] for v in rows]
            for field in ("accuracy", "macro_f1", "unsafe_to_safe_rate", "safe_to_unsafe_rate", "hit_at_1", "recall_at_5", "mrr"):
                values = [v[field] for v in rows if v[field] is not None]
                r[field] = averages(values) if values else None
            results.append(r)
        return results

    def completeness(self):
        expected = {}
        for section, methods, family in (("main", ["dual", "scene", "learned_prototypes", "cbert", "cross"], "coarse"),
                                         ("anonymous", ["dual_anonymous"], "coarse"),
                                         ("atomic", ["dual", "scene", "learned_prototypes", "cross"], "atomic")):
            keys = [f"{section}/{family}_{method}_{aug}_s{seed}" for method in methods for aug in ("clean", "mixed") for seed in (42, 43, 44)]
            expected[section] = {"expected_neural_evaluations": len(keys), "complete_neural_evaluations": sum(k in self.runs for k in keys),
                                 "missing": [k for k in keys if k not in self.runs]}
        expected["updates"] = {"expected": 28, "complete": sum(r['section'] == 'updates' for r in self.runs.values())}
        expected["benchmarks"] = {"expected": 3, "complete": len(list((self.artifacts / 'main/benchmarks').glob('*.json')))}
        return expected

    def benchmark_table(self):
        rows = []
        for path in sorted((self.artifacts / "main/benchmarks").glob("*.json")):
            d = data.read_json(path)
            for r in d["libraries"]:
                rows.append({"method": d["method"], "library": r["library"], "candidates": r["candidates"], "workers": r["workers"],
                             "online_seconds": r["online_seconds"]["mean"], "online_std_seconds": r["online_seconds"]["std"],
                             "workers_per_second": r["workers_per_second"], "precompute_seconds": r["offline_candidate_precompute_seconds"],
                             "peak_cuda_bytes": r["peak_cuda_allocated_bytes"], "device": d["device"], "batch_size": d["batch_size"]})
        return rows

    def safe_retrieval(self):
        results = []
        for method in ("dual", "cross"):
            for variant in ("base", "paraphrase", "paraphrase_no_generic"):
                values = self.predictions[(f"updates/coarse_{method}_mixed_s42/{variant}", "clean")]
                for name, mask in (("safe", ~self.unsafe), ("unsafe", self.unsafe)):
                    results.append({"method": method, "library": variant, "stratum": name, "workers": int(mask.sum()),
                                    "class_accuracy": float(values['correct'][mask].mean()),
                                    "state_hit_at_1": float(values['hit_at_1'][mask].mean())})
        return results

    def unseen_texts(self):
        rows = []
        for key, run in self.runs.items():
            if run["section"] not in ("main", "anonymous"):
                continue
            arrays = self.predictions[(key, "clean")]
            rows.append({"section": run["section"], "run": key, "method": run["method"],
                         "augmentation": run["metadata"]["augmentation"], "seed": run["metadata"]["training_seed"],
                         "unseen_workers": int(self.unseen.sum()),
                         "accuracy_unseen": float(arrays["correct"][self.unseen].mean()),
                         "seen_workers": int((~self.unseen).sum()),
                         "accuracy_seen": float(arrays["correct"][~self.unseen].mean())})
        return rows

    def winning_prototypes(self):
        rows = []
        for method in ("dual", "learned_prototypes", "cross"):
            for variant in ("base", "no_generic"):
                path = self.artifacts / "main/updates" / f"coarse_{method}_mixed_s42" / variant / "clean/predictions.jsonl"
                counts = Counter()
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        r = json.loads(line)
                        counts[(r["true_label"], r["ranking"][0]["candidate_id"], r["pred_label"])] += 1
                rows.extend({"method": method, "library": variant, "true_label": label, "top_candidate": cid,
                             "pred_label": pred, "workers": count} for (label, cid, pred), count in sorted(counts.items()))
        return rows

    def plots(self, grouped, benchmark):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                             "pdf.fonttype": 42, "svg.fonttype": "none"})
        colors = {"dual": "#0072B2", "scene": "#009E73", "learned_prototypes": "#D55E00", "dual_anonymous": "#CC79A7"}
        labels = {"dual": "Dual", "scene": "Scene-only", "learned_prototypes": "Learned prototypes", "dual_anonymous": "Anonymous dual"}
        markers = {"dual": "o", "scene": "s", "learned_prototypes": "^", "dual_anonymous": "D"}
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        for ax, aug in zip(axes, ("clean", "mixed")):
            for method in colors:
                section = "anonymous" if method == "dual_anonymous" else "main"
                values = [next(r for r in grouped if r['section'] == section and r['method'] == method and
                               r['augmentation'] == aug and r['condition'] == c)['accuracy'] for c in ('clean','d1_0.10','d1_0.30','d1_0.50')]
                ax.errorbar([0, .1, .3, .5], [100*v['mean'] for v in values], yerr=[100*(v['std'] or 0) for v in values],
                            color=colors[method], marker=markers[method], label=labels[method], capsize=3)
            ax.set(title=f"{aug.title()} training (3 seeds)", xlabel="D1 edge-deletion probability", ylabel="Accuracy (%)", ylim=(70, 101))
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8, loc='lower left')
        fig.savefig(self.output / 'robustness.png', dpi=180)
        fig.savefig(self.output / 'robustness.pdf')
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        for method, color, marker in [('dual','#0072B2','o'), ('cross','#D55E00','s'), ('cbert','#009E73','^')]:
            points = sorted([r for r in benchmark if r['method']==method], key=lambda r:r['candidates'])
            axes[0].errorbar([r['candidates'] for r in points], [r['online_seconds'] for r in points],
                             yerr=[r['online_std_seconds'] for r in points], color=color, marker=marker,label=method, capsize=3)
        axes[0].set(xlabel='Candidate count', ylabel='Seconds / 128 workers', ylim=(0,2.2), title='Warm online scoring (seed 42)')
        axes[0].legend()
        for method, color in [('dual','#0072B2'), ('cross','#D55E00')]:
            rows = [r for r in self.safe_retrieval() if r['method']==method and r['stratum']=='safe' and r['library'] in ('base','paraphrase')]
            axes[1].plot([0,1],[100*r['state_hit_at_1'] for r in rows], marker='o' if method=='dual' else 's', color=color, label=method)
        axes[1].set(xticks=[0,1], xticklabels=['Base wording','Paraphrase'], ylabel='Safe-context Hit@1 (%)', ylim=(0,105),
                    title='Correct context retrieval is not class accuracy')
        axes[1].legend()
        for ax in axes:
            ax.grid(axis='y',alpha=.2)
        fig.savefig(self.output / 'efficiency_and_retrieval.png',dpi=180)
        fig.savefig(self.output / 'efficiency_and_retrieval.pdf')
        plt.close(fig)

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.load()
        grouped = self.group_summary()
        paired = self.paired()
        benchmark = self.benchmark_table()
        safe = self.safe_retrieval()
        data.write_json(self.output / "audit.json", {"completeness": self.completeness(), "runs": self.audit,
            "suite_manifest_sha256": self.suite_hash, "bootstrap_repeats": self.repeats, "bootstrap_seed": self.seed,
            "source_code_sha256": data.file_hash(__file__), "notes": ["smoke excluded", "all 24 planned atomic evaluations complete",
                "MRR cannot be reproduced from saved top5; classification, Hit@1 and Recall@5 verified",
                "CIs are exploratory, image-conditional; no multiple-testing correction or new normative gold"]})
        data.write_json(self.output / "grouped_metrics.json", grouped)
        data.write_json(self.output / "paired_differences.json", paired)
        csv_file(self.output / "run_metrics.csv", self.metrics_rows)
        csv_file(self.output / "benchmark.csv", benchmark)
        csv_file(self.output / "safe_context_retrieval.csv", safe)
        csv_file(self.output / "unseen_text_metrics.csv", self.unseen_texts())
        csv_file(self.output / "winning_prototypes.csv", self.winning_prototypes())
        self.plots(grouped, benchmark)
        print(json.dumps(self.completeness(), ensure_ascii=False, indent=2), flush=True)
        print("Paired evidence saved to " + str(self.output / "paired_differences.json"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Directory containing suite_v2 and aic_knowledge_decoupling")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261014)
    args = p.parse_args()
    Analysis(args.input, args.output, args.bootstrap_repeats, args.seed).run()


if __name__ == "__main__":
    main()
