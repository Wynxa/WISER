"""Sequential server runner. Plans are the default; --execute launches jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AiC_resubmission.knowledge_decoupling_experiment.data import HAZARD, file_hash, read_json, write_json
from AiC_resubmission.knowledge_decoupling_experiment.models import NEURAL

ENTRY = Path(__file__).with_name("experiment.py")


def jobs(args):
    if args.batch_size % args.cross_batch_size:
        raise ValueError("batch-size must be divisible by cross-batch-size for equal effective training batches")
    manifest = read_json(args.suite / "manifest.json")
    result = []
    by_group = {}
    for seed in args.seeds:
        for augmentation in args.augmentations:
            group = []
            for method in args.methods:
                if method == "rule" and (seed != args.seeds[0] or augmentation != args.augmentations[0]):
                    continue
                if method not in NEURAL and seed != args.seeds[0]:
                    continue
                variant = method + "_anonymous" if method == "dual" and getattr(args, "anonymous_rules", False) else method
                tag = f"{args.family}_{variant}_{augmentation}_s{seed}"
                checkpoint = args.output_root / "checkpoints" / tag
                output = args.output_root / "results" / tag
                common = ["--suite", str(args.suite), "--family", args.family, "--method", method]
                neural = ["--hazard-project-root", str(args.hazard_project_root), "--model-cache-dir", str(args.model_cache_dir),
                          "--device", args.device, "--batch-size", str(args.cross_batch_size if method == "cross" else args.batch_size),
                          "--pair-batch-size", str(args.pair_batch_size)]
                if method in NEURAL and "train" in args.stages:
                    shared = ["--share-backbone"] if args.share_backbone and method == "dual" else []
                    anonymous = ["--anonymous-rules"] if method == "dual" and getattr(args, "anonymous_rules", False) else []
                    result.append({"name": tag + "_train", "kind": "train", "done": checkpoint / "train_config.json",
                        "command": [sys.executable, str(ENTRY), "train", *common, *neural,
                                    "--augmentation", augmentation, "--seed", str(seed), "--epochs", str(args.epochs),
                                    "--gradient-accumulation", str(args.batch_size // args.cross_batch_size if method == "cross" else 1),
                                    "--alignment-loss", args.alignment_loss, *shared, *anonymous, "--output-dir", str(checkpoint)]})
                weights = ["--checkpoint", str(checkpoint / "best_model.pt"), "--train-config", str(checkpoint / "train_config.json")] if method in NEURAL else []
                fitting = ["--baseline-train-conditions", *manifest["conditions"]] if augmentation == "mixed" and method in ("lookup", "tfidf") else []
                if "evaluate" in args.stages:
                    cmd = [sys.executable, str(ENTRY), "evaluate", *common, *neural, *weights, *fitting, "--output-dir", str(output)]
                    if args.predicted_dir:
                        cmd += ["--predicted-dir", str(args.predicted_dir)]
                    result.append({"name": tag + "_evaluate", "kind": "evaluate", "done": output / "run.json", "command": cmd})
                group.append(output)
                if "update" in args.stages and method in ("dual", "learned_prototypes", "cross", "tfidf") and seed == args.seeds[0] and augmentation == ("mixed" if "mixed" in args.augmentations else args.augmentations[0]):
                    calibration = args.output_root / "calibrations" / (tag + ".json")
                    result.append({"name": tag + "_calibrate", "kind": "calibrate", "done": calibration,
                        "command": [sys.executable, str(ENTRY), "calibrate", *common, *neural, *weights, *fitting,
                                    "--output", str(calibration)]})
                    updated_paths = []
                    for variant in args.update_libraries:
                        updated = args.output_root / "updates" / tag / variant
                        updated_paths.append(updated)
                        result.append({"name": tag + "_update_" + variant, "kind": "evaluate", "done": updated / "run.json",
                            "command": [sys.executable, str(ENTRY), "evaluate", *common, *neural, *weights, *fitting,
                                        "--library-name", variant, "--conditions", "clean", "--calibration", str(calibration),
                                        "--output-dir", str(updated)]})
                    if len(updated_paths) > 1:
                        comparison = args.output_root / "update_comparisons" / tag
                        result.append({"name": tag + "_update_compare", "kind": "compare", "done": comparison / "comparisons.json",
                            "command": [sys.executable, str(ENTRY), "compare", "--results", *map(str, updated_paths),
                                        "--output-dir", str(comparison)]})
                if "benchmark" in args.stages and augmentation == args.augmentations[0] and seed == args.seeds[0] and method in NEURAL:
                    target = args.output_root / "benchmarks" / (tag + ".json")
                    result.append({"name": tag + "_benchmark", "kind": "benchmark", "done": target,
                                   "command": [sys.executable, str(ENTRY), "benchmark", *common, *neural, *weights, "--output", str(target)]})
            by_group[f"{augmentation}_s{seed}"] = group
    if "compare" in args.stages:
        for name, paths in by_group.items():
            if len(paths) < 2:
                continue
            target = args.output_root / "comparisons" / (args.family + "_" + name)
            result.append({"name": name + "_compare", "kind": "compare", "done": target / "comparisons.json",
                "command": [sys.executable, str(ENTRY), "compare", "--results", *map(str, paths), "--output-dir", str(target)]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--family", choices=["coarse", "atomic"], default="coarse")
    p.add_argument("--methods", nargs="+", choices=["rule", "lookup", "tfidf", *NEURAL], default=["dual", "scene", "learned_prototypes", "cbert", "cross", "rule", "lookup", "tfidf"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--augmentations", nargs="+", choices=["clean", "mixed"], default=["clean", "mixed"])
    p.add_argument("--stages", nargs="+", choices=["train", "evaluate", "update", "benchmark", "compare"], default=["train", "evaluate", "compare"])
    p.add_argument("--update-libraries", nargs="+", default=["base", "paraphrase", "no_generic", "paraphrase_no_generic", "add_paraphrases", "drop_label", "restore"])
    p.add_argument("--hazard-project-root", type=Path, default=HAZARD)
    p.add_argument("--model-cache-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cross-batch-size", type=int, default=4)
    p.add_argument("--pair-batch-size", type=int, default=32)
    p.add_argument("--alignment-loss", choices=["multi_positive", "balanced_bce"], default="multi_positive")
    p.add_argument("--share-backbone", action="store_true", help="Use one shared BERT in the dual control; use a separate output root for unshared runs")
    p.add_argument("--anonymous-rules", action="store_true", help="Replace dual rule inputs with anonymous IDs; keep identical architecture")
    p.add_argument("--predicted-dir", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--plan-file", type=Path)
    p.add_argument("--resume", action="store_true", help="Skip completed matching jobs; partial jobs are not resumed")
    args = p.parse_args()
    if any(len(values) != len(set(values)) for values in (args.methods, args.seeds, args.augmentations, args.stages)):
        raise ValueError("Duplicate matrix entries")
    plan = jobs(args)
    serial = [{**j, "done": str(j["done"])} for j in plan]
    if args.plan_file:
        write_json(args.plan_file, serial)
    print(json.dumps({"execute": args.execute, "jobs": serial}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    write_json(args.output_root / "job_plan.json", serial)
    suite_hash = file_hash(args.suite / "manifest.json")
    for job in plan:
        done = job["done"]
        receipt = args.output_root / "logs" / (job["name"] + ".completed.json")
        identity = {"command": job["command"], "suite_manifest_sha256": suite_hash}
        if args.resume and done.exists() and receipt.exists():
            if read_json(receipt) != identity:
                raise ValueError("Completed job has another command/protocol: " + job["name"])
            print("SKIP " + job["name"], flush=True)
            continue
        if done.exists():
            raise ValueError("Existing output without matching completion receipt: " + str(done))
        log = args.output_root / "logs" / (job["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        print("RUN " + job["name"], flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not done.exists():
            raise RuntimeError("Job exited without expected artifact: " + job["name"])
        write_json(receipt, identity)
    write_json(args.output_root / "matrix_completed.json", {"suite_manifest_sha256": suite_hash, "completed_jobs": len(plan)})


if __name__ == "__main__":
    main()
