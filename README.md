# WISER: Worker-centered Inference with Safety-rule Evidence Retrieval

Data and evaluation configurations for the paper:

> **Worker-Centered Construction Hazard Inference via Decoupled Retrieval of Safety-Rule Prototypes**

This repository is the authoritative source for the safety-rule prototype library and the worker-centered scene data used in the paper.

---

## 1. Contents

| Folder | Contents | Size |
|---|---|---:|
| `annotations/` | Raw construction scene graph and worker-level hazard labels with relations | 21.0 MB |
| `samples/` | Task-level train / dev / test worker samples for six-class inference | 21.1 MB |
| `rules/` | Safety-rule prototype libraries and context policies | 0.06 MB |
| `protocol/` | Controlled retrieval protocol (`suite_v2`): splits and coarse / atomic candidate libraries | 12.0 MB |
| `baselines/` | Rule-graph baseline data (`train` / `val` / `test`) | 25.4 MB |
| `code/` | Training, evaluation, benchmarking, and analysis scripts | 0.13 MB |

### File descriptions

- `annotations/scene_graphs_vg_with_relations.json` — full construction scene graph annotations (entities, relations).
- `annotations/worker_hazard_labels_v4.json` — worker-centered samples with labels and relation evidence.
- `samples/train.jsonl`, `dev.jsonl`, `test.jsonl` — worker-level task samples used by the dual encoder (`scene_text`, `rule_text`, labels, positive prototype IDs).
- `rules/rule_prototypes_v1.json` — original 33-candidate coarse prototype library.
- `rules/rule_prototypes_six_class_v1.json` — six-class prototype library.
- `rules/hazard_context_policies_v4.json`, `hazard_context_policies_six_class_v1.json` — context policies used to derive rule-defined labels.
- `rules/hazard_rulebook_v1.json` — rule book with working conditions and required PPE.
- `protocol/suite_v2/` — controlled retrieval protocol: `train.jsonl`, `dev.jsonl`, `test.jsonl`, `manifest.json`, `policy.json`, `input_diagnostics.json`, and `libraries/coarse|atomic/` (base, paraphrase, paraphrase_no_generic, no_generic, drop_label, add_paraphrases, restore).
- `baselines/` — rule-graph baseline data in the six-class split.
- `code/` — `experiment.py`, `run_matrix.py`, `data.py`, `models.py`, `metrics.py`, `analyze_results.py`, `train_dual_encoder.py`, `evaluate_dual_encoder.py`, `server_commands.sh`.

---

## 2. Data source and license

- The construction **images** are from the ConstructionSite 10k dataset (Chen & Zou, 2026). They are **not redistributed** here because the source dataset is released under a non-commercial license.
- Image identifiers and filenames in this repository are relative (e.g., `train_000001.jpg`). Place the original images under a local image root and map them by filename.
- This repository provides the **annotations, worker-centered samples, split lists, rule prototypes, candidate libraries, and configurations** needed to reproduce the reported experiments.

## 3. Reproduction outline

1. Obtain ConstructionSite 10k images from the original release and set a local image root.
2. Task-level experiments: use `samples/{train,dev,test}.jsonl` with `code/train_dual_encoder.py` and `code/evaluate_dual_encoder.py`.
3. Controlled retrieval experiments: use `protocol/suite_v2/{train,dev,test}.jsonl` and `protocol/suite_v2/libraries/{coarse,atomic}/`; see `code/server_commands.sh` and `code/run_matrix.py`.
4. Aggregate and audit results: `code/analyze_results.py`.

## 4. Citation

> Song, L., Wang, Y., & Ma, Z. Worker-Centered Construction Hazard Inference via Decoupled Retrieval of Safety-Rule Prototypes. *Automation in Construction* (under review).

## 5. Contact

Lin Song — songlin@xauat.edu.cn

Xi'an University of Architecture and Technology, Xi'an, China
