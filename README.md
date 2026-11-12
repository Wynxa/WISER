# WISER

WISER provides the worker-level dataset and safety-rule prototype library used in the study. The sample files follow the shared image-level protocol: all workers from an image belong to the same partition, and no image ID appears in more than one partition.

| File | Partition | Worker samples | Source images |
| --- | --- | ---: | ---: |
| [samples/train.jsonl](samples/train.jsonl) | Training | 13,727 | 3,690 |
| [samples/dev.jsonl](samples/dev.jsonl) | Validation | 3,929 | 1,028 |
| [samples/test.jsonl](samples/test.jsonl) | Test | 1,968 | 523 |

Each JSONL record represents one worker and includes the image ID, observed scene-graph evidence, safety-state label, and IDs of the matching rule prototypes. In the validation file, the record field `split` is set to `val`.

The [six-class rule prototype library](rules/rule_prototypes_six_class_v1.json) and its [context policies](rules/hazard_context_policies_six_class_v1.json) accompany these samples. Source images are available from [ConstructionSite 10k](https://huggingface.co/datasets/LouisChen15/ConstructionSite).
