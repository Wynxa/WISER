#!/usr/bin/env bash
set -euo pipefail

HAZARD_ROOT="${HAZARD_ROOT:-/root/hazard_reasoning}"
MODEL_CACHE="${MODEL_CACHE:-/root/autodl-tmp/hf_models}"
OUT="${OUT:-/root/autodl-tmp/aic_knowledge_decoupling}"
export OMP_NUM_THREADS=4
cd "$HAZARD_ROOT"
BASE="AiC_resubmission/knowledge_decoupling_experiment"
SUITE="${SUITE:-$HAZARD_ROOT/$BASE/outputs/suite_v2}"
MODE="${1:-plan}"

COMMON=(--suite "$SUITE" --hazard-project-root "$HAZARD_ROOT" --model-cache-dir "$MODEL_CACHE")
MATRIX=(python "$BASE/run_matrix.py" "${COMMON[@]}" --share-backbone)

case "$MODE" in
  plan)
    "${MATRIX[@]}" --output-root "$OUT/main" --plan-file "$OUT/main_plan.json"
    ;;
  smoke)
    "${MATRIX[@]}" --output-root "$OUT/smoke" --methods dual scene learned_prototypes \
      --seeds 42 --augmentations mixed --epochs 1 --stages train evaluate --execute --resume
    ;;
  main)
    "${MATRIX[@]}" --output-root "$OUT/main" --stages train evaluate compare --execute --resume
    ;;
  update)
    "${MATRIX[@]}" --output-root "$OUT/main" --methods dual learned_prototypes cross tfidf \
      --seeds 42 --augmentations mixed --stages update --execute --resume
    ;;
  anonymous)
    "${MATRIX[@]}" --output-root "$OUT/anonymous" --methods dual --anonymous-rules \
      --stages train evaluate --execute --resume
    ;;
  benchmark)
    "${MATRIX[@]}" --output-root "$OUT/main" --methods dual cbert cross \
      --seeds 42 --augmentations clean --batch-size 4 --cross-batch-size 4 \
      --stages benchmark --execute --resume
    ;;
  atomic)
    "${MATRIX[@]}" --output-root "$OUT/atomic" --family atomic --alignment-loss balanced_bce \
      --methods dual scene learned_prototypes cross --stages train evaluate compare --execute --resume
    ;;
  *)
    printf 'Supported modes: plan smoke main anonymous update benchmark atomic\n' >&2
    exit 2
    ;;
esac
