#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/processed/study3/v2/prompts"
for granularity in g1 g2; do
  python -m ghm.study3.prompts \
    --input "$MGHDA_DATA_ROOT/processed/study3/v2/items/study3_${granularity}_multiselect_items.jsonl" \
    --model-inputs-output "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${granularity}_multiselect_model_inputs.jsonl" \
    --eval-metadata-output "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl"
done

mkdir -p "$MGHDA_DATA_ROOT/processed/study3/v2/smoke"
python -m ghm.study3.smoke \
  --model-inputs \
    "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_g1_multiselect_model_inputs.jsonl" \
    "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_g2_multiselect_model_inputs.jsonl" \
  --eval-metadata \
    "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_g1_multiselect_eval_metadata.jsonl" \
    "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_g2_multiselect_eval_metadata.jsonl" \
  --model-inputs-output "$MGHDA_DATA_ROOT/processed/study3/v2/smoke/model_inputs.jsonl" \
  --eval-metadata-output "$MGHDA_DATA_ROOT/processed/study3/v2/smoke/eval_metadata.jsonl" \
  --seed "${STUDY3_SAMPLE_SEED:-42}"
