#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/processed/study3/prompts"
for granularity in g1 g2; do
  python -m ghm.study3.prompts \
    --input "$MGHDA_DATA_ROOT/processed/study3/items/study3_${granularity}_multiselect_items.jsonl" \
    --model-inputs-output "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_model_inputs.jsonl" \
    --eval-metadata-output "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl"
done
