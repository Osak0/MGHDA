#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_env MEDGEMMA_MODEL_PATH
require_study3_identity
: "${RUN_NAME:=medgemma_study3_multiselect_v1}"
: "${SEED:=42}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/raw_responses"
for granularity in g1 g2; do
  limit_args=()
  if [[ -n "${LIMIT:-}" ]]; then
    limit_args=(--limit "$LIMIT")
  fi
  python -m ghm.inference.medgemma_runner \
    --input "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/raw_responses/${RUN_NAME}_${granularity}_raw.jsonl" \
    --model-path "$MEDGEMMA_MODEL_PATH" \
    --data-root "$MGHDA_DATA_ROOT" \
    "${limit_args[@]}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --dtype "${DTYPE:-bfloat16}" \
    --device "${DEVICE:-cuda}" \
    --seed "$SEED" \
    --checkpoint "$MGHDA_DATA_ROOT/outputs/study3/raw_responses/${RUN_NAME}_${granularity}.checkpoint.jsonl" \
    --resume
done
