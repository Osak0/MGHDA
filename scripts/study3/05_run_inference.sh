#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
MODEL_PATH="${MODEL_PATH:-${MEDGEMMA_MODEL_PATH:-}}"
require_env MODEL_PATH
require_env MODEL_NAME
require_study3_identity
: "${RUN_NAME:=model_study3_multiselect_v2}"
: "${SEED:=42}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/v2/raw_responses"
if [[ "${STUDY3_MODE:-full}" == "smoke" ]]; then
  splits=(smoke)
else
  splits=(g1 g2)
fi
for granularity in "${splits[@]}"; do
  if [[ "$granularity" == "smoke" ]]; then
    input_path="$MGHDA_DATA_ROOT/processed/study3/v2/smoke/model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study3/v2/smoke/eval_metadata.jsonl"
  else
    input_path="$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${granularity}_multiselect_model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl"
  fi
  limit_args=()
  if [[ -n "${LIMIT:-}" ]]; then
    limit_args=(--limit "$LIMIT")
  fi
  python -m ghm.inference.medgemma_runner \
    --input "$input_path" \
    --eval-metadata "$metadata_path" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/v2/raw_responses/${RUN_NAME}_${granularity}_raw.jsonl" \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --data-root "$MGHDA_DATA_ROOT" \
    "${limit_args[@]}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --dtype "${DTYPE:-bfloat16}" \
    --device "${DEVICE:-cuda}" \
    --seed "$SEED" \
    --checkpoint "$MGHDA_DATA_ROOT/outputs/study3/v2/raw_responses/${RUN_NAME}_${granularity}.checkpoint.jsonl" \
    --resume
done
