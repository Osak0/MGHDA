#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env MEDGEMMA_MODEL_PATH
: "${RUN_NAME:=medgemma_study2}"
: "${SEED:=42}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/raw_responses"

for split in study2_g1 study2_g2; do
  limit_args=()
  if [[ -n "${LIMIT:-}" ]]; then
    limit_args=(--limit "$LIMIT")
  fi
  python -m ghm.inference.medgemma_runner \
    --input "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_eval_metadata.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}.jsonl" \
    --model-path "$MEDGEMMA_MODEL_PATH" \
    --data-root "$MGHDA_DATA_ROOT" \
    "${limit_args[@]}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --dtype "${DTYPE:-bfloat16}" \
    --device "${DEVICE:-cuda}" \
    --seed "$SEED" \
    --checkpoint "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}.checkpoint.jsonl" \
    --resume
done
