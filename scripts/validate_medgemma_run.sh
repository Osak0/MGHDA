#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
: "${RUN_NAME:=medgemma_study2}"

for split in study2_g1 study2_g2; do
  python -m ghm.evaluation.validate_run \
    --model-inputs "$MGHDA_DATA_ROOT/processed/prompts/${split}_claim_verification_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/processed/prompts/${split}_claim_verification_eval_metadata.jsonl" \
    --raw "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}.jsonl" \
    --parsed "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}_parsed.jsonl" \
    --scored "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_${split}_scored.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_${split}_validation.json"
done
