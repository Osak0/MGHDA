#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env MEDGEMMA_MODEL_PATH

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"

for split in study2_g1 study2_g2; do
  python -m ghm.inference.medgemma_runner \
    --input "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_eval_metadata.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/audits/medgemma_dry_run_${split}.jsonl" \
    --model-path "$MEDGEMMA_MODEL_PATH" \
    --data-root "$MGHDA_DATA_ROOT" \
    --limit "${LIMIT:-20}" \
    --dry-run
done
