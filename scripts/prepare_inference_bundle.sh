#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/transfer"
python -m ghm.migration.bundle manifest \
  --data-root "$MGHDA_DATA_ROOT" \
  --output-dir "$MGHDA_DATA_ROOT/outputs/transfer" \
  --git-commit "$(git rev-parse HEAD)" \
  --model-inputs \
    "$MGHDA_DATA_ROOT/processed/prompts/study2_g1_claim_verification_model_inputs.jsonl" \
    "$MGHDA_DATA_ROOT/processed/prompts/study2_g2_claim_verification_model_inputs.jsonl" \
  --eval-metadata \
    "$MGHDA_DATA_ROOT/processed/prompts/study2_g1_claim_verification_eval_metadata.jsonl" \
    "$MGHDA_DATA_ROOT/processed/prompts/study2_g2_claim_verification_eval_metadata.jsonl"

echo "No images were copied or moved."
