#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/data/processed/prompts"

for split in study2_g1 study2_g2; do
  python -m ghm.prompts.build_prompts \
    --input "$MGHDA_DATA_ROOT/data/processed/items/${split}_claim_verification_items_linked.jsonl" \
    --model-inputs-output "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_model_inputs.jsonl" \
    --eval-metadata-output "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_eval_metadata.jsonl"
done
