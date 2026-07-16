#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/private/study2-bundle" >&2
  exit 2
fi

cd "$MGHDA_ROOT"
python -m ghm.migration.bundle create \
  --data-root "$MGHDA_DATA_ROOT" \
  --output-dir "$1" \
  --git-commit "$(git rev-parse HEAD)" \
  --model-inputs \
    "$MGHDA_DATA_ROOT/data/processed/prompts/study2_g1_claim_verification_model_inputs.jsonl" \
    "$MGHDA_DATA_ROOT/data/processed/prompts/study2_g2_claim_verification_model_inputs.jsonl" \
  --eval-metadata \
    "$MGHDA_DATA_ROOT/data/processed/prompts/study2_g1_claim_verification_eval_metadata.jsonl" \
    "$MGHDA_DATA_ROOT/data/processed/prompts/study2_g2_claim_verification_eval_metadata.jsonl"
