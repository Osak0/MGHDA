#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity

cd "$MGHDA_ROOT"
transfer_dir="$MGHDA_DATA_ROOT/outputs/study3/transfer"
mkdir -p "$transfer_dir"
if find "$transfer_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "error: Study 3 transfer directory is not empty: $transfer_dir" >&2
  exit 2
fi
python -m ghm.study3.transfer manifest \
  --data-root "$MGHDA_DATA_ROOT" \
  --output-dir "$transfer_dir" \
  --git-commit "$(git rev-parse HEAD)" \
  --model-inputs \
    "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_g1_multiselect_model_inputs.jsonl" \
    "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_g2_multiselect_model_inputs.jsonl" \
  --eval-metadata \
    "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_g1_multiselect_eval_metadata.jsonl" \
    "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_g2_multiselect_eval_metadata.jsonl"

echo "No images were copied or moved."
echo "Transfer list: outputs/study3/transfer/study3_files_from.txt"
