#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_env MEDGEMMA_MODEL_PATH
require_study3_identity

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/audits"
python -m ghm.study3.transfer verify \
  --data-root "$MGHDA_DATA_ROOT" \
  --checksum "$MGHDA_DATA_ROOT/outputs/study3/transfer/study3_files.sha256" \
  --summary "$MGHDA_DATA_ROOT/outputs/study3/transfer/study3_transfer_summary.json" \
  --expected-git-commit "$(git rev-parse HEAD)"
python -m ghm.migration.preflight \
  --data-root "$MGHDA_DATA_ROOT" \
  --model-path "$MEDGEMMA_MODEL_PATH" \
  --output "$MGHDA_DATA_ROOT/outputs/study3/audits/medgemma_study3_preflight.json"
for granularity in g1 g2; do
  python -m ghm.study3.validation \
    --model-inputs "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl" \
    --data-root "$MGHDA_DATA_ROOT" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/audits/study3_${granularity}_input_validation.json"
done
