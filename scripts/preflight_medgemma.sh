#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env MEDGEMMA_MODEL_PATH

mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"
python -m ghm.migration.preflight \
  --data-root "$MGHDA_DATA_ROOT" \
  --model-path "$MEDGEMMA_MODEL_PATH" \
  --output "$MGHDA_DATA_ROOT/outputs/audits/medgemma_preflight.json"
