#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
: "${MODELS_CONFIG:=$MGHDA_ROOT/configs/models.env}"
if [[ ! -f "$MODELS_CONFIG" ]]; then
  echo "error: copy configs/models.env.example to configs/models.env" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$MODELS_CONFIG"
require_env MEDGEMMA4_MODEL_PATH
require_env MEDGEMMA15_MODEL_PATH
require_env QWEN3_VL_MODEL_PATH

mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"
python -m ghm.migration.preflight \
  --data-root "$MGHDA_DATA_ROOT" \
  --model-path "$MEDGEMMA4_MODEL_PATH" \
  --model-path "$MEDGEMMA15_MODEL_PATH" \
  --model-path "$QWEN3_VL_MODEL_PATH" \
  --output "$MGHDA_DATA_ROOT/outputs/audits/three_model_preflight.json"
