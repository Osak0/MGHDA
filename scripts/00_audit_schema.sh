#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

cd "$MGHDA_ROOT"
python -m ghm.data.audit_chest_imagenome \
  --objects "$MGHDA_DATA_ROOT/interim/ci_objects.parquet" \
  --assertions "$MGHDA_DATA_ROOT/interim/ci_attribute_assertions.parquet" \
  --output-dir "$MGHDA_DATA_ROOT/outputs/audits"
