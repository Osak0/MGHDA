#!/usr/bin/env bash
set -euo pipefail

: "${MGHDA_DATA_ROOT:=.}"

python -m ghm.data.audit_chest_imagenome \
  --objects "$MGHDA_DATA_ROOT/data/interim/ci_objects.parquet" \
  --assertions "$MGHDA_DATA_ROOT/data/interim/ci_attribute_assertions.parquet" \
  --output-dir "$MGHDA_DATA_ROOT/outputs/audits"
