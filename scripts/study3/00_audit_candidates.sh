#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity
: "${MIMIC_METADATA_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv}"
: "${MIMIC_SPLIT_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-split.csv}"
: "${MIMIC_JPG_FILES_ROOT:=$MGHDA_DATA_ROOT/files}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/v2/audits"
python -m ghm.study3.candidates \
  --attributes "$MGHDA_DATA_ROOT/interim/ci_attribute_assertions.parquet" \
  --objects "$MGHDA_DATA_ROOT/interim/ci_objects.parquet" \
  --mimic-metadata "$MIMIC_METADATA_CSV" \
  --mimic-split "$MIMIC_SPLIT_CSV" \
  --files-root "$MIMIC_JPG_FILES_ROOT" \
  --image-path-root "files" \
  --output "$MGHDA_DATA_ROOT/outputs/study3/v2/audits/study3_candidate_audit.json"
