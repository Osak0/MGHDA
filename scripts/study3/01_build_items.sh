#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity
: "${MIMIC_METADATA_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv}"
: "${MIMIC_SPLIT_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-split.csv}"
: "${MIMIC_JPG_FILES_ROOT:=$MGHDA_DATA_ROOT/files}"
: "${STUDY3_NATURAL_ANCHORS_PER_GRANULARITY:=250}"
: "${STUDY3_CONTROLLED_ANCHORS_PER_GRANULARITY:=50}"
: "${STUDY3_CONTROLLED_K:=2,3,4,5}"
: "${STUDY3_SAMPLE_SEED:=42}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/processed/study3/v2/items"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/v2/audits"
python -m ghm.study3.build_items \
  --attributes "$MGHDA_DATA_ROOT/interim/ci_attribute_assertions.parquet" \
  --objects "$MGHDA_DATA_ROOT/interim/ci_objects.parquet" \
  --mimic-metadata "$MIMIC_METADATA_CSV" \
  --mimic-split "$MIMIC_SPLIT_CSV" \
  --files-root "$MIMIC_JPG_FILES_ROOT" \
  --image-path-root "files" \
  --g1-output "$MGHDA_DATA_ROOT/processed/study3/v2/items/study3_g1_multiselect_items.jsonl" \
  --g2-output "$MGHDA_DATA_ROOT/processed/study3/v2/items/study3_g2_multiselect_items.jsonl" \
  --summary "$MGHDA_DATA_ROOT/outputs/study3/v2/audits/study3_item_summary.json" \
  --natural-anchors "$STUDY3_NATURAL_ANCHORS_PER_GRANULARITY" \
  --controlled-anchors "$STUDY3_CONTROLLED_ANCHORS_PER_GRANULARITY" \
  --controlled-k "$STUDY3_CONTROLLED_K" \
  --seed "$STUDY3_SAMPLE_SEED"
