#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
: "${MISSING_FINDING_SAMPLE_SIZE:=2}"
: "${MISSING_FINDING_SEED:=42}"
: "${MIMIC_METADATA_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv}"
: "${MIMIC_SPLIT_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-split.csv}"
: "${MIMIC_JPG_FILES_ROOT:=$MGHDA_DATA_ROOT/files}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/processed/items"
mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"

python -m ghm.granularity.build_g1_items \
  --input "$MGHDA_DATA_ROOT/interim/ci_attribute_assertions.parquet" \
  --image-index "$MGHDA_DATA_ROOT/interim/image_index.parquet" \
  --output "$MGHDA_DATA_ROOT/processed/items/study2_g1_claim_verification_items.jsonl" \
  --summary "$MGHDA_DATA_ROOT/outputs/audits/study2_candidate_summary.json" \
  --missing-finding-sample-size "$MISSING_FINDING_SAMPLE_SIZE" \
  --missing-finding-seed "$MISSING_FINDING_SEED"

python -m ghm.granularity.build_g2_items \
  --attributes "$MGHDA_DATA_ROOT/interim/ci_attribute_assertions.parquet" \
  --objects "$MGHDA_DATA_ROOT/interim/ci_objects.parquet" \
  --image-index "$MGHDA_DATA_ROOT/interim/image_index.parquet" \
  --output "$MGHDA_DATA_ROOT/processed/items/study2_g2_claim_verification_items.jsonl" \
  --summary "$MGHDA_DATA_ROOT/outputs/audits/study2_candidate_summary.json" \
  --missing-finding-sample-size "$MISSING_FINDING_SAMPLE_SIZE" \
  --missing-finding-seed "$MISSING_FINDING_SEED"

bash "$MGHDA_ROOT/scripts/01b_link_and_sample_items.sh"
