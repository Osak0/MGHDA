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

python -m ghm.data.link_and_download_mimic_jpg \
  --study2-only \
  --study2-g1-items "$MGHDA_DATA_ROOT/processed/items/study2_g1_claim_verification_items.jsonl" \
  --study2-g2-items "$MGHDA_DATA_ROOT/processed/items/study2_g2_claim_verification_items.jsonl" \
  --study2-g1-output "$MGHDA_DATA_ROOT/processed/items/study2_g1_claim_verification_items_linked.jsonl" \
  --study2-g2-output "$MGHDA_DATA_ROOT/processed/items/study2_g2_claim_verification_items_linked.jsonl" \
  --metadata "$MIMIC_METADATA_CSV" \
  --split "$MIMIC_SPLIT_CSV" \
  --files-root "$MIMIC_JPG_FILES_ROOT" \
  --image-path-root "files" \
  --needed-index "$MGHDA_DATA_ROOT/interim/study2_needed_mimic_jpg_index.parquet" \
  --manifest "$MGHDA_DATA_ROOT/interim/study2_needed_mimic_jpg_download_manifest.csv" \
  --url-list "$MGHDA_DATA_ROOT/interim/study2_needed_mimic_jpg_urls.txt" \
  --summary "$MGHDA_DATA_ROOT/outputs/audits/study2_mimic_jpg_link_summary.json" \
  --link-only-existing
