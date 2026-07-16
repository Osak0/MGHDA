#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
: "${STUDY2_MAX_ITEMS_TOTAL:=960}"
: "${STUDY2_SAMPLE_SEED:=42}"
: "${MIMIC_METADATA_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv}"
: "${MIMIC_SPLIT_CSV:=$MGHDA_DATA_ROOT/raw/MIMIC-CXR/mimic-cxr-2.0.0-split.csv}"
: "${MIMIC_JPG_FILES_ROOT:=$MGHDA_DATA_ROOT/files}"

if (( STUDY2_MAX_ITEMS_TOTAL < 4 || STUDY2_MAX_ITEMS_TOTAL % 2 != 0 )); then
  echo "error: STUDY2_MAX_ITEMS_TOTAL must be an even integer of at least 4" >&2
  exit 2
fi
MAX_ITEMS_PER_GROUP=$((STUDY2_MAX_ITEMS_TOTAL / 2))

cd "$MGHDA_ROOT"
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
  --max-linked-items-per-group "$MAX_ITEMS_PER_GROUP" \
  --sampling-seed "$STUDY2_SAMPLE_SEED" \
  --link-only-existing
