#!/usr/bin/env bash
set -euo pipefail

: "${MGHDA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA}"
: "${MGHDA_DATA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA-data}"
: "${MISSING_FINDING_SAMPLE_SIZE:=2}"
: "${MISSING_FINDING_SEED:=42}"
: "${MIMIC_METADATA_CSV:=$MGHDA_DATA_ROOT/data/interim/mimic_metadata.csv}"
: "${MIMIC_SPLIT_CSV:=$MGHDA_DATA_ROOT/data/interim/mimic_split.csv}"
: "${MIMIC_JPG_FILES_ROOT:=$MGHDA_DATA_ROOT/data/files}"

cd "$MGHDA_ROOT"
export PYTHONPATH="$MGHDA_ROOT/src"

mkdir -p "$MGHDA_DATA_ROOT/data/processed/items"
mkdir -p "$MGHDA_DATA_ROOT/data/processed/prompts"
mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"

python -m ghm.granularity.build_g1_items \
  --input "$MGHDA_DATA_ROOT/data/interim/ci_attribute_assertions.parquet" \
  --image-index "$MGHDA_DATA_ROOT/data/interim/image_index.parquet" \
  --output "$MGHDA_DATA_ROOT/data/processed/items/study2_g1_claim_verification_items.jsonl" \
  --summary "$MGHDA_DATA_ROOT/outputs/audits/study2_candidate_summary.json" \
  --missing-finding-sample-size "$MISSING_FINDING_SAMPLE_SIZE" \
  --missing-finding-seed "$MISSING_FINDING_SEED"

python -m ghm.granularity.build_g2_items \
  --attributes "$MGHDA_DATA_ROOT/data/interim/ci_attribute_assertions.parquet" \
  --objects "$MGHDA_DATA_ROOT/data/interim/ci_objects.parquet" \
  --image-index "$MGHDA_DATA_ROOT/data/interim/image_index.parquet" \
  --output "$MGHDA_DATA_ROOT/data/processed/items/study2_g2_claim_verification_items.jsonl" \
  --summary "$MGHDA_DATA_ROOT/outputs/audits/study2_candidate_summary.json" \
  --missing-finding-sample-size "$MISSING_FINDING_SAMPLE_SIZE" \
  --missing-finding-seed "$MISSING_FINDING_SEED"

python -m ghm.data.link_and_download_mimic_jpg \
  --study2-only \
  --study2-g1-items "$MGHDA_DATA_ROOT/data/processed/items/study2_g1_claim_verification_items.jsonl" \
  --study2-g2-items "$MGHDA_DATA_ROOT/data/processed/items/study2_g2_claim_verification_items.jsonl" \
  --study2-g1-output "$MGHDA_DATA_ROOT/data/processed/items/study2_g1_claim_verification_items_linked.jsonl" \
  --study2-g2-output "$MGHDA_DATA_ROOT/data/processed/items/study2_g2_claim_verification_items_linked.jsonl" \
  --metadata "$MIMIC_METADATA_CSV" \
  --split "$MIMIC_SPLIT_CSV" \
  --files-root "$MIMIC_JPG_FILES_ROOT" \
  --needed-index "$MGHDA_DATA_ROOT/data/interim/study2_needed_mimic_jpg_index.parquet" \
  --manifest "$MGHDA_DATA_ROOT/data/interim/study2_needed_mimic_jpg_download_manifest.csv" \
  --url-list "$MGHDA_DATA_ROOT/data/interim/study2_needed_mimic_jpg_urls.txt" \
  --summary "$MGHDA_DATA_ROOT/outputs/audits/study2_mimic_jpg_link_summary.json" \
  --link-only-existing

for split in study2_g1 study2_g2; do
  python -m ghm.prompts.build_prompts \
    --input "$MGHDA_DATA_ROOT/data/processed/items/${split}_claim_verification_items_linked.jsonl" \
    --model-inputs-output "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_model_inputs.jsonl" \
    --eval-metadata-output "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_claim_verification_eval_metadata.jsonl"
done
