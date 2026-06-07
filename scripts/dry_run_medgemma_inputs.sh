#!/usr/bin/env bash
set -euo pipefail

: "${MGHDA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA}"
: "${MGHDA_DATA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA-data}"
: "${MGHDA_MODEL_ROOT:=/xjtu-mlp-vepfs/wangruiyang/models}"
: "${MEDGEMMA_MODEL_PATH:=$MGHDA_MODEL_ROOT/google/medgemma-4b-it}"

cd "$MGHDA_ROOT"
export PYTHONPATH="$MGHDA_ROOT/src"

mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"

for split in g1_h1 g1_h2 g2_h1 g2_h2; do
  python -m ghm.inference.medgemma_runner \
    --input "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/data/processed/prompts/${split}_eval_metadata.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/audits/medgemma_dry_run_${split}.jsonl" \
    --model-path "$MEDGEMMA_MODEL_PATH" \
    --data-root "$MGHDA_DATA_ROOT" \
    --limit "${LIMIT:-20}" \
    --dry-run
done
