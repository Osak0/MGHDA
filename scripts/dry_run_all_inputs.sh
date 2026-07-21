#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
output_root="$MGHDA_DATA_ROOT/outputs/input_dry_run"
mkdir -p "$output_root"

for version in v1 v2; do
  for split in g1 g2; do
    python -m ghm.inference.medgemma_runner \
      --input "$MGHDA_DATA_ROOT/processed/study2/$version/study2_${split}_model_inputs.jsonl" \
      --eval-metadata "$MGHDA_DATA_ROOT/processed/study2/$version/study2_${split}_eval_metadata.jsonl" \
      --output "$output_root/study2_${version}_${split}.jsonl" \
      --model-path . --model-name input-dry-run \
      --data-root "$MGHDA_DATA_ROOT" --dry-run
    python -m ghm.inference.medgemma_runner \
      --input "$MGHDA_DATA_ROOT/processed/study2/ablation/$version/${split}_model_inputs.jsonl" \
      --eval-metadata "$MGHDA_DATA_ROOT/processed/study2/ablation/$version/${split}_eval_metadata.jsonl" \
      --output "$output_root/study2_ablation_${version}_${split}.jsonl" \
      --model-path . --model-name input-dry-run \
      --data-root "$MGHDA_DATA_ROOT" --dry-run
  done
done

for split in g1 g2; do
  python -m ghm.inference.medgemma_runner \
    --input "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${split}_multiselect_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${split}_multiselect_eval_metadata.jsonl" \
    --output "$output_root/study3_${split}.jsonl" \
    --model-path . --model-name input-dry-run \
    --data-root "$MGHDA_DATA_ROOT" --dry-run
done
python -m ghm.inference.medgemma_runner \
  --input "$MGHDA_DATA_ROOT/processed/study3/v2/smoke/model_inputs.jsonl" \
  --eval-metadata "$MGHDA_DATA_ROOT/processed/study3/v2/smoke/eval_metadata.jsonl" \
  --output "$output_root/study3_smoke.jsonl" \
  --model-path . --model-name input-dry-run \
  --data-root "$MGHDA_DATA_ROOT" --dry-run

echo "All Study 2 v1/v2 and Study 3 input dry-runs passed."
