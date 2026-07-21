#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

cd "$MGHDA_ROOT"
v1_root="$MGHDA_DATA_ROOT/processed/study2/v1"
v2_root="$MGHDA_DATA_ROOT/processed/study2/v2"
ablation_root="$MGHDA_DATA_ROOT/processed/study2/ablation"
mkdir -p "$v1_root" "$v2_root" "$ablation_root"

for split in study2_g1 study2_g2; do
  for version in v1 v2; do
    output_root="$MGHDA_DATA_ROOT/processed/study2/$version"
    python -m ghm.prompts.build_prompts \
      --input "$MGHDA_DATA_ROOT/processed/items/${split}_claim_verification_items_linked.jsonl" \
      --model-inputs-output "$output_root/${split}_model_inputs.jsonl" \
      --eval-metadata-output "$output_root/${split}_eval_metadata.jsonl" \
      --template-version "$version"
  done
done

python -m ghm.evaluation.study2_ablation build \
  --inputs \
    "$MGHDA_DATA_ROOT/processed/items/study2_g1_claim_verification_items_linked.jsonl" \
    "$MGHDA_DATA_ROOT/processed/items/study2_g2_claim_verification_items_linked.jsonl" \
  --output-root "$ablation_root" \
  --per-stratum "${STUDY2_ABLATION_PER_STRATUM:-10}" \
  --seed "${STUDY2_ABLATION_SEED:-42}"
