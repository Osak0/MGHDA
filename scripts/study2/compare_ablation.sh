#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env V1_RUN_NAME
require_env V2_RUN_NAME
: "${ABLATION_COMPARISON_NAME:=${V1_RUN_NAME}_vs_${V2_RUN_NAME}}"

output_dir="$MGHDA_DATA_ROOT/outputs/study2/ablation_comparisons"
mkdir -p "$output_dir"
python -m ghm.evaluation.study2_ablation compare \
  --v1-scored \
    "$MGHDA_DATA_ROOT/outputs/study2/full_v1/$V1_RUN_NAME/ablation/scored/g1.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/study2/full_v1/$V1_RUN_NAME/ablation/scored/g2.jsonl" \
  --v2-scored \
    "$MGHDA_DATA_ROOT/outputs/study2/full_v2/$V2_RUN_NAME/ablation/scored/g1.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/study2/full_v2/$V2_RUN_NAME/ablation/scored/g2.jsonl" \
  --output "$output_dir/${ABLATION_COMPARISON_NAME}.json" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --seed "${SEED:-42}"
