#!/usr/bin/env bash
set -euo pipefail

: "${MGHDA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA}"
: "${MGHDA_DATA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA-data}"
: "${RUN_NAME:=medgemma_study2}"

cd "$MGHDA_ROOT"
export PYTHONPATH="$MGHDA_ROOT/src"

python -m ghm.evaluation.visualize_results \
  --inputs "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_study2_g1_g2_score_summary.json" \
  --output-dir "$MGHDA_DATA_ROOT/outputs/reports/${RUN_NAME}_study2"
