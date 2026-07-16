#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
: "${RUN_NAME:=medgemma_study2}"

cd "$MGHDA_ROOT"
python -m ghm.evaluation.visualize_results \
  --inputs "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_study2_g1_g2_score_summary.json" \
  --output-dir "$MGHDA_DATA_ROOT/outputs/reports/${RUN_NAME}_study2"
