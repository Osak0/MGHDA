#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env RUN_NAME
: "${STUDY2_SET:=full_v2}"
: "${RUN_PHASE:=full}"
phase_root="$MGHDA_DATA_ROOT/outputs/study2/$STUDY2_SET/$RUN_NAME/$RUN_PHASE"
python -m ghm.evaluation.visualize_results \
  --inputs "$phase_root/audits/combined_score_summary.json" \
  --run-names "$RUN_NAME-$RUN_PHASE" \
  --output-dir "$phase_root/reports"
