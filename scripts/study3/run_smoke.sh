#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=model_study3_multiselect_v2_smoke}"
: "${STUDY3_MODE:=smoke}"
export RUN_NAME STUDY3_MODE
unset LIMIT

bash "$(dirname "$0")/05_run_inference.sh"
bash "$(dirname "$0")/06_score_validate.sh"
