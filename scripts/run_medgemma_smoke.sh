#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study2_smoke}"
: "${LIMIT:=20}"
export RUN_NAME LIMIT

bash "$(dirname "$0")/02_run_inference.sh"
bash "$(dirname "$0")/03_score_outputs.sh"
bash "$(dirname "$0")/validate_medgemma_run.sh"
