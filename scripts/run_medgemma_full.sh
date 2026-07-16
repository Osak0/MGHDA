#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study2_full}"
export RUN_NAME
unset LIMIT

bash "$(dirname "$0")/02_run_inference.sh"
bash "$(dirname "$0")/03_score_outputs.sh"
bash "$(dirname "$0")/validate_medgemma_run.sh"
bash "$(dirname "$0")/04_make_figures.sh"
