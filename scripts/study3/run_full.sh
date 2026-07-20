#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study3_multiselect_v1}"
export RUN_NAME
unset LIMIT

bash "$(dirname "$0")/05_run_inference.sh"
bash "$(dirname "$0")/06_score_validate.sh"
bash "$(dirname "$0")/07_make_figures.sh"
