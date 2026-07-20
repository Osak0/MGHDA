#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study3_multiselect_v1_smoke}"
: "${LIMIT:=8}"
export RUN_NAME LIMIT

bash "$(dirname "$0")/05_run_inference.sh"
