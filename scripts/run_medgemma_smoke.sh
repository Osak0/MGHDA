#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study2_smoke}"
: "${LIMIT:=20}"
export RUN_NAME LIMIT

bash "$(dirname "$0")/02_run_inference.sh"
