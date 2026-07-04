#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study2_full}"
export RUN_NAME
unset LIMIT

bash "$(dirname "$0")/02_run_inference.sh"
