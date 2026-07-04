#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study2}"
export RUN_NAME

bash "$(dirname "$0")/03_score_outputs.sh"
