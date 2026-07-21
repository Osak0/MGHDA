#!/usr/bin/env bash
set -euo pipefail

: "${STUDY2_SET:=full_v2}"
: "${RUN_MODE:=dry-run}"
: "${RUN_NAME:=study2_v2_dry_run}"
export STUDY2_SET RUN_MODE RUN_NAME
exec bash "$(dirname "$0")/study2/run.sh"
