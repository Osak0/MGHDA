#!/usr/bin/env bash
set -euo pipefail

echo "note: delegating to the versioned Study 2 runner" >&2
: "${STUDY2_SET:=full_v2}"
export STUDY2_SET
exec bash "$(dirname "$0")/study2/run.sh"
