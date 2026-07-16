#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

python -m ghm.migration.bundle verify-transfer \
  --data-root "$MGHDA_DATA_ROOT" \
  --checksum "$MGHDA_DATA_ROOT/outputs/transfer/study2_files.sha256"
