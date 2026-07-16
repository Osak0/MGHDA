#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

python -m ghm.migration.bundle verify --bundle-root "$MGHDA_DATA_ROOT"
