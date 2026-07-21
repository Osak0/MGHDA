#!/usr/bin/env bash
set -euo pipefail

echo "note: v1/v2 packaging creates both clean replacement archives" >&2
exec bash "$(dirname "$0")/package_release.sh"
