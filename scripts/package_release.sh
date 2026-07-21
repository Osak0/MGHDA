#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
cd "$MGHDA_ROOT"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "error: code archive requires a clean Git commit" >&2
  exit 2
fi
commit="$(git rev-parse HEAD)"
output_dir="${PACKAGE_OUTPUT_DIR:-$MGHDA_ROOT/artifacts/release-$commit}"
mkdir -p "$output_dir"
code_name="MGHDA-study2v1v2-study3v2-$commit.tar.gz"
if [[ -e "$output_dir/$code_name" ]]; then
  echo "error: code archive already exists: $output_dir/$code_name" >&2
  exit 2
fi

git archive \
  --format=tar \
  --prefix=MGHDA/ \
  --add-virtual-file="MGHDA/RELEASE_COMMIT:$commit" \
  "$commit" | gzip -n > "$output_dir/$code_name"
(
  cd "$output_dir"
  sha256sum "$code_name" > "$code_name.sha256"
)
python -m ghm.migration.joint_bundle \
  --data-root "$MGHDA_DATA_ROOT" \
  --output-dir "$output_dir" \
  --git-commit "$commit"
echo "Created code and private archives for commit $commit"
