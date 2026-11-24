#!/usr/bin/env bash
# Fetch the two upstream benchmark repositories at the pinned commits and verify
# every catalog against the repositories' own SHA-256 manifests.
# Usage: bash scripts/fetch_upstream.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external

fetch() {
  local name=$1 commit=$2
  local dir=external/$name
  if [ ! -d "$dir/.git" ]; then
    git clone -q "https://github.com/Research-SoftwareLocalization/$name.git" "$dir"
  fi
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$dir" checkout -q "$commit"
  echo "$name @ $(git -C "$dir" rev-parse HEAD)"
  (cd "$dir" && shasum -a 256 -c CHECKSUMS.txt --quiet) && echo "  checksums OK ($(wc -l < "$dir/CHECKSUMS.txt" | tr -d ' ') files)"
}

fetch cloud-native-nmt-benchmark 05a411a1507ba8b5efae8454cd3c14bff03c2532
fetch cloud-native-llm-benchmark 6f31ddd9f60f6c6243ad062d9f84ce8ef0fa872d
