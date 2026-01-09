#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS=./artifacts_smoke
DB=$ARTIFACTS/memory.sqlite

rm -rf "$ARTIFACTS"
mkdir -p "$ARTIFACTS"

export RSLM_FAST_VERIFY=1

rslm init-db --db "$DB"
rslm seed-tasks

rslm self-improve \
  --db "$DB" \
  --tasks bundled \
  --cycles 1 \
  --train-k 1 \
  --train-limit 5 \
  --heldout-size 10 \
  --backend mock \
  --memory-enabled \
  --artifacts-dir "$ARTIFACTS" \
  --seed 42 \
  --verify-mode local

test -f "$ARTIFACTS/iteration_001/report.json"
test -f "$ARTIFACTS/iteration_001/report.md"

echo "Smoke demo complete."
