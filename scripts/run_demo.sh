#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS=./artifacts
DB=$ARTIFACTS/memory.sqlite
RESULTS=$ARTIFACTS/results.json
PLOT=$ARTIFACTS/results.png

mkdir -p "$ARTIFACTS"

rslm init-db --db "$DB"
rslm seed-tasks

for i in 1 2 3; do
  echo "Running iteration $i"
  rslm run-iteration --db "$DB" --tasks bundled --k 8 --mode trainpool --backend mock
  rslm consolidate --db "$DB" --min-evidence 3 --backend mock
  rslm eval --db "$DB" --backend mock --conditions all --k 1 --heldout-size 40 --output "$RESULTS"
done

rslm plot --input "$RESULTS" --output "$PLOT"

echo "Demo complete. Results: $RESULTS"
