#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS=./artifacts
DB=$ARTIFACTS/memory.sqlite
PLOT=$ARTIFACTS/results.png

mkdir -p "$ARTIFACTS"

rslm init-db --db "$DB"
rslm seed-tasks

for i in 1 2 3; do
  echo "Running iteration $i"
  ITER=$(printf "%03d" "$i")
  RESULTS_ITER=$ARTIFACTS/results_iter${ITER}.json
  rslm run-iteration --db "$DB" --tasks bundled --k 8 --mode trainpool --backend mock --memory-enabled
  rslm consolidate --db "$DB" --min-evidence 1 --backend mock
  rslm eval --db "$DB" --backend mock --conditions all --k 1 --heldout-size 40 --output "$RESULTS_ITER"
done

rslm plot --input "$DB" --output "$PLOT"

echo "Demo complete. Results: $ARTIFACTS/results_iter*.json"
