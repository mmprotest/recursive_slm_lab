#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS=./artifacts_real
DB=$ARTIFACTS/memory.sqlite
PLOT=$ARTIFACTS/results.png

mkdir -p "$ARTIFACTS/adapters"

export RSLM_BACKEND=localhf
export RSLM_HF_MODEL_ID=${RSLM_HF_MODEL_ID:-Qwen/Qwen3-4B-Thinking-2507}
export RSLM_FAST_VERIFY=1

rslm init-db --db "$DB"
rslm seed-tasks --count 200

for i in 1 2 3 4 5; do
  echo "Running iteration $i"
  ITER=$(printf "%03d" "$i")
  RESULTS_ITER=$ARTIFACTS/results_iter${ITER}.json
  ADAPTER_DIR=$ARTIFACTS/adapters/iter${ITER}

  rslm run-iteration --db "$DB" --tasks bundled --k 2 --mode trainpool --backend localhf \
    --memory-enabled --heldout-size 40 --task-limit 25 --unseen-only --train-seed 1337 \
    --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50

  rslm consolidate --db "$DB" --min-evidence 1
  set +e
  rslm train-lora --db "$DB" --out "$ADAPTER_DIR"
  TRAIN_STATUS=$?
  set -e
  if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "Warning: train-lora failed (exit $TRAIN_STATUS). Continuing without learning for iter${ITER}." >&2
  else
    rslm set-active-adapter --db "$DB" --name "iter${ITER}"
  fi

  rslm eval --db "$DB" --backend localhf --conditions baseline,memory,learning,memory_learning \
    --k 1 --heldout-size 40 --task-limit 40 --output "$RESULTS_ITER" \
    --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50

done

rslm wipe-memory --db "$DB"
rslm eval --db "$DB" --backend localhf --conditions learning --k 1 --heldout-size 40 --task-limit 40 \
  --output "$ARTIFACTS/results_after_wipe.json" --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50

rslm plot --input "$DB" --output "$PLOT"

echo "Real demo complete. Results: $ARTIFACTS/results_iter*.json and $ARTIFACTS/results_after_wipe.json"
