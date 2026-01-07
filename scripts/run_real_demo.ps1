$ErrorActionPreference = "Stop"

$Artifacts = "./artifacts_real"
$Db = Join-Path $Artifacts "memory.sqlite"
$Plot = Join-Path $Artifacts "results.png"

New-Item -ItemType Directory -Force -Path (Join-Path $Artifacts "adapters") | Out-Null

$env:RSLM_BACKEND = "localhf"
if (-not $env:RSLM_HF_MODEL_ID) {
  $env:RSLM_HF_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
}
$env:RSLM_FAST_VERIFY = "1"

rslm init-db --db $Db
rslm seed-tasks --count 200

foreach ($i in 1..5) {
  Write-Host "Running iteration $i"
  $iter = $i.ToString("000")
  $resultsIter = Join-Path $Artifacts "results_iter$iter.json"
  $adapterDir = Join-Path $Artifacts "adapters/iter$iter"

  rslm run-iteration --db $Db --tasks bundled --k 2 --mode trainpool --backend localhf `
    --memory-enabled --heldout-size 40 --task-limit 25 --unseen-only true --train-seed 1337 `
    --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50

  rslm consolidate --db $Db --min-evidence 1 --backend localhf
  rslm train-lora --db $Db --out $adapterDir
  rslm set-active-adapter --db $Db --name "iter$iter"

  rslm eval --db $Db --backend localhf --conditions baseline,memory,learning,memory_learning `
    --k 1 --heldout-size 40 --task-limit 40 --output $resultsIter `
    --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50
}

rslm wipe-memory --db $Db
rslm eval --db $Db --backend localhf --conditions learning --k 1 --heldout-size 40 --task-limit 40 `
  --output (Join-Path $Artifacts "results_after_wipe.json") --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50

rslm plot --input $Db --output $Plot

Write-Host "Real demo complete. Results: $Artifacts/results_iter*.json and $Artifacts/results_after_wipe.json"
