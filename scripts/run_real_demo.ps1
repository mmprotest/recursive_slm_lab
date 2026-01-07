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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
rslm seed-tasks --count 200
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($i in 1..5) {
  Write-Host "Running iteration $i"
  $iter = $i.ToString("000")
  $resultsIter = Join-Path $Artifacts "results_iter$iter.json"
  $adapterDir = Join-Path $Artifacts "adapters/iter$iter"
  $adapterName = "iter$iter"
  $learnedUnavailable = $false

  rslm run-iteration --db $Db --tasks bundled --k 2 --mode trainpool --backend localhf `
    --memory-enabled --heldout-size 40 --task-limit 25 --unseen-only --train-seed 1337 `
    --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  rslm consolidate --db $Db --min-evidence 1
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  rslm train-lora --db $Db --out $adapterDir
  $trainExit = $LASTEXITCODE
  if ($trainExit -ne 0) {
    Write-Warning "train-lora failed or was skipped for $adapterName; continuing without learning."
    $learnedUnavailable = $true
  }

  $adapterRegistered = $false
  if ($trainExit -eq 0) {
    $adapterList = rslm list-adapters --db $Db
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $adapterRegistered = $adapterList | Select-String -Pattern ("^{0}:" -f [regex]::Escape($adapterName))
  }

  if ($adapterRegistered -and (Test-Path $adapterDir)) {
    rslm set-active-adapter --db $Db --name $adapterName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  } else {
    if (-not $learnedUnavailable) {
      Write-Warning "No registered adapter found for $adapterName; continuing without learning."
      $learnedUnavailable = $true
    }
  }
  if ($learnedUnavailable) {
    $adapterList = rslm list-adapters --db $Db
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $activeAdapter = $null
    foreach ($line in $adapterList) {
      if ($line -match "^(?<name>[^:]+):.*\\(active\\)$") {
        $activeAdapter = $Matches.name
        break
      }
    }
    if ($activeAdapter) {
      Write-Warning "Deactivating active adapter $activeAdapter because learning is unavailable."
      rslm rollback-adapter --db $Db --name $activeAdapter
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
  }

  rslm eval --db $Db --backend localhf --conditions baseline,memory,learning,memory_learning `
    --k 1 --heldout-size 40 --task-limit 40 --output $resultsIter `
    --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

rslm wipe-memory --db $Db
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
rslm eval --db $Db --backend localhf --conditions learning --k 1 --heldout-size 40 --task-limit 40 `
  --output (Join-Path $Artifacts "results_after_wipe.json") --max-tokens 512 --temperature 0.2 --top-p 0.9 --top-k 50
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

rslm plot --input $Db --output $Plot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Real demo complete. Results: $Artifacts/results_iter*.json and $Artifacts/results_after_wipe.json"
