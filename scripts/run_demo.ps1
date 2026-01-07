$ErrorActionPreference = "Stop"

$Artifacts = "./artifacts"
$Db = "$Artifacts/memory.sqlite"
$Plot = "$Artifacts/results.png"

New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

$env:RSLM_FAST_VERIFY = "1"

rslm init-db --db $Db
rslm seed-tasks

1..3 | ForEach-Object {
  Write-Host "Running iteration $_"
  $Iter = $_.ToString("000")
  $ResultsIter = "$Artifacts/results_iter$Iter.json"
  rslm run-iteration --db $Db --tasks bundled --k 4 --mode trainpool --backend mock --memory-enabled --heldout-size 20 --task-limit 30
  rslm consolidate --db $Db --min-evidence 1
  rslm eval --db $Db --backend mock --conditions all --k 1 --heldout-size 20 --task-limit 20 --output $ResultsIter
}

rslm plot --input $Db --output $Plot

Write-Host "Demo complete. Results: $Artifacts/results_iter*.json"
