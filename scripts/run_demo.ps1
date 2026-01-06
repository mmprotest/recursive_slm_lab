$ErrorActionPreference = "Stop"

$Artifacts = "./artifacts"
$Db = "$Artifacts/memory.sqlite"
$Plot = "$Artifacts/results.png"

New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

rslm init-db --db $Db
rslm seed-tasks

1..3 | ForEach-Object {
  Write-Host "Running iteration $_"
  $Iter = $_.ToString("000")
  $ResultsIter = "$Artifacts/results_iter$Iter.json"
  rslm run-iteration --db $Db --tasks bundled --k 8 --mode trainpool --backend mock --memory-enabled
  rslm consolidate --db $Db --min-evidence 1 --backend mock
  rslm eval --db $Db --backend mock --conditions all --k 1 --heldout-size 40 --output $ResultsIter
}

rslm plot --input $Db --output $Plot

Write-Host "Demo complete. Results: $Artifacts/results_iter*.json"
