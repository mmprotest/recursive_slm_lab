$ErrorActionPreference = "Stop"

$Artifacts = "./artifacts"
$Db = "$Artifacts/memory.sqlite"
$Results = "$Artifacts/results.json"
$Plot = "$Artifacts/results.png"

New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

rslm init-db --db $Db
rslm seed-tasks

1..3 | ForEach-Object {
  Write-Host "Running iteration $_"
  rslm run-iteration --db $Db --tasks bundled --k 8 --mode trainpool --backend mock
  rslm consolidate --db $Db --min-evidence 3 --backend mock
  rslm eval --db $Db --backend mock --conditions all --k 1 --heldout-size 40 --output $Results
}

rslm plot --input $Results --output $Plot

Write-Host "Demo complete. Results: $Results"
