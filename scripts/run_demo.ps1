$ErrorActionPreference = "Stop"

$Artifacts = "./artifacts"
$Db = "$Artifacts/memory.sqlite"
$Plot = "$Artifacts/results.png"

New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

$env:RSLM_FAST_VERIFY = "1"

rslm init-db --db $Db
rslm seed-tasks

rslm self-improve --db $Db --tasks bundled --cycles 3 --train-k 4 --train-limit 30 --heldout-size 20 --backend mock --memory-enabled --artifacts-dir $Artifacts --seed 1337 --verify-mode local

rslm plot --input $Db --output $Plot

Write-Host "Demo complete. Reports: $Artifacts/iteration_*/report.json"
