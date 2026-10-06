param(
    [int]$Epochs = 3,
    [int]$BatchSize = 8192,
    [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Cache = Join-Path $ProjectRoot "data\routecast_cache\cross_token_limit1000_h16_packed"
$Save = Join-Path $ProjectRoot "code\models\routecast_cross_token_limit1000_v3_reproduced.pt"
$Baseline = Join-Path $ProjectRoot "record\baselines\paper_heatmap_limit1000.json"

& $Python (Join-Path $ProjectRoot "code\train_universal_routecast_v3.py") `
    --cache $Cache `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --seed $Seed `
    --baseline-report $Baseline `
    --save $Save
if ($LASTEXITCODE -ne 0) { throw "RouteCast training failed." }

