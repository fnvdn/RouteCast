param(
    [int]$Epochs = 3,
    [int]$BatchSize = 8192,
    [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Cache = Join-Path $ProjectRoot "data\routecast_cache\cross_token_limit1000_h16_packed"
$Out = Join-Path $ProjectRoot "record\reproduction\baselines"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

& $Python (Join-Path $ProjectRoot "code\evaluate_paper_heatmap.py") `
    --data-root (Join-Path $ProjectRoot "data") `
    --limit-per-model 1000 `
    --output (Join-Path $Out "paper_heatmap_limit1000.json") `
    --progress-every 20
if ($LASTEXITCODE -ne 0) { throw "Paper Heatmap evaluation failed." }

& $Python (Join-Path $ProjectRoot "code\evaluate_request_aware_heatmap.py") `
    --cache $Cache `
    --output (Join-Path $Out "request_aware_heatmap_limit1000.json") `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Request-aware Heatmap evaluation failed." }

& $Python (Join-Path $ProjectRoot "code\train_tcn_baseline.py") `
    --cache $Cache `
    --save (Join-Path $ProjectRoot "code\models\tcn_cross_token_limit1000_reproduced.pt") `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --seed $Seed `
    --routecast-report (Join-Path $ProjectRoot "code\models\routecast_cross_token_limit1000_v3_stable.json")
if ($LASTEXITCODE -ne 0) { throw "TCN training failed." }

