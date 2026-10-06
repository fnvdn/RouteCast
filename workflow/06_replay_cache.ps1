param(
    [string]$Model = "",
    [string]$Calibration = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Cache = Join-Path $ProjectRoot "data\routecast_cache\cross_token_limit1000_h16_packed"
$Out = Join-Path $ProjectRoot "record\reproduction\cache"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
if (-not $Model) { $Model = Join-Path $ProjectRoot "code\models\routecast_cross_token_limit1000_v3_stable.pt" }
if (-not $Calibration) { $Calibration = Join-Path $ProjectRoot "record\reproduction\calibration\routecast_limit1000_calibration.json" }

& $Python (Join-Path $ProjectRoot "code\evaluate_adaptive_cache_v3.py") `
    --cache $Cache --model $Model --calibration $Calibration `
    --save (Join-Path $Out "routecast_limit1000_cache_replay.json") `
    --masses 0.50 0.60 0.70 0.80 0.85 0.90 0.95 `
    --selection-mode quality --recall-retention 0.98 --max-k 32 `
    --capacity-multipliers 1 2 4 --cache-decay 0.90 `
    --batch-size 8192
if ($LASTEXITCODE -ne 0) { throw "Idealized cache replay failed." }

