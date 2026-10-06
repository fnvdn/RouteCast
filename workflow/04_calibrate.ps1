param(
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Cache = Join-Path $ProjectRoot "data\routecast_cache\cross_token_limit1000_h16_packed"
$Out = Join-Path $ProjectRoot "record\reproduction\calibration"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
if (-not $Model) { $Model = Join-Path $ProjectRoot "code\models\routecast_cross_token_limit1000_v3_stable.pt" }

& $Python (Join-Path $ProjectRoot "code\calibrate_routecast_v3.py") `
    --cache $Cache `
    --model $Model `
    --save (Join-Path $Out "routecast_limit1000_calibration.json") `
    --batch-size 8192
if ($LASTEXITCODE -ne 0) { throw "Calibration failed." }

