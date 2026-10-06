param(
    [int]$LimitPerModel = 1000,
    [int]$History = 16
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Output = Join-Path $ProjectRoot "data\routecast_cache\cross_token_limit${LimitPerModel}_h${History}_packed"

& $Python (Join-Path $ProjectRoot "code\build_cross_token_cache_v3.py") `
    --data-root (Join-Path $ProjectRoot "data") `
    --output $Output `
    --limit-per-model $LimitPerModel `
    --history $History `
    --progress-every 20 `
    --resume
if ($LASTEXITCODE -ne 0) { throw "Cache construction failed." }

