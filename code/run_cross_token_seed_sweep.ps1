param(
  [int[]]$Seeds = @(2024, 2025)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CodeRoot = $PSScriptRoot
$Cache = Join-Path $ProjectRoot 'data\routecast_cache\cross_token_limit100_h16'
$Models = Join-Path $CodeRoot 'models'
$Baseline = Join-Path $ProjectRoot 'record\baselines\paper_heatmap_limit100.json'
$Output = Join-Path $ProjectRoot 'record\seed_repro\cross_token_limit100'

New-Item -ItemType Directory -Force -Path $Models | Out-Null
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Set-Location $CodeRoot

foreach ($Seed in $Seeds) {
  $ModelPath = Join-Path $Models "routecast_cross_token_limit100_v3_seed$Seed.pt"
  $LogPath = Join-Path $Output "train_seed$Seed.log"
  Write-Host "`n================ seed=$Seed START ================"
  & $Python '.\train_universal_routecast_v3.py' `
    --cache $Cache `
    --epochs 3 `
    --batch-size 256 `
    --seed $Seed `
    --save $ModelPath `
    --baseline-report $Baseline 2>&1 | Tee-Object -FilePath $LogPath
  if ($LASTEXITCODE -ne 0) {
    throw "Training failed for seed=$Seed with exit code $LASTEXITCODE"
  }
  Write-Host "================ seed=$Seed COMPLETE ================`n"
}

& $Python '.\summarize_cross_token_seeds.py' `
  --models-dir $Models `
  --baseline-report $Baseline `
  --output-dir $Output `
  --seeds 2024 2025 2026
if ($LASTEXITCODE -ne 0) {
  throw "Seed summary failed with exit code $LASTEXITCODE"
}
