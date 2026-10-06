$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Python=Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CodeRoot=$PSScriptRoot
$Cache=Join-Path $ProjectRoot 'data\routecast_cache\cross_token_limit100_h16'
$Models=Join-Path $CodeRoot 'models'
$Output=Join-Path $ProjectRoot 'record\baselines\strong_cross_token_limit100'
$Paper=Join-Path $ProjectRoot 'record\baselines\paper_heatmap_limit100.json'
$RequestAware=Join-Path $Output 'request_aware_heatmap_limit100.json'
$GruModel=Join-Path $Models 'gru_only_cross_token_limit100.pt'
New-Item -ItemType Directory -Force -Path $Output|Out-Null
Set-Location $CodeRoot

Write-Host "`n================ REQUEST-AWARE HEATMAP ================"
& $Python '.\evaluate_request_aware_heatmap.py' --cache $Cache --output $RequestAware --alpha-step 0.05 --batch-size 2048 2>&1 | Tee-Object -FilePath (Join-Path $Output 'request_aware_heatmap.log')
if($LASTEXITCODE -ne 0){throw 'Request-aware heatmap failed'}

Write-Host "`n================ GRU-ONLY ================"
& $Python '.\train_universal_routecast_v3.py' --cache $Cache --epochs 3 --batch-size 256 --seed 2026 --save $GruModel `
  --disable-branches popularity one_step two_hop prefill multiscale 2>&1 | Tee-Object -FilePath (Join-Path $Output 'gru_only_train.log')
if($LASTEXITCODE -ne 0){throw 'GRU-only training failed'}

& $Python '.\summarize_strong_baselines.py' `
  --paper $Paper --request-aware $RequestAware --gru ($GruModel.Replace('.pt','.json')) `
  --routecast (Join-Path $Models 'routecast_cross_token_limit100_v3.json') --output-dir $Output
if($LASTEXITCODE -ne 0){throw 'Strong baseline summary failed'}
