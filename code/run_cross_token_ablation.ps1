$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CodeRoot = $PSScriptRoot
$Cache = Join-Path $ProjectRoot 'data\routecast_cache\cross_token_limit100_h16'
$Models = Join-Path $CodeRoot 'models'
$Baseline = Join-Path $ProjectRoot 'record\baselines\paper_heatmap_limit100.json'
$Output = Join-Path $ProjectRoot 'record\ablations\cross_token_limit100'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Set-Location $CodeRoot

$Runs = @(
  @{Name='no_history'; Disabled=@('temporal','multiscale')},
  @{Name='no_two_hop'; Disabled=@('two_hop')},
  @{Name='no_prefill'; Disabled=@('prefill')}
)
foreach($Run in $Runs){
  $Save=Join-Path $Models ("routecast_cross_token_ablation_"+$Run.Name+'.pt')
  $Log=Join-Path $Output ("train_"+$Run.Name+'.log')
  Write-Host "`n================ ABLATION $($Run.Name) START ================"
  & $Python '.\train_universal_routecast_v3.py' `
    --cache $Cache --epochs 3 --batch-size 256 --seed 2026 `
    --save $Save --baseline-report $Baseline `
    --disable-branches $Run.Disabled 2>&1 | Tee-Object -FilePath $Log
  if($LASTEXITCODE -ne 0){throw "Ablation failed: $($Run.Name)"}
}
& $Python '.\summarize_cross_token_ablation.py' --models-dir $Models --output-dir $Output
if($LASTEXITCODE -ne 0){throw 'Ablation summary failed'}
