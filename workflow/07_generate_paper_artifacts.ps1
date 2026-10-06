$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python (Join-Path $ProjectRoot "code\summarize_limit1000_main.py")
if ($LASTEXITCODE -ne 0) { throw "Main-result summary failed." }

& $Python (Join-Path $ProjectRoot "code\generate_paper_figures.py")
if ($LASTEXITCODE -ne 0) { throw "Paper figure generation failed." }

Write-Output "Paper artifacts refreshed under record\paper."

