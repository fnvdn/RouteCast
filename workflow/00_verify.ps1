$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found. Run setup_environment.ps1 first."
}

& $Python (Join-Path $ProjectRoot "code\verify_reproducibility.py")
if ($LASTEXITCODE -ne 0) { throw "Reproducibility audit failed." }

