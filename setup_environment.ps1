param(
  [string]$BasePython = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
  & $BasePython -m venv $Venv
  if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment" }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-core.txt")
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-figures.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

& $Python (Join-Path $ProjectRoot "code\verify_reproducibility.py")

