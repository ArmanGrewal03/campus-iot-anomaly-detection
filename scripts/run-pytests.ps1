<# 
Runs pytest with coverage for all Python services and opens the HTML report.
Usage:
  pwsh -File .\scripts\run-pytests.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure we run from repo root (parent of scripts)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

# Prefer local venv Python if present
$venvPython = Join-Path -Path $PSScriptRoot -ChildPath "..\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = (Resolve-Path $venvPython).Path
} else {
    $python = "python"
}

Write-Host "Using Python: $python" -ForegroundColor Cyan

# Ensure pytest is available; install pytest and pytest-cov if missing
try {
    & $python -m pytest --version *> $null
} catch {
    Write-Host "pytest not found; installing pytest and pytest-cov..." -ForegroundColor Yellow
    & $python -m pip install --upgrade pip
    & $python -m pip install pytest pytest-cov
}

# Build pytest command
$argsList = @(
    "-m", "pytest",
    "-q",
    "--cov=01_Data_Ingestion_Service",
    "--cov=02_Model_Service",
    "--cov=04_User_Service",
    "--cov=05_Gateway_Proxy",
    "--cov=06_Live_Metrics_Service",
    "--cov-report=term-missing",
    "--cov-report=html",
    "tests"
)

Write-Host "Running tests with coverage..." -ForegroundColor Cyan
& $python @argsList
$code = $LASTEXITCODE

if (Test-Path ".\htmlcov\index.html") {
    Write-Host "Opening coverage report: .\htmlcov\index.html" -ForegroundColor Green
    Invoke-Item ".\htmlcov\index.html"
} else {
    Write-Warning "Coverage HTML not found. Check pytest output for errors."
}

exit $code

