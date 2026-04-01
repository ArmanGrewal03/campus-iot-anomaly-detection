# PowerShell script to run the 07 ML-Ops Service
# Separate service that handles model training, validation, and lifecycle management
# Proxies to Data Ingestion Service (8000) and Model Service (8001)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting 07 ML-Ops Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$serviceDir = Join-Path $projectRoot "07_ML_Ops_Service"

if (-not (Test-Path $serviceDir)) {
    Write-Host "Error: Service directory not found at $serviceDir" -ForegroundColor Red
    exit 1
}

Set-Location $serviceDir

$venvPath = Join-Path $serviceDir "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPath) -or -not (Test-Path $venvPython)) {
    if (Test-Path $venvPath) { Remove-Item -Recurse -Force $venvPath }
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

& "$venvPath\Scripts\Activate.ps1"
$requirementsFile = Join-Path $serviceDir "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing/updating requirements..." -ForegroundColor Yellow
    python -m pip install -q -r requirements.txt
}

Write-Host "Starting ML-Ops Service on http://127.0.0.1:8004" -ForegroundColor Green
Write-Host "Endpoints:" -ForegroundColor Green
Write-Host "  POST   /train (train a model)" -ForegroundColor Green
Write-Host "  POST   /test (test a model)" -ForegroundColor Green
Write-Host "  GET    /health (service health)" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn ml_ops:app --host 127.0.0.1 --port 8004 --reload
