# PowerShell script to run the 06 Live Metrics Service
# Standalone service that generates mock live time-series for dashboard tiles

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting 06 Live Metrics Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$serviceDir = Join-Path $projectRoot "06_Live_Metrics_Service"

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

Write-Host "Starting Live Metrics Service on http://127.0.0.1:8010" -ForegroundColor Green
Write-Host "GET /health  GET /metrics" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn live_metrics_service:app --host 127.0.0.1 --port 8010 --reload
