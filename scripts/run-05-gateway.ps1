# PowerShell script to run the API Gateway Service

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting API Gateway Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$gatewayDir = Join-Path $projectRoot "05_Gateway_Proxy"

# Check if gateway directory exists
if (-not (Test-Path $gatewayDir)) {
    Write-Host "Error: Gateway directory not found: $gatewayDir" -ForegroundColor Red
    exit 1
}

# Change to gateway directory
Set-Location $gatewayDir

# Check if virtual environment exists
$venvPath = Join-Path $gatewayDir "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Install/upgrade dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting API Gateway on http://127.0.0.1:8003" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Yellow
Write-Host ""

# Run the gateway service
python gateway.py
