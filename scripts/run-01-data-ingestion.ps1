# PowerShell script to run the 01 Data Ingestion Service
# This script activates the virtual environment and starts the FastAPI backend on port 8000

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting 01 Data Ingestion Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$serviceDir = Join-Path $projectRoot "01_Data_Ingestion_Service"

# Check if service directory exists
if (-not (Test-Path $serviceDir)) {
    Write-Host "Error: Service directory not found at $serviceDir" -ForegroundColor Red
    exit 1
}

# Change to service directory
Set-Location $serviceDir

# Check if virtual environment exists and is valid
$venvPath = Join-Path $serviceDir "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPath) -or -not (Test-Path $venvPython)) {
    if (Test-Path $venvPath) {
        Write-Host "Virtual environment appears corrupted. Recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    }
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "$venvPath\Scripts\Activate.ps1"

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Green
$requirementsFile = Join-Path $serviceDir "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing/updating requirements..." -ForegroundColor Yellow
    python -m pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some dependencies may not have installed correctly" -ForegroundColor Yellow
    }
}

# Start the FastAPI service
Write-Host "Starting FastAPI server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
