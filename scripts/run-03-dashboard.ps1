# PowerShell script to run the 03 Dashboard (Vue.js)
# This script installs dependencies and starts the Vue.js development server

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting 03 Dashboard (Vue.js)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$serviceDir = Join-Path $projectRoot "03_Dashboard"

# Check if service directory exists
if (-not (Test-Path $serviceDir)) {
    Write-Host "Error: Service directory not found at $serviceDir" -ForegroundColor Red
    exit 1
}

# Change to service directory
Set-Location $serviceDir

# Check if node_modules exists
$nodeModulesPath = Join-Path $serviceDir "node_modules"
if (-not (Test-Path $nodeModulesPath)) {
    Write-Host "Node modules not found. Installing dependencies..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install npm dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

# Check if npm is available
$npmCheck = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCheck) {
    Write-Host "Error: npm is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Start the Vue.js development server
Write-Host "Starting Vue.js development server..." -ForegroundColor Green
Write-Host "The dashboard will open automatically in your browser" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

npm run serve
