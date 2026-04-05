# Model Dashboard launcher script (Port 5174)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
$dashboardPath = Join-Path $projectRoot "04_Model_Dashboard"

Write-Host "Starting Model Management Dashboard (Port 5174)..." -ForegroundColor Cyan

# Setup: Create venv and install dependencies
Set-Location $dashboardPath

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Run Vite dev server
Write-Host "Dashboard running on http://localhost:5174" -ForegroundColor Green
npm run dev
