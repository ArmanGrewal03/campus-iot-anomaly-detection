# PowerShell script to run all services simultaneously
# This script starts the Data Ingestion Service, Model Service, User Service, and Dashboard in separate windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Paths to individual scripts
$script01 = Join-Path $scriptDir "run-01-data-ingestion.ps1"
$script02 = Join-Path $scriptDir "run-02-model-service.ps1"
$script04 = Join-Path $scriptDir "run-04-user-service.ps1"
$script03 = Join-Path $scriptDir "run-03-dashboard.ps1"

# Check if scripts exist
if (-not (Test-Path $script01)) {
    Write-Host "Error: Script not found: $script01" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script02)) {
    Write-Host "Error: Script not found: $script02" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script04)) {
    Write-Host "Error: Script not found: $script04" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script03)) {
    Write-Host "Error: Script not found: $script03" -ForegroundColor Red
    exit 1
}

Write-Host "Starting services in separate PowerShell windows..." -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  - Data Ingestion API: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - Model API:          http://127.0.0.1:8001" -ForegroundColor White
Write-Host "  - User Service:       http://127.0.0.1:8002 (WebSocket: ws://127.0.0.1:8002/ws/data-stream)" -ForegroundColor White
Write-Host "  - Dashboard:          http://127.0.0.1:8080 (will open automatically)" -ForegroundColor White
Write-Host ""
Write-Host "Each service will run in its own window." -ForegroundColor Yellow
Write-Host "Close the individual windows to stop each service." -ForegroundColor Yellow
Write-Host ""

# Start each service in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script01`""
Start-Sleep -Seconds 2

Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script02`""
Start-Sleep -Seconds 2

Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script04`""
Start-Sleep -Seconds 2

Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script03`""

Write-Host "All services are starting..." -ForegroundColor Green
Write-Host "Check the individual windows for service status." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit this window (services will continue running)..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
