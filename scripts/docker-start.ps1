# PowerShell script to start all Docker services

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Docker Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Set-Location $projectRoot

# Check if docker-compose.yml exists
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "Error: docker-compose.yml not found" -ForegroundColor Red
    exit 1
}

# Start services
Write-Host "Starting all services..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Services are starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  - Dashboard:        http://localhost:5173" -ForegroundColor White
Write-Host "  - API Gateway:       http://localhost:8003" -ForegroundColor White
Write-Host "  - Data Ingestion:    http://localhost:8000" -ForegroundColor White
Write-Host "  - Model Service:     http://localhost:8001" -ForegroundColor White
Write-Host "  - User Service:       http://localhost:8002" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f" -ForegroundColor White
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Cyan
Write-Host "  docker-compose down" -ForegroundColor White
