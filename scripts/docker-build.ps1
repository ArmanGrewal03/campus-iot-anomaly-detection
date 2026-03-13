# PowerShell script to build all Docker images

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Docker Images" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Set-Location $projectRoot

Write-Host "Building Data Ingestion Service..." -ForegroundColor Yellow
docker build -t campus-iot-data-ingestion ./01_Data_Ingestion_Service
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build Data Ingestion Service" -ForegroundColor Red
    exit 1
}

Write-Host "Building Model Service..." -ForegroundColor Yellow
docker build -t campus-iot-model-service ./02_Model_Service
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build Model Service" -ForegroundColor Red
    exit 1
}

Write-Host "Building User Service..." -ForegroundColor Yellow
docker build -t campus-iot-user-service ./04_User_Service
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build User Service" -ForegroundColor Red
    exit 1
}

Write-Host "Building Gateway..." -ForegroundColor Yellow
docker build -t campus-iot-gateway ./05_Gateway_Proxy
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build Gateway" -ForegroundColor Red
    exit 1
}

Write-Host "Building Dashboard..." -ForegroundColor Yellow
docker build -t campus-iot-dashboard ./03_Dashboard
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build Dashboard" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All images built successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start all services, run:" -ForegroundColor Cyan
Write-Host "  docker-compose up -d" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f" -ForegroundColor White
