# Kafka Setup Script for Windows
# This script sets up Kafka and Zookeeper for the Campus IoT Anomaly Detection project
# Usage: .\setup_kafka.ps1

param(
    [switch]$SkipDockerCheck,
    [switch]$SkipTopicCreation
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kafka Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
if (-not $SkipDockerCheck) {
    Write-Host "[1/5] Checking Docker..." -ForegroundColor Yellow
    try {
        docker info | Out-Null
        Write-Host "       ✅ Docker is running" -ForegroundColor Green
    } catch {
        Write-Host "       ❌ Docker is not running!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
        exit 1
    }
    Write-Host ""
}

# Step 2: Navigate to project root
Write-Host "[2/5] Checking project directory..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "       ❌ docker-compose.yml not found!" -ForegroundColor Red
    Write-Host "       Please run this script from the project root directory." -ForegroundColor Yellow
    exit 1
}
Write-Host "       ✅ Found docker-compose.yml" -ForegroundColor Green
Write-Host ""

# Step 3: Start Zookeeper and Kafka
Write-Host "[3/5] Starting Zookeeper and Kafka..." -ForegroundColor Yellow
Write-Host "       (This may take a minute on first run)" -ForegroundColor Gray

$startResult = docker-compose up -d zookeeper kafka 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "       ❌ Failed to start services" -ForegroundColor Red
    Write-Host $startResult
    exit 1
}

Write-Host "       ✅ Containers started" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for Kafka to be ready
Write-Host "[4/5] Waiting for Kafka to be ready..." -ForegroundColor Yellow
$maxWait = 60  # seconds
$waitInterval = 2
$elapsed = 0
$kafkaReady = $false

while ($elapsed -lt $maxWait -and -not $kafkaReady) {
    Start-Sleep -Seconds $waitInterval
    $elapsed += $waitInterval
    
    try {
        $null = docker exec campus-iot-kafka /usr/bin/kafka-broker-api-versions --bootstrap-server localhost:9092 2>&1
        if ($LASTEXITCODE -eq 0) {
            $kafkaReady = $true
        }
    } catch {
        # Continue waiting
    }
    
    if (-not $kafkaReady) {
        Write-Host "." -NoNewline -ForegroundColor Gray
    }
}

if ($kafkaReady) {
    Write-Host ""
    Write-Host "       ✅ Kafka is ready!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "       ⚠️  Kafka may not be fully ready" -ForegroundColor Yellow
    Write-Host "       Continuing anyway..." -ForegroundColor Gray
}
Write-Host ""

# Step 5: Create topic
if (-not $SkipTopicCreation) {
    Write-Host "[5/5] Creating Kafka topic..." -ForegroundColor Yellow
    
    $topicName = "prediction_queue"
    $partitions = 3
    $replicationFactor = 1
    
    # Check if container is running
    $containerStatus = docker ps --filter "name=campus-iot-kafka" --format "{{.Status}}"
    if (-not $containerStatus) {
        Write-Host "       ❌ Kafka container is not running" -ForegroundColor Red
        Write-Host "       Check logs: docker-compose logs kafka" -ForegroundColor Yellow
        exit 1
    }
    
    # Create topic
    Write-Host "       Creating topic '$topicName' with $partitions partitions..." -ForegroundColor Gray
    $createResult = docker exec campus-iot-kafka /usr/bin/kafka-topics --create `
        --bootstrap-server localhost:9092 `
        --topic $topicName `
        --partitions $partitions `
        --replication-factor $replicationFactor `
        --if-not-exists 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "       ✅ Topic created successfully" -ForegroundColor Green
    } else {
        # Check if topic already exists
        if ($createResult -match "already exists" -or $createResult -match "Created topic") {
            Write-Host "       ✅ Topic already exists or was created" -ForegroundColor Green
        } else {
            Write-Host "       ⚠️  Topic creation warning:" -ForegroundColor Yellow
            Write-Host $createResult
        }
    }
    
    # Verify topic
    Write-Host ""
    Write-Host "       Verifying topic..." -ForegroundColor Gray
    docker exec campus-iot-kafka /usr/bin/kafka-topics --describe `
        --bootstrap-server localhost:9092 `
        --topic $topicName | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "       ✅ Topic verified" -ForegroundColor Green
    }
    Write-Host ""
}

# Summary
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ Kafka Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Show container status
Write-Host "Container Status:" -ForegroundColor Cyan
docker ps --filter "name=campus-iot" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Out-String
Write-Host ""

# Show topic list
Write-Host "Available Topics:" -ForegroundColor Cyan
docker exec campus-iot-kafka /usr/bin/kafka-topics --list --bootstrap-server localhost:9092
Write-Host ""

# Next steps
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Start User Service:" -ForegroundColor White
Write-Host "     docker-compose up -d user-service" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Or start all services:" -ForegroundColor White
Write-Host "     docker-compose up -d" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Test publishing:" -ForegroundColor White
Write-Host "     curl -X POST http://localhost:8002/publish -H 'Content-Type: application/json' -d '{\"network_id\":\"NET-001\",\"data\":{\"feature1\":1.0}}'" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. View messages:" -ForegroundColor White
Write-Host "     docker exec campus-iot-kafka /usr/bin/kafka-console-consumer --bootstrap-server localhost:9092 --topic prediction_queue --from-beginning" -ForegroundColor Gray
Write-Host ""
