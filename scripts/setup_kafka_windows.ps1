# Complete Kafka Setup Script for Windows
# This script checks Docker, starts Kafka, and creates the topic

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Kafka Setup for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Docker is running
Write-Host "Step 1: Checking Docker status..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker is running" -ForegroundColor Green
    } else {
        Write-Host "❌ Docker is not running!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please start Docker Desktop and wait for it to fully start." -ForegroundColor Yellow
        Write-Host "Then run this script again." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To start Docker Desktop:" -ForegroundColor Cyan
        Write-Host "  1. Open Docker Desktop from Start Menu" -ForegroundColor White
        Write-Host "  2. Wait for the Docker icon in system tray to show 'Docker Desktop is running'" -ForegroundColor White
        Write-Host "  3. Run this script again" -ForegroundColor White
        exit 1
    }
} catch {
    Write-Host "❌ Docker is not installed or not accessible!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Navigate to project directory
Write-Host "Step 2: Checking project directory..." -ForegroundColor Yellow
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ docker-compose.yml not found in $projectRoot" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found docker-compose.yml" -ForegroundColor Green
Write-Host ""

# Step 3: Start Zookeeper and Kafka
Write-Host "Step 3: Starting Zookeeper and Kafka..." -ForegroundColor Yellow
Write-Host "This may take a minute to download images if this is the first time..." -ForegroundColor Gray

docker-compose up -d zookeeper kafka

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start Kafka services" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Zookeeper and Kafka containers started" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for Kafka to be ready
Write-Host "Step 4: Waiting for Kafka to be ready (this may take 30-60 seconds)..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$kafkaReady = $false

while ($attempt -lt $maxAttempts -and -not $kafkaReady) {
    Start-Sleep -Seconds 2
    $attempt++
    
    try {
        $result = docker exec campus-iot-kafka /usr/bin/kafka-broker-api-versions --bootstrap-server localhost:9092 2>&1
        if ($LASTEXITCODE -eq 0) {
            $kafkaReady = $true
            Write-Host "✅ Kafka is ready!" -ForegroundColor Green
        } else {
            Write-Host "." -NoNewline -ForegroundColor Gray
        }
    } catch {
        Write-Host "." -NoNewline -ForegroundColor Gray
    }
}

if (-not $kafkaReady) {
    Write-Host ""
    Write-Host "⚠️  Kafka may not be fully ready yet. Continuing anyway..." -ForegroundColor Yellow
    Write-Host "If topic creation fails, wait a bit longer and run:" -ForegroundColor Yellow
    Write-Host "  .\scripts\setup_kafka_topic.ps1" -ForegroundColor Cyan
}

Write-Host ""

# Step 5: Create Kafka topic
Write-Host "Step 5: Creating Kafka topic 'prediction_queue'..." -ForegroundColor Yellow

$KAFKA_CONTAINER = "campus-iot-kafka"
$TOPIC_NAME = "prediction_queue"
$BOOTSTRAP_SERVER = "localhost:9092"
$PARTITIONS = 3
$REPLICATION_FACTOR = 1

# Check if container is running
$containerRunning = docker ps --filter "name=$KAFKA_CONTAINER" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "❌ Kafka container '$KAFKA_CONTAINER' is not running" -ForegroundColor Red
    Write-Host "Check logs with: docker-compose logs kafka" -ForegroundColor Yellow
    exit 1
}

# Create topic
Write-Host "Creating topic with $PARTITIONS partitions..." -ForegroundColor Gray
docker exec $KAFKA_CONTAINER /usr/bin/kafka-topics --create `
  --bootstrap-server $BOOTSTRAP_SERVER `
  --topic $TOPIC_NAME `
  --partitions $PARTITIONS `
  --replication-factor $REPLICATION_FACTOR `
  --if-not-exists

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Topic created successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  Topic creation returned exit code $LASTEXITCODE" -ForegroundColor Yellow
    Write-Host "This might be okay if the topic already exists" -ForegroundColor Gray
}

Write-Host ""

# Step 6: Verify topic
Write-Host "Step 6: Verifying topic..." -ForegroundColor Yellow
docker exec $KAFKA_CONTAINER /usr/bin/kafka-topics --describe `
  --bootstrap-server $BOOTSTRAP_SERVER `
  --topic $TOPIC_NAME

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Kafka Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start the User Service:" -ForegroundColor White
    Write-Host "     docker-compose up -d user-service" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Or start all services:" -ForegroundColor White
    Write-Host "     docker-compose up -d" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. Test publishing a message:" -ForegroundColor White
    Write-Host "     curl -X POST http://localhost:8002/publish -H 'Content-Type: application/json' -d '{\"network_id\":\"NET-001\",\"data\":{\"feature1\":1.0}}'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  4. View messages:" -ForegroundColor White
    Write-Host "     docker exec campus-iot-kafka /usr/bin/kafka-console-consumer --bootstrap-server localhost:9092 --topic prediction_queue --from-beginning" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "⚠️  Could not verify topic. Check Kafka logs:" -ForegroundColor Yellow
    Write-Host "   docker-compose logs kafka" -ForegroundColor Cyan
}
