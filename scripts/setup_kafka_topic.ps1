# Setup Kafka Topic for Campus IoT Anomaly Detection (PowerShell)
# This script creates the prediction_queue topic if it doesn't exist

$KAFKA_CONTAINER = "campus-iot-kafka"
$TOPIC_NAME = "prediction_queue"
$BOOTSTRAP_SERVER = "localhost:9092"
$PARTITIONS = 3
$REPLICATION_FACTOR = 1

Write-Host "Setting up Kafka topic: $TOPIC_NAME" -ForegroundColor Cyan

# Check if Kafka container is running
$containerRunning = docker ps --filter "name=$KAFKA_CONTAINER" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "Error: Kafka container '$KAFKA_CONTAINER' is not running" -ForegroundColor Red
    Write-Host "Start it with: docker-compose up -d kafka" -ForegroundColor Yellow
    exit 1
}

# Create topic if it doesn't exist
Write-Host "Creating topic '$TOPIC_NAME' with $PARTITIONS partitions..." -ForegroundColor Yellow
docker exec $KAFKA_CONTAINER /usr/bin/kafka-topics --create `
  --bootstrap-server $BOOTSTRAP_SERVER `
  --topic $TOPIC_NAME `
  --partitions $PARTITIONS `
  --replication-factor $REPLICATION_FACTOR `
  --if-not-exists

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Topic creation may have failed. It might already exist." -ForegroundColor Yellow
}

# Verify topic was created
Write-Host ""
Write-Host "Verifying topic creation..." -ForegroundColor Cyan
docker exec $KAFKA_CONTAINER /usr/bin/kafka-topics --describe `
  --bootstrap-server $BOOTSTRAP_SERVER `
  --topic $TOPIC_NAME

Write-Host ""
Write-Host "✅ Kafka topic '$TOPIC_NAME' is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now:" -ForegroundColor Cyan
Write-Host "  - Publish messages via: POST http://localhost:8002/publish"
Write-Host "  - View messages: docker exec $KAFKA_CONTAINER /usr/bin/kafka-console-consumer --bootstrap-server $BOOTSTRAP_SERVER --topic $TOPIC_NAME --from-beginning"
