#!/bin/bash

# Setup Kafka Topic for Campus IoT Anomaly Detection
# This script creates the prediction_queue topic if it doesn't exist

set -e

KAFKA_CONTAINER="campus-iot-kafka"
TOPIC_NAME="prediction_queue"
BOOTSTRAP_SERVER="localhost:9092"
PARTITIONS=3
REPLICATION_FACTOR=1

echo "Setting up Kafka topic: $TOPIC_NAME"

# Check if Kafka container is running
if ! docker ps | grep -q "$KAFKA_CONTAINER"; then
    echo "Error: Kafka container '$KAFKA_CONTAINER' is not running"
    echo "Start it with: docker-compose up -d kafka"
    exit 1
fi

# Create topic if it doesn't exist
echo "Creating topic '$TOPIC_NAME' with $PARTITIONS partitions..."
docker exec $KAFKA_CONTAINER kafka-topics.sh --create \
  --bootstrap-server $BOOTSTRAP_SERVER \
  --topic $TOPIC_NAME \
  --partitions $PARTITIONS \
  --replication-factor $REPLICATION_FACTOR \
  --if-not-exists

# Verify topic was created
echo ""
echo "Verifying topic creation..."
docker exec $KAFKA_CONTAINER kafka-topics.sh --describe \
  --bootstrap-server $BOOTSTRAP_SERVER \
  --topic $TOPIC_NAME

echo ""
echo "✅ Kafka topic '$TOPIC_NAME' is ready!"
echo ""
echo "You can now:"
echo "  - Publish messages via: POST http://localhost:8002/publish"
echo "  - View messages: docker exec $KAFKA_CONTAINER kafka-console-consumer.sh --bootstrap-server $BOOTSTRAP_SERVER --topic $TOPIC_NAME --from-beginning"
