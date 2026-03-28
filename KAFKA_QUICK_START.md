# Kafka Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Start Kafka with Docker Compose

```bash
# Start Kafka and Zookeeper
docker-compose up -d zookeeper kafka

# Wait for Kafka to be ready (about 30 seconds)
docker-compose logs -f kafka
# Press Ctrl+C when you see "started (kafka.server.KafkaServer)"
```

### 2. Create the Topic

**Linux/Mac:**
```bash
chmod +x scripts/setup_kafka_topic.sh
./scripts/setup_kafka_topic.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\setup_kafka_topic.ps1
```

**Or manually:**
```bash
docker exec campus-iot-kafka kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists
```

### 3. Start User Service

```bash
# Start all services (Kafka will be included)
docker-compose up -d

# Or start just user-service (if Kafka is already running)
docker-compose up -d user-service
```

### 4. Verify Setup

```bash
# Check Kafka is running
docker-compose ps kafka

# Check topic exists
docker exec campus-iot-kafka kafka-topics.sh --list \
  --bootstrap-server localhost:9092

# Test publishing a message
curl -X POST http://localhost:8002/publish \
  -H "Content-Type: application/json" \
  -d '{
    "network_id": "NET-001",
    "data": {"feature1": 1.0, "feature2": 2.0}
  }'
```

## 📋 Environment Variables

The User Service needs these environment variables (already set in docker-compose.yml):

| Variable | Value | Description |
|----------|-------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9093` | Kafka broker address (use `localhost:9092` for local) |
| `KAFKA_TOPIC` | `prediction_queue` | Topic name |
| `KAFKA_CONSUMER_GROUP` | `prediction_consumer_group` | Consumer group ID |
| `KAFKA_AUTO_OFFSET_RESET` | `earliest` | Read from beginning if no offset |
| `MESSAGE_QUEUE_ENABLED` | `true` | Enable message queue |

## 🔍 Common Commands

### View Messages
```bash
docker exec campus-iot-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --from-beginning
```

### Check Consumer Status
```bash
docker exec campus-iot-kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group prediction_consumer_group \
  --describe
```

### View Logs
```bash
# Kafka logs
docker-compose logs -f kafka

# User Service logs (to see Kafka connection)
docker-compose logs -f user-service | grep -i kafka
```

## ⚠️ Troubleshooting

### Kafka won't start
```bash
# Check if Zookeeper is running
docker-compose ps zookeeper

# Check logs
docker-compose logs kafka
```

### Can't connect to Kafka
```bash
# Verify Kafka is healthy
docker-compose ps kafka

# Test connection
docker exec campus-iot-kafka kafka-broker-api-versions.sh \
  --bootstrap-server localhost:9092
```

### Topic doesn't exist
```bash
# Create it manually
docker exec campus-iot-kafka kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --partitions 3 \
  --replication-factor 1
```

## 📚 More Information

See [KAFKA_SETUP.md](./KAFKA_SETUP.md) for detailed configuration and troubleshooting.
