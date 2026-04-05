# Kafka Configuration Guide

This guide explains how to set up and configure Apache Kafka for the Campus IoT Anomaly Detection application.

## Table of Contents

1. [Quick Start with Docker](#quick-start-with-docker)
2. [Local Installation](#local-installation)
3. [Docker Compose Setup](#docker-compose-setup)
4. [Topic Configuration](#topic-configuration)
5. [Application Configuration](#application-configuration)
6. [Testing Kafka](#testing-kafka)
7. [Troubleshooting](#troubleshooting)

## Quick Start with Docker

The easiest way to run Kafka is using Docker Compose. Kafka is already configured in `docker-compose.yml`.

### Start Kafka

```bash
# Start Kafka and Zookeeper
docker-compose up -d kafka zookeeper

# Check status
docker-compose ps kafka zookeeper

# View logs
docker-compose logs -f kafka
```

### Create Topic

```bash
# Create the prediction_queue topic
docker-compose exec kafka kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# List topics
docker-compose exec kafka kafka-topics.sh --list \
  --bootstrap-server localhost:9092
```

## Local Installation

### Prerequisites

- Java 8 or higher
- At least 4GB RAM available

### Download Kafka

1. **Download Kafka** (latest stable version):
   ```bash
   # Linux/Mac
   wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz
   tar -xzf kafka_2.13-3.6.0.tgz
   cd kafka_2.13-3.6.0
   
   # Windows
   # Download from: https://kafka.apache.org/downloads
   # Extract to C:\kafka
   ```

2. **Start Zookeeper**:
   ```bash
   # Linux/Mac
   bin/zookeeper-server-start.sh config/zookeeper.properties
   
   # Windows PowerShell
   .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
   ```

3. **Start Kafka Broker** (in a new terminal):
   ```bash
   # Linux/Mac
   bin/kafka-server-start.sh config/server.properties
   
   # Windows PowerShell
   .\bin\windows\kafka-server-start.bat .\config\server.properties
   ```

### Create Topic (Local)

```bash
# Linux/Mac
bin/kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --partitions 3 \
  --replication-factor 1

# Windows PowerShell
.\bin\windows\kafka-topics.bat --create `
  --bootstrap-server localhost:9092 `
  --topic prediction_queue `
  --partitions 3 `
  --replication-factor 1
```

## Docker Compose Setup

Kafka is configured in `docker-compose.yml` with the following services:

- **Zookeeper**: Required for Kafka coordination (port 2181)
- **Kafka**: Message broker (port 9092)

### Environment Variables

The User Service needs these environment variables:

```yaml
environment:
  - KAFKA_BOOTSTRAP_SERVERS=kafka:9092  # For Docker
  # OR
  - KAFKA_BOOTSTRAP_SERVERS=localhost:9092  # For local Kafka
  - KAFKA_TOPIC=prediction_queue
  - KAFKA_CONSUMER_GROUP=prediction_consumer_group
  - KAFKA_AUTO_OFFSET_RESET=earliest
```

## Topic Configuration

### Recommended Topic Settings

For the prediction queue, use these settings:

```bash
# Create topic with optimal settings
docker-compose exec kafka kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000 \
  --if-not-exists
```

**Settings Explanation:**
- **Partitions**: 3 (allows parallel processing)
- **Replication Factor**: 1 (for single broker, use 3+ for production)
- **Retention**: 7 days (604800000 ms)
- **Segment**: 1 day (86400000 ms)

### Topic Management Commands

```bash
# List all topics
docker-compose exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Describe topic
docker-compose exec kafka kafka-topics.sh --describe \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue

# Delete topic (use with caution!)
docker-compose exec kafka kafka-topics.sh --delete \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue

# Alter topic configuration
docker-compose exec kafka kafka-configs.sh --alter \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name prediction_queue \
  --add-config retention.ms=86400000
```

## Application Configuration

### Environment Variables

Set these in your `.env` file or Docker Compose:

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address(es) | `localhost:9092` |
| `KAFKA_TOPIC` | Topic name for predictions | `prediction_queue` |
| `KAFKA_CONSUMER_GROUP` | Consumer group ID | `prediction_consumer_group` |
| `KAFKA_AUTO_OFFSET_RESET` | Offset reset policy | `earliest` |

### Example `.env` File

```env
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=prediction_queue
KAFKA_CONSUMER_GROUP=prediction_consumer_group
KAFKA_AUTO_OFFSET_RESET=earliest

# User Service Configuration
MODEL_API_URL=http://127.0.0.1:8001
MESSAGE_QUEUE_ENABLED=true
```

### Docker Compose Configuration

Update `docker-compose.yml` user-service section:

```yaml
user-service:
  environment:
    - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    - KAFKA_TOPIC=prediction_queue
    - KAFKA_CONSUMER_GROUP=prediction_consumer_group
    - KAFKA_AUTO_OFFSET_RESET=earliest
    - MESSAGE_QUEUE_ENABLED=true
  depends_on:
    - kafka
```

## Testing Kafka

### 1. Test Producer (Send Messages)

```bash
# Using kafka-console-producer
docker-compose exec kafka kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue

# Then type messages and press Enter:
# {"network_id": "NET-001", "data": {"feature1": 1.0, "feature2": 2.0}, "created_at": "2024-01-01T00:00:00"}
```

### 2. Test Consumer (Receive Messages)

```bash
# Using kafka-console-consumer
docker-compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --from-beginning \
  --group test-consumer-group
```

### 3. Test via API

```bash
# Publish a message via the User Service API
curl -X POST http://localhost:8002/publish \
  -H "Content-Type: application/json" \
  -d '{
    "network_id": "NET-001",
    "data": {
      "feature1": 1.0,
      "feature2": 2.0
    }
  }'
```

### 4. Monitor Consumer Lag

```bash
# Check consumer group lag
docker-compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group prediction_consumer_group \
  --describe
```

## Production Configuration

### Recommended Settings

For production environments:

1. **Multiple Brokers** (replication factor ≥ 3):
   ```bash
   --replication-factor 3
   ```

2. **More Partitions** (for scalability):
   ```bash
   --partitions 6
   ```

3. **Enable Compression**:
   ```yaml
   # In application code
   compression_type='gzip'
   ```

4. **Acknowledge All** (for durability):
   ```yaml
   acks='all'
   ```

5. **Idempotent Producer**:
   ```yaml
   enable_idempotence=True
   ```

### Security (SASL/SSL)

For production, enable security:

```yaml
# In docker-compose.yml
kafka:
  environment:
    - KAFKA_CFG_SASL_ENABLED_MECHANISMS=PLAIN
    - KAFKA_CFG_SASL_MECHANISM_INTER_BROKER_PROTOCOL=PLAIN
    - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:SASL_PLAINTEXT
```

## Troubleshooting

### Kafka Won't Start

**Problem**: Kafka container exits immediately

**Solutions**:
```bash
# Check logs
docker-compose logs kafka

# Common issues:
# 1. Zookeeper not running
docker-compose up -d zookeeper

# 2. Port already in use
netstat -ano | findstr :9092  # Windows
lsof -i :9092  # Linux/Mac

# 3. Insufficient memory
# Increase Docker memory limit
```

### Connection Refused

**Problem**: Application can't connect to Kafka

**Solutions**:
```bash
# 1. Verify Kafka is running
docker-compose ps kafka

# 2. Check bootstrap servers address
# Docker: use service name (kafka:9092)
# Local: use localhost:9092

# 3. Test connection
docker-compose exec kafka kafka-broker-api-versions.sh \
  --bootstrap-server localhost:9092
```

### Messages Not Consumed

**Problem**: Messages are published but not processed

**Solutions**:
```bash
# 1. Check consumer group status
docker-compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group prediction_consumer_group \
  --describe

# 2. Check if consumer is running
# Look for "Kafka consumer started" in user-service logs
docker-compose logs user-service | grep -i kafka

# 3. Verify topic exists
docker-compose exec kafka kafka-topics.sh --list \
  --bootstrap-server localhost:9092

# 4. Check offset reset policy
# Set KAFKA_AUTO_OFFSET_RESET=earliest to read from beginning
```

### High Memory Usage

**Problem**: Kafka using too much memory

**Solutions**:
```yaml
# In docker-compose.yml, add memory limits
kafka:
  deploy:
    resources:
      limits:
        memory: 2G
      reservations:
        memory: 1G
```

### Topic Not Found

**Problem**: Error "Topic 'prediction_queue' does not exist"

**Solutions**:
```bash
# Create the topic
docker-compose exec kafka kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists
```

## Useful Commands

### View Topic Messages

```bash
# Read all messages from beginning
docker-compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue \
  --from-beginning

# Read latest messages only
docker-compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue
```

### Get Topic Statistics

```bash
# Get detailed topic information
docker-compose exec kafka kafka-topics.sh --describe \
  --bootstrap-server localhost:9092 \
  --topic prediction_queue

# Get consumer group information
docker-compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group prediction_consumer_group \
  --describe
```

### Reset Consumer Group

```bash
# Reset consumer group to earliest offset
docker-compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group prediction_consumer_group \
  --reset-offsets \
  --to-earliest \
  --topic prediction_queue \
  --execute
```

## Monitoring

### Kafka UI (Optional)

For a web-based Kafka management interface, add Kafka UI to docker-compose.yml:

```yaml
kafka-ui:
  image: provectuslabs/kafka-ui:latest
  container_name: kafka-ui
  ports:
    - "8080:8080"
  environment:
    KAFKA_CLUSTERS_0_NAME: local
    KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
  depends_on:
    - kafka
  networks:
    - campus-iot-network
```

Access at: http://localhost:8080

## Next Steps

1. **Set up monitoring**: Use Kafka UI or Prometheus
2. **Configure retention**: Adjust based on your needs
3. **Set up alerts**: Monitor consumer lag and broker health
4. **Enable security**: For production deployments
5. **Optimize performance**: Tune partitions and replication

## Additional Resources

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka Best Practices](https://kafka.apache.org/documentation/#bestPractices)
- [aiokafka Documentation](https://aiokafka.readthedocs.io/)
