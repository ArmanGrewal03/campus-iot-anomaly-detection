# Docker Setup Guide

This guide explains how to containerize and run the Campus IoT Anomaly Detection application using Docker.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose 2.0 or later

## Quick Start

### 1. Build and Start All Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### 2. Access the Application

Once all services are running:

- **Frontend Dashboard**: http://localhost:5173
- **API Gateway**: http://localhost:8003
- **Data Ingestion Service**: http://localhost:8000
- **Model Service**: http://localhost:8001
- **User Service**: http://localhost:8002

## Service Details

### Data Ingestion Service
- **Port**: 8000 (HTTP), 50051 (gRPC)
- **Health Check**: http://localhost:8000/health
- **Volume**: `data-ingestion-db` (persists database)

### Model Service
- **Port**: 8001 (HTTP), 50052 (gRPC)
- **Health Check**: http://localhost:8001/health
- **Volume**: `model-models` (persists trained models)
- **Depends on**: Data Ingestion Service

### User Service
- **Port**: 8002 (HTTP/WebSocket)
- **Health Check**: http://localhost:8002/health
- **Volume**: `user-service-db` (persists user data)
- **Depends on**: Model Service

### API Gateway
- **Port**: 8003
- **Health Check**: http://localhost:8003/health
- **Routes requests to**: Data Ingestion, Model, and User services
- **Depends on**: All backend services

### Dashboard (Frontend)
- **Port**: 80 (mapped to 5173 on host)
- **Health Check**: http://localhost:5173/health
- **Built with**: Nginx serving React/Vite build
- **Depends on**: Gateway

## Building Individual Services

### Build a specific service:

```bash
# Data Ingestion Service
docker build -t campus-iot-data-ingestion ./01_Data_Ingestion_Service

# Model Service
docker build -t campus-iot-model-service ./02_Model_Service

# User Service
docker build -t campus-iot-user-service ./04_User_Service

# Gateway
docker build -t campus-iot-gateway ./05_Gateway_Proxy

# Dashboard
docker build -t campus-iot-dashboard ./03_Dashboard
```

### Run a specific service:

```bash
docker run -d -p 8000:8000 campus-iot-data-ingestion
```

## Environment Variables

### Data Ingestion Service
- `DB_PATH`: Database file path (default: `/app/data/campus_iot_data.db`)
- `GRPC_PORT`: gRPC server port (default: `50051`)
- `ERROR_ONLY_LOGGING`: Enable error-only logging (default: `false`)

### Model Service
- `API_BASE_URL`: Data Ingestion Service URL (default: `http://data-ingestion:8000`)
- `GRPC_PORT`: gRPC server port (default: `50052`)
- `DATA_INGESTION_GRPC`: Data Ingestion gRPC address (default: `data-ingestion:50051`)
- `ERROR_ONLY_LOGGING`: Enable error-only logging (default: `false`)

### User Service
- `MODEL_API_URL`: Model Service URL (default: `http://model-service:8001`)
- `MODEL_SERVICE_GRPC`: Model Service gRPC address (default: `model-service:50052`)
- `ERROR_ONLY_LOGGING`: Enable error-only logging (default: `false`)

### Gateway
- `DATA_INGESTION_SERVICE`: Data Ingestion Service URL
- `MODEL_SERVICE`: Model Service URL
- `USER_SERVICE`: User Service URL
- `ERROR_ONLY_LOGGING`: Enable error-only logging (default: `false`)

## Volumes

Docker volumes are used to persist data:

- **data-ingestion-db**: SQLite database for data ingestion
- **model-models**: Trained model files (.pkl, .json)
- **user-service-db**: User service databases (users.db, network_logs.db, etc.)

### Backup Volumes

```bash
# Backup a volume
docker run --rm -v campus-iot-anomaly-detection_data-ingestion-db:/data -v $(pwd):/backup alpine tar czf /backup/data-ingestion-backup.tar.gz /data

# Restore a volume
docker run --rm -v campus-iot-anomaly-detection_data-ingestion-db:/data -v $(pwd):/backup alpine tar xzf /backup/data-ingestion-backup.tar.gz -C /
```

## Development Mode

### Run with hot reload (for development):

Modify `docker-compose.yml` to add volume mounts for code:

```yaml
services:
  data-ingestion:
    volumes:
      - ./01_Data_Ingestion_Service:/app
      - data-ingestion-db:/app/data
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### View logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f data-ingestion

# Last 100 lines
docker-compose logs --tail=100 data-ingestion
```

## Troubleshooting

### Services won't start

1. **Check service health**:
   ```bash
   docker-compose ps
   ```

2. **View service logs**:
   ```bash
   docker-compose logs <service-name>
   ```

3. **Check port conflicts**:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   
   # Linux/Mac
   lsof -i :8000
   ```

### Database issues

If databases are corrupted or need reset:

```bash
# Stop services
docker-compose down

# Remove volumes (WARNING: This deletes all data!)
docker-compose down -v

# Restart
docker-compose up -d
```

### Rebuild after code changes

```bash
# Rebuild specific service
docker-compose build data-ingestion

# Rebuild all services
docker-compose build

# Rebuild and restart
docker-compose up -d --build
```

### Network issues

If services can't communicate:

```bash
# Check network
docker network inspect campus-iot-anomaly-detection_campus-iot-network

# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

## Production Deployment

### 1. Use environment-specific compose file:

```bash
# Create docker-compose.prod.yml
cp docker-compose.yml docker-compose.prod.yml

# Modify for production (remove dev volumes, add production configs)
```

### 2. Use secrets for sensitive data:

```yaml
services:
  user-service:
    secrets:
      - db_password
secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### 3. Use reverse proxy (nginx/traefik):

Add nginx service to handle SSL/TLS and routing.

### 4. Resource limits:

```yaml
services:
  model-service:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Cleanup

### Remove all containers and volumes:

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (deletes data!)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

### Remove only stopped containers:

```bash
docker container prune
```

## Health Checks

All services include health checks. Monitor service health:

```bash
# Check health status
docker-compose ps

# Manual health check
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## Performance Tips

1. **Use multi-stage builds** (already implemented for Dashboard)
2. **Layer caching**: Order Dockerfile commands from least to most frequently changing
3. **Use .dockerignore**: Exclude unnecessary files from build context
4. **Resource limits**: Set appropriate CPU/memory limits
5. **Volume optimization**: Use named volumes for better performance

## Next Steps

- Set up CI/CD pipeline for automated builds
- Configure monitoring (Prometheus, Grafana)
- Add logging aggregation (ELK stack, Loki)
- Set up backup automation for volumes
- Configure SSL/TLS with reverse proxy
