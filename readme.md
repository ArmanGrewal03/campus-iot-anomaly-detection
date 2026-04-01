# Campus IoT Anomaly Detection

A modular system for monitoring campus IoT network traffic and identifying anomalies using machine learning. This project implements a microservices architecture to handle data ingestion, model lifecycle management, and real-time visualization.

## Project Purpose

The primary objective of this system is to provide a robust framework for securing campus IoT infrastructures. This platform serves as an external monitoring layer that:
- Detects malicious network patterns and unauthorized device activities.
- Provides real-time visibility into network health and security events.
- Evaluates various machine learning models (Random Forest, Isolation Forest, Autoencoders).
- Offers a scalable architecture for integrating new data sources and algorithms.

## System Architecture

The project is structured into specialized microservices:

- **Data Ingestion (01)**: Port 8000. Manages CSV uploads, validation, and data persistence in SQLite.
- **Model Service (02)**: Port 8001. ML model training, testing, and predictions (Random Forest, Isolation Forest, Autoencoders).
- **Dashboard (03)**: Port 5173. React/Vite frontend for real-time monitoring and analytics.
- **Model Dashboard (04)**: Port 5174. **NEW** - Separate React UI for model management (training, testing, registry).
- **User Service (04)**: Port 8002. User management, WebSocket data streams, network telemetry.
- **API Gateway (05)**: Port 8003. Central request routing, caching, and validation.
- **ML-Ops Service (07)**: Port 8004. Dedicated backend service for model lifecycle management and training.
- **Live Metrics (06)**: Port 8010. Real-time metric generation and streaming.

### Data Flow
```
Raw Network Data → Data Ingestion (8000)
                    ↓
               SQLite Database
                    ↓
          Model Service (8001) ← ML-Ops Service (8004)
                    ↓
            Real-time Predictions
                    ↓
       Dashboard (5173) ← User Service (8002) ← Kafka Message Queue
```

### Key Features
- **Modular Architecture**: Each service independently deployable
- **Separation of Concerns**: Dashboard focuses on monitoring, ML-Ops on model management
- **Real-Time Processing**: WebSocket streams + Kafka message queue
- **Model Flexibility**: Support for supervised and unsupervised ML approaches
- **Scalable**: Docker containerization ready for cloud deployment

## Documentation

Detailed setup and configuration guides are available in the `docs/` directory:
- **[Docker Setup](docs/DOCKER_SETUP.md)** - Containerized deployment with Docker/Docker Compose
- **[Kafka Configuration](docs/KAFKA_SETUP.md)** - Message broker setup and topic management
- **[Kafka Quick Start](docs/KAFKA_QUICK_START.md)** - Fast 5-minute Kafka setup
- **[gRPC Setup](docs/GRPC_SETUP.md)** - Service-to-service communication
- **[Model Explanation](docs/MODEL_EXPLANATION.md)** - How the anomaly detection models work

Reference materials are located in `Reference Maps/`.

## Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js & npm**
- **Docker & Docker Compose** (for containerized deployment)

### Quick Start (Bash/macOS/Linux)
```bash
bash scripts/start_all.sh
# Main Dashboard opens at http://localhost:5173
# Model Dashboard opens at http://localhost:5174
```

### Automated Startup (Windows PowerShell)
```powershell
.\scripts\run-all-services.ps1
# Main Dashboard opens at http://localhost:5173
# Model Dashboard opens at http://localhost:5174
```

### Service URLs
All services available at:
- **Main Dashboard** (Monitoring): http://localhost:5173
- **Model Management Dashboard** (ML-Ops UI): http://localhost:5174
- **Data Ingestion API**: http://localhost:8000
- **Model Service API**: http://localhost:8001
- **User Service**: http://localhost:8002
- **API Gateway**: http://localhost:8003
- **ML-Ops Service (Backend)**: http://localhost:8004/docs
- **Live Metrics**: http://localhost:8010

### Docker Deployment
To run all services in containers:
```bash
docker-compose up -d
```

See [Docker Setup](docs/DOCKER_SETUP.md) for details.

## Troubleshooting

- **Port Already in Use**: If a service fails to start, check for processes on ports 8000-8002 or 5173.
  - Mac/Linux: `lsof -i :8000` then `kill -9 <PID>`
- **API Unreachable**: Verify the backend is healthy by visiting `http://localhost:8000/api/health`.
- **CORS Errors**: Ensure the frontend port (typically 5173) is listed in the backend's allowed origins.

## Machine Learning Approaches

The system evaluates traffic through three primary model types:
1. **Random Forest**: Supervised classification for known attack vectors.
2. **Isolation Forest**: Unsupervised anomaly detection for novel threats.
3. **Autoencoders**: Neural network-based reconstruction for complex pattern deviations.

## Data Source
The system utilizes the **UNSW-NB15** dataset and live campus IoT telemetry, processed through the internal data ingestion pipeline.


