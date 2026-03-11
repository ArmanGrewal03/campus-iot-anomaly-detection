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

- **Data Ingestion (01)**: Port 8000. Manages network traffic persistence.
- **Model Service (02)**: Port 8001. Handles ML training, evaluation, and deployment.
- **Dashboard (03)**: Port 5173 / 8080. A React/Vite visualization interface.
- **User Service (04)**: Port 8002. Provides authentication and WebSocket data streams.
- **Gateway & Metrics (05, 06)**: Handles service proxying and real-time metric tracking.

## Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js & npm**
- **PowerShell** (optional, for automation scripts)

### Automated Startup
Use the scripts in the `scripts/` directory to launch the environment:
- **Run all services**: `.\scripts\run-all-services.ps1`
- **Individual services**: e.g., `.\scripts\run-02-model-service.ps1`

### Manual Startup
Launch services in the following order:

1. **Backend Services (01, 02, 04)**:
   ```bash
   # Navigate to service directory
   cd 02_Model_Service
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python model_api.py
   ```

2. **Frontend Dashboard (03)**:
   ```bash
   cd 03_Dashboard
   npm install
   npm run dev
   ```

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


