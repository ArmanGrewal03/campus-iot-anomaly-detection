#!/bin/bash
# Main startup script - navigate to project root first
cd "$(dirname "$0")/.."

mkdir -p logs

echo "Stopping existing services..."
lsof -ti:8000,8001,8002,8003,8004,8010,5173,5174 | xargs kill -9 2>/dev/null || true

echo "Starting Kafka..."
brew services start kafka

echo "Starting Data Ingestion Service (Port 8000)..."
cd 01_Data_Ingestion_Service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt > /dev/null
nohup python main.py > ../logs/01_data_ingestion.log 2>&1 &
cd ..

echo "Starting Model Service (Port 8001)..."
cd 02_Model_Service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt > /dev/null
export API_BASE_URL=http://localhost:8000
export GRPC_PORT=50052
export DATA_INGESTION_GRPC=localhost:50051
nohup python model_api.py > ../logs/02_model.log 2>&1 &
cd ..

echo "Starting User Service (Port 8002)..."
cd 04_User_Service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt > /dev/null
export MODEL_API_URL=http://localhost:8001
export MODEL_SERVICE_GRPC=localhost:50052
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
nohup python user_service.py > ../logs/04_user.log 2>&1 &
cd ..

echo "Starting Gateway Proxy (Port 8003)..."
cd 05_Gateway_Proxy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt > /dev/null
export DATA_INGESTION_SERVICE=http://localhost:8000
export MODEL_SERVICE=http://localhost:8001
export USER_SERVICE=http://localhost:8002
nohup python gateway.py > ../logs/05_gateway.log 2>&1 &
cd ..

echo "Starting ML-Ops Service (Port 8004)..."
cd 07_ML_Ops_Service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt > /dev/null
export MODEL_SERVICE=http://localhost:8001
export DATA_INGESTION_SERVICE=http://localhost:8000
nohup python ml_ops.py > ../logs/07_ml_ops.log 2>&1 &
cd ..

echo "Starting Live Metrics Service (Port 8010)..."
cd 06_Live_Metrics_Service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt > /dev/null
nohup python live_metrics_service.py > ../logs/06_live_metrics.log 2>&1 &
cd ..

echo "Starting Dashboard (Port 5173)..."
cd 03_Dashboard
npm install > /dev/null
nohup npm run dev > ../logs/03_dashboard.log 2>&1 &
cd ..

echo "Starting Model Management Dashboard (Port 5174)..."
cd 04_Model_Dashboard
npm install > /dev/null
nohup npm run dev > ../logs/04_model_dashboard.log 2>&1 &
cd ..

echo "All services started! Check the logs/ directory for outputs."
echo "Main Dashboard (Monitoring) is available at http://localhost:5173"
echo "Model Management Dashboard is available at http://localhost:5174"
echo "ML-Ops Service (backend) is available at http://localhost:8004/docs"
