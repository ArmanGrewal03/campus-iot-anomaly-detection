# gRPC Setup for Service Mesh Communication

This document describes the gRPC implementation for inter-service communication.

## Overview

gRPC is used for **service-to-service** communication to minimize JSON overhead and improve performance. HTTP/JSON is still used for:
- Gateway → Backend Services (external API)
- Frontend → Gateway (external API)

## Architecture

```
Frontend (HTTP/JSON) → Gateway (HTTP/JSON) → Backend Services
                                    ↓
                            Service Mesh (gRPC)
                                    ↓
                    Model Service ←→ Data Ingestion Service
                    User Service ←→ Model Service
```

## Setup Instructions

### 1. Install Dependencies

All services have been updated with `grpcio`, `grpcio-tools`, and `protobuf` in their `requirements.txt` files.

Install dependencies:
```powershell
# Data Ingestion Service
cd 01_Data_Ingestion_Service
pip install -r requirements.txt

# Model Service
cd ..\02_Model_Service
pip install -r requirements.txt

# User Service
cd ..\04_User_Service
pip install -r requirements.txt
```

### 2. Generate gRPC Code

Run the generation script:
```powershell
cd scripts
.\generate_grpc_code.ps1
```

Or manually:
```powershell
# Data Ingestion Service
cd 01_Data_Ingestion_Service
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/data_ingestion.proto

# Model Service
cd ..\02_Model_Service
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/model_service.proto
```

### 3. Start gRPC Servers

gRPC servers run alongside FastAPI servers:

**Data Ingestion Service:**
- HTTP API: Port 8000 (FastAPI)
- gRPC: Port 50051 (default, configurable via `GRPC_PORT` env var)

**Model Service:**
- HTTP API: Port 8001 (FastAPI)
- gRPC: Port 50052 (default, configurable via `GRPC_PORT` env var)

### 4. Environment Variables

Set gRPC ports (optional, defaults shown):
```powershell
# Data Ingestion Service
$env:GRPC_PORT="50051"

# Model Service
$env:GRPC_PORT="50052"

# User Service (client only, no server needed)
$env:DATA_INGESTION_GRPC="127.0.0.1:50051"
$env:MODEL_SERVICE_GRPC="127.0.0.1:50052"
```

## Service Communication

### Model Service → Data Ingestion Service

**Before (HTTP/JSON):**
```python
url = f"{API_BASE_URL}/training"
response = await httpx.get(url, params={"limit": 1000, "offset": 0})
```

**After (gRPC):**
```python
from grpc_client import DataIngestionClient
client = DataIngestionClient("127.0.0.1:50051")
data = await client.get_training_data(limit=1000, offset=0)
```

### User Service → Model Service

**Before (HTTP/JSON):**
```python
url = f"{MODEL_API_URL}/predict"
response = await httpx.post(url, json={"data": [data]})
```

**After (gRPC):**
```python
from grpc_client import ModelServiceClient
client = ModelServiceClient("127.0.0.1:50052")
predictions = await client.predict(data=[data], model_name="model_name")
```

## Benefits

1. **Performance**: Binary protocol is faster than JSON
2. **Type Safety**: Protocol buffers provide strong typing
3. **Efficiency**: Smaller payload sizes
4. **Streaming**: Support for streaming requests/responses
5. **Service Mesh**: Better suited for microservices communication

## Migration Status

- ✅ Proto files created
- ✅ Requirements updated
- ⏳ gRPC servers implementation (in progress)
- ⏳ gRPC clients implementation (in progress)
- ⏳ HTTP → gRPC migration (in progress)
