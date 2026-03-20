#!/bin/bash

echo "Generating gRPC code from proto files..."

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Data Ingestion Service
echo "Generating Data Ingestion Service gRPC code..."
cd "$PROJECT_ROOT/01_Data_Ingestion_Service"
if [ -d "proto" ]; then
    python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/data_ingestion.proto
    if [ $? -eq 0 ]; then
        echo "✓ Data Ingestion Service gRPC code generated"
    else
        echo "✗ Failed to generate Data Ingestion Service gRPC code"
    fi
else
    echo "✗ proto directory not found in 01_Data_Ingestion_Service"
fi

# Model Service
echo "Generating Model Service gRPC code..."
cd "$PROJECT_ROOT/02_Model_Service"
if [ -d "proto" ]; then
    python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/model_service.proto
    if [ $? -eq 0 ]; then
        echo "✓ Model Service gRPC code generated"
    else
        echo "✗ Failed to generate Model Service gRPC code"
    fi
else
    echo "✗ proto directory not found in 02_Model_Service"
fi

echo "Done!"
