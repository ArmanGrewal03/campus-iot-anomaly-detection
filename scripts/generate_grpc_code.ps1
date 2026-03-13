# Script to generate gRPC Python code from proto files

Write-Host "Generating gRPC code from proto files..."

# Data Ingestion Service
Write-Host "Generating Data Ingestion Service gRPC code..."
Set-Location "..\01_Data_Ingestion_Service"
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/data_ingestion.proto
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Data Ingestion Service gRPC code generated"
} else {
    Write-Host "✗ Failed to generate Data Ingestion Service gRPC code"
}

# Model Service
Write-Host "Generating Model Service gRPC code..."
Set-Location "..\02_Model_Service"
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/model_service.proto
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Model Service gRPC code generated"
} else {
    Write-Host "✗ Failed to generate Model Service gRPC code"
}

Set-Location "..\scripts"
Write-Host "Done!"
