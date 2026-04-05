"""
ML-Ops Service
Handles model training, validation, testing, and management.
Separate from main dashboard for security and scalability.
"""

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import logging
import os
from datetime import datetime
from typing import Optional

app = FastAPI(title="Campus IoT ML-Ops Service", version="1.0.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Services
MODEL_SERVICE = os.getenv("MODEL_SERVICE", "http://127.0.0.1:8001")
DATA_INGESTION_SERVICE = os.getenv("DATA_INGESTION_SERVICE", "http://127.0.0.1:8000")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Campus IoT ML-Ops Service",
            "version": "1.0.0"
        },
        status_code=200
    )


@app.post("/upload")
async def upload_dataset(request: Request, dataset_name: str = Header(..., alias="dataset_name")):
    """Upload CSV dataset for training"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{DATA_INGESTION_SERVICE}/new",
                content=body,
                headers=headers
            )
            
        if response.status_code != 201:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get("detail", "Upload failed")
            )
        
        logger.info(f"Dataset '{dataset_name}' uploaded successfully")
        return response.json()
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.put("/validate")
async def validate_dataset(request: Request, dataset_name: str = Header(..., alias="dataset_name")):
    """Validate and split dataset into training/testing"""
    try:
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)  # Remove content-length since we're not sending body
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.put(
                f"{DATA_INGESTION_SERVICE}/validate",
                headers=headers
            )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get("detail", "Validation failed")
            )
        
        logger.info(f"Dataset '{dataset_name}' validated")
        return response.json()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/train")
async def train_model(request: Request, dataset_name: str = Header(..., alias="dataset_name"), model_name: str = Header(..., alias="model_name")):
    """Train a new model"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers["Content-Type"] = "application/json"
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{MODEL_SERVICE}/train",
                content=body,
                headers=headers
            )
        
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Training failed")
            logger.error(f"Training failed for model '{model_name}': {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)
        
        result = response.json()
        logger.info(f"Model '{model_name}' trained on dataset '{dataset_name}'")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/test")
async def test_model(request: Request, dataset_name: str = Header(..., alias="dataset_name"), model_name: str = Header(..., alias="model_name")):
    """Test a trained model"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers["Content-Type"] = "application/json"
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{MODEL_SERVICE}/test",
                content=body,
                headers=headers
            )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get("detail", "Testing failed")
            )
        
        logger.info(f"Model '{model_name}' tested on dataset '{dataset_name}'")
        return response.json()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Testing error: {e}")
        raise HTTPException(status_code=500, detail=f"Testing failed: {str(e)}")


@app.delete("/clear")
async def clear_dataset(dataset_name: Optional[str] = Header(None, alias="dataset_name")):
    """Delete a specific dataset"""
    try:
        if not dataset_name:
            raise HTTPException(status_code=400, detail="dataset_name header is required")
        
        headers = {"dataset_name": dataset_name}
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.delete(
                f"{DATA_INGESTION_SERVICE}/clear",
                headers=headers
            )
        
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Delete failed")
            raise HTTPException(status_code=response.status_code, detail=error_detail)
        
        logger.info(f"Dataset '{dataset_name}' deleted successfully")
        return response.json()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/tables")
async def get_tables():
    """Get list of available datasets"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{DATA_INGESTION_SERVICE}/tables")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to list datasets")
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@app.get("/models")
async def get_models():
    """Get list of available models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MODEL_SERVICE}/models")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to list models")
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/model-types")
async def get_model_types():
    """Get available model architectures"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MODEL_SERVICE}/model-types")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to list model types")
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error listing model types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list model types: {str(e)}")


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a trained model"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{MODEL_SERVICE}/models/{model_name}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to delete model")
        
        logger.info(f"Model '{model_name}' deleted")
        return response.json()
    
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ML-Ops Service on port 8004...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8004,
        log_level="info"
    )
