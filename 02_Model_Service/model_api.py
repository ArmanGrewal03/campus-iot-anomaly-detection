"""
FastAPI Application for Random Forest Model

Endpoints:
- GET /health - Health check endpoint
- GET /models - List all available models
- POST /train - Train the model using data from the backend API
- POST /test - Test the model and return evaluation metrics
- POST /predict - Make predictions on new data
- GET /model/status - Get model status and metadata
- GET /model/metrics - Get model evaluation metrics
"""

from fastapi import FastAPI, HTTPException, Header, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import json
import os
from datetime import datetime
import logging
import warnings
import sqlite3
warnings.filterwarnings('ignore')

app = FastAPI(title="Campus IoT Anomaly Detection Model API", version="1.0.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_DIR = "models"
MODEL_FILENAME = "random_forest_model.pkl"
METADATA_FILENAME = "model_metadata.json"

# Pydantic models for request/response
class TrainRequest(BaseModel):
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None
    random_state: Optional[int] = 42
    database_name: Optional[str] = None
    include_fields: Optional[List[str]] = None
    exclude_fields: Optional[List[str]] = None

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]

def fetch_all_data(endpoint: str, label_type: str = "all", database_name: Optional[str] = None) -> List[Dict]:
    logger.info(f"Fetching {label_type} data from {endpoint}...")
    all_data = []
    limit = 1000
    offset = 0
    
    headers = {}
    if database_name:
        headers["dataset_name"] = database_name
    
    while True:
        try:
            url = f"{API_BASE_URL}{endpoint}"
            logger.info(f"Requesting {url} with params: limit={limit}, offset={offset}, headers={headers}")
            response = requests.get(
                url,
                params={"limit": limit, "offset": offset},
                headers=headers,
                timeout=30
            )
            logger.info(f"Backend API response: {response.status_code} {response.reason} for {endpoint}")
            
            if response.status_code >= 400:
                error_detail = f"{response.status_code}"
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", error_json.get("message", f"{response.status_code}"))
                except:
                    error_detail = response.text or f"{response.status_code}"
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail
                )
            
            result = response.json()
            if result.get("status") != "success":
                error_msg = result.get("message", result.get("detail", "Unknown error"))
                logger.error(f"Error: {error_msg}")
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
            
            if offset == 0:
                total_rows = result.get("total_rows", 0)
                if total_rows == 0:
                    error_msg = result.get("message", f"No {label_type} data found. Please ensure data has been uploaded and validated.")
                    raise HTTPException(
                        status_code=400,
                        detail=error_msg
                    )
            
            data = result.get("data", [])
            if not data:
                break
            
            all_data.extend(data)
            logger.info(f"  Fetched {len(data)} rows (total: {len(all_data)})")
            
            if not result.get("has_more", False):
                break
            
            offset += limit
            
        except HTTPException:
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Error connecting to backend API: {str(e)}"
            )
    
    logger.info(f"Total {label_type} records fetched: {len(all_data)}")
    return all_data

def extract_features_and_labels(data_records: List[Dict], include_fields: Optional[List[str]] = None, exclude_fields: Optional[List[str]] = None) -> tuple:
    """Extract features and labels from API response data."""
    rows = []
    for record in data_records:
        row_data = record.get("data", {})
        if isinstance(row_data, str):
            row_data = json.loads(row_data)
        rows.append(row_data)
    
    if not rows:
        return pd.DataFrame(), np.array([]), []
    
    df = pd.DataFrame(rows)
    
    if "label" not in df.columns:
        raise ValueError("'label' column not found in data.")
    
    default_exclude_cols = ["label", "id", "attack_cat"]
    
    if include_fields is not None:
        include_fields = [f.lower() for f in include_fields]
        available_cols = [col for col in df.columns if col.lower() in include_fields]
        if not available_cols:
            raise ValueError(f"None of the specified include_fields {include_fields} were found in the data.")
        feature_cols = [col for col in available_cols if col not in default_exclude_cols]
        logger.info(f"Using include_fields: {include_fields}, resulting in {len(feature_cols)} features")
    else:
        exclude_cols = set(default_exclude_cols)
        if exclude_fields is not None:
            exclude_fields_lower = [f.lower() for f in exclude_fields]
            for col in df.columns:
                if col.lower() in exclude_fields_lower:
                    exclude_cols.add(col)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        if exclude_fields:
            logger.info(f"Using exclude_fields: {exclude_fields}, resulting in {len(feature_cols)} features")
    
    if "label" in feature_cols:
        raise ValueError("CRITICAL: 'label' was found in feature columns. This should never happen!")
    
    if not feature_cols:
        raise ValueError("No features available after filtering. Please check your include/exclude field settings.")
    
    X = df[feature_cols].copy()
    y = df["label"].copy()
    
    if "label" in X.columns:
        raise ValueError("CRITICAL: 'label' column found in feature matrix X. Removing it would cause data leakage!")
    
    y = pd.to_numeric(y, errors='coerce')
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    for col in X.columns:
        if col == "label":
            raise ValueError(f"CRITICAL: Found 'label' in feature column '{col}'. This must be excluded!")
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0)
    y = y.astype(int)
    
    if "label" in feature_cols:
        raise ValueError("CRITICAL: 'label' found in feature_cols list. This must be excluded!")
    
    logger.info(f"Extracted {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Label distribution: Safe (0) = {(y == 0).sum()}, Unsafe (1) = {(y == 1).sum()}")
    logger.info(f"VERIFIED: 'label' is NOT in feature columns. Features: {len(feature_cols)}, Label used only as target variable.")
    
    return X, y, feature_cols

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, 
                n_estimators: int = 100, max_depth: Optional[int] = None, 
                random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    logger.info(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, 
                   y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance_df.head(20).to_dict('records')
    }
    
    return metrics

def save_model(model: RandomForestClassifier, feature_names: List[str], 
               metrics: Dict[str, Any], training_params: Dict[str, Any],
               model_name: str = "random_forest_model"):
    """Save the trained model and metadata."""
    if "label" in feature_names:
        logger.error("CRITICAL ERROR: Attempting to save model with 'label' in feature_names!")
        raise ValueError("CRITICAL: 'label' must not be included in feature_names. This would cause data leakage!")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
    model_filename = f"{sanitized_model_name}.pkl"
    metadata_filename = f"{sanitized_model_name}_metadata.json"
    
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    logger.info(f"VALIDATION: Saving model with {len(feature_names)} features (label correctly excluded)")
    
    metadata = {
        'model_type': 'RandomForestClassifier',
        'model_name': model_name,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'training_date': datetime.now().isoformat(),
        'training_params': training_params,
        'metrics': metrics,
        'label_mapping': {
            '0': 'safe',
            '1': 'unsafe'
        }
    }
    
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

def load_model(model_name: str) -> tuple:
    sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
    model_filename = f"{sanitized_model_name}.pkl"
    metadata_filename = f"{sanitized_model_name}_metadata.json"
    model_path = os.path.join(MODEL_DIR, model_filename)
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    
    if not os.path.exists(model_path):
        return None, None
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model file: {e}")
        return None, None
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None, None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        return None, None
    
    return model, metadata

@app.get("/health")
async def health_check():
    os.makedirs(MODEL_DIR, exist_ok=True)
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Campus IoT Anomaly Detection Model API",
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    )

@app.get("/models")
async def list_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        model_files = []
        if os.path.exists(MODEL_DIR):
            for filename in os.listdir(MODEL_DIR):
                if filename.endswith('.pkl'):
                    model_name = filename[:-4]
                    metadata_filename = f"{model_name}_metadata.json"
                    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
                    
                    model_info = {
                        "model_name": model_name,
                        "model_file": filename,
                        "has_metadata": os.path.exists(metadata_path)
                    }
                    
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                model_info["training_date"] = metadata.get("training_date")
                                model_info["n_features"] = metadata.get("n_features")
                                if "metrics" in metadata and metadata["metrics"]:
                                    model_info["accuracy"] = metadata["metrics"].get("accuracy")
                        except Exception as e:
                            logger.warning(f"Error reading metadata for {model_name}: {e}")
                    
                    model_files.append(model_info)
        
        model_files.sort(key=lambda x: x["model_name"])
        
        logger.info(f"Retrieved {len(model_files)} models")
        
        return JSONResponse(
            content={
                "status": "success",
                "total_models": len(model_files),
                "models": model_files,
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error listing models: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

def get_database_name(
    dataset_name: str = Header(..., alias="dataset_name"),
    train_request: TrainRequest = TrainRequest()
) -> str:
    if dataset_name:
        return dataset_name
    return train_request.database_name or "default"

def get_model_name(
    model_name: str = Header(..., alias="model_name")
) -> str:
    return model_name

@app.post("/train")
async def train(
    train_request: Optional[TrainRequest] = Body(default=None),
    dataset_name: str = Depends(get_database_name),
    model_name: str = Depends(get_model_name)
):
    logger.info("Training request received")
    logger.info(f"Using dataset: {dataset_name}, model_name: {model_name}")
    
    if train_request is None:
        train_request = TrainRequest()
    
    headers = {}
    headers["dataset_name"] = dataset_name
    
    try:
        health_url = f"{API_BASE_URL}/health"
        logger.info(f"Checking backend health at {health_url} with headers: {headers}")
        response = requests.get(health_url, headers=headers, timeout=5)
        logger.info(f"Backend health check response: {response.status_code} {response.reason}")
        if response.status_code != 200:
            logger.error(f"Backend API health check failed with status {response.status_code}")
            raise HTTPException(
                status_code=503,
                detail="Backend API is not healthy. Please ensure FastAPI backend is running."
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to backend API at {API_BASE_URL}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to backend API at {API_BASE_URL}"
        )
    
    try:
        training_data = fetch_all_data("/training", "training", dataset_name)
        if not training_data:
            raise HTTPException(
                status_code=400,
                detail="No training data found. Please ensure data has been uploaded and validated."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching training data: {str(e)}")
    
    # Extract features and labels
    try:
        X_train, y_train, feature_names = extract_features_and_labels(
            training_data,
            include_fields=train_request.include_fields,
            exclude_fields=train_request.exclude_fields
        )
        if len(X_train) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid training samples found."
            )
        
        if "label" in feature_names:
            logger.error("CRITICAL ERROR: 'label' found in feature_names list!")
            raise HTTPException(
                status_code=500,
                detail="CRITICAL: 'label' must not be included in features. This would cause data leakage!"
            )
        
        if "label" in X_train.columns:
            logger.error("CRITICAL ERROR: 'label' found in training feature matrix!")
            raise HTTPException(
                status_code=500,
                detail="CRITICAL: 'label' column found in training features. This must be excluded!"
            )
        
        logger.info(f"VALIDATION PASSED: 'label' is correctly excluded from {len(feature_names)} features")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing training data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing training data: {str(e)}")
    
    # Train model
    try:
        model = train_model(
            X_train, y_train,
            n_estimators=train_request.n_estimators,
            max_depth=train_request.max_depth,
            random_state=train_request.random_state
        )
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
    
    # Save model
    training_params = {
        'n_estimators': train_request.n_estimators,
        'max_depth': train_request.max_depth,
        'random_state': train_request.random_state
    }
    
    # Create placeholder metrics (will be updated after testing)
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'confusion_matrix': [[0, 0], [0, 0]],
        'feature_importance': []
    }
    
    save_model(model, feature_names, metrics, training_params, model_name)
    
    response_content = {
        "status": "success",
        "message": "Model trained successfully",
        "dataset_name": dataset_name,
        "model_name": model_name,
        "training_samples": len(X_train),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_params": training_params,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if train_request.include_fields is not None:
        response_content["include_fields"] = train_request.include_fields
    
    if train_request.exclude_fields is not None:
        response_content["exclude_fields"] = train_request.exclude_fields
    
    return JSONResponse(
        content=response_content,
        status_code=200
    )

class TestRequest(BaseModel):
    database_name: Optional[str] = None

def get_test_database_name(
    dataset_name: Optional[str] = Header(None, alias="dataset_name"),
    test_request: TestRequest = TestRequest()
) -> Optional[str]:
    if dataset_name:
        return dataset_name
    return test_request.database_name

def get_test_model_name(
    model_name: str = Header(..., alias="model_name")
) -> str:
    return model_name

@app.post("/test")
async def test(
    test_request: TestRequest = TestRequest(),
    database_name: Optional[str] = Depends(get_test_database_name),
    model_name: str = Depends(get_test_model_name)
):
    logger.info("Testing request received")
    logger.info(f"Using model: {model_name}")
    
    model, metadata = load_model(model_name)
    if model is None or metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first using POST /train"
        )
    
    headers = {}
    if database_name:
        headers["dataset_name"] = database_name
        logger.info(f"Using database: {database_name}")
    
    try:
        health_url = f"{API_BASE_URL}/health"
        logger.info(f"Checking backend health at {health_url} with headers: {headers}")
        response = requests.get(health_url, headers=headers, timeout=5)
        logger.info(f"Backend health check response: {response.status_code} {response.reason}")
        if response.status_code != 200:
            logger.error(f"Backend API health check failed with status {response.status_code}")
            raise HTTPException(
                status_code=503,
                detail="Backend API is not healthy."
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to backend API at {API_BASE_URL}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to backend API at {API_BASE_URL}"
        )
    
    try:
        testing_data = fetch_all_data("/testing", "testing", database_name)
        if not testing_data:
            raise HTTPException(
                status_code=400,
                detail="No testing data found."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching testing data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching testing data: {str(e)}")
    
    # Extract features and labels
    try:
        X_test, y_test, _ = extract_features_and_labels(testing_data)
        if len(X_test) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid testing samples found."
            )
        
        if "label" in X_test.columns:
            logger.error("CRITICAL ERROR: 'label' found in test feature matrix!")
            raise HTTPException(
                status_code=500,
                detail="CRITICAL: 'label' column found in test features. This must be excluded!"
            )
        
        logger.info(f"VALIDATION PASSED: 'label' is correctly excluded from test features")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing testing data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing testing data: {str(e)}")
    
    # Ensure feature columns match training features exactly
    feature_names = metadata['feature_names']
    logger.info(f"Model expects {len(feature_names)} features: {feature_names[:5]}...")
    logger.info(f"Test data has {len(X_test.columns)} features: {list(X_test.columns)[:5]}...")
    
    # Create a new DataFrame with features in the exact order as training
    X_test_aligned = pd.DataFrame(index=X_test.index)
    for feature in feature_names:
        if feature in X_test.columns:
            X_test_aligned[feature] = X_test[feature]
        else:
            logger.warning(f"Feature '{feature}' not found in test data, filling with 0")
            X_test_aligned[feature] = 0
    
    # Remove any extra features that weren't in training
    missing_features = set(feature_names) - set(X_test.columns)
    if missing_features:
        logger.warning(f"Missing features in test data: {missing_features}")
    
    extra_features = set(X_test.columns) - set(feature_names)
    if extra_features:
        logger.info(f"Extra features in test data (will be ignored): {extra_features}")
    
    X_test = X_test_aligned[feature_names]
    logger.info(f"Aligned test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Evaluate model
    try:
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")
    
    # Update and save metadata with new metrics
    metadata['metrics'] = metrics
    metadata['last_test_date'] = datetime.utcnow().isoformat()
    
    sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
    metadata_filename = f"{sanitized_model_name}_metadata.json"
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Updated metrics saved to: {metadata_path}")
    
    return JSONResponse(
        content={
            "status": "success",
            "message": "Model tested successfully",
            "testing_samples": len(X_test),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(
    predict_request: PredictRequest,
    model_name: str = Depends(get_model_name)
):
    """
    Make predictions on new data.
    
    Request body should contain a list of data records with feature values.
    """
    logger.info(f"Prediction request received for {len(predict_request.data)} samples")
    logger.info(f"Using model: {model_name}")
    
    # Load model
    model, metadata = load_model(model_name)
    if model is None or metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first using POST /train"
        )
    
    feature_names = metadata['feature_names']
    
    if "label" in feature_names:
        logger.error("CRITICAL ERROR: 'label' found in model feature_names!")
        raise HTTPException(
            status_code=500,
            detail="CRITICAL: Model metadata contains 'label' in features. This model is corrupted!"
        )
    
    # Prepare features
    try:
        df = pd.DataFrame(predict_request.data)
        
        if "label" in df.columns:
            logger.warning("'label' field found in prediction request. It will be ignored as it's not a feature.")
            df = df.drop(columns=["label"], errors='ignore')
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}, filling with 0")
            for feature in missing_features:
                df[feature] = 0
        
        # Select only the features used in training
        df = df[feature_names]
        
        if "label" in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CRITICAL: 'label' must not be included in prediction features!"
            )
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values
        df = df.fillna(0)
        
        X = df.values
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise HTTPException(status_code=400, detail=f"Error preparing features: {str(e)}")
    
    # Make predictions
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        label_mapping = metadata['label_mapping']
        results = []
        
        for i in range(len(predictions)):
            results.append({
                'prediction': int(predictions[i]),
                'label': label_mapping[str(int(predictions[i]))],
                'probability_safe': float(probabilities[i][0]),
                'probability_unsafe': float(probabilities[i][1]),
                'confidence': float(max(probabilities[i]))
            })
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
    
    return JSONResponse(
        content={
            "status": "success",
            "predictions": results,
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    )

@app.get("/model/status")
async def get_model_status(model_name: str = Depends(get_model_name)):
    """Get the current status of the model."""
    logger.info(f"Getting status for model: {model_name}")
    model, metadata = load_model(model_name)
    
    if model is None or metadata is None:
        return JSONResponse(
            content={
                "status": "not_trained",
                "message": "Model has not been trained yet",
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=200
        )
    
    return JSONResponse(
        content={
            "status": "trained",
            "model_name": model_name,
            "model_type": metadata.get('model_type', 'Unknown'),
            "training_date": metadata.get('training_date', 'Unknown'),
            "n_features": metadata.get('n_features', 0),
            "last_test_date": metadata.get('last_test_date', 'Not tested yet'),
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    )

@app.get("/model/metrics")
async def get_model_metrics(model_name: str = Depends(get_model_name)):
    """Get the evaluation metrics of the trained model."""
    logger.info(f"Getting metrics for model: {model_name}")
    model, metadata = load_model(model_name)
    
    if model is None or metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first."
        )
    
    metrics = metadata.get('metrics', {})
    
    return JSONResponse(
        content={
            "status": "success",
            "model_name": model_name,
            "metrics": metrics,
            "training_date": metadata.get('training_date', 'Unknown'),
            "last_test_date": metadata.get('last_test_date', 'Not tested yet'),
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
