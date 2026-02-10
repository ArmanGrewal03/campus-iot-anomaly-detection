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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
import joblib
import json
import os
from datetime import datetime, timezone
import logging
import warnings
import sqlite3
warnings.filterwarnings('ignore')
# Suppress sklearn parallel warning about delayed/Parallel usage
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')

app = FastAPI(title="Campus IoT Anomaly Detection Model API", version="1.0.0")

# Add CORS middleware to allow React frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (default)
        "http://localhost:5174",  # Vite dev server (this project)
        "http://localhost:3000",  # Alternative React dev server
        "http://localhost:8080",  # Vue CLI dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_DIR = "models"
MODEL_TYPES_DIR = "Model_types"
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
    model_type: Optional[str] = None  # e.g., "RFv1", "IFv1", "AEv1"
    contamination: Optional[float] = 0.1  # For Isolation Forest
    hidden_layers: Optional[str] = "64,32,32,64"  # For Autoencoder (comma-separated)

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

def train_rf_model(X_train: pd.DataFrame, y_train: np.ndarray, 
                   n_estimators: int = 100, max_depth: Optional[int] = None, 
                   random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier (RFv1)."""
    logger.info(f"Training Random Forest model with n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    logger.info("Random Forest model training completed")
    return model

def train_if_model(X_train: pd.DataFrame, y_train: np.ndarray,
                   n_estimators: int = 100, contamination: float = 0.1,
                   random_state: int = 42) -> IsolationForest:
    """Train an Isolation Forest model (IFv1)."""
    logger.info(f"Training Isolation Forest model with n_estimators={n_estimators}, contamination={contamination}")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Isolation Forest is unsupervised, so we only use X_train
    model.fit(X_train)
    logger.info("Isolation Forest model training completed")
    return model

def train_ae_model(X_train: pd.DataFrame, y_train: np.ndarray,
                   hidden_layers: str = "64,32,32,64", random_state: int = 42) -> tuple:
    """Train an Autoencoder model (AEv1). Returns (model, scaler)."""
    logger.info(f"Training Autoencoder model with hidden_layers={hidden_layers}")
    
    # Scale data for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parse hidden layers
    hidden_layer_sizes = tuple(map(int, hidden_layers.split(',')))
    
    # MLPRegressor as autoencoder (input = output)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=random_state,
        verbose=True
    )
    
    # Train autoencoder: X -> X (reconstruction)
    model.fit(X_train_scaled, X_train_scaled)
    logger.info("Autoencoder model training completed")
    return model, scaler

def evaluate_rf_model(model: RandomForestClassifier, X_test: pd.DataFrame, 
                     y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate Random Forest model and return metrics."""
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

def evaluate_if_model(model: IsolationForest, X_test: pd.DataFrame, 
                     y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate Isolation Forest model and return metrics."""
    # Isolation Forest returns -1 for outliers, 1 for inliers
    y_pred_raw = model.predict(X_test)
    # Convert to: 0 for normal (inlier), 1 for anomaly (outlier)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    # Get anomaly scores using decision_function
    # decision_function: positive for inliers, negative for outliers
    # We negate it so higher score = more anomalous
    scores = -model.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # AUC metrics using scores
    try:
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'feature_importance': []  # Isolation Forest doesn't have feature importance
    }
    
    return metrics

def evaluate_ae_model(model: MLPRegressor, scaler: StandardScaler, X_test: pd.DataFrame, 
                     y_test: np.ndarray, threshold_percentile: float = 95.0) -> Dict[str, Any]:
    """Evaluate Autoencoder model and return metrics."""
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    X_pred = model.predict(X_test_scaled)
    
    # Mean Squared Error per sample (reconstruction error)
    mse = np.mean(np.power(X_test_scaled - X_pred, 2), axis=1)
    
    # Determine threshold based on percentile
    threshold = np.percentile(mse, threshold_percentile)
    logger.info(f"Reconstruction Error Threshold ({threshold_percentile}th percentile): {threshold:.4f}")
    
    # Predict anomalies based on reconstruction error
    y_pred = (mse > threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # AUC metrics using MSE scores
    try:
        roc_auc = roc_auc_score(y_test, mse)
        pr_auc = average_precision_score(y_test, mse)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'threshold': float(threshold),
        'feature_importance': []  # Autoencoder doesn't have feature importance
    }
    
    return metrics

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray, 
                  model_type: Optional[str] = None, scaler: Optional[Any] = None) -> Dict[str, Any]:
    """Evaluate model based on its type. Routes to appropriate evaluation function."""
    # Determine model type from model class or parameter
    if model_type:
        model_type_str = model_type
    elif isinstance(model, RandomForestClassifier):
        model_type_str = "RFv1"
    elif isinstance(model, IsolationForest):
        model_type_str = "IFv1"
    elif isinstance(model, MLPRegressor):
        model_type_str = "AEv1"
    else:
        # Try to infer from model class name
        model_class_name = type(model).__name__
        if "RandomForest" in model_class_name:
            model_type_str = "RFv1"
        elif "IsolationForest" in model_class_name:
            model_type_str = "IFv1"
        elif "MLPRegressor" in model_class_name:
            model_type_str = "AEv1"
        else:
            # Default to RF evaluation (will fail if not compatible)
            logger.warning(f"Unknown model type: {model_class_name}, attempting Random Forest evaluation")
            model_type_str = "RFv1"
    
    logger.info(f"Evaluating model type: {model_type_str}")
    
    if model_type_str == "RFv1":
        return evaluate_rf_model(model, X_test, y_test)
    elif model_type_str == "IFv1":
        return evaluate_if_model(model, X_test, y_test)
    elif model_type_str == "AEv1":
        if scaler is None:
            raise ValueError("Scaler is required for Autoencoder model evaluation")
        return evaluate_ae_model(model, scaler, X_test, y_test)
    else:
        # Fallback: try Random Forest evaluation
        logger.warning(f"Unknown model type {model_type_str}, attempting Random Forest evaluation")
        return evaluate_rf_model(model, X_test, y_test)

def save_model(model: Any, feature_names: List[str], 
               metrics: Dict[str, Any], training_params: Dict[str, Any],
               model_name: str = "model", scaler: Optional[Any] = None):
    """Save the trained model and metadata. Supports different model types."""
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
    
    # Save scaler if provided (for autoencoder)
    if scaler is not None:
        scaler_filename = f"{sanitized_model_name}_scaler.pkl"
        scaler_path = os.path.join(MODEL_DIR, scaler_filename)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
    
    logger.info(f"VALIDATION: Saving model with {len(feature_names)} features (label correctly excluded)")
    
    # Get model type from training_params or infer from model class
    model_type_str = training_params.get('model_type', type(model).__name__)
    
    metadata = {
        'model_type': model_type_str,
        'model_name': model_name,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'training_date': datetime.now(timezone.utc).isoformat(),
        'training_params': training_params,
        'metrics': metrics,
        'has_scaler': scaler is not None
    }
    
    # Add label mapping for supervised models
    if model_type_str in ['RandomForestClassifier']:
        metadata['label_mapping'] = {
            '0': 'safe',
            '1': 'unsafe'
        }
    
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

def load_model(model_name: str) -> tuple:
    """Load model, metadata, and scaler (if exists). Returns (model, metadata, scaler)."""
    sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
    model_filename = f"{sanitized_model_name}.pkl"
    metadata_filename = f"{sanitized_model_name}_metadata.json"
    scaler_filename = f"{sanitized_model_name}_scaler.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    scaler_path = os.path.join(MODEL_DIR, scaler_filename)
    
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model file: {e}")
        return None, None, None
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None, None, None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        return None, None, None
    
    # Load scaler if it exists (for autoencoder models)
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from: {scaler_path}")
        except Exception as e:
            logger.warning(f"Error loading scaler file: {e}")
    
    return model, metadata, scaler

@app.get("/health")
async def health_check():
    os.makedirs(MODEL_DIR, exist_ok=True)
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Campus IoT Anomaly Detection Model API",
            "timestamp": datetime.now(timezone.utc).isoformat()
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
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error listing models: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/model-types")
async def list_model_types():
    """
    List all model types available in the Model_types folder.
    Returns information about each model type directory and its contents.
    """
    MODEL_TYPES_DIR = "Model_types"
    
    try:
        model_types = []
        
        if os.path.exists(MODEL_TYPES_DIR):
            for item in os.listdir(MODEL_TYPES_DIR):
                item_path = os.path.join(MODEL_TYPES_DIR, item)
                
                # Only process directories
                if os.path.isdir(item_path):
                    model_type_info = {
                        "model_type": item,
                        "path": item_path,
                        "files": []
                    }
                    
                    # List files in the model type directory
                    try:
                        for file in os.listdir(item_path):
                            file_path = os.path.join(item_path, file)
                            if os.path.isfile(file_path):
                                file_info = {
                                    "name": file,
                                    "size": os.path.getsize(file_path),
                                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                                }
                                model_type_info["files"].append(file_info)
                    except Exception as e:
                        logger.warning(f"Error reading files in {item_path}: {e}")
                        model_type_info["error"] = str(e)
                    
                    model_types.append(model_type_info)
        
        model_types.sort(key=lambda x: x["model_type"])
        
        logger.info(f"Retrieved {len(model_types)} model types")
        
        return JSONResponse(
            content={
                "status": "success",
                "total_model_types": len(model_types),
                "model_types": model_types,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error listing model types: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing model types: {str(e)}")

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
    
    # Determine model type (default to RFv1 if not specified)
    model_type = train_request.model_type or "RFv1"
    logger.info(f"Using model architecture: {model_type}")
    
    # Validate model type exists
    model_type_path = os.path.join(MODEL_TYPES_DIR, model_type)
    if not os.path.exists(model_type_path) or not os.path.isdir(model_type_path):
        raise HTTPException(
            status_code=400,
            detail=f"Model type '{model_type}' not found. Available types can be listed via GET /model-types"
        )
    
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
    
    # Train model based on model_type
    model = None
    scaler = None
    training_params = {}
    
    try:
        if model_type == "RFv1":
            # Random Forest
            model = train_rf_model(
                X_train, y_train,
                n_estimators=train_request.n_estimators,
                max_depth=train_request.max_depth,
                random_state=train_request.random_state
            )
            training_params = {
                'model_type': 'RandomForestClassifier',
                'n_estimators': train_request.n_estimators,
                'max_depth': train_request.max_depth,
                'random_state': train_request.random_state
            }
        elif model_type == "IFv1":
            # Isolation Forest
            contamination = train_request.contamination if train_request.contamination is not None else 0.1
            model = train_if_model(
                X_train, y_train,
                n_estimators=train_request.n_estimators,
                contamination=contamination,
                random_state=train_request.random_state
            )
            training_params = {
                'model_type': 'IsolationForest',
                'n_estimators': train_request.n_estimators,
                'contamination': contamination,
                'random_state': train_request.random_state
            }
        elif model_type == "AEv1":
            # Autoencoder
            hidden_layers = train_request.hidden_layers if train_request.hidden_layers else "64,32,32,64"
            model, scaler = train_ae_model(
                X_train, y_train,
                hidden_layers=hidden_layers,
                random_state=train_request.random_state
            )
            training_params = {
                'model_type': 'Autoencoder',
                'hidden_layers': hidden_layers,
                'random_state': train_request.random_state
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {model_type}. Supported types: RFv1, IFv1, AEv1"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
    
    # Save model
    # Create placeholder metrics (will be updated after testing)
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'confusion_matrix': [[0, 0], [0, 0]],
        'feature_importance': []
    }
    
    save_model(model, feature_names, metrics, training_params, model_name, scaler=scaler)
    
    response_content = {
        "status": "success",
        "message": "Model trained successfully",
        "dataset_name": dataset_name,
        "model_name": model_name,
        "model_type": model_type,
        "training_samples": len(X_train),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_params": training_params,
        "timestamp": datetime.now(timezone.utc).isoformat()
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
    
    model, metadata, scaler = load_model(model_name)
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
        # Get model_type from metadata or infer from model class
        model_type = metadata.get('model_type')
        if model_type and model_type not in ['RFv1', 'IFv1', 'AEv1']:
            # Convert class name to model type string
            if 'RandomForest' in model_type:
                model_type = 'RFv1'
            elif 'IsolationForest' in model_type:
                model_type = 'IFv1'
            elif 'MLPRegressor' in model_type:
                model_type = 'AEv1'
        
        metrics = evaluate_model(model, X_test, y_test, model_type=model_type, scaler=scaler)
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")
    
    # Update and save metadata with new metrics
    metadata['metrics'] = metrics
    metadata['last_test_date'] = datetime.now(timezone.utc).isoformat()
    
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
            "timestamp": datetime.now(timezone.utc).isoformat()
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
    model, metadata, scaler = load_model(model_name)
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
    
    # Make predictions based on model type
    try:
        # Get model_type from metadata or infer from model class
        model_type = metadata.get('model_type')
        if model_type and model_type not in ['RFv1', 'IFv1', 'AEv1']:
            # Convert class name to model type string
            if 'RandomForest' in model_type:
                model_type = 'RFv1'
            elif 'IsolationForest' in model_type:
                model_type = 'IFv1'
            elif 'MLPRegressor' in model_type:
                model_type = 'AEv1'
        
        # Infer from model class if not in metadata
        if not model_type:
            if isinstance(model, RandomForestClassifier):
                model_type = 'RFv1'
            elif isinstance(model, IsolationForest):
                model_type = 'IFv1'
            elif isinstance(model, MLPRegressor):
                model_type = 'AEv1'
        
        logger.info(f"Making predictions with model type: {model_type}")
        
        results = []
        
        if model_type == 'RFv1':
            # Random Forest: supervised model with predict_proba
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            label_mapping = metadata.get('label_mapping', {'0': 'safe', '1': 'unsafe'})
            
            for i in range(len(predictions)):
                pred = int(predictions[i])
                results.append({
                    'prediction': pred,
                    'label': label_mapping.get(str(pred), 'unknown'),
                    'probability_safe': float(probabilities[i][0]),
                    'probability_unsafe': float(probabilities[i][1]),
                    'confidence': float(max(probabilities[i]))
                })
        
        elif model_type == 'IFv1':
            # Isolation Forest: unsupervised model
            predictions_raw = model.predict(X)  # Returns -1 (outlier) or 1 (inlier)
            scores = -model.decision_function(X)  # Negative for outliers, positive for inliers
            
            # Convert to 0 (safe/inlier) or 1 (unsafe/outlier)
            predictions = np.where(predictions_raw == -1, 1, 0)
            
            # Normalize scores to probabilities (0-1 range)
            # Higher score = more anomalous
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            
            for i in range(len(predictions)):
                pred = int(predictions[i])
                prob_unsafe = float(normalized_scores[i])
                prob_safe = 1.0 - prob_unsafe
                results.append({
                    'prediction': pred,
                    'label': 'unsafe' if pred == 1 else 'safe',
                    'probability_safe': prob_safe,
                    'probability_unsafe': prob_unsafe,
                    'confidence': max(prob_safe, prob_unsafe)
                })
        
        elif model_type == 'AEv1':
            # Autoencoder: unsupervised model using reconstruction error
            if scaler is None:
                raise ValueError("Scaler is required for Autoencoder model predictions")
            
            # Scale the input
            X_scaled = scaler.transform(X)
            X_pred = model.predict(X_scaled)
            
            # Calculate reconstruction error (MSE) per sample
            mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
            
            # Use threshold from metadata if available, otherwise use 95th percentile
            threshold = metadata.get('metrics', {}).get('threshold')
            if threshold is None:
                threshold = np.percentile(mse, 95.0)
            
            # Predict: 1 if error > threshold (anomaly), 0 otherwise
            predictions = (mse > threshold).astype(int)
            
            # Normalize MSE to probabilities (0-1 range)
            # Higher error = more anomalous
            max_error = mse.max()
            if max_error > 0:
                normalized_errors = np.clip(mse / max_error, 0, 1)
            else:
                normalized_errors = np.zeros_like(mse)
            
            for i in range(len(predictions)):
                pred = int(predictions[i])
                prob_unsafe = float(normalized_errors[i])
                prob_safe = 1.0 - prob_unsafe
                results.append({
                    'prediction': pred,
                    'label': 'unsafe' if pred == 1 else 'safe',
                    'probability_safe': prob_safe,
                    'probability_unsafe': prob_unsafe,
                    'confidence': max(prob_safe, prob_unsafe)
                })
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
    
    return JSONResponse(
        content={
            "status": "success",
            "predictions": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        status_code=200
    )

@app.get("/model/status")
async def get_model_status(model_name: str = Depends(get_model_name)):
    """Get the current status of the model."""
    logger.info(f"Getting status for model: {model_name}")
    model, metadata, scaler = load_model(model_name)
    
    if model is None or metadata is None:
        return JSONResponse(
            content={
                "status": "not_trained",
                "message": "Model has not been trained yet",
                "timestamp": datetime.now(timezone.utc).isoformat()
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        status_code=200
    )

@app.get("/model/metrics")
async def get_model_metrics(model_name: str = Depends(get_model_name)):
    """Get the evaluation metrics of the trained model."""
    logger.info(f"Getting metrics for model: {model_name}")
    model, metadata, scaler = load_model(model_name)
    
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
