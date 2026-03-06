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

from fastapi import FastAPI, HTTPException, Header, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from starlette.requests import Request
import requests
import httpx
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars

# Import Feature Store and Vectorizer
from feature_store import FeatureStore, DataVectorizer

warnings.filterwarnings('ignore')
# Suppress sklearn parallel warning about delayed/Parallel usage
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
# Suppress all UserWarnings from sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# Suppress joblib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

app = FastAPI(title="Campus IoT Anomaly Detection Model API", version="1.0.0")

# Environment variable to control error-only logging
ERROR_ONLY_LOGGING = os.getenv("ERROR_ONLY_LOGGING", "false").lower() == "true"

# Context variable to store response status code
response_status: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar('response_status', default=None)

# Custom logging filter for error-only mode
class ErrorOnlyFilter(logging.Filter):
    """Filter to only show 400 and 500 errors when ERROR_ONLY_LOGGING is enabled"""
    
    def filter(self, record):
        if not ERROR_ONLY_LOGGING:
            return True  # Show all logs when disabled
        
        # Always show ERROR level and above
        if record.levelno >= logging.ERROR:
            return True
        
        # Check if this is an HTTP status code log
        status = response_status.get(None)
        if status is not None:
            # Only show 400 and 500 errors
            if status >= 400:
                return True
            return False
        
        # For non-HTTP logs, only show ERROR and above
        return record.levelno >= logging.ERROR

# Configure logging with filter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add filter to root logger
if ERROR_ONLY_LOGGING:
    for handler in logging.root.handlers:
        handler.addFilter(ErrorOnlyFilter())
    logger.info("ERROR_ONLY_LOGGING enabled - showing only 400 and 500 errors")

# Middleware to capture response status codes
class StatusCodeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        status = response.status_code
        response_status.set(status)
        
        # Log 400 and 500 errors even in error-only mode
        if ERROR_ONLY_LOGGING and status >= 400:
            logger.error(f"{request.method} {request.url.path} - Status: {status}")
        
        return response

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

# Add status code middleware (after CORS)
app.add_middleware(StatusCodeMiddleware)


# Input validation middleware
class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize input parameters for Model Service"""
    
    def validate_integer_param(self, value: str, param_name: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Tuple[Optional[int], Optional[str]]:
        """Validate integer parameter"""
        try:
            int_value = int(value)
            if min_val is not None and int_value < min_val:
                return None, f"{param_name} must be >= {min_val}"
            if max_val is not None and int_value > max_val:
                return None, f"{param_name} must be <= {max_val}"
            return int_value, None
        except ValueError:
            return None, f"{param_name} must be a valid integer"
    
    def validate_string_param(self, value: str, param_name: str, max_len: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
        """Validate string parameter"""
        if not value or len(value.strip()) == 0:
            return None, f"{param_name} cannot be empty"
        if max_len and len(value) > max_len:
            return None, f"{param_name} must be <= {max_len} characters"
        return value.strip(), None
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation for health checks
        if request.url.path in ["/health", "/"]:
            return await call_next(request)
        
        validation_errors = []
        path = request.url.path
        
        # Validate query parameters
        if "limit" in request.query_params:
            limit, error = self.validate_integer_param(request.query_params["limit"], "limit", min_val=1, max_val=10000)
            if error:
                validation_errors.append(error)
        
        if "offset" in request.query_params:
            offset, error = self.validate_integer_param(request.query_params["offset"], "offset", min_val=0)
            if error:
                validation_errors.append(error)
        
        if "database_name" in request.query_params:
            name, error = self.validate_string_param(request.query_params["database_name"], "database_name", max_len=255)
            if error:
                validation_errors.append(error)
        
        # Validate request body for POST endpoints (train, test, predict)
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    MAX_BODY_SIZE = 50 * 1024 * 1024  # 50MB for model operations (larger than gateway)
                    if len(body) > MAX_BODY_SIZE:
                        validation_errors.append(f"Request body too large. Maximum size: {MAX_BODY_SIZE / (1024*1024):.1f}MB")
                    else:
                        try:
                            json_data = json.loads(body.decode('utf-8'))
                            
                            # Service-specific validation for model endpoints
                            if "/predict" in path:
                                # Validate predict request structure
                                if not isinstance(json_data, dict):
                                    validation_errors.append("Predict request must be a JSON object")
                                elif "data" not in json_data:
                                    validation_errors.append("Predict request must contain 'data' field")
                                elif not isinstance(json_data.get("data"), list):
                                    validation_errors.append("Predict 'data' must be an array")
                            
                            request.state.validated_json = json_data
                        except json.JSONDecodeError as e:
                            validation_errors.append(f"Invalid JSON: {str(e)}")
                except Exception as e:
                    validation_errors.append(f"Error reading request body: {str(e)}")
        
        # Path validation
        if ".." in path or "//" in path or "\x00" in path:
            validation_errors.append("Invalid path: path traversal or null byte detected")
        
        if validation_errors:
            logger.warning(f"Validation errors for {request.method} {path}: {validation_errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation failed",
                    "errors": validation_errors,
                    "path": path
                }
            )
        
        return await call_next(request)


# Add input validation middleware
app.add_middleware(InputValidationMiddleware)

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
    feature_set_name: Optional[str] = None  # Name of feature set in store to use

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]

async def fetch_all_data(endpoint: str, label_type: str = "all", database_name: Optional[str] = None) -> List[Dict]:
    logger.info(f"Fetching {label_type} data from {endpoint}...")
    limit = 1000
    
    headers = {}
    if database_name:
        headers["dataset-name"] = database_name
    
    url = f"{API_BASE_URL}{endpoint}"
    
    async def fetch_page(page_num: int) -> tuple[int, List[Dict]]:
        """Fetch a single page and return (page_num, data)"""
        page_offset = page_num * limit
        try:
            logger.info(f"Fetching page {page_num + 1} (offset={page_offset})")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    params={"limit": limit, "offset": page_offset},
                    headers=headers
                )
            
            if response.status_code >= 400:
                error_detail = f"{response.status_code}"
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", error_json.get("message", f"{response.status_code}"))
                except:
                    error_detail = response.text or f"{response.status_code}"
                    raise Exception(f"HTTP {response.status_code}: {error_detail}")
            
            result = response.json()
            if result.get("status") != "success":
                error_msg = result.get("message", result.get("detail", "Unknown error"))
                raise Exception(f"API error: {error_msg}")
            
            data = result.get("data", [])
            logger.info(f"  Fetched page {page_num + 1}: {len(data)} rows")
            return (page_num, data)
        except Exception as e:
            logger.error(f"Error fetching page {page_num + 1}: {e}")
            raise
    
    try:
        # Fetch first page to get total_rows
        logger.info(f"Requesting {url} with params: limit={limit}, offset=0, headers={headers}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            first_response = await client.get(
                url,
                params={"limit": limit, "offset": 0},
                headers=headers
            )
        
        logger.info(f"Backend API response: {first_response.status_code} for {endpoint}")
        
        if first_response.status_code >= 400:
            error_detail = f"{first_response.status_code}"
            try:
                error_json = first_response.json()
                error_detail = error_json.get("detail", error_json.get("message", f"{first_response.status_code}"))
            except:
                error_detail = first_response.text or f"{first_response.status_code}"
            raise HTTPException(
                status_code=first_response.status_code,
                detail=error_detail
            )
        
        first_result = first_response.json()
        if first_result.get("status") != "success":
            error_msg = first_result.get("message", first_result.get("detail", "Unknown error"))
            logger.error(f"Error: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        total_rows = first_result.get("total_rows", 0)
        if total_rows == 0:
            error_msg = first_result.get("message", f"No {label_type} data found. Please ensure data has been uploaded and validated.")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
            
        first_data = first_result.get("data", [])
        if not first_data:
            logger.info(f"Total {label_type} records fetched: 0")
            return []
        
        # Calculate how many pages we need
        num_pages = (total_rows + limit - 1) // limit  # Ceiling division
        logger.info(f"Total rows: {total_rows}, limit: {limit}, will fetch {num_pages} pages in parallel")
        
        # If only one page, return early
        if num_pages == 1:
            logger.info(f"Total {label_type} records fetched: {len(first_data)}")
            return first_data
        
        # Fetch remaining pages (1 to num_pages-1) in batches to avoid overwhelming the server
        # Page 0 is already fetched, so we start from page 1
        all_data = [None] * num_pages  # Pre-allocate list to maintain order
        all_data[0] = first_data  # First page is already fetched
        
        # Batch size: fetch 10 pages at a time to avoid overwhelming the server
        batch_size = 10
        remaining_pages = list(range(1, num_pages))
        
        # Fetch pages in batches with retries
        async def fetch_with_retry(page_num: int, max_retries: int = 3) -> tuple[int, List[Dict]]:
            """Fetch a page with retry logic"""
            for attempt in range(max_retries):
                try:
                    return await fetch_page(page_num)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to fetch page {page_num + 1} after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for page {page_num + 1}: {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # Process pages in batches
        failed_pages = []
        for batch_start in range(0, len(remaining_pages), batch_size):
            batch = remaining_pages[batch_start:batch_start + batch_size]
            logger.info(f"Fetching batch of {len(batch)} pages (pages {batch[0] + 1} to {batch[-1] + 1})")
            
            # Create tasks for this batch
            tasks = [fetch_with_retry(page_num) for page_num in batch]
            
            # Execute batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                page_num = batch[i]
                if isinstance(result, Exception):
                    logger.error(f"Error fetching page {page_num + 1}: {result}")
                    failed_pages.append((page_num, str(result)))
                else:
                    result_page_num, data = result
                    all_data[result_page_num] = data
            
            # Small delay between batches to avoid overwhelming the server
            if batch_start + batch_size < len(remaining_pages):
                await asyncio.sleep(0.1)
        
        # If we have failed pages, try to continue with what we have or raise error
        if failed_pages:
            failed_count = len(failed_pages)
            total_pages = num_pages
            success_count = sum(1 for data in all_data if data is not None)
            
            # If more than 10% of pages failed, raise an error
            if failed_count > total_pages * 0.1:
                logger.error(f"Too many page fetch failures: {failed_count}/{total_pages} pages failed")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Data fetch partially failed",
                        "message": f"Failed to fetch {failed_count} out of {total_pages} pages",
                        "successful_pages": success_count,
                        "failed_pages": failed_count,
                        "solution": "The Data Ingestion Service may be overloaded. Please try again or reduce the dataset size."
                    }
                )
            else:
                logger.warning(f"Some pages failed ({failed_count}/{total_pages}), but continuing with available data")
        
        # Flatten the list of lists, maintaining order
        flattened_data = []
        for page_data in all_data:
            if page_data:
                flattened_data.extend(page_data)
        
        logger.info(f"Total {label_type} records fetched: {len(flattened_data)}")
        return flattened_data
    
    except HTTPException:
        raise
    except httpx.RequestError as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Error connecting to backend API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error fetching data: {str(e)}"
        )

def extract_features_and_labels(
    data_records: List[Dict],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    cat_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> tuple:
    """Extract features and labels from API response data.
    
    Includes feature engineering (ratios, log transforms) and categorical
    encoding via sklearn LabelEncoder. When ``cat_encoders`` is None (training),
    new encoders are fitted and returned. When ``cat_encoders`` is provided
    (inference / testing), the existing encoders are reused so categories map to
    the same integers seen during training.
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Label array (0 = safe, 1 = unsafe)  — empty when label col missing
        y_attack_cat: Attack category array (strings) — None if not available
        feature_cols: List of feature column names
        cat_encoders: Dict[str, LabelEncoder] fitted encoders (pass back on save)
    """

    # ── Categorical & skewed-numeric column lists (UNSW-NB15 specific) ──
    CATEGORICAL_COLS = ['proto', 'service', 'state']
    LOG_COLS = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'dpkts',
                'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'sjit', 'djit',
                'sinpkt', 'dinpkt', 'ct_srv_src', 'ct_dst_ltm', 'ct_src_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    # ── Parse raw records ──
    rows = []
    for record in data_records:
        row_data = record.get("data", {})
        if isinstance(row_data, str):
            row_data = json.loads(row_data)
        rows.append(row_data)
    
    if not rows:
        return pd.DataFrame(), np.array([]), None, [], cat_encoders

    df = pd.DataFrame(rows)

    # Strip BOM from column names (common with CSV uploads)
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]

    logger.info(f"Available columns in data ({len(df.columns)}): {list(df.columns)[:15]}...")

    # ── Detect attack_cat column (case-insensitive) ──
    attack_cat_col = None
    for col in df.columns:
        if col.lower() in ('attack_cat', 'attackcat'):
            attack_cat_col = col
            break
    if attack_cat_col and attack_cat_col != 'attack_cat':
        df = df.rename(columns={attack_cat_col: 'attack_cat'})

    # ── Label handling ──
    has_label = "label" in df.columns
    if not has_label:
        logger.warning("'label' column not found — treating as inference data (y will be empty)")

    default_exclude_cols = {"label", "id", "attack_cat", "\ufeffid"}

    # ── Select feature columns ──
    if include_fields is not None:
        include_lower = {f.lower() for f in include_fields}
        feature_cols = [col for col in df.columns
                        if col.lower() in include_lower and col not in default_exclude_cols]
    else:
        exclude_set = set(default_exclude_cols)
        if exclude_fields:
            for col in df.columns:
                if col.lower() in {f.lower() for f in exclude_fields}:
                    exclude_set.add(col)
        feature_cols = [col for col in df.columns if col not in exclude_set]

    if "label" in feature_cols:
        feature_cols.remove("label")
    if "attack_cat" in feature_cols:
        feature_cols.remove("attack_cat")

    if not feature_cols:
        raise ValueError("No features available after filtering.")

    X = df[feature_cols].copy()

    # ── Extract y and attack_cat ──
    if has_label:
        y = pd.to_numeric(df["label"], errors='coerce')
        valid_mask = ~y.isna()
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].astype(int).reset_index(drop=True)
    else:
        y = np.array([])
        valid_mask = pd.Series([True] * len(X))

    y_attack_cat = None
    if "attack_cat" in df.columns:
        y_attack_cat = df["attack_cat"][valid_mask].fillna("Normal").astype(str).replace("", "Normal").reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════
    #  FEATURE ENGINEERING — this is the key accuracy improvement
    # ══════════════════════════════════════════════════════════════════

    # 1) Engineered ratio / aggregate features
    if 'sbytes' in X.columns and 'dbytes' in X.columns:
        sbytes = pd.to_numeric(X['sbytes'], errors='coerce').fillna(0)
        dbytes = pd.to_numeric(X['dbytes'], errors='coerce').fillna(0)
        X['total_bytes'] = sbytes + dbytes
        X['byte_ratio'] = sbytes / (X['total_bytes'] + 1)

    if 'spkts' in X.columns and 'dpkts' in X.columns:
        spkts = pd.to_numeric(X['spkts'], errors='coerce').fillna(0)
        dpkts = pd.to_numeric(X['dpkts'], errors='coerce').fillna(0)
        X['total_pkts'] = spkts + dpkts
        X['pkt_ratio'] = spkts / (X['total_pkts'] + 1)

    if 'smeansz' in X.columns and 'dmeansz' in X.columns:
        smeansz = pd.to_numeric(X['smeansz'], errors='coerce').fillna(0)
        dmeansz = pd.to_numeric(X['dmeansz'], errors='coerce').fillna(0)
        X['mean_pkt_size_ratio'] = smeansz / (dmeansz + 1)

    if 'sload' in X.columns and 'dload' in X.columns:
        sload = pd.to_numeric(X['sload'], errors='coerce').fillna(0)
        dload = pd.to_numeric(X['dload'], errors='coerce').fillna(0)
        X['load_ratio'] = sload / (dload + 1)

    if 'sttl' in X.columns and 'dttl' in X.columns:
        sttl = pd.to_numeric(X['sttl'], errors='coerce').fillna(0)
        dttl = pd.to_numeric(X['dttl'], errors='coerce').fillna(0)
        X['ttl_ratio'] = sttl / (dttl + 1)

    if 'dur' in X.columns and ('sbytes' in X.columns or 'dbytes' in X.columns):
        dur = pd.to_numeric(X['dur'], errors='coerce').fillna(0)
        sbytes = pd.to_numeric(X.get('sbytes', 0), errors='coerce').fillna(0)
        dbytes = pd.to_numeric(X.get('dbytes', 0), errors='coerce').fillna(0)
        X['bytes_per_sec'] = (sbytes + dbytes) / (dur + 0.001)

    # 2) Log1p transform for highly skewed numeric columns
    for col in LOG_COLS:
        if col in X.columns:
            X[col] = np.log1p(pd.to_numeric(X[col], errors='coerce').fillna(0).clip(lower=0))

    # 3) Categorical encoding via LabelEncoder
    is_training = (cat_encoders is None)
    if is_training:
        cat_encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        X[col] = X[col].astype(str).str.strip().str.lower()
        if is_training:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            cat_encoders[col] = le
            logger.info(f"Fitted LabelEncoder for '{col}' with {len(le.classes_)} classes")
        else:
            le = cat_encoders.get(col)
            if le is not None:
                # Map unseen categories to a known class (or the most frequent one)
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])
            else:
                X[col] = pd.factorize(X[col])[0]

    # 4) Force remaining columns to numeric
    for col in X.columns:
        if col not in CATEGORICAL_COLS:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)

    # Update feature_cols to match the actual columns after engineering
    feature_cols = list(X.columns)

    logger.info(f"Extracted {len(X)} samples with {len(feature_cols)} features (incl. engineered)")
    if has_label:
        logger.info(f"Label distribution: Safe={int((y == 0).sum())}, Unsafe={int((y == 1).sum())}")
    if y_attack_cat is not None:
        logger.info(f"Attack categories: {dict(y_attack_cat.value_counts().head(8))}")

    return X, y, y_attack_cat, feature_cols, cat_encoders

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
        verbose=1,
        class_weight='balanced'  # Improved accuracy for imbalanced data
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
    proba = model.predict_proba(X_test)
    # Handle case where model might only have one class (shouldn't happen but be defensive)
    if proba.shape[1] == 1:
        # Only one class - use that class's probability
        y_pred_proba = proba[:, 0]
    elif proba.shape[1] >= 2:
        # Binary or multi-class: use probability of class 1 (unsafe/anomaly)
        y_pred_proba = proba[:, 1]
    else:
        # Fallback: use predictions as probabilities
        y_pred_proba = y_pred.astype(float)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC AUC and PR AUC using probabilities
    try:
        # Check if we have both classes in y_test for ROC AUC calculation
        unique_classes = np.unique(y_test)
        if len(unique_classes) >= 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
        else:
            # Only one class in test data - can't calculate AUC
            logger.warning(f"Only one class ({unique_classes[0]}) in test data. Cannot calculate ROC AUC and PR AUC.")
            roc_auc = 0.0
            pr_auc = 0.0
    except ValueError as e:
        logger.warning(f"Could not calculate AUC metrics: {e}")
        roc_auc = 0.0
        pr_auc = 0.0
    
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
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
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
        # Fallback: try to infer model type from instance
        if isinstance(model, MLPRegressor):
            if scaler is None:
                raise ValueError("Scaler is required for Autoencoder model evaluation")
            logger.warning(f"Unknown model type {model_type_str}, but detected MLPRegressor - using Autoencoder evaluation")
            return evaluate_ae_model(model, scaler, X_test, y_test)
        elif isinstance(model, IsolationForest):
            logger.warning(f"Unknown model type {model_type_str}, but detected IsolationForest - using Isolation Forest evaluation")
            return evaluate_if_model(model, X_test, y_test)
        elif isinstance(model, RandomForestClassifier):
            logger.warning(f"Unknown model type {model_type_str}, but detected RandomForestClassifier - using Random Forest evaluation")
            return evaluate_rf_model(model, X_test, y_test)
        else:
            raise ValueError(f"Unknown model type {model_type_str} and cannot infer from model instance. Model class: {type(model).__name__}")

def save_model(model: Any, vectorizer: DataVectorizer, 
               metrics: Dict[str, Any], training_params: Dict[str, Any],
               model_name: str = "model",
               attack_cat_model: Optional[Any] = None, attack_cat_classes: Optional[List[str]] = None):
    """Save the trained model, vectorizer, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
    model_path = os.path.join(MODEL_DIR, f"{sanitized_model_name}.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, f"{sanitized_model_name}_vectorizer.pkl")
    metadata_path = os.path.join(MODEL_DIR, f"{sanitized_model_name}_metadata.json")
    
    # Save Model
    joblib.dump(model, model_path)
    
    # Save Vectorizer (The "DNA" of the model)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save attack category model if provided
    if attack_cat_model is not None:
        attack_cat_path = os.path.join(MODEL_DIR, f"{sanitized_model_name}_attack_cat.pkl")
        joblib.dump(attack_cat_model, attack_cat_path)
    
    # Metadata
    model_type_str = training_params.get('model_type', type(model).__name__)
    metadata = {
        'model_type': model_type_str,
        'model_name': model_name,
        'feature_names': vectorizer.feature_names,
        'n_features': len(vectorizer.feature_names),
        'training_date': datetime.now(timezone.utc).isoformat(),
        'training_params': training_params,
        'metrics': metrics,
        'has_attack_cat_model': attack_cat_model is not None,
        'attack_cat_classes': attack_cat_classes if attack_cat_classes else []
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model and DNA (Vectorizer) saved for {model_name}")

def load_model(model_name: str) -> tuple:
    """Load model, metadata, scaler, attack_cat_model, cat_encoders, and vectorizer.
    Returns (model, metadata, scaler, attack_cat_model, cat_encoders, vectorizer)."""
    sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
    model_filename = f"{sanitized_model_name}.pkl"
    metadata_filename = f"{sanitized_model_name}_metadata.json"
    scaler_filename = f"{sanitized_model_name}_scaler.pkl"
    attack_cat_filename = f"{sanitized_model_name}_attack_cat.pkl"
    cat_enc_filename = f"{sanitized_model_name}_cat_encoders.pkl"
    vectorizer_filename = f"{sanitized_model_name}_vectorizer.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    metadata_path = os.path.join(MODEL_DIR, metadata_filename)
    scaler_path = os.path.join(MODEL_DIR, scaler_filename)
    attack_cat_path = os.path.join(MODEL_DIR, attack_cat_filename)
    cat_enc_path = os.path.join(MODEL_DIR, cat_enc_filename)
    vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)
    
    if not os.path.exists(model_path):
        return None, None, None, None, None, None
    
    try:
        model = joblib.load(model_path)
    except (OSError, IOError) as e:
        logger.error(f"File error loading model: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Error loading model file (possibly corrupted): {e}")
        return None, None, None, None, None, None
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None, None, None, None, None, None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (OSError, IOError) as e:
        logger.error(f"File error loading metadata: {e}")
        return None, None, None, None, None, None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in metadata file (possibly corrupted): {e}")
        return None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        return None, None, None, None, None, None
    
    # Load scaler if it exists (for autoencoder models)
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from: {scaler_path}")
        except Exception as e:
            logger.warning(f"Error loading scaler file: {e}")
    
    # Load attack category model if it exists
    attack_cat_model = None
    if os.path.exists(attack_cat_path):
        try:
            attack_cat_model = joblib.load(attack_cat_path)
            logger.info(f"Loaded attack category model from: {attack_cat_path}")
        except Exception as e:
            logger.warning(f"Error loading attack category model: {e}")
            
    # Load categorical encoders if they exist
    cat_encoders = None
    if os.path.exists(cat_enc_path):
        try:
            cat_encoders = joblib.load(cat_enc_path)
            logger.info(f"Loaded cat encoders from: {cat_enc_path}")
        except Exception as e:
            logger.warning(f"Error loading cat encoders: {e}")
    
    # Load vectorizer if it exists
    vectorizer = None
    if os.path.exists(vectorizer_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Loaded vectorizer from: {vectorizer_path}")
        except Exception as e:
            logger.warning(f"Error loading vectorizer: {e}")
    
    return model, metadata, scaler, attack_cat_model, cat_encoders, vectorizer

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
            status_code=200,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
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
    dataset_name: str = Header(...),
    train_request: TrainRequest = TrainRequest()
) -> str:
    if dataset_name:
        return dataset_name
    return train_request.database_name or "default"

@app.get("/features")
async def list_features():
    """List all feature sets in the Feature Store."""
    fs = FeatureStore()
    features = fs.list_features()
    return JSONResponse(
        content={
            "status": "success",
            "total_feature_sets": len(features),
            "feature_sets": features,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        status_code=200
    )

@app.post("/features/vectorize")
async def vectorize_features(
    name: str = Body(..., embed=True),
    dataset_name: str = Depends(get_database_name)
):
    """Fetch data from backend, vectorize it, and save to Feature Store."""
    logger.info(f"Vectorizing features for dataset '{dataset_name}' and saving as '{name}'")
    
    try:
        # Fetch training data
        training_data = await fetch_all_data("/training", "training", dataset_name)
        if not training_data:
            raise HTTPException(status_code=400, detail="No training data found to vectorize.")
            
        # Parse into DataFrame
        rows = []
        for record in training_data:
            row_data = record.get("data", {})
            if isinstance(row_data, str):
                row_data = json.loads(row_data)
            rows.append(row_data)
        df = pd.DataFrame(rows)
        
        # Split features and label
        if 'label' not in df.columns:
            raise HTTPException(status_code=400, detail="'label' column missing from data.")
            
        y = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int).values
        X_raw = df.drop(['label', 'id', 'attack_cat'], axis=1, errors='ignore')
        
        # Fit and transform
        vectorizer = DataVectorizer()
        X_vec = vectorizer.fit(X_raw).transform(X_raw)
        
        # Save to store
        fs = FeatureStore()
        fs.save(name, X_vec, y, vectorizer)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Features '{name}' vectorized and stored successfully",
                "samples": len(X_vec),
                "features": list(X_vec.columns),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=201
        )
    except Exception as e:
        logger.error(f"Error vectorizing features: {e}")
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

def get_model_name(
    model_name: str = Header(...)
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
    headers["dataset-name"] = dataset_name
    
    try:
        health_url = f"{API_BASE_URL}/health"
        logger.info(f"Checking backend health at {health_url} with headers: {headers}")
        # Increase timeout to 10 seconds for health checks
        response = requests.get(health_url, headers=headers, timeout=10)
        logger.info(f"Backend health check response: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Backend API health check failed with status {response.status_code}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Backend service unhealthy",
                    "message": f"Data Ingestion Service returned status {response.status_code}",
                    "service": "Data Ingestion Service",
                    "port": "8000"
                }
            )
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout connecting to backend API at {API_BASE_URL}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Backend service timeout",
                "message": f"Data Ingestion Service did not respond within 10 seconds",
                "service": "Data Ingestion Service",
                "port": "8000",
                "solution": "The service may be overloaded or not responding. Check if the Data Ingestion Service is running and healthy."
            }
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to backend API at {API_BASE_URL}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Backend service unavailable",
                "message": f"Cannot connect to Data Ingestion Service at {API_BASE_URL}",
                "service": "Data Ingestion Service",
                "port": "8000",
                "solution": "Please ensure the Data Ingestion Service is running. Start it with: cd 01_Data_Ingestion_Service && uvicorn main:app --reload --port 8000"
            }
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to backend API at {API_BASE_URL}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Backend service error",
                "message": f"Error connecting to Data Ingestion Service: {str(e)}",
                "service": "Data Ingestion Service",
                "port": "8000"
            }
        )
    
    # --- UNIFIED AUTOMATED DATA LOADING ---
    fs = FeatureStore()
    X_train, y_train, vectorizer = None, None, None
    
    # Auto-identifer for the dataset's features
    feature_store_name = train_request.feature_set_name or f"auto_{dataset_name}"
    
    # Try to load from store
    X_train, y_train, vectorizer = fs.load(feature_store_name)
    
    if X_train is None:
        logger.info(f"Vectorized data not found in store for '{feature_store_name}'. Vectorizing now...")
        # Fetch fresh data from backend
        training_data = await fetch_all_data("/training", "training", dataset_name)
        if not training_data:
            raise HTTPException(status_code=400, detail="No training data found in backend.")
            
        # Standardize into DataFrame
        rows = [json.loads(r["data"]) if isinstance(r["data"], str) else r["data"] for r in training_data]
        df = pd.DataFrame(rows)
        
        if 'label' not in df.columns:
            # Fallback if label is named 'Label' or similar
            label_col = next((c for c in df.columns if c.lower() == 'label'), None)
            if label_col:
                df = df.rename(columns={label_col: 'label'})
            else:
                raise HTTPException(status_code=400, detail="Required 'label' column missing for supervised training.")
        
        y_train = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int).values
        X_raw = df.drop(['label', 'id', 'attack_cat'], axis=1, errors='ignore')
        
        # Fit logic
        vectorizer = DataVectorizer()
        X_train = vectorizer.fit(X_raw).transform(X_raw)
        
        # Automatically save to store for next time
        fs.save(feature_store_name, X_train, y_train, vectorizer)
        logger.info(f"Auto-vectorization complete. Saved to store as '{feature_store_name}'")
    else:
        logger.info(f"Loaded pre-vectorized data from store: '{feature_store_name}'")

    # --- MODEL TRAINING ---
    model = None
    training_params = {'model_type': model_type, 'feature_store_name': feature_store_name}
    
    try:
        if model_type == "RFv1":
            model = train_rf_model(X_train, y_train, 
                                 n_estimators=train_request.n_estimators,
                                 max_depth=train_request.max_depth)
        elif model_type == "IFv1":
            model = train_if_model(X_train, y_train, 
                                 contamination=train_request.contamination or 0.1)
        elif model_type == "AEv1":
            model, scaler = train_ae_model(X_train, y_train, 
                                         hidden_layers=train_request.hidden_layers or "64,32,32,64")
            # We treat the scaler inside our DataVectorizer, so AEv1 is mostly for backward compat
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        # Placeholder metrics (real ones added in /test phase usually, or we can add summary here)
        metrics = {"status": "trained", "samples": len(X_train)}
        
        save_model(model, vectorizer, metrics, training_params, model_name=model_name)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model '{model_name}' trained and saved automatically with DNA.",
                "feature_set": feature_store_name,
                "samples": len(X_train)
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'confusion_matrix': [[0, 0], [0, 0]],
        'feature_importance': []
    }
    
    # Save model (attack_cat_model is in scope from training block)
    save_model(model, feature_names, metrics, training_params, model_name, scaler=scaler,
               attack_cat_model=attack_cat_model, attack_cat_classes=attack_cat_classes,
               cat_encoders=cat_encoders)
    
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
    dataset_name: Optional[str] = Header(None),
    test_request: TestRequest = TestRequest()
) -> Optional[str]:
    if dataset_name:
        return dataset_name
    return test_request.database_name

def get_test_model_name(
    model_name: str = Header(...)
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
    
    model, metadata, scaler, attack_cat_model, cat_encoders, _ = load_model(model_name)
    if model is None or metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first using POST /train"
        )
    
    headers = {}
    if database_name:
        headers["dataset-name"] = database_name
        logger.info(f"Using database: {database_name}")
    
    try:
        health_url = f"{API_BASE_URL}/health"
        logger.info(f"Checking backend health at {health_url} with headers: {headers}")
        response = requests.get(health_url, headers=headers, timeout=5)
        logger.info(f"Backend health check response: {response.status_code}")
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
        testing_data = await fetch_all_data("/testing", "testing", database_name)
        if not testing_data:
            raise HTTPException(
                status_code=422,
                detail="No testing data found. Please validate data to create training/testing split first."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching testing data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching testing data: {str(e)}")
    
    # Extract features and labels using the training encoders
    try:
        # Pass the plain dict records into the standardized pipeline.
        raw_records = [{"data": row} for _, row in pd.DataFrame(testing_data).iterrows()] if isinstance(testing_data, pd.DataFrame) else testing_data
        
        feature_names = metadata['feature_names']
        
        X_df, y_test, _, _, _ = extract_features_and_labels(
            data_records=raw_records, 
            include_fields=feature_names,
            cat_encoders=cat_encoders
        )
        
        X_test = X_df
        if len(X_test) == 0:
            raise HTTPException(
                status_code=422,
                detail="No valid testing samples found. Testing data may be missing required features."
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
        
        # If model_type is still not set, infer from model instance
        if not model_type or model_type not in ['RFv1', 'IFv1', 'AEv1']:
            if isinstance(model, RandomForestClassifier):
                model_type = 'RFv1'
            elif isinstance(model, IsolationForest):
                model_type = 'IFv1'
            elif isinstance(model, MLPRegressor):
                model_type = 'AEv1'
            else:
                # Try to infer from class name
                model_class_name = type(model).__name__
                if 'RandomForest' in model_class_name:
                    model_type = 'RFv1'
                elif 'IsolationForest' in model_class_name:
                    model_type = 'IFv1'
                elif 'MLPRegressor' in model_class_name:
                    model_type = 'AEv1'
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unknown model type: {model_class_name}. Cannot evaluate model."
                    )
        
        logger.info(f"Evaluating model with type: {model_type}")
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
# a
    """
    Make predictions on new data.
    
    Request body should contain a list of data records with feature values.
    """
    logger.info(f"Prediction request received for {len(predict_request.data)} samples")
    logger.info(f"Using model: {model_name}")
    
    # Load model
    model, metadata, scaler, attack_cat_model, cat_encoders, vectorizer = load_model(model_name)
    if model is None or metadata is None:
        # Check if model file exists but is corrupted
        sanitized_model_name = model_name.replace('/', '_').replace('\\', '_').replace('..', '_')
        model_path = os.path.join(MODEL_DIR, f"{sanitized_model_name}.pkl")
        if os.path.exists(model_path):
            raise HTTPException(
                status_code=500,
                detail="Model file exists but could not be loaded. Model may be corrupted. Please retrain."
            )
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
    
    # Validate input data
    if not predict_request.data or len(predict_request.data) == 0:
        raise HTTPException(
            status_code=422,
            detail="Prediction request must contain at least one data record."
        )
    
    # --- AUTOMATED VECTORIZATION FOR INFERENCE ---
    try:
        # Load the vectorizer (DNA) that was used when this specific model was trained
        # This is loaded in load_model automatically as part of the tuple
        if vectorizer is None:
            raise ValueError("Model DNA (Vectorizer) is missing. Please retrain this model.")
            
        # Convert incoming data dicts into a DataFrame
        df_input = pd.DataFrame(predict_request.data)
        
        # Strip BOM and normalize columns as the vectorizer expects
        df_input.columns = [c.lstrip('\ufeff').strip().lower() for c in df_input.columns]
        
        # Transform using the EXACT rules used for training
        X_vec = vectorizer.transform(df_input)
        X = X_vec.values
        
    except Exception as e:
        logger.error(f"Inference Vectorization failed: {e}")
        raise HTTPException(status_code=400, detail=f"Data preparation failed: {str(e)}")
    
    # Make predictions based on model type
    try:
        # Get model_type from metadata or infer from model class
        model_type = metadata.get('model_type')
        if model_type and model_type not in ['RFv1', 'IFv1', 'AEv1']:
            # Convert class name or display name to model type string
            if 'RandomForest' in model_type:
                model_type = 'RFv1'
            elif 'IsolationForest' in model_type:
                model_type = 'IFv1'
            elif 'MLPRegressor' in model_type or model_type == 'Autoencoder':
                model_type = 'AEv1'
        
        # Infer from model class if not in metadata
        if not model_type or model_type not in ['RFv1', 'IFv1', 'AEv1']:
            if isinstance(model, RandomForestClassifier):
                model_type = 'RFv1'
            elif isinstance(model, IsolationForest):
                model_type = 'IFv1'
            elif isinstance(model, MLPRegressor):
                model_type = 'AEv1'
            else:
                # Last resort: try to infer from model class name
                model_class_name = type(model).__name__
                if 'RandomForest' in model_class_name:
                    model_type = 'RFv1'
                elif 'IsolationForest' in model_class_name:
                    model_type = 'IFv1'
                elif 'MLPRegressor' in model_class_name:
                    model_type = 'AEv1'
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unknown model type: {model_type or model_class_name}. Cannot make predictions."
                    )
        
        logger.info(f"Making predictions with model type: {model_type}")
        
        results = []
        
        if model_type == 'RFv1':
            # Random Forest: supervised model with predict_proba
            predictions = model.predict(X)
            proba = model.predict_proba(X)
            label_mapping = metadata.get('label_mapping', {'0': 'safe', '1': 'unsafe'})
            
            # Handle case where model might only have one class
            if proba.shape[1] == 1:
                # Only one class - check which class the model has
                model_classes = model.classes_
                prob_single = proba[:, 0]
                
                if len(model_classes) == 1:
                    # Model only has one class - determine if it's class 0 (safe) or class 1 (unsafe)
                    if model_classes[0] == 0:
                        # Model only predicts safe (0)
                        probabilities = np.column_stack([prob_single, 1.0 - prob_single])
                        logger.warning(f"Model has only class 0 (safe). Creating probabilities: safe={prob_single}, unsafe={1.0 - prob_single}")
                    elif model_classes[0] == 1:
                        # Model only predicts unsafe (1)
                        probabilities = np.column_stack([1.0 - prob_single, prob_single])
                        logger.warning(f"Model has only class 1 (unsafe). Creating probabilities: safe={1.0 - prob_single}, unsafe={prob_single}")
                    else:
                        # Unknown class - assume it's unsafe
                        probabilities = np.column_stack([1.0 - prob_single, prob_single])
                        logger.warning(f"Model has unknown single class {model_classes[0]}. Assuming unsafe probabilities.")
                else:
                    # Fallback: assume unsafe
                    probabilities = np.column_stack([1.0 - prob_single, prob_single])
                    logger.warning(f"Model predict_proba returned 1 column but classes={model_classes}. Assuming unsafe probabilities.")
            elif proba.shape[1] >= 2:
                # Binary or multi-class: use as is
                probabilities = proba
            else:
                # Fallback: create probabilities from predictions
                pred_float = predictions.astype(float)
                probabilities = np.column_stack([1.0 - pred_float, pred_float])
                logger.warning("Could not get probabilities from model, using predictions as probabilities")
            
            # Predict attack categories for unsafe samples if attack_cat_model is available
            attack_cat_predictions = None
            attack_cat_probabilities = None
            unsafe_indices = None
            if attack_cat_model is not None:
                logger.info("Attack category model found, attempting to predict attack categories for unsafe samples")
                # Only predict attack category for samples predicted as unsafe
                unsafe_indices = np.where(predictions == 1)[0]
                logger.info(f"Found {len(unsafe_indices)} unsafe samples out of {len(predictions)} total")
                if len(unsafe_indices) > 0:
                    X_unsafe = X[unsafe_indices]
                    attack_cat_predictions_raw = attack_cat_model.predict(X_unsafe)
                    attack_cat_probabilities_raw = attack_cat_model.predict_proba(X_unsafe)
                    # Get class names from metadata or use model's classes_ attribute
                    attack_cat_classes = metadata.get('attack_cat_classes', [])
                    if len(attack_cat_classes) == 0:
                        # Fallback: try to get classes from the model itself
                        if hasattr(attack_cat_model, 'classes_'):
                            attack_cat_classes = attack_cat_model.classes_.tolist()
                            logger.info(f"Using attack_cat_classes from model: {attack_cat_classes}")
                        else:
                            logger.warning("attack_cat_classes not found in metadata and model has no classes_ attribute")
                    
                    if len(attack_cat_classes) > 0:
                        # Map predictions to class names
                        attack_cat_predictions = [attack_cat_classes[int(pred)] for pred in attack_cat_predictions_raw]
                        # Get probabilities for each class
                        attack_cat_probabilities = []
                        for probs in attack_cat_probabilities_raw:
                            prob_dict = {attack_cat_classes[i]: float(probs[i]) for i in range(len(attack_cat_classes))}
                            attack_cat_probabilities.append(prob_dict)
                        logger.info(f"Successfully predicted {len(attack_cat_predictions)} attack categories: {attack_cat_predictions[:5]}")
                    else:
                        logger.warning("attack_cat_classes is empty, cannot map predictions to category names")
                else:
                    logger.info("No unsafe samples found, skipping attack category prediction")
            else:
                logger.info("No attack category model found for this model")
            
            for i in range(len(predictions)):
                pred_binary = int(predictions[i])
                # Calculate risk percentage (0-100) based on probability_unsafe
                risk_percentage = float(probabilities[i][1] * 100)
                result = {
                    'prediction': round(risk_percentage, 2),  # Risk percentage (0-100)
                    'label': label_mapping.get(str(pred_binary), 'unknown'),
                    'probability_safe': float(probabilities[i][0]),
                    'probability_unsafe': float(probabilities[i][1]),
                    'confidence': float(max(probabilities[i]))
                }
                
                # Add attack category prediction if available and sample is unsafe
                if pred_binary == 1:  # Unsafe/anomaly prediction (using binary for logic)
                    if attack_cat_model is not None and attack_cat_predictions is not None and unsafe_indices is not None:
                        # Find the position of this sample in the unsafe_indices array
                        pos_in_unsafe = np.where(unsafe_indices == i)[0]
                        if len(pos_in_unsafe) > 0:
                            idx = pos_in_unsafe[0]
                            result['attack_cat'] = attack_cat_predictions[idx]
                            result['attack_cat_probabilities'] = attack_cat_probabilities[idx] if attack_cat_probabilities else {}
                        else:
                            result['attack_cat'] = 'Unknown'
                            result['attack_cat_probabilities'] = {}
                    elif attack_cat_model is not None:
                        # Attack category model exists but predictions weren't generated (likely missing attack_cat_classes)
                        result['attack_cat'] = 'Unknown'
                        result['attack_cat_probabilities'] = {}
                        logger.debug(f"Attack category model exists but no predictions for sample {i}")
                    else:
                        # No attack category model available
                        result['attack_cat'] = None
                        result['attack_cat_probabilities'] = {}
                elif pred_binary == 0:
                    # Safe samples don't have attack categories
                    result['attack_cat'] = 'Normal'
                    result['attack_cat_probabilities'] = {}
                else:
                    result['attack_cat'] = None
                    result['attack_cat_probabilities'] = {}
                
                results.append(result)
        
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
                pred_binary = int(predictions[i])
                prob_unsafe = float(normalized_scores[i])
                prob_safe = 1.0 - prob_unsafe
                # Calculate risk percentage (0-100) based on probability_unsafe
                risk_percentage = round(prob_unsafe * 100, 2)
                results.append({
                    'prediction': risk_percentage,  # Risk percentage (0-100)
                    'label': 'unsafe' if pred_binary == 1 else 'safe',
                    'probability_safe': prob_safe,
                    'probability_unsafe': prob_unsafe,
                    'confidence': max(prob_safe, prob_unsafe),
                    'attack_cat': None,  # Isolation Forest doesn't predict attack categories
                    'attack_cat_probabilities': {}
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
                pred_binary = int(predictions[i])
                prob_unsafe = float(normalized_errors[i])
                prob_safe = 1.0 - prob_unsafe
                # Calculate risk percentage (0-100) based on probability_unsafe
                risk_percentage = round(prob_unsafe * 100, 2)
                results.append({
                    'prediction': risk_percentage,  # Risk percentage (0-100)
                    'label': 'unsafe' if pred_binary == 1 else 'safe',
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
    model, metadata, scaler, attack_cat_model, cat_encoders, _ = load_model(model_name)
    
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
    model, metadata, scaler, attack_cat_model, cat_encoders, _ = load_model(model_name)
    
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
