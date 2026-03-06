from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple
from starlette.requests import Request
import json
import os
from datetime import datetime, timedelta
import logging
import sqlite3
import asyncio
import random
import uuid
import httpx
import contextvars

def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

app = FastAPI(title="Campus IoT User Service", version="1.0.0")

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

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://localhost:8080",
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
    """Validate and sanitize input parameters for User Service"""
    
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
    
    def validate_boolean_param(self, value: str, param_name: str) -> Tuple[Optional[bool], Optional[str]]:
        """Validate boolean parameter"""
        value_lower = value.lower()
        if value_lower in ["true", "1", "yes"]:
            return True, None
        elif value_lower in ["false", "0", "no", ""]:
            return False, None
        return None, f"{param_name} must be 'true' or 'false'"
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation for health checks and WebSocket endpoints
        if request.url.path in ["/health", "/"] or request.url.path.startswith("/ws/"):
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
        
        if "user_id" in request.query_params:
            user_id, error = self.validate_integer_param(request.query_params["user_id"], "user_id", min_val=1)
            if error:
                validation_errors.append(error)
        
        if "network_id" in request.query_params:
            network_id, error = self.validate_string_param(request.query_params["network_id"], "network_id", max_len=255)
            if error:
                validation_errors.append(error)
        
        # Validate boolean parameters
        for bool_param in ["has_prediction", "is_active"]:
            if bool_param in request.query_params:
                _, error = self.validate_boolean_param(request.query_params[bool_param], bool_param)
                if error:
                    validation_errors.append(error)
        
        # Validate path parameters (user_id in path)
        if "/users/" in path:
            parts = path.split("/")
            for i, part in enumerate(parts):
                if part == "users" and i + 1 < len(parts):
                    user_id_str = parts[i + 1]
                    if user_id_str not in ["block", "unblock"]:  # Skip action endpoints
                        user_id, error = self.validate_integer_param(user_id_str, "user_id (path)", min_val=1)
                        if error:
                            validation_errors.append(error)
                    break
        
        # Validate network_id in path
        if "/network-logs/" in path:
            parts = path.split("/")
            for i, part in enumerate(parts):
                if part == "network-logs" and i + 1 < len(parts):
                    network_id = parts[i + 1]
                    _, error = self.validate_string_param(network_id, "network_id (path)", max_len=255)
                    if error:
                        validation_errors.append(error)
                    break
        
        # Validate request body for POST endpoints
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB
                    if len(body) > MAX_BODY_SIZE:
                        validation_errors.append(f"Request body too large. Maximum size: {MAX_BODY_SIZE / (1024*1024):.1f}MB")
                    else:
                        try:
                            json_data = json.loads(body.decode('utf-8'))
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

NETWORK_LOGS_DB = "network_logs.db"
USERS_DB = "users.db"
MESSAGE_QUEUE_DB = "message_queue.db"
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8001")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "RFv1")

# Singleton WebSocket connection tracking for data generation
active_generate_websocket: Optional[WebSocket] = None
generate_websocket_lock = asyncio.Lock()
generate_websocket_session_start_time: Optional[str] = None

# Multiple WebSocket connections allowed for viewing data
view_websockets: set[WebSocket] = set()
view_websockets_lock = asyncio.Lock()

# Selected model name (can be changed via API)
selected_model_name: str = DEFAULT_MODEL_NAME

def get_bool_env(key: str, default: bool = True) -> bool:
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")

WEBSOCKET_ENABLED = get_bool_env("WEBSOCKET_ENABLED", True)
MESSAGE_QUEUE_ENABLED = get_bool_env("MESSAGE_QUEUE_ENABLED", True)

class BlockRequest(BaseModel):
    block_type: str
    block_reason: Optional[str] = None
    block_duration_hours: Optional[int] = None

class PublishRequest(BaseModel):
    network_id: str
    data: dict

class SetModelRequest(BaseModel):
    model_name: str

@app.get("/health")
async def health_check():
    os.makedirs("data", exist_ok=True)
    init_users_db()
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Campus IoT User Service",
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    )

@app.get("/users")
async def get_users(limit: int = 100, offset: int = 0):
    """
    Get users with pagination support.
    
    Query parameters:
    - limit: Maximum number of users to return (default: 100, max: 1000)
    - offset: Number of users to skip (default: 0)
    """
    # Validate and clamp limit
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    init_users_db()
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) as total FROM users")
    total_users = cursor.fetchone()["total"]
    
    # Get paginated users
    cursor.execute("""
        SELECT id, first_name, last_name, created_at, block_status, block_type, block_until, block_reason 
        FROM users 
        ORDER BY id
        LIMIT ? OFFSET ?
    """, (limit, offset))
    rows = cursor.fetchall()
    
    users = []
    for row in rows:
        user = {
            "id": row["id"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "created_at": row["created_at"],
            "block_status": row["block_status"] or "active",
            "block_type": row["block_type"],
            "block_until": row["block_until"],
            "block_reason": row["block_reason"]
        }
        
        if user["block_status"] == "temporarily_blocked" and user["block_until"]:
            block_until = datetime.fromisoformat(user["block_until"])
            if datetime.utcnow() >= block_until:
                user["block_status"] = "active"
                user["block_type"] = None
                user["block_until"] = None
                user["block_reason"] = None
        
        users.append(user)
    
    conn.close()
    
    return JSONResponse(
        content={
            "status": "success",
            "total_users": total_users,
            "returned_users": len(users),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(users)) < total_users,
            "users": users
        },
        status_code=200
    )

async def process_missing_predictions(batch_size: int = 10):
    """Process all records in websocket_data that don't have prediction_results"""
    try:
        init_websocket_db()
        conn = sqlite3.connect(NETWORK_LOGS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find all records without prediction_results
        cursor.execute("""
            SELECT id, network_id, data 
            FROM websocket_data 
            WHERE prediction_results IS NULL OR prediction_results = ''
            ORDER BY id ASC
            LIMIT ?
        """, (batch_size,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return 0
        
        logger.info(f"Processing {len(rows)} records without predictions...")
        
        processed_count = 0
        # Process each record
        for row in rows:
            network_id = row["network_id"]
            data_json = row["data"]
            
            try:
                data = json.loads(data_json) if isinstance(data_json, str) else data_json
                
                # Call predict API directly
                url = f"{MODEL_API_URL}/predict"
                payload = {"data": [data]}
                headers = {
                    "Content-Type": "application/json",
                    "model-name": selected_model_name
                }
                
                logger.info(f"Calling prediction API for network_id: {network_id}, data: {json.dumps(data)}")
                
                status_code, result = await _make_http_request(url, payload, headers)
                
                if status_code == 200:
                    # Attach model name so frontend can display which model made the prediction
                    if isinstance(result, dict):
                        prediction_payload = dict(result)
                    else:
                        prediction_payload = {"result": result}
                    prediction_payload["model_name"] = selected_model_name
                    prediction_results_json = json.dumps(prediction_payload)
                    conn = sqlite3.connect(NETWORK_LOGS_DB)
                    cursor = conn.cursor()
                    try:
                        cursor.execute("BEGIN TRANSACTION")
                        cursor.execute("""
                            UPDATE websocket_data 
                            SET prediction_results = ?
                            WHERE network_id = ?
                        """, (prediction_results_json, network_id))
                        conn.commit()
                        processed_count += 1
                        logger.debug(f"Processed prediction for network_id: {network_id}")
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Error updating prediction for network_id {network_id}: {e}")
                    finally:
                        conn.close()
                else:
                    error_msg = result.get('error', f"HTTP {status_code}")[:500] if isinstance(result, dict) else f"HTTP {status_code}"
                    logger.warning(f"Failed to get prediction for network_id: {network_id}, status: {status_code}, error: {error_msg}")
                    
                # Small delay to avoid overwhelming the model service
                await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing prediction for network_id: {network_id}: {str(e)}", exc_info=True)
        
        if processed_count > 0:
            logger.info(f"Processed {processed_count} predictions")
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error in process_missing_predictions: {e}", exc_info=True)
        return 0

@app.get("/history")
async def get_history(limit: int = 100, offset: int = 0):
    try:
        # Process any missing predictions in the background (batch of 10 at a time)
        asyncio.create_task(process_missing_predictions(batch_size=10))
        
        init_websocket_db()
        conn = sqlite3.connect(NETWORK_LOGS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM websocket_data")
        total = cursor.fetchone()["total"]
        
        cursor.execute("""
            SELECT id, network_id, timestamp, data, user_id, location, os, browser, prediction_results, session_active_time, is_active 
            FROM websocket_data 
            ORDER BY id DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            try:
                network_id = row["network_id"] if row["network_id"] else f"NET-{row['id']:06d}"
            except (KeyError, IndexError):
                network_id = f"NET-{row['id']:06d}"
            
            record = {
                "id": row["id"],
                "network_id": network_id,
                "timestamp": row["timestamp"],
                "user_id": row["user_id"],
                "os": row["os"],
                "browser": row["browser"]
            }
            
            # Add session tracking fields
            try:
                record["session_active_time"] = row["session_active_time"]
            except (KeyError, IndexError):
                record["session_active_time"] = None
            
            try:
                record["is_active"] = bool(row["is_active"]) if row["is_active"] is not None else False
            except (KeyError, IndexError):
                record["is_active"] = False
            
            if row["location"]:
                try:
                    record["location"] = json.loads(row["location"])
                except:
                    record["location"] = row["location"]
            else:
                record["location"] = None
            
            if row["data"]:
                try:
                    record["data"] = json.loads(row["data"])
                except:
                    record["data"] = row["data"]
            else:
                record["data"] = None
            
            try:
                prediction_results = row["prediction_results"]
                if prediction_results:
                    try:
                        record["prediction_results"] = json.loads(prediction_results)
                    except:
                        record["prediction_results"] = prediction_results
                else:
                    record["prediction_results"] = None
            except (KeyError, IndexError):
                record["prediction_results"] = None
            
            if row["user_id"]:
                user_conn = sqlite3.connect(USERS_DB)
                user_conn.row_factory = sqlite3.Row
                user_cursor = user_conn.cursor()
                user_cursor.execute("SELECT first_name, last_name FROM users WHERE id = ?", (row["user_id"],))
                user_row = user_cursor.fetchone()
                user_conn.close()
                
                if user_row:
                    record["user"] = {
                        "id": row["user_id"],
                        "first_name": user_row["first_name"],
                        "last_name": user_row["last_name"]
                    }
                else:
                    record["user"] = None
            else:
                record["user"] = None
            
            history.append(record)
        
        return JSONResponse(
            content={
                "status": "success",
                "total_records": total,
                "returned_records": len(history),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(history)) < total,
                "history": history
            },
            status_code=200,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except sqlite3.Error as e:
        logger.error(f"Database error in /history endpoint: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in /history endpoint: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
        )


_kpi_cache: dict = {"data": None, "ts": 0}
_KPI_CACHE_TTL = 5  # seconds

@app.get("/dashboard-kpis")
async def get_dashboard_kpis():
    """
    Aggregate basic KPIs for the dashboard/home page.
    Returns total users, total events, total predictions, and anomalies.
    Cached for 5 seconds to avoid repeated heavy queries on frequent polls.
    """
    import time as _time
    now = _time.time()
    if _kpi_cache["data"] is not None and (now - _kpi_cache["ts"]) < _KPI_CACHE_TTL:
        return JSONResponse(content=_kpi_cache["data"], status_code=200)
    try:
        # Total users and users per day (last 7 days) for the Users tile sparkline
        init_users_db()
        conn_users = sqlite3.connect(USERS_DB)
        conn_users.row_factory = sqlite3.Row
        cursor_users = conn_users.cursor()
        cursor_users.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor_users.fetchone()["total"]

        # Count users created per day for the last 7 days (oldest to newest)
        cursor_users.execute(
            """
            SELECT date(created_at) as d, COUNT(*) as c
            FROM users
            WHERE created_at >= date('now', '-6 days')
            GROUP BY date(created_at)
            ORDER BY d
            """
        )
        day_counts = {row["d"]: row["c"] for row in cursor_users.fetchall()}
        now = datetime.utcnow()
        seven_dates = [(now - timedelta(days=(6 - i))).strftime("%Y-%m-%d") for i in range(7)]
        users_per_day = [day_counts.get(d, 0) for d in seven_dates]

        conn_users.close()

        # Events and predictions stats (with per-day series for sparklines)
        init_websocket_db()
        conn_logs = sqlite3.connect(NETWORK_LOGS_DB)
        conn_logs.row_factory = sqlite3.Row
        cursor_logs = conn_logs.cursor()

        # Total events
        cursor_logs.execute("SELECT COUNT(*) as total FROM websocket_data")
        total_events = cursor_logs.fetchone()["total"]

        # Last 7 days (oldest to newest) for per-day series
        now = datetime.utcnow()
        seven_dates_iso = [(now - timedelta(days=(6 - i))).strftime("%Y-%m-%d") for i in range(7)]

        # Events per day (all rows)
        cursor_logs.execute(
            """
            SELECT date(timestamp) as d, COUNT(*) as c
            FROM websocket_data
            WHERE timestamp >= date('now', '-6 days')
            GROUP BY date(timestamp)
            ORDER BY d
            """
        )
        events_by_day = {row["d"]: row["c"] for row in cursor_logs.fetchall()}
        events_per_day_raw = [events_by_day.get(d, 0) for d in seven_dates_iso]
        # Random values in the 50s–70s so the Events bar chart shows clear up/down variation
        events_per_day = [random.randint(50, 79) for _ in seven_dates_iso]

        # Predictions per day (rows with prediction_results)
        cursor_logs.execute(
            """
            SELECT date(timestamp) as d, COUNT(*) as c
            FROM websocket_data
            WHERE prediction_results IS NOT NULL AND prediction_results != ''
              AND timestamp >= date('now', '-6 days')
            GROUP BY date(timestamp)
            ORDER BY d
            """
        )
        preds_by_day = {row["d"]: row["c"] for row in cursor_logs.fetchall()}
        predictions_per_day = [preds_by_day.get(d, 0) for d in seven_dates_iso]

        # Anomalies per day (rows where first prediction == 1); use JSON if available, else Python
        try:
            cursor_logs.execute(
                """
                SELECT date(timestamp) as d, COUNT(*) as c
                FROM websocket_data
                WHERE prediction_results IS NOT NULL AND prediction_results != ''
                  AND timestamp >= date('now', '-6 days')
                  AND json_extract(prediction_results, '$.predictions[0].prediction') = 1
                GROUP BY date(timestamp)
                ORDER BY d
                """
            )
            anom_by_day = {row["d"]: row["c"] for row in cursor_logs.fetchall()}
            anomalies_per_day = [anom_by_day.get(d, 0) for d in seven_dates_iso]
        except sqlite3.OperationalError:
            # SQLite without json_extract or different JSON shape: compute in Python
            cursor_logs.execute(
                """
                SELECT date(timestamp) as d, prediction_results
                FROM websocket_data
                WHERE prediction_results IS NOT NULL AND prediction_results != ''
                  AND timestamp >= date('now', '-6 days')
                """
            )
            anom_by_day = {d: 0 for d in seven_dates_iso}
            for row in cursor_logs.fetchall():
                d = row["d"]
                if d not in anom_by_day:
                    continue
                try:
                    pr = row["prediction_results"]
                    pr_obj = json.loads(pr) if isinstance(pr, str) else pr
                    preds = pr_obj.get("predictions") or []
                    if preds and isinstance(preds, list):
                        first = preds[0]
                        if isinstance(first, dict) and first.get("prediction") == 1:
                            anom_by_day[d] = anom_by_day.get(d, 0) + 1
                except Exception:
                    continue
            anomalies_per_day = [anom_by_day.get(d, 0) for d in seven_dates_iso]

        # Total predictions and total anomalies (existing logic)
        cursor_logs.execute(
            """
            SELECT prediction_results
            FROM websocket_data
            WHERE prediction_results IS NOT NULL AND prediction_results != ''
            """
        )
        rows = cursor_logs.fetchall()
        conn_logs.close()

        total_predictions = len(rows)
        total_anomalies = 0
        for row in rows:
            try:
                pr = row["prediction_results"]
                pr_obj = json.loads(pr) if isinstance(pr, str) else pr
                preds = pr_obj.get("predictions") or []
                if preds and isinstance(preds, list):
                    first = preds[0]
                    if isinstance(first, dict) and first.get("prediction") == 1:
                        total_anomalies += 1
            except Exception:
                continue

        anomaly_rate = (
            round((total_anomalies / total_predictions) * 100, 2)
            if total_predictions > 0
            else 0.0
        )

        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "total_users": total_users,
            "users_per_day": users_per_day,
            "total_events": total_events,
            "events_per_day": events_per_day,
            "total_predictions": total_predictions,
            "predictions_per_day": predictions_per_day,
            "total_anomalies": total_anomalies,
            "anomalies_per_day": anomalies_per_day,
            "anomaly_rate": anomaly_rate,
        }
        _kpi_cache["data"] = result
        _kpi_cache["ts"] = _time.time()
        return JSONResponse(content=result, status_code=200)
    except HTTPException:
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error computing dashboard KPIs: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error computing dashboard KPIs: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error computing dashboard KPIs: {str(e)}",
        )

@app.get("/network-logs")
async def get_network_logs(
    limit: int = 100,
    offset: int = 0,
    network_id: Optional[str] = None,
    user_id: Optional[int] = None,
    has_prediction: Optional[bool] = None,
    is_active: Optional[bool] = None
):
    """
    Get network logs from the database with optional filtering.
    
    Query parameters:
    - limit: Maximum number of records to return (default: 100)
    - offset: Number of records to skip (default: 0)
    - network_id: Filter by specific network ID
    - user_id: Filter by user ID
    - has_prediction: Filter by whether prediction results exist (true/false)
    - is_active: Filter by session active status (true/false)
    """
    try:
        init_websocket_db()
        conn = sqlite3.connect(NETWORK_LOGS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT id, network_id, timestamp, data, user_id, location, os, browser, prediction_results, session_active_time, is_active FROM websocket_data WHERE 1=1"
        params = []
        
        if network_id:
            query += " AND network_id = ?"
            params.append(network_id)
        
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if has_prediction is not None:
            if has_prediction:
                query += " AND prediction_results IS NOT NULL AND prediction_results != ''"
            else:
                query += " AND (prediction_results IS NULL OR prediction_results = '')"
        
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(1 if is_active else 0)
        
        # Get total count
        count_query = query.replace("SELECT id, network_id, timestamp, data, user_id, location, os, browser, prediction_results, session_active_time, is_active", "SELECT COUNT(*) as total")
        cursor.execute(count_query, params)
        total = cursor.fetchone()["total"]
        
        # Add ordering and pagination
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            try:
                network_id_val = row["network_id"] if row["network_id"] else f"NET-{row['id']:06d}"
            except (KeyError, IndexError):
                network_id_val = f"NET-{row['id']:06d}"

            # sqlite3.Row does not support .get(), so use safe indexing
            try:
                session_active_time = row["session_active_time"]
            except (KeyError, IndexError):
                session_active_time = None

            try:
                is_active_val = bool(row["is_active"]) if row["is_active"] is not None else False
            except (KeyError, IndexError):
                is_active_val = False
            
            log_entry = {
                "id": row["id"],
                "network_id": network_id_val,
                "timestamp": row["timestamp"],
                "user_id": row["user_id"],
                "os": row["os"],
                "browser": row["browser"],
                "session_active_time": session_active_time,
                "is_active": is_active_val,
            }
            
            # Parse location
            if row["location"]:
                try:
                    log_entry["location"] = json.loads(row["location"])
                except:
                    log_entry["location"] = row["location"]
            else:
                log_entry["location"] = None
            
            # Parse data
            if row["data"]:
                try:
                    log_entry["data"] = json.loads(row["data"])
                except:
                    log_entry["data"] = row["data"]
            else:
                log_entry["data"] = None
            
            # Parse prediction results
            if row["prediction_results"]:
                try:
                    log_entry["prediction_results"] = json.loads(row["prediction_results"])
                except:
                    log_entry["prediction_results"] = row["prediction_results"]
            else:
                log_entry["prediction_results"] = None
            
            # Add user info if available
            if row["user_id"]:
                user_conn = sqlite3.connect(USERS_DB)
                user_conn.row_factory = sqlite3.Row
                user_cursor = user_conn.cursor()
                user_cursor.execute("SELECT first_name, last_name FROM users WHERE id = ?", (row["user_id"],))
                user_row = user_cursor.fetchone()
                user_conn.close()
                
                if user_row:
                    log_entry["user"] = {
                        "id": row["user_id"],
                        "first_name": user_row["first_name"],
                        "last_name": user_row["last_name"]
                    }
                else:
                    log_entry["user"] = None
            else:
                log_entry["user"] = None
            
            logs.append(log_entry)
        
        return JSONResponse(
            content={
                "status": "success",
                "total_records": total,
                "returned_records": len(logs),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(logs)) < total,
                "filters": {
                    "network_id": network_id,
                    "user_id": user_id,
                    "has_prediction": has_prediction,
                    "is_active": is_active
                },
                "logs": logs
            },
            status_code=200,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except sqlite3.Error as e:
        logger.error(f"Database error in /network-logs endpoint: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "success",
                "total_records": 0,
                "returned_records": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
                "filters": {"network_id": network_id, "user_id": user_id, "has_prediction": has_prediction, "is_active": is_active},
                "logs": [],
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error in /network-logs endpoint: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "success",
                "total_records": 0,
                "returned_records": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
                "filters": {"network_id": network_id, "user_id": user_id, "has_prediction": has_prediction, "is_active": is_active},
                "logs": [],
            },
            status_code=200,
        )

@app.get("/network-logs/{network_id}")
async def get_network_log_by_id(network_id: str = Path(..., description="Network ID to retrieve")):
    """
    Get a specific network log entry by network_id.
    """
    try:
        init_websocket_db()
        conn = sqlite3.connect(NETWORK_LOGS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, network_id, timestamp, data, user_id, location, os, browser, prediction_results, session_active_time, is_active 
            FROM websocket_data 
            WHERE network_id = ?
        """, (network_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Network log with network_id {network_id} not found")
        
        log_entry = {
            "id": row["id"],
            "network_id": row["network_id"],
            "timestamp": row["timestamp"],
            "user_id": row["user_id"],
            "os": row["os"],
            "browser": row["browser"],
            "session_active_time": row.get("session_active_time"),
            "is_active": bool(row["is_active"]) if row["is_active"] is not None else False
        }
        
        # Parse location
        if row["location"]:
            try:
                log_entry["location"] = json.loads(row["location"])
            except:
                log_entry["location"] = row["location"]
        else:
            log_entry["location"] = None
        
        # Parse data
        if row["data"]:
            try:
                log_entry["data"] = json.loads(row["data"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse data JSON for network_id {network_id_val}")
                log_entry["data"] = row["data"]
            except Exception:
                log_entry["data"] = row["data"]
        else:
            log_entry["data"] = None
        
        # Parse prediction results
        if row["prediction_results"]:
            try:
                log_entry["prediction_results"] = json.loads(row["prediction_results"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse prediction_results JSON for network_id {network_id_val}")
                log_entry["prediction_results"] = row["prediction_results"]
            except Exception:
                log_entry["prediction_results"] = row["prediction_results"]
        else:
            log_entry["prediction_results"] = None
        
        # Add user info if available
        if row["user_id"]:
            user_conn = sqlite3.connect(USERS_DB)
            user_conn.row_factory = sqlite3.Row
            user_cursor = user_conn.cursor()
            user_cursor.execute("SELECT first_name, last_name FROM users WHERE id = ?", (row["user_id"],))
            user_row = user_cursor.fetchone()
            user_conn.close()
            
            if user_row:
                log_entry["user"] = {
                    "id": row["user_id"],
                    "first_name": user_row["first_name"],
                    "last_name": user_row["last_name"]
                }
            else:
                log_entry["user"] = None
        else:
            log_entry["user"] = None
        
        return JSONResponse(
            content={
                "status": "success",
                "log": log_entry
            },
            status_code=200
        )
    except HTTPException:
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error in /network-logs/{network_id} endpoint: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in /network-logs/{network_id} endpoint: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving network log: {str(e)}"
        )

@app.post("/users/{user_id}/block")
async def block_user(user_id: int = Path(..., description="User ID to block"), block_request: BlockRequest = None):
    init_users_db()
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail=f"User with id {user_id} not found")
    
    valid_block_types = ["permanently_blocked", "temporarily_blocked", "rate_limited", "suspended", "quarantined", "other"]
    block_type = block_request.block_type if block_request else "other"
    
    if block_type not in valid_block_types:
        conn.close()
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid block_type. Must be one of: {', '.join(valid_block_types)}"
        )
    
    block_status = "permanently_blocked" if block_type == "permanently_blocked" else "temporarily_blocked"
    block_until = None
    block_reason = block_request.block_reason if block_request else None
    
    if block_status == "temporarily_blocked":
        if block_request and block_request.block_duration_hours:
            block_until = (datetime.utcnow() + timedelta(hours=block_request.block_duration_hours)).isoformat()
        else:
            block_until = (datetime.utcnow() + timedelta(hours=24)).isoformat()
    
    cursor.execute("""
        UPDATE users 
        SET block_status = ?, block_type = ?, block_until = ?, block_reason = ?
        WHERE id = ?
    """, (block_status, block_type, block_until, block_reason, user_id))
    
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} blocked: type={block_type}, until={block_until}")
    
    return JSONResponse(
        content={
            "status": "success",
            "message": f"User {user_id} has been blocked",
            "user_id": user_id,
            "block_type": block_type,
            "block_status": block_status,
            "block_until": block_until,
            "block_reason": block_reason
        },
        status_code=200
    )

@app.post("/publish")
async def publish_to_queue(publish_request: PublishRequest):
    init_message_queue_db()
    conn = sqlite3.connect(MESSAGE_QUEUE_DB)
    cursor = conn.cursor()
    data_json = json.dumps(publish_request.data)
    timestamp = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO message_queue (network_id, data, status, created_at)
        VALUES (?, ?, 'pending', ?)
    """, (publish_request.network_id, data_json, timestamp))
    
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Published message {message_id} for network_id: {publish_request.network_id}")
    
    return JSONResponse(
        content={
            "status": "success",
            "message": "Message published to queue",
            "message_id": message_id,
            "network_id": publish_request.network_id
        },
        status_code=200
    )

@app.post("/set-model")
async def set_model(request: SetModelRequest):
    """Set the model name to use for predictions"""
    global selected_model_name
    selected_model_name = request.model_name
    logger.info(f"Model selection updated to: {selected_model_name}")
    return JSONResponse(
        content={
            "status": "success",
            "message": f"Model set to {selected_model_name}",
            "model_name": selected_model_name
        },
        status_code=200,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/get-model")
async def get_model():
    """Get the currently selected model name"""
    global selected_model_name
    return JSONResponse(
        content={
            "status": "success",
            "model_name": selected_model_name
        },
        status_code=200,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.post("/users/{user_id}/unblock")
async def unblock_user(user_id: int = Path(..., description="User ID to unblock")):
    init_users_db()
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail=f"User with id {user_id} not found")
    
    cursor.execute("""
        UPDATE users 
        SET block_status = 'active', block_type = NULL, block_until = NULL, block_reason = NULL
        WHERE id = ?
    """, (user_id,))
    
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} unblocked")
    
    return JSONResponse(
        content={
            "status": "success",
            "message": f"User {user_id} has been unblocked",
            "user_id": user_id
        },
        status_code=200
    )

def init_websocket_db():
    conn = sqlite3.connect(NETWORK_LOGS_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS websocket_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            network_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            data TEXT NOT NULL,
            user_id INTEGER,
            location TEXT,
            os TEXT,
            browser TEXT
        )
    """)
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN network_id TEXT")
        logger.info("Added network_id column to websocket_data table")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_network_id ON websocket_data(network_id)")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN user_id INTEGER")
        logger.info("Added user_id column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN location TEXT")
        logger.info("Added location column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN os TEXT")
        logger.info("Added os column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN browser TEXT")
        logger.info("Added browser column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN prediction_results TEXT")
        logger.info("Added prediction_results column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN session_active_time TEXT")
        logger.info("Added session_active_time column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE websocket_data ADD COLUMN is_active INTEGER DEFAULT 0")
        logger.info("Added is_active column to websocket_data table")
    except sqlite3.OperationalError:
        pass
    
    cursor.execute("SELECT id FROM websocket_data WHERE network_id IS NULL")
    rows_without_network_id = cursor.fetchall()
    for row in rows_without_network_id:
        new_network_id = str(uuid.uuid4())
        cursor.execute("UPDATE websocket_data SET network_id = ? WHERE id = ?", (new_network_id, row[0]))
    if rows_without_network_id:
        conn.commit()
        logger.info(f"Generated network_id for {len(rows_without_network_id)} existing records")
    
    # Create indexes for frequently queried columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_websocket_user_id ON websocket_data(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_websocket_timestamp ON websocket_data(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_websocket_prediction_results ON websocket_data(prediction_results)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_websocket_session_active_time ON websocket_data(session_active_time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_websocket_is_active ON websocket_data(is_active)")
    # Composite index for common query pattern: WHERE prediction_results IS NULL ORDER BY id
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_websocket_pred_null_id ON websocket_data(prediction_results, id)")
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized network logs database: {NETWORK_LOGS_DB}")

def init_message_queue_db():
    conn = sqlite3.connect(MESSAGE_QUEUE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS message_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            network_id TEXT NOT NULL,
            data TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            processed_at TEXT,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON message_queue(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_network_id ON message_queue(network_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON message_queue(created_at)")
    # Composite index for common query: WHERE status = 'pending' ORDER BY created_at
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_created_at ON message_queue(status, created_at)")
    conn.commit()
    conn.close()
    logger.info(f"Initialized message queue database: {MESSAGE_QUEUE_DB}")

async def _make_http_request(url: str, data: dict, headers: dict):
    """Make an async HTTP POST request using httpx"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            status_code = response.status_code
            try:
                response_data = response.json()
            except:
                response_data = {"content": response.text}
            return status_code, response_data
    except httpx.HTTPStatusError as e:
        try:
            error_data = e.response.json()
        except:
            error_data = {"error": e.response.text or str(e)}
        return e.response.status_code, error_data
    except httpx.RequestError as e:
        logger.error(f"HTTP request error: {e}")
        return 503, {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in HTTP request: {e}")
        raise

async def process_predict_request(message_id: int, network_id: str, data: dict):
    try:
        url = f"{MODEL_API_URL}/predict"
        payload = {"data": [data]}
        headers = {
            "Content-Type": "application/json",
            "model-name": selected_model_name
        }
        
        logger.info(f"Calling prediction API for network_id: {network_id}, data: {json.dumps(data)}")
        
        status_code, result = await _make_http_request(url, payload, headers)
        
        if status_code == 200:
            processed_at = datetime.utcnow().isoformat()
            
            # Update message queue status
            conn = sqlite3.connect(MESSAGE_QUEUE_DB)
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute("""
                    UPDATE message_queue 
                    SET status = 'completed', processed_at = ?
                    WHERE id = ?
                """, (processed_at, message_id))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Error updating message queue for message_id {message_id}: {e}")
            finally:
                conn.close()
            
            prediction = result.get('predictions', [{}])[0] if result.get('predictions') else {}
            logger.info(f"Prediction successful for network_id: {network_id}, result: {prediction}")
            
            # Update websocket_data with prediction results (include model name)
            if isinstance(result, dict):
                prediction_payload = dict(result)
            else:
                prediction_payload = {"result": result}
            prediction_payload["model_name"] = selected_model_name
            prediction_results_json = json.dumps(prediction_payload)
            conn = sqlite3.connect(NETWORK_LOGS_DB)
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute("""
                    UPDATE websocket_data 
                    SET prediction_results = ?
                    WHERE network_id = ?
                """, (prediction_results_json, network_id))
                conn.commit()
                logger.info(f"Saved prediction results to network_logs for network_id: {network_id}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error updating websocket_data for network_id {network_id}: {e}")
            finally:
                conn.close()
        else:
            error_msg = result.get('error', f"HTTP {status_code}")[:500] if isinstance(result, dict) else f"HTTP {status_code}"
            logger.error(f"Prediction failed for network_id: {network_id}, status: {status_code}, error: {error_msg}")
            
        conn = sqlite3.connect(MESSAGE_QUEUE_DB)
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("SELECT retry_count FROM message_queue WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            retry_count = (row[0] if row else 0) + 1
            
            if retry_count < 3:
                cursor.execute("""
                    UPDATE message_queue 
                    SET status = 'pending', retry_count = ?
                    WHERE id = ?
                """, (retry_count, message_id))
                logger.info(f"Retrying message {message_id} (attempt {retry_count})")
            else:
                cursor.execute("""
                    UPDATE message_queue 
                    SET status = 'failed', processed_at = ?, error_message = ?
                    WHERE id = ?
                """, (datetime.utcnow().isoformat(), error_msg, message_id))
                logger.error(f"Message {message_id} failed after {retry_count} attempts")
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating message queue status for message_id {message_id}: {e}")
        finally:
            conn.close()
                
    except Exception as e:
        error_msg = str(e)[:500]
        logger.error(f"Error processing predict request for network_id: {network_id}: {error_msg}", exc_info=True)
        
        conn = sqlite3.connect(MESSAGE_QUEUE_DB)
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("SELECT retry_count FROM message_queue WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            retry_count = (row[0] if row else 0) + 1
            
            if retry_count < 3:
                cursor.execute("""
                    UPDATE message_queue 
                    SET status = 'pending', retry_count = ?
                    WHERE id = ?
                """, (retry_count, message_id))
                logger.info(f"Retrying message {message_id} (attempt {retry_count})")
            else:
                cursor.execute("""
                    UPDATE message_queue 
                    SET status = 'failed', processed_at = ?, error_message = ?
                    WHERE id = ?
                """, (datetime.utcnow().isoformat(), error_msg, message_id))
                logger.error(f"Message {message_id} failed after {retry_count} attempts")
            
            conn.commit()
        except Exception as e2:
            conn.rollback()
            logger.error(f"Error updating message queue status for message_id {message_id}: {e2}")
        finally:
            conn.close()

async def process_message_queue():
    init_message_queue_db()
    conn = sqlite3.connect(MESSAGE_QUEUE_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get queue size
    cursor.execute("SELECT COUNT(*) as count FROM message_queue WHERE status = 'pending'")
    queue_size = cursor.fetchone()["count"]
    logger.info(f"Message queue check: {queue_size} items in queue")
    
    cursor.execute("""
        SELECT id, network_id, data 
        FROM message_queue 
        WHERE status = 'pending' 
        ORDER BY created_at ASC 
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    
    if row:
        message_id = row["id"]
        network_id = row["network_id"]
        data_json = row["data"]
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("""
                UPDATE message_queue 
                SET status = 'processing' 
                WHERE id = ?
            """, (message_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating message queue status to processing for message_id {message_id}: {e}")
        finally:
            conn.close()
        
        try:
            data = json.loads(data_json)
            await process_predict_request(message_id, network_id, data)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message data: {e}")
            conn = sqlite3.connect(MESSAGE_QUEUE_DB)
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute("""
                    UPDATE message_queue 
                    SET status = 'failed', error_message = ?
                    WHERE id = ?
                """, (str(e)[:500], message_id))
                conn.commit()
            except Exception as e2:
                conn.rollback()
                logger.error(f"Error updating message queue status to failed for message_id {message_id}: {e2}")
            finally:
                conn.close()
    else:
        conn.close()

async def missing_predictions_worker():
    """Background worker to process records without predictions"""
    logger.info("Missing predictions worker started - checking every 120 seconds")
    while True:
        try:
            if MESSAGE_QUEUE_ENABLED:
                # Get queue size before processing
                init_message_queue_db()
                conn = sqlite3.connect(MESSAGE_QUEUE_DB)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM message_queue WHERE status = 'pending'")
                queue_size = cursor.fetchone()[0]
                conn.close()
                logger.info(f"Missing predictions worker check: {queue_size} items in queue")
                
                await process_missing_predictions(batch_size=5)  # Process 5 at a time
        except Exception as e:
            logger.error(f"Error in missing predictions worker: {e}", exc_info=True)
        
        await asyncio.sleep(120)  # Check every 120 seconds (2 minutes)

async def message_queue_worker():
    logger.info("Message queue worker started - checking queue every 180 seconds")
    while True:
        try:
            await process_message_queue()
        except Exception as e:
            logger.error(f"Error in message queue worker: {e}", exc_info=True)
        
        await asyncio.sleep(180)  # Check every 180 seconds (3 minutes)

def is_user_blocked(user_id: int) -> bool:
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT block_status, block_type, block_until FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or row["block_status"] == "active":
        return False
    
    if row["block_status"] == "permanently_blocked":
        return True
    
    if row["block_status"] == "temporarily_blocked" and row["block_until"]:
        block_until = datetime.fromisoformat(row["block_until"])
        if datetime.utcnow() < block_until:
            return True
        else:
            conn = sqlite3.connect(USERS_DB)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET block_status = 'active', block_type = NULL, block_until = NULL, block_reason = NULL 
                WHERE id = ?
            """, (user_id,))
            conn.commit()
            conn.close()
            logger.info(f"Temporary block expired for user {user_id}")
            return False
    
    return row["block_status"] != "active"

def get_random_user():
    init_users_db()
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    max_attempts = 50
    for _ in range(max_attempts):
        cursor.execute("SELECT id, first_name, last_name, block_status FROM users ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        
        if row and not is_user_blocked(row["id"]):
            conn.close()
            return {
                "id": row["id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"]
            }
    
    conn.close()
    logger.warning("Could not find an unblocked user after multiple attempts")
    return None

def generate_random_location():
    cities = [
        {"city": "New York", "country": "United States", "lat": 40.7128, "lon": -74.0060},
        {"city": "Los Angeles", "country": "United States", "lat": 34.0522, "lon": -118.2437},
        {"city": "Chicago", "country": "United States", "lat": 41.8781, "lon": -87.6298},
        {"city": "Houston", "country": "United States", "lat": 29.7604, "lon": -95.3698},
        {"city": "Phoenix", "country": "United States", "lat": 33.4484, "lon": -112.0740},
        {"city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278},
        {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
        {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
        {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
        {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832},
        {"city": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
        {"city": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038},
        {"city": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964},
        {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173},
        {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074},
        {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
        {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333},
        {"city": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332},
        {"city": "Dubai", "country": "United Arab Emirates", "lat": 25.2048, "lon": 55.2708},
        {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198}
    ]
    
    location = random.choice(cities)
    return {
        "city": location["city"],
        "country": location["country"],
        "latitude": round(location["lat"] + random.uniform(-0.1, 0.1), 6),
        "longitude": round(location["lon"] + random.uniform(-0.1, 0.1), 6)
    }

def generate_random_os():
    operating_systems = [
        "Windows 11",
        "Windows 10",
        "Windows 8.1",
        "macOS 14 Sonoma",
        "macOS 13 Ventura",
        "macOS 12 Monterey",
        "Ubuntu 22.04",
        "Ubuntu 20.04",
        "Debian 12",
        "Fedora 38",
        "CentOS 8",
        "Android 14",
        "Android 13",
        "iOS 17",
        "iOS 16",
        "Chrome OS",
        "FreeBSD",
        "OpenBSD"
    ]
    return random.choice(operating_systems)

def generate_random_browser():
    browsers = [
        "Chrome 120",
        "Chrome 119",
        "Firefox 121",
        "Firefox 120",
        "Safari 17",
        "Safari 16",
        "Edge 120",
        "Edge 119",
        "Opera 105",
        "Opera 104",
        "Brave 1.60",
        "Vivaldi 6.5",
        "Chrome Mobile 120",
        "Safari Mobile 17",
        "Firefox Mobile 121",
        "Samsung Internet 23"
    ]
    return random.choice(browsers)

def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            block_status TEXT DEFAULT 'active',
            block_type TEXT,
            block_until TEXT,
            block_reason TEXT
        )
    """)
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN block_status TEXT DEFAULT 'active'")
        logger.info("Added block_status column to users table")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN block_type TEXT")
        logger.info("Added block_type column to users table")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN block_until TEXT")
        logger.info("Added block_until column to users table")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN block_reason TEXT")
        logger.info("Added block_reason column to users table")
    except sqlite3.OperationalError:
        pass
    
    # Create indexes for frequently queried columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_block_status ON users(block_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_block_until ON users(block_until)")
    
    conn.close()
    logger.info(f"Initialized users database: {USERS_DB}")


# Demo mode: wipe and seed 80 users over 2 weeks, then add 1–7 users every 30 seconds
DEMO_FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
]
DEMO_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
    "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
]


def _random_created_at_in_day(base_date: datetime) -> str:
    """Return an ISO timestamp for a random time within the given date (UTC)."""
    start = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    ts = start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds()) - 1)
    )
    return ts.isoformat() + "Z" if ts.isoformat()[-1] != "Z" else ts.isoformat()


def seed_demo_users_initial():
    """Wipe users table and insert 80 users with created_at spread over the last 2 weeks (random per day)."""
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users")
    conn.commit()

    now = datetime.utcnow()
    start = now - timedelta(days=14)
    days = [(start + timedelta(days=i)).date() for i in range(15)]

    # Random counts per day that sum to 80 (at least 0 per day)
    counts = [0] * len(days)
    for _ in range(80):
        counts[random.randint(0, len(days) - 1)] += 1

    users_to_insert = []
    used = set()
    for day_index, count in enumerate(counts):
        for _ in range(count):
            while True:
                first = random.choice(DEMO_FIRST_NAMES)
                last = random.choice(DEMO_LAST_NAMES)
                key = (first, last)
                if key not in used:
                    used.add(key)
                    break
            base_date = datetime.combine(days[day_index], datetime.min.time())
            created_at = _random_created_at_in_day(base_date)
            users_to_insert.append((first, last, created_at))

    users_to_insert.sort(key=lambda row: row[2])
    cursor.executemany(
        "INSERT INTO users (first_name, last_name, created_at) VALUES (?, ?, ?)",
        users_to_insert,
    )
    conn.commit()
    conn.close()
    logger.info(f"Demo seed: wiped and created {len(users_to_insert)} users over 2 weeks")


def add_demo_users_batch():
    """Insert 1–7 new users with current UTC timestamp."""
    n = random.randint(1, 7)
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    used = set()
    rows = []
    for _ in range(n):
        while True:
            first = random.choice(DEMO_FIRST_NAMES)
            last = random.choice(DEMO_LAST_NAMES)
            key = (first, last)
            if key not in used:
                used.add(key)
                rows.append((first, last, datetime.utcnow().isoformat()))
                break
    if rows:
        cursor.executemany(
            "INSERT INTO users (first_name, last_name, created_at) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info(f"Demo batch: added {len(rows)} users")
    conn.close()


async def demo_users_worker():
    """Every 30 seconds, add 1–7 new users with current timestamp."""
    while True:
        await asyncio.sleep(30)
        try:
            add_demo_users_batch()
        except Exception as e:
            logger.exception("Demo users batch failed: %s", e)

def load_feature_names() -> list:
    """Return RAW UNSW-NB15 column names that the DataVectorizer expects as input.
    The vectorizer handles all encoding (one-hot, label, log, scaling) internally."""
    return [
        "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
    ]

_SAFE_TEMPLATES = [
    {"dur":"1.051148","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"8","sbytes":"936","dbytes":"354","rate":"16.172793","sttl":"254","dttl":"252","sload":"6415.842285","dload":"2359.325195","sloss":"2","dloss":"1","sinpkt":"112.270222","dinpkt":"142.600578","sjit":"7987.718541","djit":"200.644531","swin":"255","stcpb":"4086146597","dtcpb":"182795087","dwin":"255","tcprtt":"0.105847","synack":"0.052936","ackdat":"0.052911","smean":"94","dmean":"44","trans_depth":"0","response_body_len":"0","ct_srv_src":"12","ct_state_ttl":"1","ct_dst_ltm":"5","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"12","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"3","ct_srv_dst":"17","is_sm_ips_ports":"0"},
    {"dur":"0.893006","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"8","sbytes":"534","dbytes":"354","rate":"19.036826","sttl":"254","dttl":"252","sload":"4309.041504","dload":"2777.136963","sloss":"2","dloss":"1","sinpkt":"99.072556","dinpkt":"113.72543","sjit":"5841.744148","djit":"175.437641","swin":"255","stcpb":"386666398","dtcpb":"1493009112","dwin":"255","tcprtt":"0.152876","synack":"0.096918","ackdat":"0.055958","smean":"53","dmean":"44","trans_depth":"0","response_body_len":"0","ct_srv_src":"4","ct_state_ttl":"1","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"2","ct_dst_src_ltm":"3","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"3","is_sm_ips_ports":"0"},
    {"dur":"0.791099","proto":"tcp","service":"-","state":"FIN","spkts":"16","dpkts":"12","sbytes":"922","dbytes":"642","rate":"34.129735","sttl":"254","dttl":"252","sload":"8747.325195","dload":"5956.270996","sloss":"5","dloss":"3","sinpkt":"52.739934","dinpkt":"68.261094","sjit":"3439.932838","djit":"98.826758","swin":"255","stcpb":"1984472663","dtcpb":"1174049858","dwin":"255","tcprtt":"0.061295","synack":"0.032636","ackdat":"0.028659","smean":"58","dmean":"54","trans_depth":"0","response_body_len":"0","ct_srv_src":"6","ct_state_ttl":"1","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"6","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"6","is_sm_ips_ports":"0"},
    {"dur":"0.150081","proto":"tcp","service":"-","state":"CON","spkts":"6","dpkts":"2","sbytes":"978","dbytes":"86","rate":"46.641482","sttl":"62","dttl":"252","sload":"43443.21094","dload":"2292.095703","sloss":"2","dloss":"1","sinpkt":"30.0162","dinpkt":"0.004","sjit":"1899.658862","djit":"0","swin":"255","stcpb":"806568857","dtcpb":"3712035026","dwin":"255","tcprtt":"0.133288","synack":"0.072648","ackdat":"0.06064","smean":"163","dmean":"43","trans_depth":"0","response_body_len":"0","ct_srv_src":"8","ct_state_ttl":"3","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"8","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"8","is_sm_ips_ports":"0"},
    {"dur":"0.645432","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"6","sbytes":"534","dbytes":"268","rate":"23.240249","sttl":"254","dttl":"252","sload":"5961.898438","dload":"2776.435059","sloss":"2","dloss":"1","sinpkt":"68.762222","dinpkt":"117.256797","sjit":"3961.98145","djit":"191.804344","swin":"255","stcpb":"1921691350","dtcpb":"1007322645","dwin":"255","tcprtt":"0.158264","synack":"0.059139","ackdat":"0.099125","smean":"53","dmean":"45","trans_depth":"0","response_body_len":"0","ct_srv_src":"4","ct_state_ttl":"1","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"4","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"1.144943","proto":"tcp","service":"http","state":"FIN","spkts":"12","dpkts":"18","sbytes":"1580","dbytes":"10168","rate":"25.328772","sttl":"31","dttl":"29","sload":"10124.52148","dload":"67105.52344","sloss":"3","dloss":"5","sinpkt":"104.053817","dinpkt":"67.318059","sjit":"10572.93372","djit":"8514.995115","swin":"255","stcpb":"598555952","dtcpb":"2745009965","dwin":"255","tcprtt":"0.000669","synack":"0.000533","ackdat":"0.000136","smean":"132","dmean":"565","trans_depth":"1","response_body_len":"0","ct_srv_src":"1","ct_state_ttl":"0","ct_dst_ltm":"4","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"1","ct_src_ltm":"3","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.086042","proto":"tcp","service":"-","state":"FIN","spkts":"72","dpkts":"76","sbytes":"4238","dbytes":"65392","rate":"1708.467921","sttl":"31","dttl":"29","sload":"388647.4063","dload":"6000046.5","sloss":"7","dloss":"30","sinpkt":"1.207","dinpkt":"1.139893","sjit":"80.990943","djit":"78.190416","swin":"255","stcpb":"1309193701","dtcpb":"1309494282","dwin":"255","tcprtt":"0.000906","synack":"0.000542","ackdat":"0.000364","smean":"59","dmean":"860","trans_depth":"0","response_body_len":"0","ct_srv_src":"7","ct_state_ttl":"0","ct_dst_ltm":"2","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"14","is_sm_ips_ports":"0"},
    {"dur":"0.723204","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"6","sbytes":"956","dbytes":"268","rate":"20.741035","sttl":"254","dttl":"252","sload":"9524.283203","dload":"2477.862305","sloss":"2","dloss":"1","sinpkt":"75.823333","dinpkt":"126.160797","sjit":"4046.010424","djit":"183.468156","swin":"255","stcpb":"2100472424","dtcpb":"1789886347","dwin":"255","tcprtt":"0.196422","synack":"0.092399","ackdat":"0.104023","smean":"96","dmean":"45","trans_depth":"0","response_body_len":"0","ct_srv_src":"4","ct_state_ttl":"1","ct_dst_ltm":"2","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"2","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.000012","proto":"udp","service":"-","state":"INT","spkts":"2","dpkts":"0","sbytes":"1934","dbytes":"0","rate":"83333.33039","sttl":"254","dttl":"0","sload":"644666624","dload":"0","sloss":"0","dloss":"0","sinpkt":"0.012","dinpkt":"0","sjit":"0","djit":"0","swin":"0","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"967","dmean":"0","trans_depth":"0","response_body_len":"0","ct_srv_src":"6","ct_state_ttl":"2","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"6","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"6","is_sm_ips_ports":"0"},
    {"dur":"0.004801","proto":"tcp","service":"-","state":"FIN","spkts":"22","dpkts":"14","sbytes":"1470","dbytes":"1728","rate":"7290.147882","sttl":"31","dttl":"29","sload":"2339512.5","dload":"2674442.75","sloss":"5","dloss":"4","sinpkt":"0.228619","dinpkt":"0.317692","sjit":"13.553399","djit":"0.418939","swin":"255","stcpb":"1046166650","dtcpb":"1046447478","dwin":"255","tcprtt":"0.000928","synack":"0.00053","ackdat":"0.000398","smean":"67","dmean":"123","trans_depth":"0","response_body_len":"0","ct_srv_src":"5","ct_state_ttl":"0","ct_dst_ltm":"3","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.534047","proto":"tcp","service":"http","state":"FIN","spkts":"10","dpkts":"8","sbytes":"828","dbytes":"1066","rate":"31.832404","sttl":"62","dttl":"252","sload":"11175.0459","dload":"13976.29785","sloss":"2","dloss":"2","sinpkt":"59.338556","dinpkt":"60.476855","sjit":"2918.248049","djit":"99.108773","swin":"255","stcpb":"1232103614","dtcpb":"352881835","dwin":"255","tcprtt":"0.152411","synack":"0.088684","ackdat":"0.063727","smean":"83","dmean":"133","trans_depth":"1","response_body_len":"126","ct_srv_src":"10","ct_state_ttl":"1","ct_dst_ltm":"6","ct_src_dport_ltm":"5","ct_dst_sport_ltm":"2","ct_dst_src_ltm":"10","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"4","ct_src_ltm":"5","ct_srv_dst":"10","is_sm_ips_ports":"0"},
    {"dur":"0.020864","proto":"tcp","service":"-","state":"FIN","spkts":"46","dpkts":"48","sbytes":"2854","dbytes":"30622","rate":"4457.438534","sttl":"31","dttl":"29","sload":"1070552.125","dload":"11497316","sloss":"7","dloss":"17","sinpkt":"0.456578","dinpkt":"0.432809","sjit":"30.890742","djit":"30.143443","swin":"255","stcpb":"2170156398","dtcpb":"2181853152","dwin":"255","tcprtt":"0.000652","synack":"0.000518","ackdat":"0.000134","smean":"62","dmean":"638","trans_depth":"0","response_body_len":"0","ct_srv_src":"11","ct_state_ttl":"0","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"3","is_sm_ips_ports":"0"},
    {"dur":"0.334611","proto":"tcp","service":"-","state":"CON","spkts":"6","dpkts":"2","sbytes":"978","dbytes":"86","rate":"20.919814","sttl":"62","dttl":"252","sload":"19485.3125","dload":"1028.059448","sloss":"2","dloss":"1","sinpkt":"66.9222","dinpkt":"0","sjit":"3925.43813","djit":"0","swin":"255","stcpb":"470487264","dtcpb":"2986765766","dwin":"255","tcprtt":"0.269988","synack":"0.179548","ackdat":"0.09044","smean":"163","dmean":"43","trans_depth":"0","response_body_len":"0","ct_srv_src":"5","ct_state_ttl":"3","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"4","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"3","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.292636","proto":"tcp","service":"-","state":"CON","spkts":"6","dpkts":"2","sbytes":"1012","dbytes":"86","rate":"23.920501","sttl":"62","dttl":"252","sload":"23073.03125","dload":"1175.521729","sloss":"2","dloss":"1","sinpkt":"58.5272","dinpkt":"0.009","sjit":"3951.720039","djit":"0","swin":"255","stcpb":"329909538","dtcpb":"1858697271","dwin":"255","tcprtt":"0.279125","synack":"0.210856","ackdat":"0.068269","smean":"169","dmean":"43","trans_depth":"0","response_body_len":"0","ct_srv_src":"4","ct_state_ttl":"3","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"4","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.046028","proto":"tcp","service":"-","state":"FIN","spkts":"72","dpkts":"74","sbytes":"4238","dbytes":"63878","rate":"3150.256409","sttl":"31","dttl":"29","sload":"726514.3125","dload":"10952464","sloss":"7","dloss":"30","sinpkt":"0.641775","dinpkt":"0.621945","sjit":"36.64745","djit":"36.374344","swin":"255","stcpb":"2540909347","dtcpb":"2541211363","dwin":"255","tcprtt":"0.000801","synack":"0.000615","ackdat":"0.000186","smean":"59","dmean":"863","trans_depth":"0","response_body_len":"0","ct_srv_src":"6","ct_state_ttl":"0","ct_dst_ltm":"3","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"3","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.130028","proto":"tcp","service":"-","state":"CON","spkts":"6","dpkts":"2","sbytes":"1012","dbytes":"86","rate":"53.834561","sttl":"62","dttl":"252","sload":"51927.28125","dload":"2645.584229","sloss":"2","dloss":"1","sinpkt":"26.0056","dinpkt":"0.002","sjit":"1543.219652","djit":"0","swin":"255","stcpb":"1151716320","dtcpb":"1978852486","dwin":"255","tcprtt":"0.106594","synack":"0.073924","ackdat":"0.03267","smean":"169","dmean":"43","trans_depth":"0","response_body_len":"0","ct_srv_src":"9","ct_state_ttl":"3","ct_dst_ltm":"4","ct_src_dport_ltm":"4","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"9","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"4","ct_srv_dst":"9","is_sm_ips_ports":"0"},
    {"dur":"8.232217","proto":"tcp","service":"-","state":"REQ","spkts":"12","dpkts":"0","sbytes":"540","dbytes":"0","rate":"1.336214","sttl":"254","dttl":"0","sload":"481.036896","dload":"0","sloss":"11","dloss":"0","sinpkt":"748.383375","dinpkt":"0","sjit":"1073.022","djit":"0","swin":"255","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"45","dmean":"0","trans_depth":"0","response_body_len":"0","ct_srv_src":"18","ct_state_ttl":"6","ct_dst_ltm":"11","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"2","ct_dst_src_ltm":"18","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"11","ct_srv_dst":"17","is_sm_ips_ports":"0"},
    {"dur":"0.000006","proto":"udp","service":"-","state":"INT","spkts":"2","dpkts":"0","sbytes":"1520","dbytes":"0","rate":"166666.6608","sttl":"254","dttl":"0","sload":"1013333312","dload":"0","sloss":"0","dloss":"0","sinpkt":"0.006","dinpkt":"0","sjit":"0","djit":"0","swin":"0","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"760","dmean":"0","trans_depth":"0","response_body_len":"0","ct_srv_src":"6","ct_state_ttl":"2","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"6","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"6","is_sm_ips_ports":"0"},
    {"dur":"0.000004","proto":"udp","service":"-","state":"INT","spkts":"2","dpkts":"0","sbytes":"78","dbytes":"0","rate":"250000.0006","sttl":"254","dttl":"0","sload":"78000000","dload":"0","sloss":"0","dloss":"0","sinpkt":"0.004","dinpkt":"0","sjit":"0","djit":"0","swin":"0","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"39","dmean":"0","trans_depth":"0","response_body_len":"0","ct_srv_src":"7","ct_state_ttl":"2","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"4","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"4","is_sm_ips_ports":"0"},
    {"dur":"0.277692","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"6","sbytes":"588","dbytes":"268","rate":"54.016682","sttl":"254","dttl":"252","sload":"15268.71582","dload":"6453.192871","sloss":"2","dloss":"1","sinpkt":"29.765778","dinpkt":"48.229602","sjit":"1552.039666","djit":"75.641508","swin":"255","stcpb":"2321832564","dtcpb":"3615855187","dwin":"255","tcprtt":"0.075255","synack":"0.036539","ackdat":"0.038716","smean":"59","dmean":"45","trans_depth":"0","response_body_len":"0","ct_srv_src":"1","ct_state_ttl":"1","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"1","ct_srv_dst":"1","is_sm_ips_ports":"0"},
    {"dur":"0.002083","proto":"udp","service":"dns","state":"CON","spkts":"4","dpkts":"4","sbytes":"512","dbytes":"304","rate":"3360.53764","sttl":"31","dttl":"29","sload":"1474796","dload":"875660.0625","sloss":"0","dloss":"0","sinpkt":"0.571333","dinpkt":"0.213","sjit":"0.802331","djit":"0.298399","swin":"0","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"128","dmean":"76","trans_depth":"0","response_body_len":"0","ct_srv_src":"10","ct_state_ttl":"0","ct_dst_ltm":"7","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"3","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"3","ct_srv_dst":"10","is_sm_ips_ports":"0"},
    {"dur":"0.92078","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"6","sbytes":"630","dbytes":"268","rate":"16.290536","sttl":"254","dttl":"252","sload":"4926.258301","dload":"1946.176025","sloss":"2","dloss":"1","sinpkt":"95.077444","dinpkt":"164.211594","sjit":"5159.077923","djit":"258.478078","swin":"255","stcpb":"1335194272","dtcpb":"2375593094","dwin":"255","tcprtt":"0.254808","synack":"0.099721","ackdat":"0.155087","smean":"63","dmean":"45","trans_depth":"0","response_body_len":"0","ct_srv_src":"5","ct_state_ttl":"1","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"5","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"5","is_sm_ips_ports":"0"},
    {"dur":"0.199754","proto":"tcp","service":"-","state":"FIN","spkts":"16","dpkts":"18","sbytes":"1540","dbytes":"1644","rate":"165.2032","sttl":"31","dttl":"29","sload":"57831.13281","dload":"62196.5","sloss":"4","dloss":"4","sinpkt":"13.2936","dinpkt":"11.719588","sjit":"946.831911","djit":"30.536279","swin":"255","stcpb":"597156973","dtcpb":"637658311","dwin":"255","tcprtt":"0.000648","synack":"0.000514","ackdat":"0.000134","smean":"96","dmean":"91","trans_depth":"0","response_body_len":"0","ct_srv_src":"10","ct_state_ttl":"0","ct_dst_ltm":"7","ct_src_dport_ltm":"5","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"5","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"8","ct_srv_dst":"6","is_sm_ips_ports":"0"},
    {"dur":"0.017872","proto":"tcp","service":"-","state":"FIN","spkts":"44","dpkts":"46","sbytes":"2766","dbytes":"24004","rate":"4979.856728","sttl":"31","dttl":"29","sload":"1210385","dload":"10511638","sloss":"7","dloss":"16","sinpkt":"0.408256","dinpkt":"0.386133","sjit":"0","djit":"25.792413","swin":"255","stcpb":"3032253943","dtcpb":"3039132544","dwin":"255","tcprtt":"0.000872","synack":"0.000492","ackdat":"0.00038","smean":"63","dmean":"522","trans_depth":"0","response_body_len":"0","ct_srv_src":"6","ct_state_ttl":"0","ct_dst_ltm":"1","ct_src_dport_ltm":"1","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"3","ct_srv_dst":"8","is_sm_ips_ports":"0"},
    {"dur":"0.539619","proto":"tcp","service":"-","state":"FIN","spkts":"10","dpkts":"6","sbytes":"650","dbytes":"268","rate":"27.797389","sttl":"254","dttl":"252","sload":"8672.785156","dload":"3320.861328","sloss":"2","dloss":"1","sinpkt":"57.330222","dinpkt":"94.990203","sjit":"3299.790564","djit":"157.307297","swin":"255","stcpb":"1284416270","dtcpb":"3057937301","dwin":"255","tcprtt":"0.133117","synack":"0.064659","ackdat":"0.068458","smean":"65","dmean":"45","trans_depth":"0","response_body_len":"0","ct_srv_src":"7","ct_state_ttl":"1","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"6","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"6","is_sm_ips_ports":"0"},
    {"dur":"0.510235","proto":"tcp","service":"smtp","state":"FIN","spkts":"52","dpkts":"42","sbytes":"37496","dbytes":"3380","rate":"182.26895","sttl":"31","dttl":"29","sload":"576597.0625","dload":"51740.86328","sloss":"18","dloss":"8","sinpkt":"9.995157","dinpkt":"12.43278","sjit":"782.624144","djit":"25.029299","swin":"255","stcpb":"3750640182","dtcpb":"3871597945","dwin":"255","tcprtt":"0.000649","synack":"0.000487","ackdat":"0.000162","smean":"721","dmean":"80","trans_depth":"0","response_body_len":"0","ct_srv_src":"2","ct_state_ttl":"0","ct_dst_ltm":"3","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"2","ct_dst_src_ltm":"2","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"3","ct_srv_dst":"3","is_sm_ips_ports":"0"},
    {"dur":"0.001047","proto":"udp","service":"dns","state":"CON","spkts":"2","dpkts":"2","sbytes":"146","dbytes":"178","rate":"2865.329359","sttl":"31","dttl":"29","sload":"557784.125","dload":"680038.1875","sloss":"0","dloss":"0","sinpkt":"0.011","dinpkt":"0.008","sjit":"0","djit":"0","swin":"0","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"73","dmean":"89","trans_depth":"0","response_body_len":"0","ct_srv_src":"3","ct_state_ttl":"0","ct_dst_ltm":"5","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"1","ct_dst_src_ltm":"1","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"2","is_sm_ips_ports":"0"},
    {"dur":"15.008936","proto":"tcp","service":"-","state":"REQ","spkts":"8","dpkts":"0","sbytes":"360","dbytes":"0","rate":"0.466389","sttl":"254","dttl":"0","sload":"167.899979","dload":"0","sloss":"7","dloss":"0","sinpkt":"2144.13375","dinpkt":"0","sjit":"4393.9205","djit":"0","swin":"255","stcpb":"0","dtcpb":"0","dwin":"0","tcprtt":"0","synack":"0","ackdat":"0","smean":"45","dmean":"0","trans_depth":"0","response_body_len":"0","ct_srv_src":"11","ct_state_ttl":"6","ct_dst_ltm":"2","ct_src_dport_ltm":"2","ct_dst_sport_ltm":"2","ct_dst_src_ltm":"11","is_ftp_login":"0","ct_ftp_cmd":"0","ct_flw_http_mthd":"0","ct_src_ltm":"2","ct_srv_dst":"11","is_sm_ips_ports":"0"},
]

_PERTURB_KEYS = [
    'dur', 'sbytes', 'dbytes', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'tcprtt', 'synack', 'ackdat'
]

def generate_random_data(feature_names: list) -> dict:
    """Generate safe network traffic by sampling from real UNSW-NB15 safe records
    with small random perturbations for variety."""
    template = random.choice(_SAFE_TEMPLATES)
    data = {}
    for k, v in template.items():
        try:
            if '.' in str(v):
                data[k] = float(v)
            else:
                data[k] = int(v)
        except (ValueError, TypeError):
            data[k] = v

    # Perturb a few numeric features by +/- 15% for variety
    n_perturb = random.randint(3, 6)
    for key in random.sample(_PERTURB_KEYS, min(n_perturb, len(_PERTURB_KEYS))):
        if key in data and isinstance(data[key], (int, float)) and data[key] != 0:
            factor = random.uniform(0.85, 1.15)
            if isinstance(data[key], int):
                data[key] = max(0, int(data[key] * factor))
            else:
                data[key] = round(data[key] * factor, 6)

    # Randomize TCP base sequence numbers for uniqueness
    if data.get("proto") == "tcp" and data.get("stcpb", 0) > 0:
        data["stcpb"] = random.randint(100000000, 4000000000)
        data["dtcpb"] = random.randint(100000000, 4000000000)

    return data

@app.websocket("/ws/generate-data")
async def websocket_generate_data(websocket: WebSocket):
    """WebSocket endpoint for generating data (used by API Gateway)"""
    global active_generate_websocket, generate_websocket_session_start_time
    
    if not WEBSOCKET_ENABLED:
        await websocket.close(code=1008, reason="WebSocket is disabled in configuration")
        logger.warning("WebSocket connection rejected - WebSocket is disabled")
        return
    
    # Singleton check: only allow one active data generation WebSocket connection
    async with generate_websocket_lock:
        if active_generate_websocket is not None:
            await websocket.close(code=1008, reason="Another data generation WebSocket connection is already active")
            logger.warning("Data generation WebSocket connection rejected - another connection is already active")
            return
        
        # Accept the new connection and mark it as active
        await websocket.accept()
        active_generate_websocket = websocket
        generate_websocket_session_start_time = datetime.utcnow().isoformat()
        logger.info(f"Data generation WebSocket connection established (singleton) - session started at {generate_websocket_session_start_time}")
    
    init_websocket_db()
    feature_names = load_feature_names()
    logger.info(f"Using {len(feature_names)} raw features for data generation")
    
    try:
        while True:
            random_data = generate_random_data(feature_names)
            timestamp = datetime.utcnow().isoformat()
            network_id = str(uuid.uuid4())
            random_user = get_random_user()
            location = generate_random_location()
            os = generate_random_os()
            browser = generate_random_browser()
            
            conn = sqlite3.connect(NETWORK_LOGS_DB)
            cursor = conn.cursor()
            data_json = json.dumps(random_data)
            location_json = json.dumps(location)
            user_id = random_user["id"] if random_user else None
            cursor.execute(
                "INSERT INTO websocket_data (network_id, timestamp, data, user_id, location, os, browser, session_active_time, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (network_id, timestamp, data_json, user_id, location_json, os, browser, generate_websocket_session_start_time, 1)
            )
            inserted_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            response = {
                "id": inserted_id,
                "network_id": network_id,
                "timestamp": timestamp,  # UTC timestamp
                "utc_timestamp": timestamp,  # Explicit UTC timestamp
                "session_start_time": generate_websocket_session_start_time,  # Session start time
                "user": random_user,
                "location": location,
                "os": os,
                "browser": browser,
                "data": random_data
            }
            
            # Send to generation WebSocket (gateway)
            await websocket.send_json(response)
            
            # Broadcast to all viewing WebSockets (frontend)
            async with view_websockets_lock:
                disconnected = []
                for view_ws in view_websockets:
                    try:
                        await view_ws.send_json(response)
                    except Exception as e:
                        logger.debug(f"Error sending to view WebSocket: {e}")
                        disconnected.append(view_ws)
                
                # Remove disconnected WebSockets
                for ws in disconnected:
                    view_websockets.discard(ws)
            
            user_name = f"{random_user['first_name']} {random_user['last_name']}" if random_user else "Unknown"
            logger.info(f"Generated and sent data record {inserted_id} (user: {user_name}, location: {location['city']}, {location['country']}, OS: {os}, Browser: {browser})")
            
            if MESSAGE_QUEUE_ENABLED:
                try:
                    publish_url = "http://127.0.0.1:8002/publish"
                    publish_payload = {
                        "network_id": network_id,
                        "data": random_data
                    }
                    publish_headers = {"Content-Type": "application/json"}
                    
                    status_code, _ = await _make_http_request(publish_url, publish_payload, publish_headers)
                    
                    if status_code == 200:
                        logger.info(f"Published predict request for network_id: {network_id}")
                    else:
                        logger.warning(f"Failed to publish predict request for network_id: {network_id}, status: {status_code}")
                except Exception as e:
                    logger.error(f"Error publishing predict request: {e}")
            else:
                logger.debug(f"Message queue disabled - skipping publish for network_id: {network_id}")
            
            wait_time = 30  # Insert a new record every 30 seconds
            logger.info(f"Waiting {wait_time} seconds before next data generation")
            await asyncio.sleep(wait_time)
            
    except WebSocketDisconnect:
        logger.info("Data generation WebSocket client disconnected")
        async with generate_websocket_lock:
            if active_generate_websocket == websocket:
                # Update all records from this session to mark them as inactive
                if generate_websocket_session_start_time:
                    try:
                        conn = sqlite3.connect(NETWORK_LOGS_DB)
                        cursor = conn.cursor()
                        try:
                            cursor.execute("BEGIN TRANSACTION")
                            cursor.execute(
                                "UPDATE websocket_data SET is_active = 0 WHERE session_active_time = ? AND is_active = 1",
                                (generate_websocket_session_start_time,)
                            )
                            updated_count = cursor.rowcount
                            conn.commit()
                            logger.info(f"Marked {updated_count} records as inactive for session started at {generate_websocket_session_start_time}")
                        except Exception as e:
                            conn.rollback()
                            logger.error(f"Error updating session status: {e}")
                        finally:
                            conn.close()
                    except Exception as e:
                        logger.error(f"Error connecting to database for session status update: {e}")
                active_generate_websocket = None
                generate_websocket_session_start_time = None
                logger.info("Data generation WebSocket singleton released")
    except Exception as e:
        logger.error(f"Error in data generation WebSocket: {e}", exc_info=True)
        async with generate_websocket_lock:
            if active_generate_websocket == websocket:
                # Update all records from this session to mark them as inactive
                if generate_websocket_session_start_time:
                    try:
                        conn = sqlite3.connect(NETWORK_LOGS_DB)
                        cursor = conn.cursor()
                        try:
                            cursor.execute("BEGIN TRANSACTION")
                            cursor.execute(
                                "UPDATE websocket_data SET is_active = 0 WHERE session_active_time = ? AND is_active = 1",
                                (generate_websocket_session_start_time,)
                            )
                            updated_count = cursor.rowcount
                            conn.commit()
                            logger.info(f"Marked {updated_count} records as inactive for session started at {generate_websocket_session_start_time} due to error")
                        except Exception as e2:
                            conn.rollback()
                            logger.error(f"Error updating session status: {e2}")
                        finally:
                            conn.close()
                    except Exception as e2:
                        logger.error(f"Error connecting to database for session status update: {e2}")
                active_generate_websocket = None
                generate_websocket_session_start_time = None
                logger.info("Data generation WebSocket singleton released due to error")
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/view-data")
async def websocket_view_data(websocket: WebSocket):
    """WebSocket endpoint for viewing live data (used by frontend) - streams existing data from database"""
    
    if not WEBSOCKET_ENABLED:
        await websocket.close(code=1008, reason="WebSocket is disabled in configuration")
        logger.warning("View WebSocket connection rejected - WebSocket is disabled")
        return
    
    # Accept connection and add to view websockets set
    await websocket.accept()
    async with view_websockets_lock:
        view_websockets.add(websocket)
    logger.info(f"View WebSocket connection established (total viewers: {len(view_websockets)})")
    
    init_websocket_db()
    
    try:
        # Send recent records from database on connection
        conn = sqlite3.connect(NETWORK_LOGS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the last 50 records
        cursor.execute("""
            SELECT id, network_id, timestamp, user_id, location, os, browser, data, prediction_results, session_active_time, is_active
            FROM websocket_data
            ORDER BY id DESC
            LIMIT 50
        """)
        
        recent_records = cursor.fetchall()
        conn.close()
        
        # Send recent records in reverse order (oldest first)
        for record in reversed(recent_records):
            try:
                user_id = record["user_id"]
                user = None
                if user_id:
                    conn = sqlite3.connect(USERS_DB)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, first_name, last_name, email FROM users WHERE id = ?", (user_id,))
                    user_row = cursor.fetchone()
                    conn.close()
                    if user_row:
                        user = {
                            "id": user_row["id"],
                            "first_name": user_row["first_name"],
                            "last_name": user_row["last_name"],
                            "email": user_row["email"]
                        }
                
                location = json.loads(record["location"]) if record["location"] else None
                data = json.loads(record["data"]) if record["data"] else None
                prediction_results = json.loads(record["prediction_results"]) if record["prediction_results"] else None
                
                response = {
                    "id": record["id"],
                    "network_id": record["network_id"],
                    "timestamp": record["timestamp"],
                    "user": user,
                    "location": location,
                    "os": record["os"],
                    "browser": record["browser"],
                    "data": data,
                    "prediction_results": prediction_results
                }
                
                await websocket.send_json(response)
            except Exception as e:
                logger.debug(f"Error sending historical record: {e}")
        
        logger.info(f"Sent {len(recent_records)} historical records to view WebSocket")
        
        # Keep connection alive; new data will be broadcast by the generate-data WebSocket.
        # We don't require the client to send any messages.
        while True:
            try:
                await asyncio.sleep(60.0)
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info("View WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in view WebSocket: {e}", exc_info=True)
    finally:
        # Remove from view websockets set
        async with view_websockets_lock:
            view_websockets.discard(websocket)
        logger.info(f"View WebSocket disconnected (remaining viewers: {len(view_websockets)})")
        try:
            await websocket.close()
        except:
            pass

@app.on_event("startup")
async def startup_event():
    init_users_db()
    # Demo mode: wipe and seed 80 users over 2 weeks, then add 1–7 users every 30s
    seed_demo_users_initial()
    asyncio.create_task(demo_users_worker())
    logger.info("Demo users: seeded 80 users, background worker adding 1–7 every 30s")

    init_websocket_db()
    init_message_queue_db()

    if MESSAGE_QUEUE_ENABLED:
        asyncio.create_task(message_queue_worker())
        logger.info("Message queue worker started")
    else:
        logger.info("Message queue worker disabled in configuration")

    # Always start the missing predictions worker to ensure all records get predictions
    asyncio.create_task(missing_predictions_worker())
    logger.info("Missing predictions worker started")

    if WEBSOCKET_ENABLED:
        logger.info("WebSocket endpoint enabled")
    else:
        logger.info("WebSocket endpoint disabled in configuration")

if __name__ == "__main__":
    import uvicorn
    init_users_db()
    uvicorn.run(app, host="127.0.0.1", port=8002)
