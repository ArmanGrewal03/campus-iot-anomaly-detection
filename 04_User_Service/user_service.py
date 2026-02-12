from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
from datetime import datetime, timedelta
import logging
import sqlite3
import asyncio
import random
import uuid
import urllib.request
import urllib.parse
import urllib.error

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NETWORK_LOGS_DB = "network_logs.db"
USERS_DB = "users.db"
MESSAGE_QUEUE_DB = "message_queue.db"
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8001")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "A")

# Singleton WebSocket connection tracking
active_websocket: Optional[WebSocket] = None
websocket_lock = asyncio.Lock()
websocket_session_start_time: Optional[str] = None

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
async def get_users():
    init_users_db()
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, first_name, last_name, created_at, block_status, block_type, block_until, block_reason 
        FROM users 
        ORDER BY id
    """)
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
            "total_users": len(users),
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
                    "model_name": selected_model_name
                }
                
                logger.info(f"Calling prediction API for network_id: {network_id}, data: {json.dumps(data)}")
                
                status_code, result = await asyncio.to_thread(_make_http_request, url, payload, headers)
                
                if status_code == 200:
                    prediction_results_json = json.dumps(result)
                    conn = sqlite3.connect(NETWORK_LOGS_DB)
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE websocket_data 
                        SET prediction_results = ?
                        WHERE network_id = ?
                    """, (prediction_results_json, network_id))
                    conn.commit()
                    conn.close()
                    processed_count += 1
                    logger.debug(f"Processed prediction for network_id: {network_id}")
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
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in /history endpoint: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
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
            
            log_entry = {
                "id": row["id"],
                "network_id": network_id_val,
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
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in /network-logs endpoint: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving network logs: {str(e)}"
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
        
        return JSONResponse(
            content={
                "status": "success",
                "log": log_entry
            },
            status_code=200
        )
    except HTTPException:
        raise
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
        status_code=200
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
        status_code=200
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
    conn.commit()
    conn.close()
    logger.info(f"Initialized message queue database: {MESSAGE_QUEUE_DB}")

def _make_http_request(url: str, data: dict, headers: dict):
    req_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=req_data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            response_data = response.read().decode('utf-8')
            return status_code, json.loads(response_data)
    except urllib.error.HTTPError as e:
        error_data = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
        return e.code, {"error": error_data}
    except Exception as e:
        raise e

async def process_predict_request(message_id: int, network_id: str, data: dict):
    try:
        url = f"{MODEL_API_URL}/predict"
        payload = {"data": [data]}
        headers = {
            "Content-Type": "application/json",
            "model_name": selected_model_name
        }
        
        logger.info(f"Calling prediction API for network_id: {network_id}, data: {json.dumps(data)}")
        
        status_code, result = await asyncio.to_thread(_make_http_request, url, payload, headers)
        
        if status_code == 200:
            processed_at = datetime.utcnow().isoformat()
            
            conn = sqlite3.connect(MESSAGE_QUEUE_DB)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE message_queue 
                SET status = 'completed', processed_at = ?
                WHERE id = ?
            """, (processed_at, message_id))
            conn.commit()
            conn.close()
            
            prediction = result.get('predictions', [{}])[0] if result.get('predictions') else {}
            logger.info(f"Prediction successful for network_id: {network_id}, result: {prediction}")
            
            prediction_results_json = json.dumps(result)
            conn = sqlite3.connect(NETWORK_LOGS_DB)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE websocket_data 
                SET prediction_results = ?
                WHERE network_id = ?
            """, (prediction_results_json, network_id))
            conn.commit()
            conn.close()
            logger.info(f"Saved prediction results to network_logs for network_id: {network_id}")
        else:
            error_msg = result.get('error', f"HTTP {status_code}")[:500] if isinstance(result, dict) else f"HTTP {status_code}"
            logger.error(f"Prediction failed for network_id: {network_id}, status: {status_code}, error: {error_msg}")
            
            conn = sqlite3.connect(MESSAGE_QUEUE_DB)
            cursor = conn.cursor()
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
            conn.close()
                
    except Exception as e:
        error_msg = str(e)[:500]
        logger.error(f"Error processing predict request for network_id: {network_id}: {error_msg}", exc_info=True)
        
        conn = sqlite3.connect(MESSAGE_QUEUE_DB)
        cursor = conn.cursor()
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
        
        cursor.execute("""
            UPDATE message_queue 
            SET status = 'processing' 
            WHERE id = ?
        """, (message_id,))
        conn.commit()
        conn.close()
        
        try:
            data = json.loads(data_json)
            await process_predict_request(message_id, network_id, data)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message data: {e}")
            conn = sqlite3.connect(MESSAGE_QUEUE_DB)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE message_queue 
                SET status = 'failed', error_message = ?
                WHERE id = ?
            """, (str(e)[:500], message_id))
            conn.commit()
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
    
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    if count == 0:
        first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
            "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor",
            "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Sanchez",
            "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
            "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams"
        ]
        
        users = []
        used_combinations = set()
        
        while len(users) < 20:
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            combination = (first_name, last_name)
            
            if combination not in used_combinations:
                used_combinations.add(combination)
                users.append((first_name, last_name, datetime.utcnow().isoformat()))
        
        cursor.executemany(
            "INSERT INTO users (first_name, last_name, created_at) VALUES (?, ?, ?)",
            users
        )
        conn.commit()
        logger.info(f"Created {len(users)} users in database")
    else:
        logger.info(f"Users database already contains {count} users")
    
    conn.close()
    logger.info(f"Initialized users database: {USERS_DB}")

def load_feature_names() -> list:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_names_path = os.path.join(base_dir, "..", "A-DataIngestion", "Processed", "feature_names.json")
    feature_names_path = os.path.normpath(feature_names_path)
    
    if os.path.exists(feature_names_path):
        try:
            with open(feature_names_path, 'r') as f:
                features = json.load(f)
                logger.info(f"Loaded {len(features)} features from {feature_names_path}")
                return features
        except Exception as e:
            logger.error(f"Error loading feature names: {e}")
    else:
        logger.warning(f"Feature names file not found at {feature_names_path}")
    
    logger.info("Using default feature set")
    return [
        "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
    ]

def generate_random_data(feature_names: list) -> dict:
    data = {}
    
    # Occasionally generate anomalous patterns (5% chance)
    is_anomaly_pattern = random.random() < 0.05
    
    proto_features = [f for f in feature_names if f.startswith("proto_")]
    state_features = [f for f in feature_names if f.startswith("state_")]
    service_features = [f for f in feature_names if f.startswith("service_")]
    
    for feature in feature_names:
        if feature == "dur":
            # Duration: vary from 0 to 10000, with occasional very long durations
            if is_anomaly_pattern and random.random() < 0.3:
                data[feature] = round(random.uniform(5000.0, 50000.0), 6)
            else:
                data[feature] = round(random.uniform(0.0, 5000.0), 6)
        elif feature.startswith("proto_"):
            # Protocol features: vary probability based on pattern
            prob = 0.3 if is_anomaly_pattern else random.uniform(0.05, 0.25)
            data[feature] = 1 if random.random() < prob else 0
        elif feature.startswith("state_"):
            # State features: vary probability
            prob = 0.5 if is_anomaly_pattern else random.uniform(0.1, 0.4)
            data[feature] = 1 if random.random() < prob else 0
        elif feature.startswith("service_"):
            # Service features: vary probability
            prob = 0.4 if is_anomaly_pattern else random.uniform(0.05, 0.3)
            data[feature] = 1 if random.random() < prob else 0
        elif feature in ["Spkts", "Dpkts", "sbytes", "dbytes", "sttl", "dttl", 
                         "sloss", "dloss", "swin", "stcpb", "dtcpb", "dwin",
                         "tcprtt", "synack", "ackdat", "trans_depth", "res_bdy_len",
                         "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
                         "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd", "ct_flw_http_mthd",
                         "ct_src_ltm", "ct_srv_dst"]:
            # Packet/connection features: wide range with occasional extremes
            if is_anomaly_pattern and random.random() < 0.4:
                # Anomaly: very high values
                data[feature] = random.randint(50000, 1000000)
            else:
                # Normal: varied range
                max_val = random.choice([1000, 5000, 10000, 50000])
                data[feature] = random.randint(0, max_val)
        elif feature in ["rate", "Sload", "Dload", "Sintpkt", "Dintpkt", "Sjit", "Djit", "smeansz", "dmeansz"]:
            # Rate/load features: vary ranges significantly
            if is_anomaly_pattern and random.random() < 0.3:
                # Anomaly: extreme values
                data[feature] = round(random.uniform(1000000.0, 10000000.0), 2)
            else:
                # Normal: varied ranges
                max_val = random.choice([1000.0, 10000.0, 100000.0, 1000000.0])
                data[feature] = round(random.uniform(0.0, max_val), 2)
        elif feature in ["is_ftp_login", "is_sm_ips_ports"]:
            # Boolean features: vary probability
            prob = 0.8 if is_anomaly_pattern else random.uniform(0.0, 0.3)
            data[feature] = 1 if random.random() < prob else 0
        elif feature in ["byte_ratio", "pkt_ratio", "flow_rate", "pkt_rate"]:
            # Ratio features: vary ranges
            if is_anomaly_pattern:
                # Anomaly: extreme ratios
                data[feature] = round(random.uniform(10.0, 100.0), 4)
            else:
                # Normal: varied ratios
                max_ratio = random.choice([1.0, 5.0, 10.0, 20.0])
                data[feature] = round(random.uniform(0.0, max_ratio), 4)
        else:
            # Default: vary the range
            max_val = random.choice([100, 500, 1000, 5000, 10000])
            data[feature] = random.randint(0, max_val)
    
    return data

@app.websocket("/ws/data-stream")
async def websocket_data_stream(websocket: WebSocket):
    global active_websocket, websocket_session_start_time
    
    if not WEBSOCKET_ENABLED:
        await websocket.close(code=1008, reason="WebSocket is disabled in configuration")
        logger.warning("WebSocket connection rejected - WebSocket is disabled")
        return
    
    # Singleton check: only allow one active WebSocket connection
    async with websocket_lock:
        if active_websocket is not None:
            await websocket.close(code=1008, reason="Another WebSocket connection is already active")
            logger.warning("WebSocket connection rejected - another connection is already active")
            return
        
        # Accept the new connection and mark it as active
        await websocket.accept()
        active_websocket = websocket
        websocket_session_start_time = datetime.utcnow().isoformat()
        logger.info(f"WebSocket connection established (singleton) - session started at {websocket_session_start_time}")
    
    init_websocket_db()
    feature_names = load_feature_names()
    logger.info(f"Loaded {len(feature_names)} features for data generation")
    
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
                (network_id, timestamp, data_json, user_id, location_json, os, browser, websocket_session_start_time, 1)
            )
            inserted_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            response = {
                "id": inserted_id,
                "network_id": network_id,
                "timestamp": timestamp,
                "user": random_user,
                "location": location,
                "os": os,
                "browser": browser,
                "data": random_data
            }
            
            await websocket.send_json(response)
            user_name = f"{random_user['first_name']} {random_user['last_name']}" if random_user else "Unknown"
            logger.info(f"Sent data record {inserted_id} via WebSocket (user: {user_name}, location: {location['city']}, {location['country']}, OS: {os}, Browser: {browser})")
            
            if MESSAGE_QUEUE_ENABLED:
                try:
                    publish_url = "http://127.0.0.1:8002/publish"
                    publish_payload = {
                        "network_id": network_id,
                        "data": random_data
                    }
                    publish_headers = {"Content-Type": "application/json"}
                    
                    status_code, _ = await asyncio.to_thread(_make_http_request, publish_url, publish_payload, publish_headers)
                    
                    if status_code == 200:
                        logger.info(f"Published predict request for network_id: {network_id}")
                    else:
                        logger.warning(f"Failed to publish predict request for network_id: {network_id}, status: {status_code}")
                except Exception as e:
                    logger.error(f"Error publishing predict request: {e}")
            else:
                logger.debug(f"Message queue disabled - skipping publish for network_id: {network_id}")
            
            wait_time = random.uniform(60, 90)  # Wait 60-90 seconds between data generation
            logger.info(f"Waiting {wait_time:.2f} seconds before next data generation")
            await asyncio.sleep(wait_time)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        async with websocket_lock:
            if active_websocket == websocket:
                # Update all records from this session to mark them as inactive
                if websocket_session_start_time:
                    try:
                        conn = sqlite3.connect(NETWORK_LOGS_DB)
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE websocket_data SET is_active = 0 WHERE session_active_time = ? AND is_active = 1",
                            (websocket_session_start_time,)
                        )
                        updated_count = cursor.rowcount
                        conn.commit()
                        conn.close()
                        logger.info(f"Marked {updated_count} records as inactive for session started at {websocket_session_start_time}")
                    except Exception as e:
                        logger.error(f"Error updating session status: {e}")
                active_websocket = None
                websocket_session_start_time = None
                logger.info("WebSocket singleton released")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}", exc_info=True)
        async with websocket_lock:
            if active_websocket == websocket:
                # Update all records from this session to mark them as inactive
                if websocket_session_start_time:
                    try:
                        conn = sqlite3.connect(NETWORK_LOGS_DB)
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE websocket_data SET is_active = 0 WHERE session_active_time = ? AND is_active = 1",
                            (websocket_session_start_time,)
                        )
                        updated_count = cursor.rowcount
                        conn.commit()
                        conn.close()
                        logger.info(f"Marked {updated_count} records as inactive for session started at {websocket_session_start_time} due to error")
                    except Exception as e2:
                        logger.error(f"Error updating session status: {e2}")
                active_websocket = None
                websocket_session_start_time = None
                logger.info("WebSocket singleton released due to error")
        try:
            await websocket.close()
        except:
            pass

@app.on_event("startup")
async def startup_event():
    init_users_db()
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
