from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from datetime import datetime
import sqlite3
import csv
import io
import json
import logging
import random
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from training_manager import train_model_dispatch
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
import numpy as np

app = FastAPI(title="Campus IoT Anomaly Detection API", version="1.0.0")

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainRequest(BaseModel):
    model_name: str
    dataset_name: Optional[str] = None
    features: List[str]
    model_type: str

class PredictRequest(BaseModel):
    model_path: str
    data: List[Dict[str, Any]]

DEFAULT_DB_NAME = "campus_iot_data.db"

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Request headers: {dict(request.headers)}")
    logger.error(f"Validation errors: {exc.errors()}")
    
    try:
        body = await request.body()
        logger.error(f"Request body type: {type(body)}")
        logger.error(f"Request body length: {len(body) if body else 0}")
        if body:
            body_preview = body[:500] if isinstance(body, bytes) else str(body)[:500]
            logger.error(f"Request body preview: {body_preview}")
    except Exception as e:
        logger.error(f"Could not read request body: {e}")
    
    return await request_validation_exception_handler(request, exc)

def get_db_name(database_name: Optional[str] = Header(None, alias="X-Database-Name")) -> str:
    if database_name is None:
        return DEFAULT_DB_NAME
    
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '', database_name)
    
    if not sanitized.endswith('.db'):
        sanitized = sanitized + '.db'
    
    if sanitized == '.db' or len(sanitized) < 4:
        logger.warning(f"Invalid database name '{database_name}', using default")
        return DEFAULT_DB_NAME
    
    logger.info(f"Using database: {sanitized}")
    return sanitized

def get_db_path(db_name: str) -> str:
    return db_name

def get_db_connection(db_name: str):
    db_path = get_db_path(db_name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_name: str = DEFAULT_DB_NAME):
    conn = get_db_connection(db_name)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS csv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_timestamp TEXT NOT NULL,
            row_data TEXT NOT NULL
        )
    """)
    
    try:
        cursor.execute("ALTER TABLE csv_data ADD COLUMN T TEXT")
        logger.info(f"Added T column to csv_data table in {db_name}")
    except sqlite3.OperationalError:
        pass
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inserted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_timestamp TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized database: {db_name}")

@app.on_event("startup")
async def startup_event():
    init_db(DEFAULT_DB_NAME)


@app.get("/api/health")
async def health_check(database_name: str = Depends(get_db_name)):
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Campus IoT Anomaly Detection API",
            "database": database_name
        },
        status_code=200
    )


@app.post("/new")
async def upload_csv(
    request: Request, 
    file: UploadFile = File(None),
    database_name: str = Depends(get_db_name)
):
    logger.info(f"Received file upload request")
    logger.info(f"Content-Type header: {request.headers.get('content-type', 'not set')}")
    
    contents = None
    filename = None
    
    content_type = request.headers.get('content-type', '').lower()
    
    if file is not None:
        logger.info("Processing multipart/form-data upload")
        logger.info(f"Filename: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        
        filename = file.filename or "uploaded_file.csv"
        
        if file.filename and not file.filename.endswith('.csv'):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        logger.info("Reading file contents...")
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file")
        
    elif 'text/csv' in content_type or 'application/csv' in content_type:
        logger.info("Processing raw CSV data upload")
        filename = "raw_upload.csv"
        
        logger.info("Reading raw request body...")
        contents = await request.body()
        logger.info(f"Read {len(contents)} bytes from request body")
        
    else:
        logger.info("No file parameter and no CSV content-type, attempting to read raw body...")
        contents = await request.body()
        if contents:
            logger.info(f"Read {len(contents)} bytes from request body (assuming CSV)")
            filename = "raw_upload.csv"
        else:
            raise HTTPException(
                status_code=400, 
                detail="No file provided. Send as multipart/form-data with field 'file' or as raw CSV with Content-Type: text/csv"
            )
    
    try:
        
        if not contents:
            logger.error("Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info("Decoding file contents...")
        csv_string = contents.decode('utf-8')
        logger.info(f"Decoded CSV string length: {len(csv_string)}")
        
        logger.info("Parsing CSV...")
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        
        init_db(database_name)
        
        logger.info(f"Connecting to database: {database_name}")
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        upload_timestamp = datetime.utcnow().isoformat()
        rows_inserted = 0
        
        logger.info("Inserting rows into database...")
        for row in csv_reader:
            row_json = json.dumps(row)
            cursor.execute(
                "INSERT INTO csv_data (upload_timestamp, row_data) VALUES (?, ?)",
                (upload_timestamp, row_json)
            )
            rows_inserted += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully inserted {rows_inserted} rows")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully uploaded and stored {rows_inserted} rows from CSV file",
                "filename": filename,
                "upload_timestamp": upload_timestamp,
                "rows_inserted": rows_inserted
            },
            status_code=200
        )
    
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error: {e}")
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except csv.Error as e:
        logger.error(f"CSV parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/view")
async def view_data(
    limit: int = 1000,
    offset: int = 0,
    database_name: str = Depends(get_db_name)
):
    if limit < 1:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing data: limit={limit}, offset={offset}")
    
    try:
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM csv_data")
        total_count = cursor.fetchone()['total']
        
        try:
            cursor.execute("""
                SELECT id, upload_timestamp, row_data, T 
                FROM csv_data 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        except sqlite3.OperationalError:
            cursor.execute("""
                SELECT id, upload_timestamp, row_data 
                FROM csv_data 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        data = []
        for row in rows:
            try:
                row_data = json.loads(row['row_data'])
                row_dict = {
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "data": row_data
                }
                if 'T' in row.keys() and row['T']:
                    row_dict["T"] = row['T']
                data.append(row_dict)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for row {row['id']}: {e}")
                row_dict = {
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "data": {"error": "Failed to parse row data", "raw": row['row_data']}
                }
                if 'T' in row.keys() and row['T']:
                    row_dict["T"] = row['T']
                data.append(row_dict)
        
        logger.info(f"Retrieved {len(data)} rows from database")
        
        return JSONResponse(
            content={
                "status": "success",
                "total_rows": total_count,
                "returned_rows": len(data),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(data)) < total_count,
                "data": data
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error retrieving data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")


@app.get("/training")
async def get_training_data(
    limit: int = 100, 
    offset: int = 0,
    database_name: str = Depends(get_db_name)
):
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing training data: limit={limit}, offset={offset}")
    
    try:
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(csv_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        cursor.execute("SELECT COUNT(*) as total FROM csv_data WHERE T = ?", ("training",))
        total_count = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT id, upload_timestamp, row_data, T 
            FROM csv_data 
            WHERE T = ?
            ORDER BY id 
            LIMIT ? OFFSET ?
        """, ("training", limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        data = []
        for row in rows:
            try:
                row_data = json.loads(row['row_data'])
                data.append({
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    "data": row_data
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for row {row['id']}: {e}")
                data.append({
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    "data": {"error": "Failed to parse row data", "raw": row['row_data']}
                })
        
        logger.info(f"Retrieved {len(data)} training rows from database")
        
        return JSONResponse(
            content={
                "status": "success",
                "label": "training",
                "total_rows": total_count,
                "returned_rows": len(data),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(data)) < total_count,
                "data": data
            },
            status_code=200
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving training data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving training data: {str(e)}")


@app.get("/testing")
async def get_testing_data(
    limit: int = 100, 
    offset: int = 0,
    database_name: str = Depends(get_db_name)
):
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing testing data: limit={limit}, offset={offset}")
    
    try:
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(csv_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        cursor.execute("SELECT COUNT(*) as total FROM csv_data WHERE T = ?", ("testing",))
        total_count = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT id, upload_timestamp, row_data, T 
            FROM csv_data 
            WHERE T = ?
            ORDER BY id 
            LIMIT ? OFFSET ?
        """, ("testing", limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        data = []
        for row in rows:
            try:
                row_data = json.loads(row['row_data'])
                data.append({
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    "data": row_data
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for row {row['id']}: {e}")
                data.append({
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    "data": {"error": "Failed to parse row data", "raw": row['row_data']}
                })
        
        logger.info(f"Retrieved {len(data)} testing rows from database")
        
        return JSONResponse(
            content={
                "status": "success",
                "label": "testing",
                "total_rows": total_count,
                "returned_rows": len(data),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(data)) < total_count,
                "data": data
            },
            status_code=200
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving testing data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving testing data: {str(e)}")


@app.put("/validate")
async def validate_data(database_name: str = Depends(get_db_name)):
    logger.info(f"Starting data validation and assignment for database: {database_name}")
    
    try:
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute("ALTER TABLE csv_data ADD COLUMN T TEXT")
            logger.info("Added T column to csv_data table")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("T column already exists")
            else:
                raise
        
        cursor.execute("SELECT id FROM csv_data")
        all_rows = cursor.fetchall()
        total_rows = len(all_rows)
        
        if total_rows == 0:
            logger.warning("No rows found in database")
            conn.close()
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "No rows to validate",
                    "total_rows": 0,
                    "training_rows": 0,
                    "testing_rows": 0
                },
                status_code=200
            )
        
        training_count = int(total_rows * 0.3)
        testing_count = total_rows - training_count
        
        logger.info(f"Total rows: {total_rows}, Training: {training_count}, Testing: {testing_count}")
        
        row_ids = [row['id'] for row in all_rows]
        random.shuffle(row_ids)
        
        training_ids = set(row_ids[:training_count])
        testing_ids = set(row_ids[training_count:])
        
        updated_training = 0
        updated_testing = 0
        
        for row_id in row_ids:
            if row_id in training_ids:
                cursor.execute("UPDATE csv_data SET T = ? WHERE id = ?", ("training", row_id))
                updated_training += 1
            else:
                cursor.execute("UPDATE csv_data SET T = ? WHERE id = ?", ("testing", row_id))
                updated_testing += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Validation complete: {updated_training} training, {updated_testing} testing")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Data validation and assignment completed",
                "total_rows": total_rows,
                "training_rows": updated_training,
                "testing_rows": updated_testing,
                "training_percentage": round((updated_training / total_rows) * 100, 2),
                "testing_percentage": round((updated_testing / total_rows) * 100, 2)
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error during validation: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during validation: {str(e)}")


@app.post("/clear")
async def clear_database(database_name: str = Depends(get_db_name)):
    logger.warning(f"Clearing database {database_name} - all data will be deleted")
    
    try:
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM csv_data")
        total_rows = cursor.fetchone()['total']
        
        cursor.execute("DELETE FROM csv_data")
        
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='csv_data'")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database {database_name} cleared: {total_rows} rows deleted")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Database cleared successfully",
                "rows_deleted": total_rows
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error clearing database: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")


@app.post("/insert")
async def insert_data(
    data: Dict[str, Any],
    database_name: str = Depends(get_db_name)
):
    logger.info(f"Inserting new row with fields: {list(data.keys())} into database: {database_name}")
    
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Request body cannot be empty")
        
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        
        upload_timestamp = datetime.utcnow().isoformat()
        row_data_json = json.dumps(data)
        
        cursor.execute("""
            INSERT INTO csv_data (upload_timestamp, row_data)
            VALUES (?, ?)
        """, (upload_timestamp, row_data_json))
        
        inserted_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully inserted row with ID: {inserted_id} into csv_data")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Row inserted successfully",
                "id": inserted_id,
                "upload_timestamp": upload_timestamp,
                "data": data
            },
            status_code=201
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inserting data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error inserting data: {str(e)}")


@app.get("/api/stats")
async def get_stats(database_name: str = Depends(get_db_name)):
    """Get aggregated statistics for KPI display"""
    try:
        init_db(database_name)
        stats = {}
        
        # Get total records
        try:
            conn = get_db_connection(database_name)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM csv_data")
            stats['total_records'] = cursor.fetchone()['total']
            conn.close()
        except Exception as e:
            logger.warning(f"Error getting total records: {e}")
            stats['total_records'] = 0
        
        # Get training records count
        try:
            conn = get_db_connection(database_name)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(csv_data)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'T' in columns:
                cursor.execute("SELECT COUNT(*) as total FROM csv_data WHERE T = ?", ("training",))
                stats['training_records'] = cursor.fetchone()['total']
            else:
                stats['training_records'] = 0
            conn.close()
        except Exception as e:
            logger.warning(f"Error getting training records: {e}")
            stats['training_records'] = 0
        
        # Get testing records count
        try:
            conn = get_db_connection(database_name)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(csv_data)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'T' in columns:
                cursor.execute("SELECT COUNT(*) as total FROM csv_data WHERE T = ?", ("testing",))
                stats['testing_records'] = cursor.fetchone()['total']
            else:
                stats['testing_records'] = 0
            conn.close()
        except Exception as e:
            logger.warning(f"Error getting testing records: {e}")
            stats['testing_records'] = 0
        
        # Calculate percentages if total > 0
        if stats['total_records'] > 0:
            stats['training_percentage'] = round((stats['training_records'] / stats['total_records']) * 100, 1)
            stats['testing_percentage'] = round((stats['testing_records'] / stats['total_records']) * 100, 1)
        else:
            stats['training_percentage'] = 0
            stats['testing_percentage'] = 0
        
        # API is always online if we got here
        stats['api_online'] = True
        
        return JSONResponse(content=stats, status_code=200)
    
    except Exception as e:
        logger.error(f"Error getting stats: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            content={
                "error": str(e),
                "total_records": 0,
                "training_records": 0,
                "testing_records": 0,
                "training_percentage": 0,
                "testing_percentage": 0,
                "api_online": False
            },
            status_code=500
        )


@app.post("/api/train")
async def train_model_endpoint(
    request: TrainRequest,
    database_name: str = Depends(get_db_name)
):
    """Train a model based on selected features and dataset"""
    db_to_use = request.dataset_name if request.dataset_name else database_name
    logger.info(f"Training request for model '{request.model_name}' on dataset '{db_to_use}'")
    
    try:
        init_db(db_to_use)
        conn = get_db_connection(db_to_use)
        
        # Check if T column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(csv_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(status_code=400, detail="Dataset not validated. Please run validation first.")
            
        # Fetch training data
        cursor.execute("SELECT row_data FROM csv_data WHERE T = ?", ("training",))
        rows = cursor.fetchall()
        
        if len(rows) < 10:
            conn.close()
            # If not enough data, return a simulated success for demonstration if requested, 
            # but here we'll try to be "real" or return a helpful error.
            logger.warning(f"Insufficient training data: {len(rows)} rows")
            
        # Parse data into DataFrame
        data_list = []
        for r in rows:
            try:
                data_list.append(json.loads(r['row_data']))
            except:
                continue
                
        if not data_list:
            conn.close()
            raise HTTPException(status_code=400, detail="No valid training data found.")
            
        df = pd.DataFrame(data_list)
        conn.close()
        
        # Filter for requested features + label
        if not request.features:
            # AUTO-SELECT FEATURES (rfV1 logic)
            exclude_cols = ["label", "id", "attack_cat", "upload_timestamp", "T"]
            available_features = [col for col in df.columns if col not in exclude_cols]
            logger.info(f"Auto-selected {len(available_features)} features for training")
        else:
            available_features = [f for f in request.features if f in df.columns]
            
        if not available_features:
            raise HTTPException(status_code=400, detail="No suitable features found for training.")
            
        label_col = 'label' if 'label' in df.columns else None
        if not label_col:
            # Try to find a label-like column
            for col in df.columns:
                if col.lower() in ['label', 'target', 'class', 'anomaly']:
                    label_col = col
                    break
        
        if not label_col:
            raise HTTPException(status_code=400, detail="No label/target column found in dataset.")
            
        # exclude label from X
        if label_col in available_features:
            available_features.remove(label_col)
            
        # ALWAYS EXCLUDE LABEL from features list passed to model
        if label_col in available_features:
            available_features.remove(label_col)

        # Delegate to Training Manager
        try:
            metrics = train_model_dispatch(
                model_type=request.model_type,
                model_name=request.model_name,
                df=df,
                features=available_features,
                label_col=label_col
            )
        except Exception as e:
            logger.error(f"Error in training manager: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
        
        # Add extra context to metrics
        metrics["dataset"] = db_to_use
        metrics["features"] = ", ".join(available_features)
        metrics["row_count"] = len(df)
        metrics["timestamp"] = datetime.utcnow().isoformat()
        
        return JSONResponse(content=metrics, status_code=200)
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict_endpoint(request: PredictRequest):
    """Make predictions using a trained model"""
    logger.info(f"Prediction request using model: {request.model_path}")
    
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
            
        model = joblib.load(request.model_path)
        
        # Load feature names from metadata
        meta_path = request.model_path.replace(".joblib", "_meta.json")
        features = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    features = meta.get("features")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        df = pd.DataFrame(request.data)
        
        # Filter columns to match training features
        if features:
            # Ensure all features exist in input, fill missing with 0
            for f in features:
                if f not in df.columns:
                    df[f] = 0
            X = df[features].copy()
        else:
            # Fallback for old models: remove known labels/IDs
            exclude_cols = ["label", "id", "attack_cat", "upload_timestamp", "T"]
            X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
            
        # Preprocess input data to match model expectations
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
        
        # Prediction Logic based on Model Type
        predictions = []
        probabilities = [] # Confidence scores
        
        is_if = isinstance(model, IsolationForest)
        
        # Handle Autoencoder
        # Since we use standard MLPRegressor from sklearn, it doesn't have a special type attribute 
        # unless we check the class name or metadata.
        is_ae = isinstance(model, MLPRegressor)
        
        if is_if:
            # Isolation Forest
            if os.path.exists(request.model_path.replace(".joblib", "_scaler.joblib")):
                scaler = joblib.load(request.model_path.replace(".joblib", "_scaler.joblib"))
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X # Should not happen now
                
            # predict returns -1 (anomaly) and 1 (normal)
            raw_preds = model.predict(X_scaled)
            # decision_function returns anomaly score (lower = more anomalous)
            scores = model.decision_function(X_scaled)
            
            predictions = np.where(raw_preds == -1, 1, 0)
            # Normalize confidence roughly
            probabilities = (scores.max() - scores) / (scores.max() - scores.min() + 1e-6)
            # Or just use the score magnitude:
            # High anomaly score (very negative) -> High confidence in anomaly
             
        elif is_ae:
            # Autoencoder
            if os.path.exists(request.model_path.replace(".joblib", "_scaler.joblib")):
                scaler = joblib.load(request.model_path.replace(".joblib", "_scaler.joblib"))
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X # Should not happen if trained correctly
                
            X_recon = model.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - X_recon, 2), axis=1)
            
            # Retrieve threshold if possible
            threshold = 0.1 # Default fallback
            meta_path = request.model_path.replace(".joblib", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    m = json.load(f)
                    threshold = m.get("threshold", 0.1)
            
            predictions = (mse > threshold).astype(int)
            # Confidence based on distance from threshold
            probabilities = mse 
            
        else:
            # Random Forest / Classifier
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] # Prob of class 1
        
        results = []
        for i, pred in enumerate(predictions):
            conf = float(probabilities[i]) if isinstance(probabilities, np.ndarray) else 0.0
            
            # Logic for readable confidence:
            # If pred=0 (Normal), we want confidence of it being Normal.
            # If pred=1 (Anomaly), we want confidence of it being Anomaly.
            
            if is_ae:
                 # AE returns MSE as "probabilities". It's not a % confidence.
                 # Let's just return it raw or normalized if possible.
                 # For now, let's keep it raw but maybe cap it for the UI bar.
                 pass
            elif is_if:
                 # IF returns normalized anomaly score. 
                 # If pred is Normal (0), score was likely high (in decision_function).
                 # If pred is Anomaly (1), score was low.
                 # Our 'probabilities' var maps roughly to anomaly likelihood.
                 if pred == 0:
                     conf = 1.0 - conf
            else:
                 # Random Forest (Standard Classifier)
                 # probabilities is P(Anomaly)
                 if pred == 0:
                     conf = 1.0 - conf
            
            # Cap confidence for display
            if conf > 1.0: conf = 1.0
            if conf < 0.0: conf = 0.0
            
            results.append({
                "index": i,
                "prediction": int(pred),
                "label": "anomaly" if pred == 1 else "normal",
                "confidence": conf
            })
            
        return JSONResponse(content={
            "status": "success",
            "results": results,
            "model_type": type(model).__name__
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def extract_types_from_chunk(rows_data):
    """Extract type values and training/testing labels from rows"""
    type_counts = {}
    type_training = {}
    type_testing = {}
    
    for row in rows_data:
        if isinstance(row, dict) and 'data' in row:
            row_data = row['data']
            if isinstance(row_data, dict):
                # Find type field (case-insensitive)
                type_value = None
                for key in row_data.keys():
                    if key.lower() == 'type':
                        type_value = row_data[key]
                        break
                
                # If not found, try other common field names
                if type_value is None:
                    for key in row_data.keys():
                        if key.lower() in ['label', 'category', 'class']:
                            type_value = row_data[key]
                            break
                
                if type_value is not None:
                    type_str = str(type_value).strip()
                    if type_str and type_str.lower() not in ['none', 'null', 'nan', '', 'undefined']:
                        # Count by type
                        type_counts[type_str] = type_counts.get(type_str, 0) + 1
                        
                        # Count by training/testing split
                        t_label = row.get('T', '').lower() if 'T' in row else None
                        if t_label == 'training':
                            type_training[type_str] = type_training.get(type_str, 0) + 1
                        elif t_label == 'testing':
                            type_testing[type_str] = type_testing.get(type_str, 0) + 1
    
    return type_counts, type_training, type_testing


def fetch_chunk_data(db_name, offset, limit):
    """Fetch a chunk of data from the database"""
    try:
        conn = get_db_connection(db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, upload_timestamp, row_data, T 
                FROM csv_data 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        except sqlite3.OperationalError:
            cursor.execute("""
                SELECT id, upload_timestamp, row_data 
                FROM csv_data 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        data = []
        for row in rows:
            try:
                row_data = json.loads(row['row_data'])
                row_dict = {
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "data": row_data
                }
                if 'T' in row.keys() and row['T']:
                    row_dict["T"] = row['T']
                data.append(row_dict)
            except json.JSONDecodeError:
                row_dict = {
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "data": {"error": "Failed to parse row data"}
                }
                if 'T' in row.keys() and row['T']:
                    row_dict["T"] = row['T']
                data.append(row_dict)
        
        return data
    except Exception as e:
        logger.error(f"Error fetching chunk at offset {offset}: {e}")
        return []


@app.get("/api/type-stats")
async def get_type_stats(database_name: str = Depends(get_db_name)):
    """Get type distribution statistics - processes all rows to find all types"""
    try:
        init_db(database_name)
        
        conn = get_db_connection(database_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM csv_data")
        total_rows = cursor.fetchone()['total']
        
        if total_rows == 0:
            conn.close()
            return JSONResponse(
                content={
                    "type_distribution": {},
                    "sample_size": 0,
                    "total_rows": 0
                },
                status_code=200
            )
        
        type_counts = {}
        type_training = {}
        type_testing = {}
        
        # Process all rows in batches for efficiency
        batch_size = 5000
        processed_count = 0
        
        logger.info(f"Processing all {total_rows} rows to find all types...")
        
        # Fetch all rows in batches
        try:
            cursor.execute("""
                SELECT id, upload_timestamp, row_data, T 
                FROM csv_data 
                ORDER BY id
            """)
        except sqlite3.OperationalError:
            # T column might not exist
            cursor.execute("""
                SELECT id, upload_timestamp, row_data 
                FROM csv_data 
                ORDER BY id
            """)
        
        # Process in batches
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            # Extract types from this batch
            batch_data = []
            for row in rows:
                try:
                    row_data = json.loads(row['row_data'])
                    row_dict = {
                        "id": row['id'],
                        "upload_timestamp": row['upload_timestamp'],
                        "data": row_data
                    }
                    if 'T' in row.keys() and row['T']:
                        row_dict["T"] = row['T']
                    batch_data.append(row_dict)
                except json.JSONDecodeError:
                    continue
            
            # Extract types from batch
            chunk_types, chunk_train, chunk_test = extract_types_from_chunk(batch_data)
            
            # Merge type counts
            for type_val, count in chunk_types.items():
                type_counts[type_val] = type_counts.get(type_val, 0) + count
            for type_val, count in chunk_train.items():
                type_training[type_val] = type_training.get(type_val, 0) + count
            for type_val, count in chunk_test.items():
                type_testing[type_val] = type_testing.get(type_val, 0) + count
            
            processed_count += len(batch_data)
        
        conn.close()
        
        # Calculate percentages
        total_with_types = sum(type_counts.values())
        type_percentages = {}
        if total_with_types > 0:
            for type_val, count in type_counts.items():
                type_percentages[type_val] = round((count / total_with_types) * 100, 2)
        
        logger.info(f"Found {len(type_counts)} unique types from {processed_count} rows: {list(type_counts.keys())}")
        
        return JSONResponse(
            content={
                "type_distribution": type_counts,
                "type_percentages": type_percentages,
                "type_training": type_training,
                "type_testing": type_testing,
                "sample_size": processed_count,
                "total_rows": total_rows,
                "sampled": False  # We processed all rows
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error getting type stats: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e), "type_distribution": {}},
            status_code=500
        )
