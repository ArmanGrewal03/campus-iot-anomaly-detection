from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header, Depends
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, Tuple
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
import contextvars

app = FastAPI(title="Campus IoT Anomaly Detection API", version="1.0.0")

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
    """Validate and sanitize input parameters for Data Ingestion Service"""
    
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
    
    def validate_string_param(self, value: str, param_name: str, min_len: int = 1, max_len: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
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
        # limit and offset validation
        if "limit" in request.query_params:
            limit, error = self.validate_integer_param(request.query_params["limit"], "limit", min_val=1, max_val=10000)
            if error:
                validation_errors.append(error)
        
        if "offset" in request.query_params:
            offset, error = self.validate_integer_param(request.query_params["offset"], "offset", min_val=0)
            if error:
                validation_errors.append(error)
        
        # dataset_name validation
        if "dataset_name" in request.query_params:
            name, error = self.validate_string_param(request.query_params["dataset_name"], "dataset_name", max_len=255)
            if error:
                validation_errors.append(error)
        
        # Validate request body for POST/PUT
        if request.method in ["POST", "PUT"]:
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
            elif "multipart/form-data" in content_type:
                # File upload validation - check Content-Length
                content_length = request.headers.get("content-length")
                if content_length:
                    try:
                        size = int(content_length)
                        MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
                        if size > MAX_UPLOAD_SIZE:
                            validation_errors.append(f"Upload too large. Maximum size: {MAX_UPLOAD_SIZE / (1024*1024):.1f}MB")
                    except ValueError:
                        pass
        
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

def get_dataset_name(dataset_name: str = Header(...)) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', dataset_name.replace('-', '_'))
    
    if not sanitized or len(sanitized) < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset name '{dataset_name}'. Dataset name must contain at least one alphanumeric character or underscore."
        )
    
    logger.info(f"Using dataset: {sanitized}")
    return sanitized

def get_optional_dataset_name(dataset_name: Optional[str] = Header(None)) -> Optional[str]:
    if dataset_name is None:
        return None
    
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', dataset_name.replace('-', '_'))
    
    if not sanitized or len(sanitized) < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset name '{dataset_name}'. Dataset name must contain at least one alphanumeric character or underscore."
        )
    
    logger.info(f"Using dataset: {sanitized}")
    return sanitized

def get_table_name(table_base: str, dataset_name: str) -> str:
    return f"{table_base}_{dataset_name}"

def get_db_connection():
    conn = sqlite3.connect(DEFAULT_DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(dataset_name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    csv_table = get_table_name("csv_data", dataset_name)
    inserted_table = get_table_name("inserted_data", dataset_name)
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {csv_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_timestamp TEXT NOT NULL,
            row_data TEXT NOT NULL
        )
    """)
    
    try:
        cursor.execute(f"ALTER TABLE {csv_table} ADD COLUMN T TEXT")
        logger.info(f"Added T column to {csv_table} table")
    except sqlite3.OperationalError:
        pass
    
    # Create indexes for frequently queried columns
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{csv_table}_T ON {csv_table}(T)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{csv_table}_upload_timestamp ON {csv_table}(upload_timestamp)")
    # Composite index for common query: WHERE T = ? ORDER BY id
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{csv_table}_T_id ON {csv_table}(T, id)")
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {inserted_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_timestamp TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)
    
    # Create index for inserted_data table
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{inserted_table}_created_timestamp ON {inserted_table}(created_timestamp)")
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized tables for dataset: {dataset_name}")

@app.on_event("startup")
async def startup_event():
    pass


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Campus IoT Anomaly Detection API",
            "database": DEFAULT_DB_NAME
        },
        status_code=200
    )


@app.get("/tables")
async def get_tables():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        
        all_tables = [row['name'] for row in cursor.fetchall()]
        
        tables = []
        for t in all_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) as c FROM [{t}]")
                if cursor.fetchone()['c'] > 0:
                    tables.append(t)
            except Exception:
                pass
        
        conn.close()
        
        logger.info(f"Retrieved {len(tables)} tables from database")
        
        return JSONResponse(
            content={
                "status": "success",
                "database": DEFAULT_DB_NAME,
                "total_tables": len(tables),
                "tables": tables
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error retrieving tables: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving tables: {str(e)}")

@app.get("/fields")
async def get_fields(dataset_name: str = Depends(get_dataset_name)):
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{csv_table}'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Table '{csv_table}' does not exist for dataset '{dataset_name}'"
            )
        
        cursor.execute(f"SELECT row_data FROM {csv_table} LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"No data found in table '{csv_table}' for dataset '{dataset_name}'"
            )
        
        try:
            row_data = json.loads(row['row_data'])
            if not isinstance(row_data, dict):
                raise HTTPException(
                    status_code=500,
                    detail=f"Row data is not a valid JSON object for dataset '{dataset_name}'"
                )
            
            field_names = list(row_data.keys())
            conn.close()
            
            logger.info(f"Retrieved {len(field_names)} fields from dataset: {dataset_name}")
            
            return JSONResponse(
                content={
                    "status": "success",
                    "dataset": dataset_name,
                    "table": csv_table,
                    "total_fields": len(field_names),
                    "fields": field_names
                },
                status_code=200
            )
        except json.JSONDecodeError as e:
            conn.close()
            logger.error(f"Error parsing row_data JSON for dataset {dataset_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing row data JSON: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving fields for dataset {dataset_name}: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving fields: {str(e)}")

@app.post("/new")
async def upload_csv(
    request: Request, 
    file: UploadFile = File(None),
    dataset_name: str = Depends(get_dataset_name)
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
        try:
            # use utf-8-sig to automatically handle BOM (from Excel exports)
            csv_string = contents.decode('utf-8-sig')
            logger.info(f"Decoded CSV string length: {len(csv_string)}")
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error: {e}")
            # Try latin-1 as a fallback for older Excel CSVs
            try:
                csv_string = contents.decode('latin-1')
                logger.warning("Decoded with latin-1 fallback")
            except:
                raise HTTPException(status_code=400, detail="File must be UTF-8 or Latin-1 encoded")
        
        logger.info("Parsing CSV...")
        try:
            # Create a reader to get the fieldnames first
            f = io.StringIO(csv_string)
            raw_reader = csv.reader(f)
            headers = next(raw_reader, None)
            
            if not headers:
                raise HTTPException(status_code=400, detail="CSV file has no headers")
                
            # Clean headers: trim whitespace and remove BOM remnants if any
            clean_headers = [h.strip().lstrip('\ufeff') for h in headers]
            logger.info(f"Detected {len(clean_headers)} columns: {clean_headers[:5]}...")
            
            # Re-read using DictReader with cleaned headers
            f.seek(0)
            next(f) # skip raw header line
            csv_reader = csv.DictReader(f, fieldnames=clean_headers)
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        csv_table = get_table_name("csv_data", dataset_name)
        logger.info(f"Connecting to database: {DEFAULT_DB_NAME}, dataset: {dataset_name}, table: {csv_table}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (csv_table,))
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            cursor.execute(f"SELECT COUNT(*) as count FROM {csv_table}")
            row_count = cursor.fetchone()['count']
            
            if row_count > 0:
                logger.info(f"Table {csv_table} already exists with {row_count} rows. Dropping and re-creating for re-upload.")
                cursor.execute(f"DROP TABLE IF EXISTS {csv_table}")
                try:
                    cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{csv_table}'")
                except Exception:
                    pass
                conn.commit()
                conn.close()
                table_exists = False
        
        if not table_exists:
            init_db(dataset_name)
            conn = get_db_connection()
            cursor = conn.cursor()
        
        upload_timestamp = datetime.utcnow().isoformat()
        rows_inserted = 0
        
        logger.info("Inserting rows into database...")
        try:
            # Prepare batch for executemany
            batch = []
            for row in csv_reader:
                batch.append((upload_timestamp, json.dumps(row)))
                
                # Insert in chunks of 5000 to balance speed and memory
                if len(batch) >= 5000:
                    cursor.executemany(
                        f"INSERT INTO {csv_table} (upload_timestamp, row_data) VALUES (?, ?)",
                        batch
                    )
                    batch = []
                    rows_inserted += 5000
            
            # Insert remaining rows
            if batch:
                cursor.executemany(
                    f"INSERT INTO {csv_table} (upload_timestamp, row_data) VALUES (?, ?)",
                    batch
                )
                rows_inserted += len(batch)
            
            conn.commit()
            logger.info(f"Successfully inserted {rows_inserted} rows")
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Error inserting rows: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error inserting rows: {str(e)}")
        
        conn.close()
        
        logger.info(f"Successfully inserted {rows_inserted} rows")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully uploaded and stored {rows_inserted} rows from CSV file",
                "filename": filename,
                "upload_timestamp": upload_timestamp,
                "rows_inserted": rows_inserted,
                "dataset": dataset_name,
                "table": csv_table
            },
            status_code=201
        )
    
    except HTTPException:
        raise
    except csv.Error as e:
        logger.error(f"CSV parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/view")
async def view_data(
    limit: int = 100, 
    offset: int = 0,
    dataset_name: str = Depends(get_dataset_name)
):
    # Frontend controls limits - backend just validates reasonable bounds
    if limit < 1:
        limit = 100  # Default to 100 if invalid
    if limit > 10000:  # Cap at reasonable maximum to prevent abuse
        limit = 10000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing data: limit={limit}, offset={offset}")
    
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (csv_table,))
        if cursor.fetchone() is None:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found. Please upload data first using POST /new"
            )
        
        cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table}")
        total_count = cursor.fetchone()['total']
        
        try:
            cursor.execute(f"""
                SELECT id, upload_timestamp, row_data, T 
                FROM {csv_table} 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        except sqlite3.OperationalError:
            cursor.execute(f"""
                SELECT id, upload_timestamp, row_data 
                FROM {csv_table} 
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
    
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")


@app.get("/training")
async def get_training_data(
    limit: int = 100, 
    offset: int = 0,
    dataset_name: str = Depends(get_dataset_name)
):
    # Frontend controls limits - backend just validates reasonable bounds
    if limit < 1:
        limit = 100  # Default to 100 if invalid
    if limit > 10000:  # Cap at reasonable maximum to prevent abuse
        limit = 10000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing training data: limit={limit}, offset={offset}")
    
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({csv_table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table} WHERE T = ?", ("training",))
        total_count = cursor.fetchone()['total']
        
        try:
            cursor.execute(f"""
                SELECT COUNT(*) as count 
                FROM {csv_table} 
                WHERE T = ? AND (json_extract(row_data, '$.label') = '0' OR CAST(json_extract(row_data, '$.label') AS INTEGER) = 0)
            """, ("training",))
            label_0_count = cursor.fetchone()['count']
        except sqlite3.OperationalError:
            label_0_count = 0
        
        try:
            cursor.execute(f"""
                SELECT COUNT(*) as count 
                FROM {csv_table} 
                WHERE T = ? AND (json_extract(row_data, '$.label') = '1' OR CAST(json_extract(row_data, '$.label') AS INTEGER) = 1)
            """, ("training",))
            label_1_count = cursor.fetchone()['count']
        except sqlite3.OperationalError:
            label_1_count = 0
        
        cursor.execute(f"""
            SELECT id, upload_timestamp, row_data, T 
            FROM {csv_table} 
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
                "label_counts": {
                    "label_0": label_0_count,
                    "label_1": label_1_count
                },
                "data": data
            },
            status_code=200
        )
    
    except HTTPException:
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving training data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving training data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving training data: {str(e)}")


@app.get("/testing")
async def get_testing_data(
    limit: int = 100, 
    offset: int = 0,
    dataset_name: str = Depends(get_dataset_name)
):
    # Frontend controls limits - backend just validates reasonable bounds
    if limit < 1:
        limit = 100  # Default to 100 if invalid
    if limit > 10000:  # Cap at reasonable maximum to prevent abuse
        limit = 10000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing testing data: limit={limit}, offset={offset}")
    
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({csv_table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table} WHERE T = ?", ("testing",))
        total_count = cursor.fetchone()['total']
        
        try:
            cursor.execute(f"""
                SELECT COUNT(*) as count 
                FROM {csv_table} 
                WHERE T = ? AND (json_extract(row_data, '$.label') = '0' OR CAST(json_extract(row_data, '$.label') AS INTEGER) = 0)
            """, ("testing",))
            label_0_count = cursor.fetchone()['count']
        except sqlite3.OperationalError:
            label_0_count = 0
        
        try:
            cursor.execute(f"""
                SELECT COUNT(*) as count 
                FROM {csv_table} 
                WHERE T = ? AND (json_extract(row_data, '$.label') = '1' OR CAST(json_extract(row_data, '$.label') AS INTEGER) = 1)
            """, ("testing",))
            label_1_count = cursor.fetchone()['count']
        except sqlite3.OperationalError:
            label_1_count = 0
        
        cursor.execute(f"""
            SELECT id, upload_timestamp, row_data, T 
            FROM {csv_table} 
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
                "label_counts": {
                    "label_0": label_0_count,
                    "label_1": label_1_count
                },
                "data": data
            },
            status_code=200
        )
    
    except HTTPException:
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving testing data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving testing data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving testing data: {str(e)}")


@app.get("/random-test")
async def get_random_test_data(
    dataset_name: str = Depends(get_dataset_name)
):
    """
    Get a single random test data record using ORDER BY RANDOM() LIMIT 1.
    This is more efficient than using offset pagination for random selection.
    """
    logger.info(f"/random-test endpoint called with dataset_name: {dataset_name}")
    try:
        init_db(dataset_name)
        logger.info(f"Database initialized for dataset: {dataset_name}")
        
        csv_table = get_table_name("csv_data", dataset_name)
        logger.info(f"Using table: {csv_table}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({csv_table})")
        columns = [col[1] for col in cursor.fetchall()]
        logger.info(f"Table columns: {columns}")
        
        if 'T' not in columns:
            conn.close()
            logger.error(f"T column does not exist in table {csv_table}. Columns found: {columns}")
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        # Check total count of testing records
        cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table} WHERE T = ?", ("testing",))
        total_testing = cursor.fetchone()['total']
        logger.info(f"Total testing records in database: {total_testing}")
        
        if total_testing == 0:
            # Also check total records and what T values exist
            cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table}")
            total_all = cursor.fetchone()['total']
            logger.warning(f"No testing records found. Total records in table: {total_all}")
            
            # Check what T values exist
            cursor.execute(f"SELECT DISTINCT T FROM {csv_table}")
            t_values = [row['T'] for row in cursor.fetchall()]
            logger.warning(f"Existing T column values: {t_values}")
            
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"No testing data found in the database. Total records: {total_all}, T values: {t_values}"
            )
        
        # Get a random test record using ORDER BY RANDOM() LIMIT 1
        logger.info(f"Querying for random test record from {total_testing} testing records...")
        cursor.execute(f"""
            SELECT id, upload_timestamp, row_data, T 
            FROM {csv_table} 
            WHERE T = ?
            ORDER BY RANDOM()
            LIMIT 1
        """, ("testing",))
        
        row = cursor.fetchone()
        logger.info(f"Query executed, row fetched: {row is not None}")
        
        if not row:
            conn.close()
            logger.error("Query returned no row despite COUNT showing testing records exist")
            raise HTTPException(
                status_code=404,
                detail="No testing data found in the database."
            )
        
        logger.info(f"Successfully retrieved random test record with id: {row['id']}")
        conn.close()
        
        try:
            row_data = json.loads(row['row_data'])
            data = {
                "id": row['id'],
                "upload_timestamp": row['upload_timestamp'],
                "T": row['T'],
                "data": row_data
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for row {row['id']}: {e}")
            data = {
                "id": row['id'],
                "upload_timestamp": row['upload_timestamp'],
                "T": row['T'],
                "data": {"error": "Failed to parse row data", "raw": row['row_data']}
            }
        
        logger.info(f"Retrieved random test record (id: {row['id']}) from database")
        
        return JSONResponse(
            content={
                "status": "success",
                "data": data
            },
            status_code=200
        )
    
    except HTTPException:
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving random test data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving random test data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving random test data: {str(e)}")

@app.put("/validate")
async def validate_data(
    dataset_name: str = Depends(get_dataset_name),
    label_0_percent: Optional[float] = Header(None, alias="X-Label-0-Percent"),
    label_1_percent: Optional[float] = Header(None, alias="X-Label-1-Percent"),
    training_percent: Optional[float] = Header(None, alias="X-Training-Percent"),
    testing_percent: Optional[float] = Header(None, alias="X-Testing-Percent")
):
    """
    Validate and assign labels and training/testing split.
    
    Headers:
    - X-Label-0-Percent: Percentage of data to label as 0 (0-100). If not provided, labels are not modified.
    - X-Label-1-Percent: Percentage of data to label as 1 (0-100). If not provided, labels are not modified.
    - X-Training-Percent: Percentage of data to mark as training (0-100). Default: 80.
    - X-Testing-Percent: Percentage of data to mark as testing (0-100). Default: 20.
    """
    logger.info(f"Starting data validation and assignment for dataset: {dataset_name}")
    logger.info(f"Headers - Label 0%: {label_0_percent}, Label 1%: {label_1_percent}, Training%: {training_percent}, Testing%: {testing_percent}")
    
    # Validate header values are in valid range (0-100)
    if label_0_percent is not None and (label_0_percent < 0 or label_0_percent > 100):
        raise HTTPException(status_code=400, detail=f"X-Label-0-Percent must be between 0 and 100. Got: {label_0_percent}")
    if label_1_percent is not None and (label_1_percent < 0 or label_1_percent > 100):
        raise HTTPException(status_code=400, detail=f"X-Label-1-Percent must be between 0 and 100. Got: {label_1_percent}")
    if training_percent is not None and (training_percent < 0 or training_percent > 100):
        raise HTTPException(status_code=400, detail=f"X-Training-Percent must be between 0 and 100. Got: {training_percent}")
    if testing_percent is not None and (testing_percent < 0 or testing_percent > 100):
        raise HTTPException(status_code=400, detail=f"X-Testing-Percent must be between 0 and 100. Got: {testing_percent}")
    
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Add T column if it doesn't exist
        try:
            cursor.execute(f"ALTER TABLE {csv_table} ADD COLUMN T TEXT")
            logger.info(f"Added T column to {csv_table} table")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("T column already exists")
            else:
                raise
        
        # Fetch all rows with their data
        cursor.execute(f"SELECT id, row_data FROM {csv_table}")
        all_rows = cursor.fetchall()
        total_rows = len(all_rows)
        
        if total_rows == 0:
            logger.warning("No rows found in database")
            conn.close()
            raise HTTPException(
                status_code=422,
                detail="No data found in dataset. Cannot perform validation on empty dataset."
            )
        
        # Validate and set training/testing percentages
        if training_percent is not None and testing_percent is not None:
            if abs(training_percent + testing_percent - 100.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail=f"Training and testing percentages must sum to 100. Got training={training_percent}%, testing={testing_percent}%"
                )
            train_pct = training_percent / 100.0
            test_pct = testing_percent / 100.0
        elif training_percent is not None:
            train_pct = training_percent / 100.0
            test_pct = 1.0 - train_pct
        elif testing_percent is not None:
            test_pct = testing_percent / 100.0
            train_pct = 1.0 - test_pct
        else:
            # Default: 80% training, 20% testing
            train_pct = 0.8
            test_pct = 0.2
        
        # Validate label percentages if provided
        if label_0_percent is not None and label_1_percent is not None:
            if abs(label_0_percent + label_1_percent - 100.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail=f"Label 0 and label 1 percentages must sum to 100. Got label_0={label_0_percent}%, label_1={label_1_percent}%"
                )
            label_0_pct = label_0_percent / 100.0
            label_1_pct = label_1_percent / 100.0
            assign_labels = True
        elif label_0_percent is not None:
            label_0_pct = label_0_percent / 100.0
            label_1_pct = 1.0 - label_0_pct
            assign_labels = True
        elif label_1_percent is not None:
            label_1_pct = label_1_percent / 100.0
            label_0_pct = 1.0 - label_1_pct
            assign_labels = True
        else:
            assign_labels = False
        
        # Calculate counts
        training_count = int(total_rows * train_pct)
        testing_count = total_rows - training_count
        
        if assign_labels:
            label_0_count = int(total_rows * label_0_pct)
            label_1_count = total_rows - label_0_count
            logger.info(f"Total rows: {total_rows}, Label 0: {label_0_count}, Label 1: {label_1_count}, Training: {training_count}, Testing: {testing_count}")
        else:
            logger.info(f"Total rows: {total_rows}, Training: {training_count}, Testing: {testing_count} (labels not modified)")
        
        # Separate rows by existing label for stratified split
        label_0_rows = []
        label_1_rows = []
        label_unknown_rows = []
        
        for row in all_rows:
            try:
                row_data = json.loads(row['row_data'])
                label = row_data.get('label')
                if label == 0 or label == '0':
                    label_0_rows.append((row['id'], row['row_data']))
                elif label == 1 or label == '1':
                    label_1_rows.append((row['id'], row['row_data']))
                else:
                    label_unknown_rows.append((row['id'], row['row_data']))
            except (json.JSONDecodeError, KeyError):
                label_unknown_rows.append((row['id'], row['row_data']))
        
        # Shuffle each group separately for stratified split
        random.shuffle(label_0_rows)
        random.shuffle(label_1_rows)
        random.shuffle(label_unknown_rows)
        
        # Begin transaction for atomic label and training/testing assignment
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Assign labels if requested
            if assign_labels:
                # Combine all rows and shuffle for label assignment
                all_rows_list = label_0_rows + label_1_rows + label_unknown_rows
                random.shuffle(all_rows_list)
                
                label_0_assigned = all_rows_list[:label_0_count]
                label_1_assigned = all_rows_list[label_0_count:]
                
                # BATCH UPDATE LABELS
                label_0_updates = []
                for row_id, row_data_json in label_0_assigned:
                    try:
                        row_data = json.loads(row_data_json)
                        row_data['label'] = 0
                        label_0_updates.append((json.dumps(row_data), row_id))
                    except: pass
                
                if label_0_updates:
                    cursor.executemany(f"UPDATE {csv_table} SET row_data = ? WHERE id = ?", label_0_updates)

                label_1_updates = []
                for row_id, row_data_json in label_1_assigned:
                    try:
                        row_data = json.loads(row_data_json)
                        row_data['label'] = 1
                        label_1_updates.append((json.dumps(row_data), row_id))
                    except: pass
                
                if label_1_updates:
                    cursor.executemany(f"UPDATE {csv_table} SET row_data = ? WHERE id = ?", label_1_updates)
                
                # Re-separate for split logic
                label_0_rows = label_0_assigned
                label_1_rows = label_1_assigned
                label_unknown_rows = []
            
            # Stratified split calculations
            label_0_train_count = int(len(label_0_rows) * train_pct)
            label_1_train_count = int(len(label_1_rows) * train_pct)
            unknown_train_count = int(len(label_unknown_rows) * train_pct)
            
            training_ids = [r[0] for r in label_0_rows[:label_0_train_count]] + \
                           [r[0] for r in label_1_rows[:label_1_train_count]] + \
                           [r[0] for r in label_unknown_rows[:unknown_train_count]]
            
            testing_ids = [r[0] for r in label_0_rows[label_0_train_count:]] + \
                          [r[0] for r in label_1_rows[label_1_train_count:]] + \
                          [r[0] for r in label_unknown_rows[unknown_train_count:]]
            
            # BATCH UPDATE SPLITS
            if training_ids:
                cursor.executemany(f"UPDATE {csv_table} SET T = 'training' WHERE id = ?", [(rid,) for rid in training_ids])
            if testing_ids:
                cursor.executemany(f"UPDATE {csv_table} SET T = 'testing' WHERE id = ?", [(rid,) for rid in testing_ids])
            
            conn.commit()
            updated_training = len(training_ids)
            updated_testing = len(testing_ids)
            logger.info(f"Validation complete: {updated_training} training, {updated_testing} testing")
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Error during validation transaction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during validation: {str(e)}")
        
        conn.close()
        
        result = {
            "status": "success",
            "message": "Data validation and assignment completed",
            "total_rows": total_rows,
            "training_rows": updated_training,
            "testing_rows": updated_testing,
            "training_percentage": round((updated_training / total_rows) * 100, 2),
            "testing_percentage": round((updated_testing / total_rows) * 100, 2)
        }
        
        if assign_labels:
            result["label_0_rows"] = label_0_count
            result["label_1_rows"] = label_1_count
            result["label_0_percentage"] = round((label_0_count / total_rows) * 100, 2)
            result["label_1_percentage"] = round((label_1_count / total_rows) * 100, 2)
        
        logger.info(f"Validation complete: {updated_training} training, {updated_testing} testing")
        if assign_labels:
            logger.info(f"Labels assigned: {label_0_count} labeled as 0, {label_1_count} labeled as 1")
        
        return JSONResponse(
            content=result,
            status_code=200
        )
    
    except HTTPException:
        raise
    except sqlite3.Error as e:
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        logger.error(f"Database error during validation: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        logger.error(f"Error during validation: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during validation: {str(e)}")


@app.delete("/clear")
async def clear_database(dataset_name: Optional[str] = Depends(get_optional_dataset_name)):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if dataset_name is None:
            logger.warning("Dropping ALL tables - all tables will be removed")
            
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
            """)
            
            all_tables = [row['name'] for row in cursor.fetchall()]
            total_rows_deleted = 0
            deleted_tables = []
            
            try:
                cursor.execute("BEGIN TRANSACTION")
                for table_name in all_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) as total FROM {table_name}")
                        row_count = cursor.fetchone()['total']
                        
                        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
                        
                        total_rows_deleted += row_count
                        deleted_tables.append(table_name)
                        logger.info(f"Dropped table {table_name}: {row_count} rows")
                    except Exception as e:
                        logger.warning(f"Error dropping table {table_name}: {e}")
                        continue
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                conn.close()
                logger.error(f"Error during clear operation: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")
            
            conn.close()
            
            logger.info(f"All tables dropped: {total_rows_deleted} total rows from {len(deleted_tables)} tables")
            
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "All tables dropped successfully",
                    "tables_dropped": deleted_tables,
                    "total_tables": len(deleted_tables),
                    "total_rows_deleted": total_rows_deleted
                },
                status_code=200
            )
        else:
            logger.warning(f"Dropping dataset {dataset_name} tables - tables will be removed")
            
            csv_table = get_table_name("csv_data", dataset_name)
            inserted_table = get_table_name("inserted_data", dataset_name)
            
            total_rows_deleted = 0
            deleted_tables = []
            
            try:
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table}")
                    csv_rows = cursor.fetchone()['total']
                    cursor.execute(f"DROP TABLE IF EXISTS {csv_table}")
                    cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{csv_table}'")
                    total_rows_deleted += csv_rows
                    deleted_tables.append(csv_table)
                    logger.info(f"Dropped table {csv_table}: {csv_rows} rows")
                except sqlite3.OperationalError:
                    logger.info(f"Table {csv_table} does not exist, skipping")
                
                try:
                    cursor.execute(f"SELECT COUNT(*) as total FROM {inserted_table}")
                    inserted_rows = cursor.fetchone()['total']
                    cursor.execute(f"DROP TABLE IF EXISTS {inserted_table}")
                    cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{inserted_table}'")
                    total_rows_deleted += inserted_rows
                    deleted_tables.append(inserted_table)
                    logger.info(f"Dropped table {inserted_table}: {inserted_rows} rows")
                except sqlite3.OperationalError:
                    logger.info(f"Table {inserted_table} does not exist, skipping")
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                conn.close()
                logger.error(f"Error during clear operation: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")
            
            conn.close()
            
            logger.info(f"Dataset {dataset_name} tables dropped: {total_rows_deleted} rows from {len(deleted_tables)} tables")
            
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"Dataset {dataset_name} tables dropped successfully",
                    "dataset": dataset_name,
                    "tables_dropped": deleted_tables,
                    "total_rows_deleted": total_rows_deleted
                },
                status_code=200
            )
    
    except HTTPException:
        if 'conn' in locals():
            conn.close()
        raise
    except sqlite3.Error as e:
        if 'conn' in locals():
            conn.close()
        logger.error(f"Database error clearing data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        logger.error(f"Error clearing database: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")


@app.get("/stats")
async def get_stats(dataset_name: str = Depends(get_dataset_name)):
    """Get aggregated statistics for KPI display"""
    try:
        stats = {}
        
        csv_table = get_table_name("csv_data", dataset_name)
        
        # Get total records
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table}")
            stats['total_records'] = cursor.fetchone()['total']
            conn.close()
        except Exception as e:
            logger.warning(f"Error getting total records: {e}")
            stats['total_records'] = 0
        
        # Get training records count
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({csv_table})")
            columns = [col[1] for col in cursor.fetchall()]
            if 'T' in columns:
                cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table} WHERE T = ?", ("training",))
                stats['training_records'] = cursor.fetchone()['total']
            else:
                stats['training_records'] = 0
            conn.close()
        except Exception as e:
            logger.warning(f"Error getting training records: {e}")
            stats['training_records'] = 0
        
        # Get testing records count
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({csv_table})")
            columns = [col[1] for col in cursor.fetchall()]
            if 'T' in columns:
                cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table} WHERE T = ?", ("testing",))
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


def fetch_chunk_data(dataset_name, offset, limit):
    """Fetch a chunk of data from the database"""
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"""
                SELECT id, upload_timestamp, row_data, T 
                FROM {csv_table} 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        except sqlite3.OperationalError:
            cursor.execute(f"""
                SELECT id, upload_timestamp, row_data 
                FROM {csv_table} 
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


@app.get("/type-stats")
async def get_type_stats(dataset_name: str = Depends(get_dataset_name)):
    """Get type distribution statistics - processes all rows to find all types"""
    try:
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table}")
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
            cursor.execute(f"""
                SELECT id, upload_timestamp, row_data, T 
                FROM {csv_table} 
                ORDER BY id
            """)
        except sqlite3.OperationalError:
            # T column might not exist
            cursor.execute(f"""
                SELECT id, upload_timestamp, row_data 
                FROM {csv_table} 
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
