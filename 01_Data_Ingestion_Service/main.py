from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
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

app = FastAPI(title="Campus IoT Anomaly Detection API", version="1.0.0")

# Add CORS middleware to allow React frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React dev server
        "http://localhost:8080",  # Vue CLI dev server
        "http://127.0.0.1:5173",
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

def get_dataset_name(dataset_name: str = Header(..., alias="dataset_name")) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '', dataset_name)
    
    if not sanitized or len(sanitized) < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset name '{dataset_name}'. Dataset name must contain at least one alphanumeric character, underscore, or hyphen."
        )
    
    logger.info(f"Using dataset: {sanitized}")
    return sanitized

def get_optional_dataset_name(dataset_name: Optional[str] = Header(None, alias="dataset_name")) -> Optional[str]:
    if dataset_name is None:
        return None
    
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '', dataset_name)
    
    if not sanitized or len(sanitized) < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset name '{dataset_name}'. Dataset name must contain at least one alphanumeric character, underscore, or hyphen."
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
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {inserted_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_timestamp TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized tables for dataset: {dataset_name}")

@app.on_event("startup")
async def startup_event():
    pass


@app.get("/health")
async def health_check(dataset_name: str = Depends(get_dataset_name)):
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Campus IoT Anomaly Detection API",
            "database": DEFAULT_DB_NAME,
            "dataset": dataset_name
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
        
        tables = [row['name'] for row in cursor.fetchall()]
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
            csv_string = contents.decode('utf-8')
            logger.info(f"Decoded CSV string length: {len(csv_string)}")
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error: {e}")
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
        
        logger.info("Parsing CSV...")
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        
        init_db(dataset_name)
        
        csv_table = get_table_name("csv_data", dataset_name)
        logger.info(f"Connecting to database: {DEFAULT_DB_NAME}, dataset: {dataset_name}, table: {csv_table}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        upload_timestamp = datetime.utcnow().isoformat()
        rows_inserted = 0
        
        logger.info("Inserting rows into database...")
        for row in csv_reader:
            row_json = json.dumps(row)
            cursor.execute(
                f"INSERT INTO {csv_table} (upload_timestamp, row_data) VALUES (?, ?)",
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
                "rows_inserted": rows_inserted,
                "dataset": dataset_name,
                "table": csv_table
            },
            status_code=200
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
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing data: limit={limit}, offset={offset}")
    
    try:
        init_db(dataset_name)
        
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
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
    
    except Exception as e:
        logger.error(f"Error retrieving data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")


@app.get("/training")
async def get_training_data(
    limit: int = 100, 
    offset: int = 0,
    dataset_name: str = Depends(get_dataset_name)
):
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing training data: limit={limit}, offset={offset}")
    
    try:
        init_db(dataset_name)
        
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
    dataset_name: str = Depends(get_dataset_name)
):
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing testing data: limit={limit}, offset={offset}")
    
    try:
        init_db(dataset_name)
        
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
async def validate_data(dataset_name: str = Depends(get_dataset_name)):
    logger.info(f"Starting data validation and assignment for dataset: {dataset_name}")
    
    try:
        init_db(dataset_name)
        
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"ALTER TABLE {csv_table} ADD COLUMN T TEXT")
            logger.info(f"Added T column to {csv_table} table")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("T column already exists")
            else:
                raise
        
        cursor.execute(f"SELECT id FROM {csv_table}")
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
        
        training_count = int(total_rows * 0.8)
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
                cursor.execute(f"UPDATE {csv_table} SET T = ? WHERE id = ?", ("training", row_id))
                updated_training += 1
            else:
                cursor.execute(f"UPDATE {csv_table} SET T = ? WHERE id = ?", ("testing", row_id))
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
            
            init_db(dataset_name)
            
            csv_table = get_table_name("csv_data", dataset_name)
            inserted_table = get_table_name("inserted_data", dataset_name)
            
            total_rows_deleted = 0
            deleted_tables = []
            
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
        conn.close()
        raise
    except Exception as e:
        conn.close()
        logger.error(f"Error clearing database: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")


@app.get("/api/stats")
async def get_stats(dataset_name: str = Depends(get_dataset_name)):
    """Get aggregated statistics for KPI display"""
    try:
        init_db(dataset_name)
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


@app.get("/api/type-stats")
async def get_type_stats(dataset_name: str = Depends(get_dataset_name)):
    """Get type distribution statistics - processes all rows to find all types"""
    try:
        init_db(dataset_name)
        
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
