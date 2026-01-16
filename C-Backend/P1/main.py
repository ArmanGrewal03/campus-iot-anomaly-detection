from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header, Depends
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

app = FastAPI(title="Campus IoT Anomaly Detection API", version="1.0.0")

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


@app.get("/health")
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


@app.get("/view")
async def view_data(
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
        
        created_timestamp = datetime.utcnow().isoformat()
        
        data_json = json.dumps(data)
        
        cursor.execute("""
            INSERT INTO inserted_data (created_timestamp, data)
            VALUES (?, ?)
        """, (created_timestamp, data_json))
        
        inserted_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully inserted row with ID: {inserted_id}")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Row inserted successfully",
                "id": inserted_id,
                "created_timestamp": created_timestamp,
                "data": data
            },
            status_code=201
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inserting data: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error inserting data: {str(e)}")
