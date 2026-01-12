from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
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

app = FastAPI(title="Campus IoT Anomaly Detection API", version="1.0.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "campus_iot_data.db"

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Request headers: {dict(request.headers)}")
    logger.error(f"Validation errors: {exc.errors()}")
    
    # Try to log the body if possible
    try:
        body = await request.body()
        logger.error(f"Request body type: {type(body)}")
        logger.error(f"Request body length: {len(body) if body else 0}")
        if body:
            # For binary data, show first bytes as hex
            body_preview = body[:500] if isinstance(body, bytes) else str(body)[:500]
            logger.error(f"Request body preview: {body_preview}")
    except Exception as e:
        logger.error(f"Could not read request body: {e}")
    
    return await request_validation_exception_handler(request, exc)

@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup."""
    init_db()

def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create table to store CSV data
    # Using a flexible schema that can handle any CSV structure
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS csv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_timestamp TEXT NOT NULL,
            row_data TEXT NOT NULL
        )
    """)
    
    # Add T column if it doesn't exist (for training/testing assignment)
    try:
        cursor.execute("ALTER TABLE csv_data ADD COLUMN T TEXT")
        logger.info("Added T column to csv_data table")
    except sqlite3.OperationalError:
        # Column already exists, which is fine
        pass
    
    # Create table for inserted data (flexible schema using JSON)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inserted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_timestamp TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server is healthy.
    Returns server status and timestamp.
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Campus IoT Anomaly Detection API"
        },
        status_code=200
    )


@app.post("/new")
async def upload_csv(request: Request, file: UploadFile = File(None)):
    """
    Upload a binary CSV file and store its data in SQLite database.
    Accepts a CSV file as binary data and inserts all rows into the database.
    
    Supports two formats:
    1. multipart/form-data with field name "file" (recommended)
    2. Raw CSV data with Content-Type: text/csv
    
    Example using curl (multipart):
    curl -X POST "http://localhost:8000/new" -F "file=@yourfile.csv"
    
    Example using curl (raw CSV):
    curl -X POST "http://localhost:8000/new" -H "Content-Type: text/csv" --data-binary @yourfile.csv
    
    Example using PowerShell:
    $form = @{ file = Get-Item "yourfile.csv" }
    Invoke-RestMethod -Uri "http://localhost:8000/new" -Method Post -Form $form
    """
    logger.info(f"Received file upload request")
    logger.info(f"Content-Type header: {request.headers.get('content-type', 'not set')}")
    
    contents = None
    filename = None
    
    # Check if request is multipart/form-data (file upload) or raw CSV
    content_type = request.headers.get('content-type', '').lower()
    
    if file is not None:
        # Handle multipart/form-data upload
        logger.info("Processing multipart/form-data upload")
        logger.info(f"Filename: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        
        filename = file.filename or "uploaded_file.csv"
        
        # Validate file type
        if file.filename and not file.filename.endswith('.csv'):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read the binary file content
        logger.info("Reading file contents...")
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file")
        
    elif 'text/csv' in content_type or 'application/csv' in content_type:
        # Handle raw CSV data
        logger.info("Processing raw CSV data upload")
        filename = "raw_upload.csv"
        
        # Read the raw request body
        logger.info("Reading raw request body...")
        contents = await request.body()
        logger.info(f"Read {len(contents)} bytes from request body")
        
    else:
        # Try to read as raw data if no content-type is set
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
        
        # Check if file is empty
        if not contents:
            logger.error("Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Decode the binary content to string
        logger.info("Decoding file contents...")
        csv_string = contents.decode('utf-8')
        logger.info(f"Decoded CSV string length: {len(csv_string)}")
        
        # Parse CSV using StringIO
        logger.info("Parsing CSV...")
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        
        # Get database connection
        logger.info("Connecting to database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        upload_timestamp = datetime.utcnow().isoformat()
        rows_inserted = 0
        
        # Insert each row into the database
        logger.info("Inserting rows into database...")
        for row in csv_reader:
            # Convert row dictionary to JSON string for storage
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
async def view_data(limit: int = 100, offset: int = 0):
    """
    View data stored in the SQLite database.
    
    Query parameters:
    - limit: Maximum number of rows to return (default: 100, max: 1000)
    - offset: Number of rows to skip (default: 0)
    
    Returns all stored CSV data with pagination support.
    """
    # Validate limit
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing data: limit={limit}, offset={offset}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM csv_data")
        total_count = cursor.fetchone()['total']
        
        # Get data with pagination (include T column if it exists)
        try:
            cursor.execute("""
                SELECT id, upload_timestamp, row_data, T 
                FROM csv_data 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        except sqlite3.OperationalError:
            # T column doesn't exist yet, select without it
            cursor.execute("""
                SELECT id, upload_timestamp, row_data 
                FROM csv_data 
                ORDER BY id 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Parse the data
        data = []
        for row in rows:
            try:
                # Parse the JSON row_data back into a dictionary
                row_data = json.loads(row['row_data'])
                row_dict = {
                    "id": row['id'],
                    "upload_timestamp": row['upload_timestamp'],
                    "data": row_data
                }
                # Add T field if it exists
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
                # Add T field if it exists
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
async def get_training_data(limit: int = 100, offset: int = 0):
    """
    View data labeled as "training" from the SQLite database.
    
    Query parameters:
    - limit: Maximum number of rows to return (default: 100, max: 1000)
    - offset: Number of rows to skip (default: 0)
    
    Returns only rows where T = "training".
    """
    # Validate limit
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing training data: limit={limit}, offset={offset}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if T column exists
        cursor.execute("PRAGMA table_info(csv_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        # Get total count of training rows
        cursor.execute("SELECT COUNT(*) as total FROM csv_data WHERE T = ?", ("training",))
        total_count = cursor.fetchone()['total']
        
        # Get training data with pagination
        cursor.execute("""
            SELECT id, upload_timestamp, row_data, T 
            FROM csv_data 
            WHERE T = ?
            ORDER BY id 
            LIMIT ? OFFSET ?
        """, ("training", limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Parse the data
        data = []
        for row in rows:
            try:
                # Parse the JSON row_data back into a dictionary
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
async def get_testing_data(limit: int = 100, offset: int = 0):
    """
    View data labeled as "testing" from the SQLite database.
    
    Query parameters:
    - limit: Maximum number of rows to return (default: 100, max: 1000)
    - offset: Number of rows to skip (default: 0)
    
    Returns only rows where T = "testing".
    """
    # Validate limit
    if limit < 1:
        limit = 100
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0
    
    logger.info(f"Viewing testing data: limit={limit}, offset={offset}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if T column exists
        cursor.execute("PRAGMA table_info(csv_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail="T column does not exist. Please call PUT /validate first to assign training/testing labels."
            )
        
        # Get total count of testing rows
        cursor.execute("SELECT COUNT(*) as total FROM csv_data WHERE T = ?", ("testing",))
        total_count = cursor.fetchone()['total']
        
        # Get testing data with pagination
        cursor.execute("""
            SELECT id, upload_timestamp, row_data, T 
            FROM csv_data 
            WHERE T = ?
            ORDER BY id 
            LIMIT ? OFFSET ?
        """, ("testing", limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Parse the data
        data = []
        for row in rows:
            try:
                # Parse the JSON row_data back into a dictionary
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
async def validate_data():
    """
    Assign training/testing labels to all rows in the database.
    Adds a "T" field if it doesn't exist, then randomly assigns:
    - 30% of rows as "training"
    - 70% of rows as "testing"
    
    This assignment is randomized every time the endpoint is called.
    """
    logger.info("Starting data validation and assignment")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure T column exists
        try:
            cursor.execute("ALTER TABLE csv_data ADD COLUMN T TEXT")
            logger.info("Added T column to csv_data table")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.info("T column already exists")
            else:
                raise
        
        # Get all rows
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
        
        # Calculate split: 30% training, 70% testing
        training_count = int(total_rows * 0.3)
        testing_count = total_rows - training_count
        
        logger.info(f"Total rows: {total_rows}, Training: {training_count}, Testing: {testing_count}")
        
        # Create a list of all row IDs and shuffle it
        row_ids = [row['id'] for row in all_rows]
        random.shuffle(row_ids)
        
        # Assign first 30% as training, rest as testing
        training_ids = set(row_ids[:training_count])
        testing_ids = set(row_ids[training_count:])
        
        # Update rows in database
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
async def clear_database():
    """
    Clear all data from the database tables.
    Deletes all rows from the csv_data table.
    
    WARNING: This operation cannot be undone!
    """
    logger.warning("Clearing database - all data will be deleted")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get count before deletion for logging
        cursor.execute("SELECT COUNT(*) as total FROM csv_data")
        total_rows = cursor.fetchone()['total']
        
        # Delete all rows from csv_data table
        cursor.execute("DELETE FROM csv_data")
        
        # Reset the auto-increment counter (optional, but good practice)
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='csv_data'")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database cleared: {total_rows} rows deleted")
        
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
async def insert_data(data: Dict[str, Any]):
    """
    Insert a new row into the inserted_data table.
    
    The request body should be a JSON object where each key-value pair
    represents a field in the data. All fields will be stored as JSON.
    
    Example request body:
    {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "status": "active"
    }
    
    Returns the inserted row with its ID and timestamp.
    """
    logger.info(f"Inserting new row with fields: {list(data.keys())}")
    
    try:
        # Validate that data is not empty
        if not data:
            raise HTTPException(status_code=400, detail="Request body cannot be empty")
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create timestamp
        created_timestamp = datetime.utcnow().isoformat()
        
        # Convert data to JSON string
        data_json = json.dumps(data)
        
        # Insert into database
        cursor.execute("""
            INSERT INTO inserted_data (created_timestamp, data)
            VALUES (?, ?)
        """, (created_timestamp, data_json))
        
        # Get the inserted row ID
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
