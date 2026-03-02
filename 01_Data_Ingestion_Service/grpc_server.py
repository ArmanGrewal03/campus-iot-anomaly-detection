"""
gRPC Server for Data Ingestion Service
Handles inter-service communication via gRPC
Runs alongside FastAPI server
"""

import grpc
from concurrent import futures
import logging
import os
import sys
import json
import sqlite3

# Import existing functions from main.py
sys.path.insert(0, os.path.dirname(__file__))
try:
    from main import (
        get_table_name, get_db_connection, init_db,
        logger as main_logger, DEFAULT_DB_NAME
    )
except ImportError:
    main_logger = logging.getLogger(__name__)
    DEFAULT_DB_NAME = "campus_iot_data.db"

# Import generated gRPC code (will be generated from proto file)
try:
    import data_ingestion_pb2
    import data_ingestion_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    main_logger.warning(
        "gRPC code not generated. Run: "
        "python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/data_ingestion.proto"
    )
    data_ingestion_pb2 = None
    data_ingestion_pb2_grpc = None

logger = main_logger


def get_training_data_grpc(dataset_name: str, limit: int, offset: int):
    """Get training data - matches HTTP endpoint logic"""
    try:
        init_db(dataset_name)
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({csv_table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise ValueError("T column does not exist. Please call PUT /validate first.")
        
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
                record = {
                    "id": str(row['id']),
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    **{str(k): str(v) for k, v in row_data.items()}
                }
                data.append(record)
            except json.JSONDecodeError:
                record = {
                    "id": str(row['id']),
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    "error": "Failed to parse row data"
                }
                data.append(record)
        
        return data, total_count
    except Exception as e:
        logger.error(f"Error in get_training_data_grpc: {e}")
        raise


def get_testing_data_grpc(dataset_name: str, limit: int, offset: int):
    """Get testing data - matches HTTP endpoint logic"""
    try:
        init_db(dataset_name)
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({csv_table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'T' not in columns:
            conn.close()
            raise ValueError("T column does not exist. Please call PUT /validate first.")
        
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
                record = {
                    "id": str(row['id']),
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    **{str(k): str(v) for k, v in row_data.items()}
                }
                data.append(record)
            except json.JSONDecodeError:
                record = {
                    "id": str(row['id']),
                    "upload_timestamp": row['upload_timestamp'],
                    "T": row['T'],
                    "error": "Failed to parse row data"
                }
                data.append(record)
        
        return data, total_count
    except Exception as e:
        logger.error(f"Error in get_testing_data_grpc: {e}")
        raise


def get_view_data_grpc(dataset_name: str, limit: int, offset: int):
    """Get view data - matches HTTP endpoint logic"""
    try:
        init_db(dataset_name)
        csv_table = get_table_name("csv_data", dataset_name)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) as total FROM {csv_table}")
        total_count = cursor.fetchone()['total']
        
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
                record = {
                    "id": str(row['id']),
                    "upload_timestamp": row['upload_timestamp'],
                    **{str(k): str(v) for k, v in row_data.items()}
                }
                data.append(record)
            except json.JSONDecodeError:
                record = {
                    "id": str(row['id']),
                    "upload_timestamp": row['upload_timestamp'],
                    "error": "Failed to parse row data"
                }
                data.append(record)
        
        return data, total_count
    except Exception as e:
        logger.error(f"Error in get_view_data_grpc: {e}")
        raise


if GRPC_AVAILABLE:
    class DataIngestionServicer(data_ingestion_pb2_grpc.DataIngestionServiceServicer):
        """gRPC servicer for Data Ingestion Service"""
        
        def GetTrainingData(self, request, context):
            """Fetch training data"""
            try:
                dataset_name = request.dataset_name if request.dataset_name else "default"
                data, total_rows = get_training_data_grpc(
                    dataset_name,
                    limit=request.limit,
                    offset=request.offset
                )
                
                # Convert to proto format
                records = []
                for record in data:
                    proto_record = data_ingestion_pb2.DataRecord()
                    proto_record.fields.update({k: str(v) for k, v in record.items()})
                    records.append(proto_record)
                
                return data_ingestion_pb2.DataResponse(
                    status="success",
                    data=records,
                    total_rows=total_rows
                )
            except Exception as e:
                logger.error(f"Error in GetTrainingData: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return data_ingestion_pb2.DataResponse(
                    status="error",
                    message=str(e)
                )
        
        def GetTestingData(self, request, context):
            """Fetch testing data"""
            try:
                dataset_name = request.dataset_name if request.dataset_name else "default"
                data, total_rows = get_testing_data_grpc(
                    dataset_name,
                    limit=request.limit,
                    offset=request.offset
                )
                
                # Convert to proto format
                records = []
                for record in data:
                    proto_record = data_ingestion_pb2.DataRecord()
                    proto_record.fields.update({k: str(v) for k, v in record.items()})
                    records.append(proto_record)
                
                return data_ingestion_pb2.DataResponse(
                    status="success",
                    data=records,
                    total_rows=total_rows
                )
            except Exception as e:
                logger.error(f"Error in GetTestingData: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return data_ingestion_pb2.DataResponse(
                    status="error",
                    message=str(e)
                )
        
        def GetViewData(self, request, context):
            """Fetch all data (view)"""
            try:
                dataset_name = request.dataset_name if request.dataset_name else "default"
                data, total_rows = get_view_data_grpc(
                    dataset_name,
                    limit=request.limit,
                    offset=request.offset
                )
                
                # Convert to proto format
                records = []
                for record in data:
                    proto_record = data_ingestion_pb2.DataRecord()
                    proto_record.fields.update({k: str(v) for k, v in record.items()})
                    records.append(proto_record)
                
                return data_ingestion_pb2.DataResponse(
                    status="success",
                    data=records,
                    total_rows=total_rows
                )
            except Exception as e:
                logger.error(f"Error in GetViewData: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return data_ingestion_pb2.DataResponse(
                    status="error",
                    message=str(e)
                )
        
        def HealthCheck(self, request, context):
            """Health check endpoint"""
            return data_ingestion_pb2.HealthResponse(
                status="healthy",
                message="Data Ingestion Service is running"
            )


def serve():
    """Start the gRPC server"""
    if not GRPC_AVAILABLE:
        logger.error("gRPC code not available. Cannot start gRPC server.")
        return
    
    port = os.getenv("GRPC_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    data_ingestion_pb2_grpc.add_DataIngestionServiceServicer_to_server(
        DataIngestionServicer(), server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server shutting down")
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
