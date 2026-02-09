from flask import Flask, render_template, request, jsonify, send_file
import requests
import json
import os
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload CSV file to /new endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Send file to FastAPI endpoint
        files = {'file': (file.filename, file.stream, 'text/csv')}
        response = requests.post(f"{API_BASE_URL}/new", files=files)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/view', methods=['GET'])
def view_data():
    """View all data from /view endpoint"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        params = {'limit': limit, 'offset': offset}
        response = requests.get(f"{API_BASE_URL}/view", params=params)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate', methods=['PUT'])
def validate_data():
    """Assign training/testing labels via /validate endpoint"""
    try:
        response = requests.put(f"{API_BASE_URL}/validate")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/training', methods=['GET'])
def get_training():
    """Get training data from /training endpoint"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        params = {'limit': limit, 'offset': offset}
        response = requests.get(f"{API_BASE_URL}/training", params=params)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/testing', methods=['GET'])
def get_testing():
    """Get testing data from /testing endpoint"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        params = {'limit': limit, 'offset': offset}
        response = requests.get(f"{API_BASE_URL}/testing", params=params)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_database():
    """Clear database via /clear endpoint"""
    try:
        response = requests.post(f"{API_BASE_URL}/clear")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/insert', methods=['POST'])
def insert_data():
    """Insert new row via /insert endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        response = requests.post(f"{API_BASE_URL}/insert", json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get aggregated statistics for KPI display"""
    try:
        stats = {}
        
        # Get total records
        try:
            response = requests.get(f"{API_BASE_URL}/view", params={'limit': 1, 'offset': 0})
            if response.status_code == 200:
                data = response.json()
                stats['total_records'] = data.get('total_rows', 0)
            else:
                stats['total_records'] = 0
        except:
            stats['total_records'] = 0
        
        # Get training records count
        try:
            response = requests.get(f"{API_BASE_URL}/training", params={'limit': 1, 'offset': 0})
            if response.status_code == 200:
                data = response.json()
                stats['training_records'] = data.get('total_rows', 0)
            else:
                stats['training_records'] = 0
        except:
            stats['training_records'] = 0
        
        # Get testing records count
        try:
            response = requests.get(f"{API_BASE_URL}/testing", params={'limit': 1, 'offset': 0})
            if response.status_code == 200:
                data = response.json()
                stats['testing_records'] = data.get('total_rows', 0)
            else:
                stats['testing_records'] = 0
        except:
            stats['testing_records'] = 0
        
        # Calculate percentages if total > 0
        if stats['total_records'] > 0:
            stats['training_percentage'] = round((stats['training_records'] / stats['total_records']) * 100, 1)
            stats['testing_percentage'] = round((stats['testing_records'] / stats['total_records']) * 100, 1)
        else:
            stats['training_percentage'] = 0
            stats['testing_percentage'] = 0
        
        # Check API health
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            stats['api_online'] = health_response.status_code == 200
        except:
            stats['api_online'] = False
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_types_from_chunk(chunk_data, offset):
    """Extract type values and training/testing labels from a chunk of data"""
    type_counts = {}
    type_training = {}  # Count of each type in training set
    type_testing = {}   # Count of each type in testing set
    processed_count = 0
    
    if 'data' in chunk_data and isinstance(chunk_data['data'], list):
        for row in chunk_data['data']:
            processed_count += 1
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
                    
                    # If still not found, check if 'type' is at the top level
                    if type_value is None and 'type' in row:
                        type_value = row['type']
                    
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
    
    return type_counts, type_training, type_testing, processed_count

def fetch_chunk(offset, limit):
    """Fetch a single chunk of data"""
    try:
        response = requests.get(f"{API_BASE_URL}/view", params={'limit': limit, 'offset': offset}, timeout=8)
        if response.status_code == 200:
            return response.json(), offset
        return None, offset
    except Exception as e:
        print(f"Error fetching chunk at offset {offset}: {str(e)}")
        return None, offset

@app.route('/api/type-stats', methods=['GET'])
def get_type_stats():
    """Get type distribution statistics - uses intelligent sampling with parallel processing for speed"""
    try:
        # First, get the total count
        response = requests.get(f"{API_BASE_URL}/view", params={'limit': 1, 'offset': 0}, timeout=5)
        
        if response.status_code != 200:
            return jsonify({"type_distribution": {}}), 200
        
        total_rows = response.json().get('total_rows', 0)
        
        if total_rows == 0:
            return jsonify({"type_distribution": {}, "sample_size": 0, "total_rows": 0}), 200
        
        chunk_size = 1000
        type_counts = {}
        type_training = {}
        type_testing = {}
        processed_count = 0
        
        # For large datasets, use intelligent sampling instead of processing everything
        # This ensures we get all types without processing all 587k rows
        if total_rows > 10000:
            # Sample strategically: beginning, end, and multiple points throughout
            # This ensures we capture all types even if they're clustered
            num_samples = min(50, total_rows // 1000)  # Sample up to 50 chunks (50k rows)
            offsets = []
            
            # Always include beginning and end
            offsets.append(0)
            
            # Add evenly distributed samples throughout
            if num_samples > 2:
                step = total_rows // (num_samples - 1)
                for i in range(1, num_samples - 1):
                    offsets.append(i * step)
                offsets.append(max(0, total_rows - chunk_size))  # End
            
            # Remove duplicates and sort
            offsets = sorted(list(set(offsets)))
            
            print(f"Sampling {len(offsets)} chunks from {total_rows} total rows...")
            
            # Fetch chunks in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_offset = {
                    executor.submit(fetch_chunk, offset, chunk_size): offset 
                    for offset in offsets if offset < total_rows
                }
                
                for future in as_completed(future_to_offset):
                    chunk_data, offset = future.result()
                    if chunk_data:
                        chunk_types, chunk_train, chunk_test, chunk_processed = extract_types_from_chunk(chunk_data, offset)
                        processed_count += chunk_processed
                        
                        # Merge type counts
                        for type_val, count in chunk_types.items():
                            type_counts[type_val] = type_counts.get(type_val, 0) + count
                        for type_val, count in chunk_train.items():
                            type_training[type_val] = type_training.get(type_val, 0) + count
                        for type_val, count in chunk_test.items():
                            type_testing[type_val] = type_testing.get(type_val, 0) + count
        else:
            # For smaller datasets (<10k rows), process all but still use parallel processing
            chunks_to_process = (total_rows // chunk_size) + (1 if total_rows % chunk_size > 0 else 0)
            offsets = [i * chunk_size for i in range(chunks_to_process) if i * chunk_size < total_rows]
            
            print(f"Processing all {chunks_to_process} chunks from {total_rows} rows in parallel...")
            
            # Fetch chunks in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_offset = {
                    executor.submit(fetch_chunk, offset, min(chunk_size, total_rows - offset)): offset 
                    for offset in offsets
                }
                
                for future in as_completed(future_to_offset):
                    chunk_data, offset = future.result()
                    if chunk_data:
                        chunk_types, chunk_train, chunk_test, chunk_processed = extract_types_from_chunk(chunk_data, offset)
                        processed_count += chunk_processed
                        
                        # Merge type counts
                        for type_val, count in chunk_types.items():
                            type_counts[type_val] = type_counts.get(type_val, 0) + count
                        for type_val, count in chunk_train.items():
                            type_training[type_val] = type_training.get(type_val, 0) + count
                        for type_val, count in chunk_test.items():
                            type_testing[type_val] = type_testing.get(type_val, 0) + count
        
        # Calculate percentages
        total_with_types = sum(type_counts.values())
        type_percentages = {}
        if total_with_types > 0:
            for type_val, count in type_counts.items():
                type_percentages[type_val] = round((count / total_with_types) * 100, 2)
        
        print(f"Found {len(type_counts)} unique types: {list(type_counts.keys())}")
        
        return jsonify({
            "type_distribution": type_counts,
            "type_percentages": type_percentages,
            "type_training": type_training,
            "type_testing": type_testing,
            "sample_size": processed_count,
            "total_rows": total_rows,
            "sampled": total_rows > processed_count
        }), 200
    except Exception as e:
        return jsonify({"error": str(e), "type_distribution": {}}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
