from flask import Flask, render_template, request, jsonify, send_file
import requests
import json
import os
from werkzeug.utils import secure_filename

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
