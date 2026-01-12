# Campus IoT Anomaly Detection Dashboard

A Flask web application that provides a user-friendly interface to interact with the FastAPI backend.

## Features

- **Health Check**: Monitor API status
- **CSV Upload**: Upload CSV files to the database
- **View Data**: Browse all stored data with pagination
- **Validate Data**: Assign training/testing labels (30%/70% split)
- **Training Data**: View only training-labeled data
- **Testing Data**: View only testing-labeled data
- **Insert Data**: Add new rows to the database
- **Clear Database**: Delete all data (with safety confirmations)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your FastAPI backend is running on `http://localhost:8000`

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Configuration

The API base URL is configured in `app.py`. To change it, modify:
```python
API_BASE_URL = "http://localhost:8000"
```

## API Endpoints

The Flask app provides the following endpoints that proxy to the FastAPI backend:

- `GET /api/health` - Check API health
- `POST /api/upload` - Upload CSV file
- `GET /api/view` - View all data
- `PUT /api/validate` - Assign training/testing labels
- `GET /api/training` - Get training data
- `GET /api/testing` - Get testing data
- `POST /api/insert` - Insert new row
- `POST /api/clear` - Clear database

## Notes

- The dashboard automatically checks API health on page load
- All operations include error handling and user feedback
- File uploads are limited to 100MB
- Clear database operation requires double confirmation
