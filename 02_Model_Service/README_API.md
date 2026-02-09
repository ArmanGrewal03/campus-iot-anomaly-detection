# Random Forest Model FastAPI

FastAPI application for training, testing, and deploying the Random Forest anomaly detection model.

## Endpoints

### Health Check
- **GET** `/health` - Check API health status

### Model Training
- **POST** `/train` - Train the Random Forest model
  - Request body (optional):
    ```json
    {
      "n_estimators": 100,
      "max_depth": null,
      "random_state": 42
    }
    ```
  - Fetches training data from backend API (`GET /training`)
  - Trains the model and saves it

### Model Testing
- **POST** `/test` - Test the trained model
  - Fetches testing data from backend API (`GET /testing`)
  - Evaluates the model and returns metrics
  - Updates model metadata with test results

### Model Prediction
- **POST** `/predict` - Make predictions on new data
  - Request body:
    ```json
    {
      "data": [
        {
          "dur": 0.1,
          "proto": "tcp",
          "service": "http",
          ...
        }
      ]
    }
    ```
  - Returns predictions with probabilities

### Model Status
- **GET** `/model/status` - Get model training status and metadata

### Model Metrics
- **GET** `/model/metrics` - Get evaluation metrics from last test

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure backend API is running:**
   - Backend should be at `http://localhost:8000`
   - Data should be uploaded and validated

3. **Run the API:**
```bash
python model_api.py
```

Or with uvicorn:
```bash
uvicorn model_api:app --host 0.0.0.0 --port 8001 --reload
```

The API will run on `http://localhost:8001`

## Usage Examples

### Train the Model
```bash
curl -X POST "http://localhost:8001/train" \
  -H "Content-Type: application/json" \
  -d '{"n_estimators": 100, "max_depth": null, "random_state": 42}'
```

### Test the Model
```bash
curl -X POST "http://localhost:8001/test"
```

### Make Predictions
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "dur": 0.1,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "spkts": 10,
        "dpkts": 8,
        ...
      }
    ]
  }'
```

### Check Model Status
```bash
curl "http://localhost:8001/model/status"
```

### Get Model Metrics
```bash
curl "http://localhost:8001/model/metrics"
```

## Configuration

Set the backend API URL via environment variable:
```bash
export API_BASE_URL=http://localhost:8000
```

Or modify the `API_BASE_URL` constant in `model_api.py`.

## Model Storage

- Model file: `models/random_forest_model.pkl`
- Metadata: `models/model_metadata.json`

## Workflow

1. **Upload data** to backend API (`POST /new`)
2. **Validate data** to assign training/testing labels (`PUT /validate`)
3. **Train model** (`POST /train`)
4. **Test model** (`POST /test`)
5. **Deploy model** for predictions (`POST /predict`)
