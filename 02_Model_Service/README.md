# Random Forest Model for Campus IoT Anomaly Detection

This model implements a Random Forest classifier to detect anomalies in IoT network traffic data.

## Label Mapping
- **0** = Safe (normal traffic)
- **1** = Unsafe (anomaly/attack)

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure FastAPI backend is running:**
   - The backend should be running on `http://localhost:8000`
   - Data should be uploaded via `POST /new`
   - Labels should be assigned via `PUT /validate`

## Usage

Run the training script:
```bash
python random_forest_model.py
```

The script will:
1. Fetch training data from `GET /training` endpoint
2. Fetch testing data from `GET /testing` endpoint
3. Extract features and labels from the data
4. Train a Random Forest classifier
5. Evaluate the model on test data
6. Save the trained model to `models/random_forest_model.pkl`
7. Save metadata to `models/model_metadata.json`

## Model Output

- **Model file**: `models/random_forest_model.pkl` - Saved scikit-learn model
- **Metadata file**: `models/model_metadata.json` - Contains:
  - Feature names
  - Evaluation metrics (accuracy, precision, recall, F1-score)
  - Confusion matrix
  - Feature importance rankings
  - Training date

## Model Parameters

Default parameters:
- `n_estimators=100` - Number of trees in the forest
- `max_depth=None` - No limit on tree depth
- `random_state=42` - For reproducibility

You can modify these in the `train_random_forest()` function call in `main()`.

## Evaluation Metrics

The model outputs:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted unsafe, how many are actually unsafe
- **Recall**: Of actual unsafe, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **Feature Importance**: Top features contributing to predictions

## Notes

- The model automatically handles missing values and non-numeric data
- Only common features between training and testing sets are used
- The model uses all available CPU cores for training (`n_jobs=-1`)
