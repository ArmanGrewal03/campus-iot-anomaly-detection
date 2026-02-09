"""
Random Forest Model for Campus IoT Anomaly Detection

This script:
1. Fetches training and testing data from the FastAPI endpoints
2. Trains a Random Forest classifier
3. Evaluates the model on test data
4. Saves the trained model

Label mapping:
- 0 = safe
- 1 = unsafe (anomaly)
"""

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# API Configuration
API_BASE_URL = "http://localhost:8000"
MODEL_DIR = "models"
MODEL_FILENAME = "random_forest_model.pkl"

def fetch_all_data(endpoint, label_type="all"):
    """
    Fetch all data from an API endpoint with pagination.
    
    Args:
        endpoint: API endpoint ('/training' or '/testing')
        label_type: Type of data being fetched (for logging)
    
    Returns:
        List of data records
    """
    print(f"Fetching {label_type} data from {endpoint}...")
    all_data = []
    limit = 1000  # Maximum per request
    offset = 0
    
    while True:
        try:
            response = requests.get(
                f"{API_BASE_URL}{endpoint}",
                params={"limit": limit, "offset": offset},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") != "success":
                print(f"Error: {result.get('message', 'Unknown error')}")
                break
            
            data = result.get("data", [])
            if not data:
                break
            
            all_data.extend(data)
            print(f"  Fetched {len(data)} rows (total: {len(all_data)})")
            
            if not result.get("has_more", False):
                break
            
            offset += limit
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    print(f"Total {label_type} records fetched: {len(all_data)}")
    return all_data

def extract_features_and_labels(data_records):
    """
    Extract features and labels from API response data.
    
    Args:
        data_records: List of data records from API
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Label array (0 = safe, 1 = unsafe)
    """
    rows = []
    for record in data_records:
        # Extract the actual data from the nested structure
        row_data = record.get("data", {})
        if isinstance(row_data, str):
            row_data = json.loads(row_data)
        
        rows.append(row_data)
    
    if not rows:
        return pd.DataFrame(), np.array([])
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Check if label column exists
    if "label" not in df.columns:
        raise ValueError("'label' column not found in data. Please ensure data has been uploaded and validated.")
    
    # Separate features and labels
    # Exclude non-feature columns
    exclude_cols = ["label", "id", "attack_cat"]  # Exclude label and metadata
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df["label"].copy()
    
    # Convert label to numeric (handle string labels)
    y = pd.to_numeric(y, errors='coerce')
    
    # Handle missing values in labels
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Convert features to numeric, handling non-numeric columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill NaN values with 0 (or median/mean if preferred)
    X = X.fillna(0)
    
    # Ensure labels are integers (0 or 1)
    y = y.astype(int)
    
    print(f"Extracted {len(X)} samples with {len(feature_cols)} features")
    print(f"Label distribution: Safe (0) = {(y == 0).sum()}, Unsafe (1) = {(y == 1).sum()}")
    
    return X, y, feature_cols

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed
    
    Returns:
        Trained Random Forest model
    """
    print("\nTraining Random Forest model...")
    print(f"  Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating model on test data...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of unsafe (class 1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Safe  Unsafe")
    print(f"Actual Safe    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Unsafe  {cm[1][0]:4d}  {cm[1][1]:4d}")
    print("="*60)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"]))
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    
    # Get top 10 most important features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance_df.to_dict('records')
    }
    
    return metrics

def save_model(model, feature_names, metrics):
    """
    Save the trained model and metadata.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        metrics: Evaluation metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'label_mapping': {
            '0': 'safe',
            '1': 'unsafe'
        }
    }
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

def main():
    """Main function to train and evaluate the model."""
    print("="*60)
    print("Campus IoT Anomaly Detection - Random Forest Model")
    print("="*60)
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("ERROR: API is not healthy. Please ensure FastAPI backend is running.")
            return
        print("✓ API is healthy")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Cannot connect to API at {API_BASE_URL}")
        print(f"Please ensure the FastAPI backend is running.")
        return
    
    # Fetch training data
    training_data = fetch_all_data("/training", "training")
    if not training_data:
        print("ERROR: No training data found. Please ensure:")
        print("  1. Data has been uploaded via POST /new")
        print("  2. Labels have been assigned via PUT /validate")
        return
    
    # Fetch testing data
    testing_data = fetch_all_data("/testing", "testing")
    if not testing_data:
        print("ERROR: No testing data found. Please ensure:")
        print("  1. Data has been uploaded via POST /new")
        print("  2. Labels have been assigned via PUT /validate")
        return
    
    # Extract features and labels
    print("\nProcessing training data...")
    try:
        X_train, y_train, feature_names = extract_features_and_labels(training_data)
        if len(X_train) == 0:
            print("ERROR: No valid training samples found.")
            return
    except Exception as e:
        print(f"ERROR processing training data: {e}")
        return
    
    print("\nProcessing testing data...")
    try:
        X_test, y_test, _ = extract_features_and_labels(testing_data)
        if len(X_test) == 0:
            print("ERROR: No valid testing samples found.")
            return
    except Exception as e:
        print(f"ERROR processing testing data: {e}")
        return
    
    # Ensure feature columns match
    common_features = set(X_train.columns) & set(X_test.columns)
    X_train = X_train[list(common_features)]
    X_test = X_test[list(common_features)]
    feature_names = list(common_features)
    
    print(f"\nUsing {len(feature_names)} common features for training")
    
    # Train model
    model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=None)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, feature_names, metrics)
    
    print("\n" + "="*60)
    print("Model training and evaluation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
