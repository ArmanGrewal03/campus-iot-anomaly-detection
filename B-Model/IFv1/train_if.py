import argparse
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)

def load_data(data_dir):
    """
    Load preprocessed data artifacts.
    """
    print(f"Loading data from {data_dir}...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        with open(os.path.join(data_dir, 'feature_names.json'), 'r') as f:
            feature_names = json.load(f)
            
        print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_names
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def train_model(X_train, args):
    """
    Train Isolation Forest. 
    Note: IF is unsupervised, but we pass X_train (which might contain anomalies if not cleaned).
    Ideally trained on 'normal' data only, but IF is robust to some contamination.
    """
    print(f"Training Isolation Forest model (contamination={args.contamination})...")
    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=1
    )
    
    start_time = time.time()
    model.fit(X_train)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds.")
    return model, train_time

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    """
    print(f"Evaluating model...")
    
    # Get anomaly scores (lower is more anomalous, but sklearn returns opposite)
    # decision_function: positive for inliers, negative for outliers
    # We negate it so higher score = more anomalous
    scores = -model.decision_function(X_test)
    
    # Predict: 1 for inlier, -1 for outlier/anomaly
    # We convert to: 0 for normal, 1 for anomaly
    y_pred_raw = model.predict(X_test)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    print("\n--- Evaluation Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
    
    return metrics

def save_artifacts(model, metrics, args, feature_names, train_time, out_dir):
    """
    Save model and metadata.
    """
    # 1. Save Model
    model_path = os.path.join(out_dir, "isolation_forest_model.joblib")
    joblib.dump(model, model_path)
    
    # 2. Save Metadata
    metadata = {
        "model": "IsolationForest",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "contamination": args.contamination,
            "random_state": args.random_state
        },
        "metrics": metrics,
        "training_time_sec": train_time,
        "features": feature_names
    }
    
    meta_path = os.path.join(out_dir, "training_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest for Anomaly Detection")
    
    # Paths
    parser.add_argument('--data_dir', default='/Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Processed', help='Directory containing preprocessed npy files')
    parser.add_argument('--out_dir', default='.', help='Directory to save model artifacts')
    
    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.1) # Expected proportion of outliers
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)
    
    args = parser.parse_args()
    
    if args.out_dir != '.':
        os.makedirs(args.out_dir, exist_ok=True)
        
    # 1. Load Data
    X_train, X_test, y_train, y_test, feature_names = load_data(args.data_dir)
    
    # 2. Train Model
    model, train_time = train_model(X_train, args)
    
    # 3. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)
    
    # 4. Save Artifacts
    model_path = save_artifacts(model, metrics, args, feature_names, train_time, args.out_dir)
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    main()
