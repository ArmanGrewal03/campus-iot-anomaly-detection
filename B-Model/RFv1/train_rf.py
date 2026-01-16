import argparse
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, 
    precision_recall_curve
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

def train_model(X_train, y_train, args):
    """
    Train Random Forest Classifier.
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        class_weight=args.class_weight,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds.")
    return model, train_time

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model performance on test set.
    """
    print(f"Evaluating model (Threshold={threshold})...")
    
    # Get probabilities for the positive class (anomaly)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Apply threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC metrics (independent of threshold)
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
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
    
    return metrics, y_scores

def threshold_sweep(model, X_test, y_test):
    """
    Sweep thresholds to find optimal operating point.
    """
    print("\n--- Threshold Sweep ---")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 40)
    
    y_scores = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        print(f"{thresh:<10.2f} {p:<10.4f} {r:<10.4f} {f:<10.4f}")
        
def generate_json_output(model, X_test, threshold, out_path):
    """
    Generate a sample JSON output for backend compatibility.
    """
    # Use a small subset for demonstration
    subset_size = min(10, len(X_test))
    X_subset = X_test[:subset_size]
    
    scores = model.predict_proba(X_subset)[:, 1]
    flags = (scores >= threshold).astype(int)
    
    results = []
    for i in range(subset_size):
        results.append({
            "event_id": i,
            "anomaly_score": float(scores[i]),
            "anomaly_flag": int(flags[i])
        })
        
    output = {
        "request_id": "rf-test-001",
        "model_version": "random-forest-v1.0",
        "threshold": threshold,
        "results": results
    }
    
    # Save purely for verification
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSample JSON output saved to {out_path}")

def save_artifacts(model, metrics, args, feature_names, train_time, out_dir):
    """
    Save model, metadata, and top features.
    """
    # 1. Save Model
    model_path = os.path.join(out_dir, "random_forest_model.joblib")
    joblib.dump(model, model_path)
    
    # 2. Extract Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_10 = []
    print("\n--- Top 10 Features ---")
    for i in range(min(10, len(feature_names))):
        feat_name = feature_names[indices[i]]
        score = importances[indices[i]]
        top_10.append({"feature": feat_name, "importance": float(score)})
        print(f"{i+1}. {feat_name}: {score:.4f}")
        
    # 3. Save Metadata
    metadata = {
        "model": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "class_weight": args.class_weight,
            "random_state": args.random_state
        },
        "metrics": metrics,
        "threshold": args.threshold,
        "training_time_sec": train_time,
        "top_features": top_10
    }
    
    meta_path = os.path.join(out_dir, "training_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Train Random Forest for Anomaly Detection")
    
    # Paths
    parser.add_argument('--data_dir', default='/Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Processed', help='Directory containing preprocessed npy files')
    parser.add_argument('--out_dir', default='.', help='Directory to save model artifacts')
    
    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--class_weight', default='balanced')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)
    
    # Thresholding
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--threshold_sweep', action='store_true', help="Perform a threshold sweep")
    
    args = parser.parse_args()
    
    # Ensure output directory exists (if not current)
    if args.out_dir != '.':
        os.makedirs(args.out_dir, exist_ok=True)
        
    # 1. Load Data
    X_train, X_test, y_train, y_test, feature_names = load_data(args.data_dir)
    
    # 2. Train Model
    model, train_time = train_model(X_train, y_train, args)
    
    # 3. Evaluate Model
    metrics, y_scores = evaluate_model(model, X_test, y_test, threshold=args.threshold)
    
    # 4. Optional Threshold Sweep
    if args.threshold_sweep:
        threshold_sweep(model, X_test, y_test)
        
    # 5. Save Artifacts
    model_path = save_artifacts(model, metrics, args, feature_names, train_time, args.out_dir)
    
    # 6. Generate JSON Output
    json_out_path = os.path.join(args.out_dir, "sample_output.json")
    generate_json_output(model, X_test, args.threshold, json_out_path)
    
    # 7. Final Summary
    # Compute latency (rough estimate per sample)
    latency_start = time.time()
    _ = model.predict_proba(X_test[:100])
    latency_end = time.time()
    latency_ms = ((latency_end - latency_start) / 100) * 1000
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {os.path.abspath(model_path)}")
    print(f"Selected Threshold: {args.threshold}")
    print(f"Inference Latency: {latency_ms:.4f} ms/sample")

if __name__ == "__main__":
    main()
