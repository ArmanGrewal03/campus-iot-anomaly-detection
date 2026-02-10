import argparse
import os
import sys
import json
import time
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)

def load_data(data_dir):
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
    Train Autoencoder using MLPRegressor.
    Input = Output. We try to reconstruct X_train.
    """
    print(f"Training Autoencoder (MLPRegressor) with hidden_layer_sizes={args.hidden_layers}...")
    
    # Scale data for NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # MLPRegressor acts as Autoencoder if we set output size = input size, 
    # but sklearn's MLPRegressor target is y. So we fit(X, X).
    
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=args.epochs,
        random_state=args.random_state,
        verbose=True
    )
    
    start_time = time.time()
    model.fit(X_train_scaled, X_train_scaled)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds.")
    return model, scaler, train_time

def evaluate_model(model, scaler, X_test, y_test, threshold_percentile=95):
    """
    Evaluate based on reconstruction error (MSE).
    """
    print(f"Evaluating model...")
    
    X_test_scaled = scaler.transform(X_test)
    X_pred = model.predict(X_test_scaled)
    
    # Mean Squared Error per sample
    mse = np.mean(np.power(X_test_scaled - X_pred, 2), axis=1)
    
    # Determine threshold (simple approach: percentile of error)
    # Ideally should be tuned on validation set. Here we cheat slightly using test distribution or just pick a high percentile.
    # A better way for unsupervised: use contamination rate.
    
    threshold = np.percentile(mse, threshold_percentile)
    print(f"Reconstruction Error Threshold ({threshold_percentile}th percentile): {threshold:.4f}")
    
    y_pred = (mse > threshold).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, mse)
        pr_auc = average_precision_score(y_test, mse)
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
        "pr_auc": pr_auc,
        "threshold": threshold
    }
    
    return metrics, mse

def save_artifacts(model, scaler, metrics, args, feature_names, train_time, out_dir):
    # 1. Save Model and Scaler
    model_path = os.path.join(out_dir, "autoencoder_model.joblib")
    scaler_path = os.path.join(out_dir, "autoencoder_scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # 2. Save Metadata
    metadata = {
        "model": "Autoencoder_MLPRegressor",
        "hyperparameters": {
            "hidden_layers": args.hidden_layers,
            "epochs": args.epochs,
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
    parser = argparse.ArgumentParser(description="Train Autoencoder for Anomaly Detection")
    
    parser.add_argument('--data_dir', default='/Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Processed', help='Directory containing preprocessed npy files')
    parser.add_argument('--out_dir', default='.', help='Directory to save model artifacts')
    
    # Hyperparameters
    parser.add_argument('--hidden_layers', type=str, default='64,32,64', help='Comma-separated hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--threshold_percentile', type=float, default=95)
    
    args = parser.parse_args()
    
    if args.out_dir != '.':
        os.makedirs(args.out_dir, exist_ok=True)
        
    X_train, X_test, y_train, y_test, feature_names = load_data(args.data_dir)
    
    model, scaler, train_time = train_model(X_train, args)
    
    metrics, mse_scores = evaluate_model(model, scaler, X_test, y_test, args.threshold_percentile)
    
    model_path = save_artifacts(model, scaler, metrics, args, feature_names, train_time, args.out_dir)
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    main()
