import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_LABEL_COL = "label"
DEFAULT_EVENT_ID_COL = "event_id"
MODEL_VERSION = "autoencoder-v1.0"

def load_dataset(train_csv, test_csv, label_col=DEFAULT_LABEL_COL):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    non_feature_cols = {label_col, DEFAULT_EVENT_ID_COL, "time", "timestamp", "device_id"}
    feature_cols = [
        c for c in train_df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    if label_col not in train_df.columns:
        raise ValueError(f"Training CSV must contain label column '{label_col}'.")

    train_norm_df = train_df[train_df[label_col] == 0].copy()
    X_train_norm = train_norm_df[feature_cols].values

    X_test = test_df[feature_cols].values

    y_test = test_df[label_col].values if label_col in test_df.columns else None

    meta_cols = [c for c in [DEFAULT_EVENT_ID_COL] if c in test_df.columns]
    test_meta = test_df[meta_cols].copy()

    return X_train_norm, X_test, y_test, test_meta, feature_cols

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_autoencoder(input_dim, encoding_dim=16, dropout_rate=0.0):
    inputs = keras.Input(shape=(input_dim,), name="input_layer")
    x = layers.Dense(encoding_dim * 2, activation="relu")(inputs)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    encoded = layers.Dense(encoding_dim, activation="relu", name="encoded")(x)
    x = layers.Dense(encoding_dim * 2, activation="relu")(encoded)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return autoencoder

def train_autoencoder(model, X_train, epochs=50, batch_size=256, validation_split=0.1):
    history = model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        verbose=1,
    )
    return history

def reconstruction_error(model, X):
    recon = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - recon), axis=1)
    return errors

def compute_threshold(train_errors, percentile=95):
    return float(np.percentile(train_errors, percentile))

def normalize_scores(errors):
    e_min = errors.min()
    e_max = errors.max()
    denom = (e_max - e_min) + 1e-12
    return ((errors - e_min) / denom).astype(float)

def make_json_response(
    request_id,
    event_ids,
    anomaly_scores,
    flags,
    threshold,
    model_version=MODEL_VERSION,
):
    results = []
    for eid, score, flag in zip(event_ids, anomaly_scores, flags):
        results.append(
            {
                "event_id": str(eid) if eid is not None else None,
                "anomaly_score": float(score),
                "anomaly_flag": int(flag),
            }
        )
    return {
        "request_id": request_id,
        "model_version": model_version,
        "threshold": threshold,
        "results": results,
    }

def evaluate(y_true, anomaly_scores, flags):
    print("\n=== Autoencoder Evaluation (using threshold-based flags) ===")
    print(f"Precision: {precision_score(y_true, flags, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_true, flags, zero_division=0):.3f}")
    print(f"F1       : {f1_score(y_true, flags, zero_division=0):.3f}")
    try:
        print(f"ROC-AUC  : {roc_auc_score(y_true, anomaly_scores):.3f}")
    except Exception as e:
        print(f"ROC-AUC  : N/A ({e})")
    try:
        print(f"PR-AUC   : {average_precision_score(y_true, anomaly_scores):.3f}")
    except Exception as e:
        print(f"PR-AUC   : N/A ({e})")

def run_autoencoder_experiment(
    train_csv,
    test_csv,
    request_id="autoencoder-test-001",
    label_col=DEFAULT_LABEL_COL,
    threshold_percentile=95,
    epochs=30,
    batch_size=256,
):
    X_train_norm, X_test, y_test, test_meta, feature_cols = load_dataset(
        train_csv, test_csv, label_col=label_col
    )

    print(f"[INFO] Training autoencoder on {X_train_norm.shape[0]} normal samples")
    print(f"[INFO] Using {len(feature_cols)} features: {feature_cols}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_norm, X_test)

    model = build_autoencoder(input_dim=X_train_scaled.shape[1], encoding_dim=16)

    start_time = time.time()
    train_autoencoder(
        model,
        X_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
    )
    train_time_s = time.time() - start_time
    print(f"[INFO] Training completed in {train_time_s:.2f} seconds")

    train_errors = reconstruction_error(model, X_train_scaled)
    threshold = compute_threshold(train_errors, percentile=threshold_percentile)
    print(f"[INFO] Train error mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
    print(f"[INFO] Threshold (p{threshold_percentile}) = {threshold:.6f}")

    test_errors = reconstruction_error(model, X_test_scaled)
    anomaly_scores = normalize_scores(test_errors)
    anomaly_flags = (test_errors > threshold).astype(int)

    if y_test is not None:
        evaluate(y_test, anomaly_scores, anomaly_flags)

    if DEFAULT_EVENT_ID_COL in test_meta.columns:
        event_ids = test_meta[DEFAULT_EVENT_ID_COL].values
    else:
        event_ids = [f"evt-{i:06d}" for i in range(len(anomaly_scores))]

    response_json = make_json_response(
        request_id=request_id,
        event_ids=event_ids,
        anomaly_scores=anomaly_scores,
        flags=anomaly_flags,
        threshold=threshold,
        model_version=MODEL_VERSION,
    )

    return response_json

def main():
    parser = argparse.ArgumentParser(description="Autoencoder anomaly detection test")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV (with labels)")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV (with or without labels)")
    parser.add_argument("--output_json", required=True, help="Where to save JSON results")
    parser.add_argument("--request_id", default="autoencoder-test-001", help="Request ID for JSON schema")
    parser.add_argument("--label_col", default=DEFAULT_LABEL_COL, help="Name of label column (0 normal, 1 anomaly)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)

    args = parser.parse_args()

    response_json = run_autoencoder_experiment(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        request_id=args.request_id,
        label_col=args.label_col,
        threshold_percentile=args.threshold_percentile,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    with open(args.output_json, "w") as f:
        json.dump(response_json, f, indent=2)

    print(f"[INFO] Saved autoencoder JSON results to {args.output_json}")

if __name__ == "__main__":
    main()
