import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Save to shared B-Model/SavedModels directory
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../B-Model/SavedModels"))
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _save_artifacts(self, model, scaler, features, metrics, extra_meta=None):
        # Paths
        base_name = f"{self.models_dir}/{self.model_name}_{self.timestamp}"
        model_path = f"{base_name}.joblib"
        meta_path = f"{base_name}_meta.json"
        
        # Save Model
        joblib.dump(model, model_path)
        
        # Save Scaler if exists
        if scaler:
            joblib.dump(scaler, f"{base_name}_scaler.joblib")
            metrics["scaler_path"] = f"{base_name}_scaler.joblib"
            
        # Metadata
        meta_data = {
            "features": features,
            "model_type": self.model_type,
            "training_date": self.timestamp,
            **(extra_meta or {})
        }
        
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)
            
        metrics["model_path"] = model_path
        metrics["meta_path"] = meta_path
        return metrics

    def train(self, df, features, label_col=None):
        raise NotImplementedError("Subclasses must implement train")

class RandomForestTrainer(ModelTrainer):
    def train(self, df, features, label_col):
        X = df[features].copy()
        y = df[label_col].copy()
        
        # Encoding
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
            
        # Split (Internal Validation)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        start = datetime.utcnow()
        model.fit(X_train, y_train)
        duration = (datetime.utcnow() - start).total_seconds()
        
        y_pred = model.predict(X_val)
        
        metrics = {
            "status": "success",
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, average='weighted', zero_division=0)),
            "training_time": duration
        }
        
        return self._save_artifacts(model, None, features, metrics)

class IsolationForestTrainer(ModelTrainer):
    def train(self, df, features, label_col=None):
        # IF is unsupervised, but we use label to estimate contamination if available
        X = df[features].copy()
        
        # Encoding
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
                
        # Scaling is important for IF
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate contamination
        contamination = 0.1
        if label_col and label_col in df.columns:
            # Assume 1 is anomaly
            y = df[label_col].copy()
            if y.dtype == 'object': y = pd.factorize(y)[0]
            contamination = float(sum(y == 1) / len(y))
            contamination = max(0.01, min(0.4, contamination)) # Camp betwen 1% and 40%
            
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        
        start = datetime.utcnow()
        model.fit(X_scaled)
        duration = (datetime.utcnow() - start).total_seconds()
        
        # No 'accuracy' in unsupervised sense unless we check against labels
        # Let's check against the same dataset for basic metrics
        y_pred_split = model.predict(X_scaled)
        y_pred = np.where(y_pred_split == -1, 1, 0)
        
        metrics = {
             "status": "success",
             "training_time": duration,
             "contamination": contamination
        }
        
        # If we have labels, add classification metrics
        if label_col and label_col in df.columns:
             y = df[label_col].copy()
             if y.dtype == 'object': y = pd.factorize(y)[0]
             metrics["accuracy"] = float(accuracy_score(y, y_pred))
             metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
             metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
        
        return self._save_artifacts(model, scaler, features, metrics, {"is_isolation_forest": True})

class AutoEncoderTrainer(ModelTrainer):
    def train(self, df, features, label_col=None):
        X = df[features].copy()
        
        # Encoding
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
                
        # Scaling is CRITICAL for Autoencoders
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        input_dim = X.shape[1]
        hidden_layers = (max(2, input_dim // 2), max(1, input_dim // 4), max(2, input_dim // 2))
        
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42
        )
        
        start = datetime.utcnow()
        model.fit(X_scaled, X_scaled) # Fit to self
        duration = (datetime.utcnow() - start).total_seconds()
        
        # Calculate Threshold
        X_pred = model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        threshold = float(np.percentile(mse, 95))
        model.ae_threshold_ = threshold
        
        y_pred = (mse > threshold).astype(int)
        
        metrics = {
            "status": "success",
            "training_time": duration,
            "threshold": threshold
        }
        
        if label_col and label_col in df.columns:
             y = df[label_col].copy()
             if y.dtype == 'object': y = pd.factorize(y)[0]
             metrics["accuracy"] = float(accuracy_score(y, y_pred))
             metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
             metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))

        return self._save_artifacts(model, scaler, features, metrics, {"is_autoencoder": True, "threshold": threshold})

def train_model_dispatch(model_type, model_name, df, features, label_col):
    if "Isolation Forest" in model_type:
        trainer = IsolationForestTrainer(model_name, model_type)
        return trainer.train(df, features, label_col)
    elif "Autoencoder" in model_type:
        trainer = AutoEncoderTrainer(model_name, model_type)
        return trainer.train(df, features, label_col)
    else:
        trainer = RandomForestTrainer(model_name, model_type)
        return trainer.train(df, features, label_col)
