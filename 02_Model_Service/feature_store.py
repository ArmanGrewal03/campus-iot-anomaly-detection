import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

class DataVectorizer:
    """
    Handles converting raw network traffic data into numerical vectors
    suitable for machine learning models.
    """
    def __init__(self):
        self.cat_encoders = {}
        self.one_hot_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Specific column configurations for UNSW-NB15
        self.CATEGORICAL_COLS_LOW = ['service', 'state']
        self.CATEGORICAL_COLS_HIGH = ['proto']
        self.LOG_COLS = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'dpkts',
                        'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'sjit', 'djit',
                        'sinpkt', 'dinpkt', 'ct_srv_src', 'ct_dst_ltm', 'ct_src_ltm',
                        'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    def fit(self, df: pd.DataFrame):
        """Fit the vectorizer on training data."""
        logger.info("Fitting DataVectorizer...")
        X = df.copy()
        
        # 1. Feature Engineering (Ratios)
        X = self._add_engineered_features(X)
        
        # 2. Log Transforms
        for col in self.LOG_COLS:
            if col in X.columns:
                X[col] = np.log1p(pd.to_numeric(X[col], errors='coerce').fillna(0).clip(lower=0))
        
        # 3. Categorical Encoding (Label for High, One-Hot for Low)
        for col in self.CATEGORICAL_COLS_HIGH:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = X[col].astype(str).str.strip().str.lower()
                X[col] = le.fit_transform(X[col])
                self.cat_encoders[col] = le
                
        for col in self.CATEGORICAL_COLS_LOW:
            if col in X.columns:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                X_col = X[[col]].astype(str).fillna('none')
                ohe_result = ohe.fit_transform(X_col)
                ohe_cols = [f"{col}_{c}" for c in ohe.get_feature_names_out()]
                ohe_df = pd.DataFrame(ohe_result, columns=ohe_cols, index=X.index)
                X = pd.concat([X.drop(col, axis=1), ohe_df], axis=1)
                self.one_hot_encoders[col] = ohe
                
        # 4. Fill remaining na and force numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 5. Fit Scaler
        self.feature_names = list(X.columns)
        self.scaler.fit(X)
        self.is_fitted = True
        logger.info(f"Vectorizer fitted with {len(self.feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders and scalers."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform.")
            
        X = df.copy()
        
        # 1. Feature Engineering
        X = self._add_engineered_features(X)
        
        # 2. Log Transforms
        for col in self.LOG_COLS:
            if col in X.columns:
                X[col] = np.log1p(pd.to_numeric(X[col], errors='coerce').fillna(0).clip(lower=0))
                
        # 3. Categorical Encoding
        for col, le in self.cat_encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip().str.lower()
                # Handle unseen labels
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0]) # Default to first class if unknown
                X[col] = le.transform(X[col])
                
        for col, ohe in self.one_hot_encoders.items():
            if col in X.columns:
                X_col = X[[col]].astype(str).fillna('none')
                ohe_result = ohe.transform(X_col)
                ohe_cols = [f"{col}_{c}" for c in ohe.get_feature_names_out()]
                ohe_df = pd.DataFrame(ohe_result, columns=ohe_cols, index=X.index)
                X = pd.concat([X.drop(col, axis=1), ohe_df], axis=1)
                
        # 4. Align features with training
        X_aligned = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                X_aligned[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            else:
                X_aligned[col] = 0.0
                
        # 5. Scale
        X_scaled = self.scaler.transform(X_aligned)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    def _add_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal helper for feature engineering."""
        # Standardize column names
        X.columns = [c.lstrip('\ufeff').strip().lower() for c in X.columns]
        
        if 'sbytes' in X.columns and 'dbytes' in X.columns:
            sbytes = pd.to_numeric(X['sbytes'], errors='coerce').fillna(0)
            dbytes = pd.to_numeric(X['dbytes'], errors='coerce').fillna(0)
            X['total_bytes'] = sbytes + dbytes
            X['byte_ratio'] = sbytes / (X['total_bytes'] + 1)
            
        if 'spkts' in X.columns and 'dpkts' in X.columns:
            spkts = pd.to_numeric(X['spkts'], errors='coerce').fillna(0)
            dpkts = pd.to_numeric(X['dpkts'], errors='coerce').fillna(0)
            X['total_pkts'] = spkts + dpkts
            X['pkt_ratio'] = spkts / (X['total_pkts'] + 1)
            
        if 'smeansz' in X.columns and 'dmeansz' in X.columns:
            smeansz = pd.to_numeric(X['smeansz'], errors='coerce').fillna(0)
            dmeansz = pd.to_numeric(X['dmeansz'], errors='coerce').fillna(0)
            X['mean_pkt_size_ratio'] = smeansz / (dmeansz + 1)

        if 'sload' in X.columns and 'dload' in X.columns:
            sload = pd.to_numeric(X['sload'], errors='coerce').fillna(0)
            dload = pd.to_numeric(X['dload'], errors='coerce').fillna(0)
            X['load_ratio'] = sload / (dload + 1)

        if 'sttl' in X.columns and 'dttl' in X.columns:
            sttl = pd.to_numeric(X['sttl'], errors='coerce').fillna(0)
            dttl = pd.to_numeric(X['dttl'], errors='coerce').fillna(0)
            X['ttl_ratio'] = sttl / (dttl + 1)

        if 'sbytes' in X.columns and 'spkts' in X.columns:
            X['s_avg_pkt_size'] = pd.to_numeric(X['sbytes'], errors='coerce').fillna(0) / (pd.to_numeric(X['spkts'], errors='coerce').fillna(0) + 1)

        if 'dbytes' in X.columns and 'dpkts' in X.columns:
            X['d_avg_pkt_size'] = pd.to_numeric(X['dbytes'], errors='coerce').fillna(0) / (pd.to_numeric(X['dpkts'], errors='coerce').fillna(0) + 1)

        if 'dur' in X.columns and ('sbytes' in X.columns or 'dbytes' in X.columns):
            dur = pd.to_numeric(X['dur'], errors='coerce').fillna(0)
            sbytes = pd.to_numeric(X.get('sbytes', 0), errors='coerce').fillna(0)
            dbytes = pd.to_numeric(X.get('dbytes', 0), errors='coerce').fillna(0)
            X['bytes_per_sec'] = (sbytes + dbytes) / (dur + 0.001)
            
        return X

class FeatureStore:
    """
    Centralized store for pre-processed features and metadata.
    """
    def __init__(self, base_dir: str = "feature_store"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save(self, name: str, X: pd.DataFrame, y: np.ndarray, vectorizer: DataVectorizer):
        """Save features and vectorizer to disk."""
        target_dir = os.path.join(self.base_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Save data
        joblib.dump(X, os.path.join(target_dir, "X.joblib"))
        joblib.dump(y, os.path.join(target_dir, "y.joblib"))
        
        # Save vectorizer
        joblib.dump(vectorizer, os.path.join(target_dir, "vectorizer.joblib"))
        
        # Save metadata
        metadata = {
            "name": name,
            "samples": len(X),
            "features": len(X.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "columns": list(X.columns)
        }
        with open(os.path.join(target_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Features for '{name}' saved to {target_dir}")

    def load(self, name: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[DataVectorizer]]:
        """Load features and vectorizer from disk."""
        target_dir = os.path.join(self.base_dir, name)
        if not os.path.exists(target_dir):
            return None, None, None
            
        try:
            X = joblib.load(os.path.join(target_dir, "X.joblib"))
            y = joblib.load(os.path.join(target_dir, "y.joblib"))
            vectorizer = joblib.load(os.path.join(target_dir, "vectorizer.joblib"))
            return X, y, vectorizer
        except Exception as e:
            logger.error(f"Error loading features '{name}': {e}")
            return None, None, None

    def list_features(self) -> List[str]:
        """List all stored feature sets."""
        if not os.path.exists(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
