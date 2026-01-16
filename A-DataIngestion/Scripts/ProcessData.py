import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Features to exclude from X (metadata/identifiers)
EXCLUDE_FROM_X = ['id', 'attack_cat', 'Label']

# Categorical features to encode
CAT_FEATURES = ['proto', 'state', 'service']

# Validation: Columns that MUST exist after renaming
REQUIRED_COLS = [
    'id', 'dur', 'proto', 'service', 'state', 'Spkts', 'Dpkts', 'sbytes', 'dbytes', 
    'rate', 'sttl', 'dttl', 'Sload', 'Dload', 'sloss', 'dloss', 'Sintpkt', 'Dintpkt', 
    'Sjit', 'Djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'ct_srv_src', 'ct_state_ttl', 
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 
    'is_sm_ips_ports', 'attack_cat', 'Label'
]

def load_data(input_path):
    """
    Load CSV data, normalize headers, and validate.
    """
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        sys.exit(1)
    
    if os.path.getsize(input_path) == 0:
        print("Error: File is empty.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    
    # 1. Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip()
    
    # 2. Rename columns to match internal logic expectations
    rename_map = {
        'label': 'Label',
        'spkts': 'Spkts',
        'dpkts': 'Dpkts',
        'sload': 'Sload',
        'dload': 'Dload',
        'sjit': 'Sjit',
        'djit': 'Djit',
        'sinpkt': 'Sintpkt',
        'dinpkt': 'Dintpkt',
        'smean': 'smeansz',
        'dmean': 'dmeansz',
        'response_body_len': 'res_bdy_len'
    }
    df.rename(columns=rename_map, inplace=True)

    # Force Label to be integer
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(int)
    
    # 3. Validate against the expected header list (after renaming)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    
    if missing_cols:
        print(f"Error: Dataset is missing required columns: {missing_cols}")
        print("Please ensure the CSV headers match the UNSW-NB15 schema.")
        sys.exit(1)
        
    print(f"Initial dataset shape: {df.shape}")
    return df

def feature_engineering(df):
    """
    Create additional ratios and rate features.
    """
    print("Performing feature engineering...")
    
    # Avoid division by zero by adding a small epsilon or 1 depending on context
    # byte_ratio = sbytes / (dbytes + 1)
    df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
    
    # pkt_ratio = Spkts / (Dpkts + 1)
    df['pkt_ratio'] = df['Spkts'] / (df['Dpkts'] + 1)
    
    # flow_rate = (sbytes + dbytes) / (dur + 1e-6)
    df['flow_rate'] = (df['sbytes'] + df['dbytes']) / (df['dur'] + 1e-6)
    
    # pkt_rate = (Spkts + Dpkts) / (dur + 1e-6)
    df['pkt_rate'] = (df['Spkts'] + df['Dpkts']) / (df['dur'] + 1e-6)
    
    return df

def clean_data(df):
    """
    Basic data cleaning: infinity handling and categorical nan filling.
    """
    print("Cleaning data...")
    # Replace inf/-inf with NaN to be handled by imputer
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill missing values in categorical columns with 'unknown'
    cat_cols_all = df.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols_all:
        df[c] = df[c].fillna('unknown')
        
    return df

def build_preprocess_pipeline(numeric_features, categorical_features):
    """
    Constructs the sklearn preprocessing pipeline.
    """
    print("Building preprocessing pipeline...")
    
    # Numeric pipeline: 
    # 1. Impute missing values with median
    # 2. Scale features using StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline:
    # 1. Impute missing with 'unknown'
    # 2. One-Hot Encode (handle_unknown='ignore')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    return preprocessor

def transform_and_save(df, args):
    """
    Applies transformation and saves all artifacts.
    """
    # 4. Identify Feature Types for Pipeline
    # Identify numeric columns including newly created ones
    numeric_candidates = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    final_numeric = []
    for c in numeric_candidates:
        # Exclude metadata/labels and categorical features
        if c not in EXCLUDE_FROM_X and c not in CAT_FEATURES:
            final_numeric.append(c)
            
    # Ensure engineered features are included if dtypes matched
    engineered_cols = ['byte_ratio', 'pkt_ratio', 'flow_rate', 'pkt_rate']
    for ec in engineered_cols:
        if ec in df.columns and ec not in final_numeric:
            final_numeric.append(ec)
            
    final_categorical = CAT_FEATURES
    
    print(f"Selected {len(final_numeric)} numeric features and {len(final_categorical)} categorical features.")
    
    # 5. Extract Targets and Metadata
    y = df['Label'].values
    
    # Keep id and attack_cat for metadata
    meta_cols = [c for c in ['id', 'attack_cat'] if c in df.columns]
    meta_df = df[meta_cols]
    
    # 6. Fit and Transform
    pipeline = build_preprocess_pipeline(final_numeric, final_categorical)
    
    print("Fitting pipeline and transforming data...")
    X_processed = pipeline.fit_transform(df)
    
    # Get feature names
    try:
        if hasattr(pipeline, 'get_feature_names_out'):
             feature_names = pipeline.get_feature_names_out().tolist()
        else:
            feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
    except AttributeError:
        feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]

    print(f"Processed data shape: {X_processed.shape}")
    
    # 7. Save Artifacts
    print(f"Saving outputs to {args.out_dir}...")
    
    # Save Pipeline
    joblib.dump(pipeline, os.path.join(args.out_dir, 'preprocessing_pipeline.joblib'))
    
    # Save Feature Names
    with open(os.path.join(args.out_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=4)
        
    # Save Metadata
    meta_df.to_csv(os.path.join(args.out_dir, 'meta.csv'), index=False)
    
    # Save Full Matrices
    np.save(os.path.join(args.out_dir, 'X.npy'), X_processed.astype(np.float32))
    np.save(os.path.join(args.out_dir, 'y.npy'), y.astype(np.int32))
    
    # 8. Optional Split
    if args.make_split:
        print(f"Splitting data (test_size={args.test_size}, stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=args.test_size, 
            random_state=args.seed,
            stratify=y
        )
        
        np.save(os.path.join(args.out_dir, 'X_train.npy'), X_train.astype(np.float32))
        np.save(os.path.join(args.out_dir, 'X_test.npy'), X_test.astype(np.float32))
        np.save(os.path.join(args.out_dir, 'y_train.npy'), y_train.astype(np.int32))
        np.save(os.path.join(args.out_dir, 'y_test.npy'), y_test.astype(np.int32))
        
        print(f"Split created -> Train: {X_train.shape}, Test: {X_test.shape}")

    return X_processed, y, feature_names

def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 Preprocessing Pipeline")
    parser.add_argument('--input_csv', required=True, help="Path to the input CSV file.")
    parser.add_argument('--out_dir', required=True, help="Directory to save the processed outputs.")
    parser.add_argument('--make_split', action='store_true', help="Whether to generate train/test splits.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Execution Flow
    df = load_data(args.input_csv)
    df = clean_data(df)
    df = feature_engineering(df)
    
    X, y, feats = transform_and_save(df, args)
    
    # Final Summary
    # Final Summary
    unique, counts = np.unique(y, return_counts=True)
    class_balance = dict(zip(unique.tolist(), counts.tolist()))
    
    print("\n--- Final Summary ---")
    print(f"Total Samples: {X.shape[0]}")
    print(f"Total Features: {X.shape[1]}")
    print(f"Class Balance (Label): {class_balance}")
    print(f"Label Data Type: {y.dtype}")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
