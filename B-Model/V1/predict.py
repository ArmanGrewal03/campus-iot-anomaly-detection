"""
Prediction script for Random Forest Model

Use this script to make predictions on new data using the trained model.
"""

import joblib
import pandas as pd
import numpy as np
import json
import os
import sys

MODEL_DIR = "models"
MODEL_FILENAME = "random_forest_model.pkl"
METADATA_FILENAME = "model_metadata.json"

def load_model():
    """Load the trained model and metadata."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    metadata_path = os.path.join(MODEL_DIR, METADATA_FILENAME)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Please train the model first.")
    
    # Load model
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Metadata loaded from {metadata_path}")
    
    return model, metadata

def prepare_features(data, feature_names):
    """
    Prepare features for prediction.
    
    Args:
        data: Dictionary or DataFrame with feature values
        feature_names: List of expected feature names
    
    Returns:
        Prepared feature array
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default value 0
        for feature in missing_features:
            df[feature] = 0
    
    # Select only the features used in training
    df = df[feature_names]
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df.values

def predict(model, metadata, data):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        metadata: Model metadata
        data: Input data (dict, list of dicts, or DataFrame)
    
    Returns:
        Predictions and probabilities
    """
    feature_names = metadata['feature_names']
    
    # Prepare features
    X = prepare_features(data, feature_names)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Map predictions to labels
    label_mapping = metadata['label_mapping']
    prediction_labels = [label_mapping[str(int(pred))] for pred in predictions]
    
    # Get probability of unsafe (class 1)
    unsafe_probabilities = probabilities[:, 1]
    
    results = []
    for i in range(len(predictions)):
        results.append({
            'prediction': int(predictions[i]),
            'label': prediction_labels[i],
            'probability_safe': float(probabilities[i][0]),
            'probability_unsafe': float(unsafe_probabilities[i]),
            'confidence': float(max(probabilities[i]))
        })
    
    return results

def predict_from_json(json_file):
    """Make predictions from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    model, metadata = load_model()
    results = predict(model, metadata, data)
    
    return results

def predict_from_dict(data_dict):
    """Make predictions from a dictionary."""
    model, metadata = load_model()
    results = predict(model, metadata, data_dict)
    
    return results

def main():
    """Command-line interface for predictions."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file.json>")
        print("   or: python predict.py --interactive")
        sys.exit(1)
    
    if sys.argv[1] == "--interactive":
        # Interactive mode
        print("\nInteractive Prediction Mode")
        print("="*60)
        model, metadata = load_model()
        
        print(f"\nModel trained on {len(metadata['feature_names'])} features")
        print(f"Training date: {metadata['training_date']}")
        print(f"\nModel metrics:")
        metrics = metadata['metrics']
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        print("\nEnter feature values (or 'quit' to exit):")
        print("Example: Enter JSON like: {\"dur\": 0.1, \"proto\": \"tcp\", ...}")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                data = json.loads(user_input)
                results = predict(model, metadata, data)
                
                for i, result in enumerate(results):
                    print(f"\nPrediction {i+1}:")
                    print(f"  Label: {result['label'].upper()}")
                    print(f"  Prediction: {result['prediction']} ({'Unsafe' if result['prediction'] == 1 else 'Safe'})")
                    print(f"  Probability Safe: {result['probability_safe']:.4f}")
                    print(f"  Probability Unsafe: {result['probability_unsafe']:.4f}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        # File mode
        json_file = sys.argv[1]
        if not os.path.exists(json_file):
            print(f"Error: File not found: {json_file}")
            sys.exit(1)
        
        print(f"Loading data from {json_file}...")
        results = predict_from_json(json_file)
        
        print("\nPredictions:")
        print("="*60)
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"  Label: {result['label'].upper()}")
            print(f"  Prediction: {result['prediction']} ({'Unsafe' if result['prediction'] == 1 else 'Safe'})")
            print(f"  Probability Safe: {result['probability_safe']:.4f}")
            print(f"  Probability Unsafe: {result['probability_unsafe']:.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()
