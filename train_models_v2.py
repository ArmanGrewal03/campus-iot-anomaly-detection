#!/usr/bin/env python3
"""
Second iteration: Fine-tune models based on learnings from first iteration
- IF: Higher contamination (0.15) works better, try even higher
- AE: Lower threshold percentile (95) helps, try even lower
- Focus on balanced precision/recall
"""
import requests
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import sys

DATA_INGESTION_URL = "http://127.0.0.1:8000"
MODEL_SERVICE_URL = "http://127.0.0.1:8001"

results = []

def log(msg: str, level: str = "INFO"):
    """Log a message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

def train_model(dataset: str, model_name: str, model_type: str, params: Dict) -> Optional[Dict]:
    """Train a model"""
    log(f"Training {model_type} model: {model_name}")
    try:
        headers = {
            "dataset_name": dataset,
            "model_name": model_name,
            "Content-Type": "application/json"
        }
        payload = {"model_type": model_type, **params}
        
        start_time = time.time()
        response = requests.post(
            f"{MODEL_SERVICE_URL}/train",
            headers=headers,
            json=payload,
            timeout=300
        )
        training_duration = time.time() - start_time
        
        if response.status_code == 200:
            log(f"Training completed in {training_duration:.2f}s")
            return response.json().get("training_params", {})
        else:
            log(f"Training failed: {response.text[:200]}", "ERROR")
            return None
    except Exception as e:
        log(f"Training error: {e}", "ERROR")
        return None

def test_model(dataset: str, model_name: str) -> Optional[Dict]:
    """Test a model"""
    log(f"Testing model: {model_name}")
    try:
        headers = {
            "dataset_name": dataset,
            "model_name": model_name
        }
        start_time = time.time()
        response = requests.post(
            f"{MODEL_SERVICE_URL}/test",
            headers=headers,
            timeout=300
        )
        testing_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            metrics = result.get("metrics", {})
            log(f"Testing completed in {testing_duration:.2f}s")
            log(f"  Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1_score', 0):.4f}, ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            return metrics
        else:
            log(f"Testing failed: {response.text[:200]}", "ERROR")
            return None
    except Exception as e:
        log(f"Testing error: {e}", "ERROR")
        return None

log("="*80)
log("SECOND ITERATION: FINE-TUNING BASED ON FIRST ITERATION LEARNINGS")
log("="*80)

dataset = "april_1"
iteration = 11  # Continue from previous iteration

# Improved IF configurations (higher contamination)
if_configs_v2 = [
    {
        "name": "IF-Very-High-Contamination",
        "params": {
            "n_estimators": 100,
            "contamination": 0.20
        }
    },
    {
        "name": "IF-Very-High-Cont-Trees",
        "params": {
            "n_estimators": 200,
            "contamination": 0.20
        }
    },
    {
        "name": "IF-Extreme-Contamination",
        "params": {
            "n_estimators": 150,
            "contamination": 0.25
        }
    },
    {
        "name": "IF-Fine-Tuned",
        "params": {
            "n_estimators": 175,
            "contamination": 0.12
        }
    },
]

# Improved AE configurations (lower thresholds, better architecture)
ae_configs_v2 = [
    {
        "name": "AE-Very-Low-Threshold",
        "params": {
            "hidden_layers": "64,32,32,64",
            "ae_train_normal_only": True,
            "ae_threshold_percentile": 90.0,  # Even lower
            "ae_max_iterations": 300,
            "ae_patience": 20
        }
    },
    {
        "name": "AE-Balanced-Arch",
        "params": {
            "hidden_layers": "48,24,24,48",
            "ae_train_normal_only": True,
            "ae_threshold_percentile": 95.0,
            "ae_max_iterations": 300,
            "ae_patience": 20
        }
    },
    {
        "name": "AE-Extended-Training",
        "params": {
            "hidden_layers": "64,32,32,64",
            "ae_train_normal_only": True,
            "ae_threshold_percentile": 95.0,
            "ae_max_iterations": 500,
            "ae_patience": 30
        }
    },
    {
        "name": "AE-Aggressive-Threshold",
        "params": {
            "hidden_layers": "64,32,32,64",
            "ae_train_normal_only": True,
            "ae_threshold_percentile": 85.0,
            "ae_max_iterations": 300,
            "ae_patience": 20
        }
    },
    {
        "name": "AE-Medium-Depth",
        "params": {
            "hidden_layers": "96,48,24,48,96",
            "ae_train_normal_only": True,
            "ae_threshold_percentile": 95.0,
            "ae_max_iterations": 300,
            "ae_patience": 20
        }
    },
    {
        "name": "AE-All-Data-Lower-Threshold",
        "params": {
            "hidden_layers": "64,32,32,64",
            "ae_train_normal_only": False,
            "ae_threshold_percentile": 90.0,
            "ae_max_iterations": 300,
            "ae_patience": 20
        }
    },
]

# Train IF models
log("\nTRAINING IMPROVED ISOLATION FOREST MODELS")
log("="*80)
for config in if_configs_v2:
    iteration += 1
    model_name = f"{config['name']}_v{iteration}"
    
    train_result = train_model(dataset, model_name, "IFv1", config["params"])
    if train_result is None:
        continue
    
    test_result = test_model(dataset, model_name)
    if test_result is None:
        continue
    
    result_entry = {
        "iteration": iteration,
        "type": "IF",
        "name": config["name"],
        "model_name": model_name,
        "params": config["params"],
        "metrics": test_result,
        "timestamp": datetime.now().isoformat()
    }
    results.append(result_entry)
    log(f"Saved: {model_name}")
    time.sleep(2)

# Train AE models
log("\nTRAINING IMPROVED AUTOENCODER MODELS")
log("="*80)
for config in ae_configs_v2:
    iteration += 1
    model_name = f"{config['name']}_v{iteration}"
    
    train_result = train_model(dataset, model_name, "AEv1", config["params"])
    if train_result is None:
        continue
    
    test_result = test_model(dataset, model_name)
    if test_result is None:
        continue
    
    result_entry = {
        "iteration": iteration,
        "type": "AE",
        "name": config["name"],
        "model_name": model_name,
        "params": config["params"],
        "metrics": test_result,
        "timestamp": datetime.now().isoformat()
    }
    results.append(result_entry)
    log(f"Saved: {model_name}")
    time.sleep(2)

# Save second iteration results
if results:
    with open("/Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/model_training_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSecond iteration results saved: {len(results)} models")

log("\nSecond iteration completed!")
