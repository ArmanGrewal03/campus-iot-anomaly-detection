#!/usr/bin/env python3
"""
Script to train and optimize AE and IF models across different hyperparameters.
Logs results to track performance improvements.
"""
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Configuration
DATA_INGESTION_URL = "http://127.0.0.1:8000"
MODEL_SERVICE_URL = "http://127.0.0.1:8001"

# Results tracker
results = []

def log(msg: str, level: str = "INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

def get_available_datasets() -> List[str]:
    """Get list of available datasets"""
    log("Fetching available datasets...")
    try:
        response = requests.get(f"{DATA_INGESTION_URL}/tables", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        datasets = set()
        if "tables" in data:
            for table in data["tables"]:
                if table.startswith("csv_data_"):
                    datasets.add(table.replace("csv_data_", ""))
                elif table.startswith("inserted_data_"):
                    datasets.add(table.replace("inserted_data_", ""))
        
        datasets_list = sorted(list(datasets))
        log(f"Found {len(datasets_list)} dataset(s): {datasets_list}")
        return datasets_list
    except Exception as e:
        log(f"Error fetching datasets: {e}", "ERROR")
        return []

def datasets_have_data(dataset: str) -> Tuple[bool, str]:
    """Check if dataset has training and testing data"""
    log(f"Checking data status for dataset: {dataset}")
    try:
        headers = {"dataset_name": dataset}
        
        # Check training data
        train_response = requests.get(
            f"{DATA_INGESTION_URL}/training?limit=1&offset=0",
            headers=headers,
            timeout=10
        )
        train_response.raise_for_status()
        train_data = train_response.json()
        has_training = train_data.get("returned_rows", 0) > 0
        
        # Check testing data
        test_response = requests.get(
            f"{DATA_INGESTION_URL}/testing?limit=1&offset=0",
            headers=headers,
            timeout=10
        )
        test_response.raise_for_status()
        test_data = test_response.json()
        has_testing = test_data.get("returned_rows", 0) > 0
        
        if has_training and has_testing:
            return True, f"Training: {train_data.get('total_rows', '?')} rows, Testing: {test_data.get('total_rows', '?')} rows"
        else:
            return False, f"Training: {has_training}, Testing: {has_testing}"
    except Exception as e:
        log(f"Error checking data status: {e}", "ERROR")
        return False, str(e)

def train_model(dataset: str, model_name: str, model_type: str, params: Dict) -> Optional[Dict]:
    """Train a model with given parameters"""
    log(f"Training {model_type} model: {model_name}")
    log(f"  Parameters: {json.dumps(params, indent=2)}")
    
    try:
        headers = {
            "dataset_name": dataset,
            "model_name": model_name,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model_type": model_type,
            **params
        }
        
        start_time = time.time()
        response = requests.post(
            f"{MODEL_SERVICE_URL}/train",
            headers=headers,
            json=payload,
            timeout=300  # 5 minute timeout for training
        )
        training_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            log(f"Training completed in {training_duration:.2f}s")
            return result.get("training_params", {})
        else:
            log(f"Training failed with status {response.status_code}: {response.text}", "ERROR")
            return None
    except requests.exceptions.Timeout:
        log(f"Training timed out after 300s", "ERROR")
        return None
    except Exception as e:
        log(f"Training error: {e}", "ERROR")
        return None

def test_model(dataset: str, model_name: str) -> Optional[Dict]:
    """Test a trained model and return metrics"""
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
            timeout=300  # 5 minute timeout for testing
        )
        testing_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            metrics = result.get("metrics", {})
            log(f"Testing completed in {testing_duration:.2f}s")
            log(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            log(f"  Precision: {metrics.get('precision', 0):.4f}")
            log(f"  Recall: {metrics.get('recall', 0):.4f}")
            log(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            return metrics
        else:
            log(f"Testing failed with status {response.status_code}: {response.text}", "ERROR")
            return None
    except requests.exceptions.Timeout:
        log(f"Testing timed out after 300s", "ERROR")
        return None
    except Exception as e:
        log(f"Testing error: {e}", "ERROR")
        return None

def run_training_iterations(dataset: str):
    """Run multiple training iterations with different hyperparameters"""
    log(f"\n{'='*80}")
    log(f"Starting training iterations for dataset: {dataset}")
    log(f"{'='*80}\n")
    
    # Best known Isolation Forest configuration from prior tuning
    if_configs = [
        {
            "name": "IF-Extreme-Contamination",
            "params": {
                "n_estimators": 100,
                "contamination": 0.25
            }
        },
    ]
    
    # Best known Autoencoder configuration from prior tuning
    ae_configs = [
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
    ]
    
    iteration = 0
    
    # Train and test Isolation Forest models
    log("\n" + "="*80)
    log("ISOLATION FOREST MODELS")
    log("="*80 + "\n")
    
    for config in if_configs:
        iteration += 1
        model_name = f"{config['name']}_v{iteration}"
        
        log(f"\n[Iteration {iteration}] {config['name']}")
        
        # Train
        train_result = train_model(dataset, model_name, "IFv1", config["params"])
        if train_result is None:
            log(f"Skipping test due to training failure", "WARN")
            continue
        
        # Test
        test_result = test_model(dataset, model_name)
        if test_result is None:
            log(f"Skipping save due to testing failure", "WARN")
            continue
        
        # Save result
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
        
        log(f"Result saved: {model_name}")
        time.sleep(2)  # Brief pause between iterations
    
    # Train and test Autoencoder models
    log("\n" + "="*80)
    log("AUTOENCODER MODELS")
    log("="*80 + "\n")
    
    for config in ae_configs:
        iteration += 1
        model_name = f"{config['name']}_v{iteration}"
        
        log(f"\n[Iteration {iteration}] {config['name']}")
        
        # Train
        train_result = train_model(dataset, model_name, "AEv1", config["params"])
        if train_result is None:
            log(f"Skipping test due to training failure", "WARN")
            continue
        
        # Test
        test_result = test_model(dataset, model_name)
        if test_result is None:
            log(f"Skipping save due to testing failure", "WARN")
            continue
        
        # Save result
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
        
        log(f"Result saved: {model_name}")
        time.sleep(2)  # Brief pause between iterations

def print_summary():
    """Print summary of all results"""
    log("\n" + "="*80)
    log("SUMMARY OF ALL TRAIN RUNS")
    log("="*80 + "\n")
    
    if not results:
        log("No results to display")
        return
    
    # Group by type
    if_results = [r for r in results if r["type"] == "IF"]
    ae_results = [r for r in results if r["type"] == "AE"]
    
    # Print IF results
    if if_results:
        log("\nISOLATION FOREST RESULTS:")
        log("-" * 80)
        log(f"{'Name':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        log("-" * 80)
        
        best_if = None
        for r in sorted(if_results, key=lambda x: x["metrics"].get("f1_score", 0), reverse=True):
            metrics = r["metrics"]
            log(f"{r['name']:<30} {metrics.get('accuracy', 0):<12.4f} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1_score', 0):<12.4f}")
            if best_if is None:
                best_if = r
        
        if best_if:
            log(f"\nBest IF: {best_if['name']}")
            log(f"  F1-Score: {best_if['metrics'].get('f1_score', 0):.4f}")
            log(f"  ROC-AUC: {best_if['metrics'].get('roc_auc', 0):.4f}")
    
    # Print AE results
    if ae_results:
        log("\nAUTOENCODER RESULTS:")
        log("-" * 80)
        log(f"{'Name':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        log("-" * 80)
        
        best_ae = None
        for r in sorted(ae_results, key=lambda x: x["metrics"].get("f1_score", 0), reverse=True):
            metrics = r["metrics"]
            log(f"{r['name']:<30} {metrics.get('accuracy', 0):<12.4f} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1_score', 0):<12.4f}")
            if best_ae is None:
                best_ae = r
        
        if best_ae:
            log(f"\nBest AE: {best_ae['name']}")
            log(f"  F1-Score: {best_ae['metrics'].get('f1_score', 0):.4f}")
            log(f"  ROC-AUC: {best_ae['metrics'].get('roc_auc', 0):.4f}")
    
    # Save full results to file
    results_file = "model_training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nFull results saved to: {results_file}")

def main():
    """Main entry point"""
    log("Starting Model Training and Optimization Script")
    
    # Get available datasets
    datasets = get_available_datasets()
    if not datasets:
        log("No datasets found", "ERROR")
        return
    
    # Find first dataset with training/testing data
    selected_dataset = None
    for dataset in datasets:
        has_data, status = datasets_have_data(dataset)
        if has_data:
            log(f"Dataset '{dataset}' has data: {status}")
            selected_dataset = dataset
            break
        else:
            log(f"Dataset '{dataset}' status: {status}")
    
    if selected_dataset is None:
        log("No dataset with training/testing data found", "ERROR")
        log("Please upload and validate a dataset first", "ERROR")
        return
    
    # Run training iterations
    run_training_iterations(selected_dataset)
    
    # Print summary
    print_summary()
    
    log("\nScript completed successfully!")

if __name__ == "__main__":
    main()
