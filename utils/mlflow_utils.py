"""
MLflow Utilities - Functions to read MLflow experiment data
"""
import os
import yaml
from typing import Dict, List, Optional, Tuple
import pandas as pd

MLRUNS_DIR = "mlruns"

def get_experiment_id() -> Optional[str]:
    """Auto-detect experiment ID from mlruns directory"""
    if not os.path.exists(MLRUNS_DIR):
        return None
    
    # Look for experiment directories (numeric IDs), skip "0" and ".trash"
    candidates = []
    for item in os.listdir(MLRUNS_DIR):
        item_path = os.path.join(MLRUNS_DIR, item)
        if os.path.isdir(item_path) and item.isdigit() and item != "0":
            # Check if it has a meta.yaml (experiment metadata)
            meta_path = os.path.join(item_path, "meta.yaml")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = yaml.safe_load(f)
                        if meta and 'name' in meta:
                            candidates.append((item, meta.get('name', '')))
                except:
                    continue
    
    # Prefer experiment named "EMIPredict_AI" or return the first valid one
    for exp_id, exp_name in candidates:
        if exp_name == "EMIPredict_AI":
            return exp_id
    if candidates:
        return candidates[0][0]
    return None

def get_experiment_id_cached():
    """Get experiment ID with caching"""
    if not hasattr(get_experiment_id_cached, '_cached_id'):
        get_experiment_id_cached._cached_id = get_experiment_id()
    return get_experiment_id_cached._cached_id

# Initialize at module load, but can be refreshed
EXPERIMENT_ID = get_experiment_id()


def read_mlflow_metric(run_id: str, metric_name: str) -> Optional[float]:
    """Read a metric value from MLflow run"""
    exp_id = get_experiment_id() if EXPERIMENT_ID == "0" or not EXPERIMENT_ID else EXPERIMENT_ID
    if not exp_id or exp_id == "0":
        return None
    metric_path = os.path.join(MLRUNS_DIR, exp_id, run_id, "metrics", metric_name)
    if os.path.exists(metric_path):
        try:
            with open(metric_path, 'r') as f:
                # MLflow metrics format: timestamp value
                line = f.read().strip().split('\n')[-1]  # Get last line (latest value)
                return float(line.split()[1])
        except:
            return None
    return None


def read_mlflow_param(run_id: str, param_name: str) -> Optional[str]:
    """Read a parameter value from MLflow run"""
    exp_id = get_experiment_id() if EXPERIMENT_ID == "0" or not EXPERIMENT_ID else EXPERIMENT_ID
    if not exp_id or exp_id == "0":
        return None
    param_path = os.path.join(MLRUNS_DIR, exp_id, run_id, "params", param_name)
    if os.path.exists(param_path):
        try:
            with open(param_path, 'r') as f:
                return f.read().strip()
        except:
            return None
    return None


def read_mlflow_tag(run_id: str, tag_name: str) -> Optional[str]:
    """Read a tag value from MLflow run"""
    exp_id = get_experiment_id() if EXPERIMENT_ID == "0" or not EXPERIMENT_ID else EXPERIMENT_ID
    if not exp_id or exp_id == "0":
        return None
    tag_path = os.path.join(MLRUNS_DIR, exp_id, run_id, "tags", tag_name)
    if os.path.exists(tag_path):
        try:
            with open(tag_path, 'r') as f:
                return f.read().strip()
        except:
            return None
    return None


def get_run_metadata(run_id: str) -> Dict:
    """Get all metadata for a run"""
    meta_path = os.path.join(MLRUNS_DIR, EXPERIMENT_ID, run_id, "meta.yaml")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    return {}


def get_all_runs() -> List[Dict]:
    """Get all runs from the experiment"""
    runs = []
    exp_id = get_experiment_id() if EXPERIMENT_ID == "0" or not EXPERIMENT_ID else EXPERIMENT_ID
    if not exp_id or exp_id == "0":
        return runs
    exp_dir = os.path.join(MLRUNS_DIR, exp_id)
    
    if not os.path.exists(exp_dir):
        return runs
    
    for item in os.listdir(exp_dir):
        run_path = os.path.join(exp_dir, item)
        if os.path.isdir(run_path) and item not in ['models', 'meta.yaml']:
            try:
                run_name = read_mlflow_tag(item, "mlflow.runName") or item
                
                # Check if it's classification or regression
                has_accuracy = read_mlflow_metric(item, "accuracy") is not None
                has_rmse = read_mlflow_metric(item, "rmse") is not None
                
                run_data = {
                    "run_id": item,
                    "run_name": run_name,
                    "type": "classification" if has_accuracy else "regression" if has_rmse else "unknown",
                    "metrics": {},
                    "params": {}
                }
                
                # Read all metrics
                metrics_dir = os.path.join(run_path, "metrics")
                if os.path.exists(metrics_dir):
                    for metric_file in os.listdir(metrics_dir):
                        value = read_mlflow_metric(item, metric_file)
                        if value is not None:
                            run_data["metrics"][metric_file] = value
                
                # Read all parameters
                params_dir = os.path.join(run_path, "params")
                if os.path.exists(params_dir):
                    for param_file in os.listdir(params_dir):
                        value = read_mlflow_param(item, param_file)
                        if value:
                            run_data["params"][param_file] = value
                
                runs.append(run_data)
            except Exception as e:
                continue
    
    return runs


def get_model_metrics(model_type: str = "final") -> Dict:
    """
    Get metrics for the final models (XGBoost)
    model_type: "final", "classification", or "regression"
    """
    runs = get_all_runs()
    
    if not runs:
        return {}
    
    if model_type == "final":
        # Find XGBoost models (usually have "XGBoost" in name or most recent)
        xgb_runs = [r for r in runs if "XGBoost" in r.get("run_name", "").upper() or 
                   "xgboost" in r.get("run_name", "").lower() or
                   "Final" in r.get("run_name", "")]
        
        # If no XGBoost found, get the best performing models
        if not xgb_runs:
            clf_runs = [r for r in runs if r["type"] == "classification"]
            reg_runs = [r for r in runs if r["type"] == "regression"]
            
            # Get best classifier (highest accuracy)
            clf_run = max(clf_runs, key=lambda x: x.get("metrics", {}).get("accuracy", 0)) if clf_runs else None
            # Get best regressor (highest r2)
            reg_run = max(reg_runs, key=lambda x: x.get("metrics", {}).get("r2", 0)) if reg_runs else None
        else:
            clf_run = next((r for r in xgb_runs if r["type"] == "classification"), None)
            reg_run = next((r for r in xgb_runs if r["type"] == "regression"), None)
        
        return {
            "classifier": clf_run.get("metrics", {}) if clf_run else {},
            "regressor": reg_run.get("metrics", {}) if reg_run else {}
        }
    elif model_type == "classification":
        clf_runs = [r for r in runs if r["type"] == "classification"]
        return {r["run_name"]: r.get("metrics", {}) for r in clf_runs}
    elif model_type == "regression":
        reg_runs = [r for r in runs if r["type"] == "regression"]
        return {r["run_name"]: r.get("metrics", {}) for r in reg_runs}
    
    return {}


def get_all_model_comparison() -> Dict:
    """Get all models for comparison"""
    runs = get_all_runs()
    
    classification_models = []
    regression_models = []
    
    for run in runs:
        model_data = {
            "name": run["run_name"],
            "metrics": run["metrics"],
            "params": run["params"]
        }
        
        if run["type"] == "classification":
            classification_models.append(model_data)
        elif run["type"] == "regression":
            regression_models.append(model_data)
    
    return {
        "classification": classification_models,
        "regression": regression_models
    }

