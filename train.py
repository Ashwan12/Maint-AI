"""
Training Pipeline for MaintAI.
Trains XGBoost, Random Forest, Isolation Forest, and K-Means models on NASA CMAPSS FD001 dataset,
evaluates on test benchmark, and serializes trained models and sample datasets.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import (
    CMAPSS_TRAIN_FILE,
    CMAPSS_TEST_FILE,
    CMAPSS_RUL_FILE,
    XGB_MODEL_FILE,
    RF_MODEL_FILE,
    ISOLATION_FOREST_FILE,
    KMEANS_FILE,
    SCALER_FILE,
    METADATA_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    ACTIVE_SENSORS
)
from src.data_loader import (
    load_cmapss_train,
    load_cmapss_test,
    load_or_create_all_sample_datasets
)
from src.feature_engineering import (
    prepare_features,
    get_feature_columns
)
from src.models import (
    PredictiveMaintenanceModel,
    calculate_metrics
)
from src.anomaly_detection import AnomalyClusterEngine

def run_training_pipeline():
    print("=" * 70)
    print("  MaintAI: Industrial Predictive Maintenance Training Pipeline")
    print("=" * 70)
    start_time = time.time()
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1/6] Loading NASA CMAPSS Turbofan Dataset...")
    df_train = load_cmapss_train(CMAPSS_TRAIN_FILE)
    df_test, df_rul = load_cmapss_test(CMAPSS_TEST_FILE, CMAPSS_RUL_FILE)
    print(f" -> Train records: {len(df_train):,} rows across {df_train['unit_id'].nunique()} engine units")
    print(f" -> Test records:  {len(df_test):,} rows across {df_test['unit_id'].nunique()} engine units")
    
    # 2. Feature Engineering
    print("\n[Step 2/6] Feature Engineering & Degradation Labeling...")
    df_train_proc = prepare_features(df_train, is_train=True, sensors=ACTIVE_SENSORS)
    df_test_proc = prepare_features(df_test, is_train=False, sensors=ACTIVE_SENSORS)
    
    feature_cols = get_feature_columns(df_train_proc)
    print(f" -> Engineered {len(feature_cols)} features (rolling means, stds, rate-of-change lags)")
    
    # Save processed parquets / csv for fast app loading
    train_cache = PROCESSED_DATA_DIR / "train_features.parquet"
    test_cache = PROCESSED_DATA_DIR / "test_features.parquet"
    df_train_proc.to_parquet(train_cache, index=False)
    df_test_proc.to_parquet(test_cache, index=False)
    print(f" -> Cached processed features to {PROCESSED_DATA_DIR}")
    
    X_train = df_train_proc[feature_cols]
    y_train = df_train_proc["RUL"]
    
    # Evaluation dataset: Last cycle of each unit in test set
    test_last_indices = df_test_proc.groupby("unit_id")["time_cycles"].idxmax()
    df_test_last = df_test_proc.loc[test_last_indices].reset_index(drop=True)
    X_test_last = df_test_last[feature_cols]
    y_test_last = df_test_last["RUL_true"].values
    
    # 3. Train XGBoost Model
    print("\n[Step 3/6] Training XGBoost Regressor for RUL Forecasting...")
    xgb_wrapper = PredictiveMaintenanceModel(model_type="xgboost")
    xgb_wrapper.train(
        X_train, 
        y_train, 
        feature_names=feature_cols,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04
    )
    xgb_preds_test, _, _ = xgb_wrapper.predict_with_confidence(X_test_last)
    xgb_metrics = calculate_metrics(y_test_last, xgb_preds_test)
    xgb_wrapper.save(XGB_MODEL_FILE)
    print(f" -> XGBoost Test RMSE: {xgb_metrics['rmse']} cycles")
    print(f" -> XGBoost Test MAE:  {xgb_metrics['mae']} cycles")
    print(f" -> XGBoost Test R²:   {xgb_metrics['r2']}")
    print(f" -> NASA CMAPSS Score: {xgb_metrics['cmapss_score']}")
    print(f" -> Saved to: {XGB_MODEL_FILE}")
    
    # 4. Train Random Forest Model
    print("\n[Step 4/6] Training Random Forest Regressor with Ensemble Uncertainty...")
    rf_wrapper = PredictiveMaintenanceModel(model_type="random_forest")
    rf_wrapper.train(
        X_train, 
        y_train, 
        feature_names=feature_cols,
        n_estimators=180,
        max_depth=15
    )
    rf_preds_test, _, _ = rf_wrapper.predict_with_confidence(X_test_last)
    rf_metrics = calculate_metrics(y_test_last, rf_preds_test)
    rf_wrapper.save(RF_MODEL_FILE)
    print(f" -> Random Forest Test RMSE: {rf_metrics['rmse']} cycles")
    print(f" -> Random Forest Test MAE:  {rf_metrics['mae']} cycles")
    print(f" -> Random Forest Test R²:   {rf_metrics['r2']}")
    print(f" -> NASA CMAPSS Score:       {rf_metrics['cmapss_score']}")
    print(f" -> Saved to: {RF_MODEL_FILE}")
    
    # 5. Train Unsupervised Anomaly & Clustering Engine
    print("\n[Step 5/6] Fitting Unsupervised Anomaly Engine (Isolation Forest, KMeans, PCA)...")
    anomaly_engine = AnomalyClusterEngine(n_clusters=3, contamination=0.06)
    anomaly_engine.fit(df_train_proc, feature_cols=ACTIVE_SENSORS)
    anomaly_engine.save(ISOLATION_FOREST_FILE)
    print(f" -> Saved Anomaly Engine to: {ISOLATION_FOREST_FILE}")
    
    # 6. Sample Datasets Preparation
    print("\n[Step 6/6] Generating CWRU Bearing & UCI Hydraulic Benchmark Datasets...")
    load_or_create_all_sample_datasets()
    print(" -> All benchmark datasets prepared in data/samples/")
    
    # Save Metadata and Benchmark Results
    metadata = {
        "dataset": "NASA CMAPSS Turbofan Engine FD001",
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_duration_seconds": round(time.time() - start_time, 2),
        "total_train_rows": len(df_train),
        "total_test_units": len(df_rul),
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "models": {
            "xgboost": {
                "file": str(XGB_MODEL_FILE.name),
                "metrics": xgb_metrics,
                "top_features": xgb_wrapper.get_feature_importances(10).to_dict(orient="records")
            },
            "random_forest": {
                "file": str(RF_MODEL_FILE.name),
                "metrics": rf_metrics,
                "top_features": rf_wrapper.get_feature_importances(10).to_dict(orient="records")
            }
        },
        "anomaly_clusters": anomaly_engine.cluster_names
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Model Summary Saved to: {METADATA_FILE}")
    print(f"Elapsed Time: {metadata['training_duration_seconds']}s")
    print("=" * 70)

if __name__ == "__main__":
    run_training_pipeline()

