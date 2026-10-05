"""
Automated Unit and Integration Tests for MaintAI.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    XGB_MODEL_FILE,
    RF_MODEL_FILE,
    ISOLATION_FOREST_FILE,
    ACTIVE_SENSORS,
    SAMPLES_DATA_DIR
)
from src.data_loader import (
    load_cmapss_train,
    generate_bearing_dataset,
    generate_hydraulic_dataset,
    parse_user_csv
)
from src.feature_engineering import (
    add_remaining_useful_life,
    compute_rolling_features,
    prepare_features
)
from src.models import (
    PredictiveMaintenanceModel,
    calculate_metrics
)
from src.anomaly_detection import AnomalyClusterEngine
from src.visualizer import (
    plot_health_gauge,
    plot_rul_forecast,
    plot_sensor_trends
)

def test_data_loader_bearing_and_hydraulic():
    df_b = generate_bearing_dataset()
    assert not df_b.empty
    assert "vibration_de_g" in df_b.columns
    assert "RUL" in df_b.columns
    
    df_h = generate_hydraulic_dataset()
    assert not df_h.empty
    assert "pump_pressure_bar" in df_h.columns
    assert "cooler_efficiency_pct" in df_h.columns

def test_feature_engineering_rul():
    dummy_data = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2],
        "sensor_2": [642.1, 642.5, 643.0, 641.8, 642.2]
    })
    df_rul = add_remaining_useful_life(dummy_data, clip_threshold=125)
    assert "RUL" in df_rul.columns
    # For unit 1, max cycle is 3, so RUL for cycle 1 should be 2
    assert df_rul.loc[0, "RUL"] == 2
    assert df_rul.loc[2, "RUL"] == 0

def test_rolling_features():
    dummy = pd.DataFrame({
        "unit_id": [1, 1, 1, 1, 1],
        "time_cycles": [1, 2, 3, 4, 5],
        "sensor_2": [10.0, 12.0, 14.0, 16.0, 18.0]
    })
    df_roll = compute_rolling_features(dummy, sensors=["sensor_2"], windows=[3])
    assert "sensor_2_roll_mean_3" in df_roll.columns
    assert "sensor_2_roll_std_3" in df_roll.columns

def test_models_exist_and_predict():
    assert XGB_MODEL_FILE.exists(), "XGBoost model file not found!"
    assert RF_MODEL_FILE.exists(), "Random Forest model file not found!"
    
    xgb_mod = PredictiveMaintenanceModel.load(XGB_MODEL_FILE)
    rf_mod = PredictiveMaintenanceModel.load(RF_MODEL_FILE)
    
    # Create test input with required feature names
    dummy_input = pd.DataFrame({col: [0.5] for col in xgb_mod.feature_names})
    
    xgb_pred, xgb_low, xgb_upp = xgb_mod.predict_with_confidence(dummy_input)
    assert len(xgb_pred) == 1
    assert xgb_upp[0] >= xgb_pred[0] >= xgb_low[0]
    
    rf_pred, rf_low, rf_upp = rf_mod.predict_with_confidence(dummy_input)
    assert len(rf_pred) == 1
    assert rf_upp[0] >= rf_pred[0] >= rf_low[0]

def test_anomaly_engine():
    assert ISOLATION_FOREST_FILE.exists(), "Isolation Forest model file not found!"
    engine = AnomalyClusterEngine.load(ISOLATION_FOREST_FILE)
    
    dummy_input = pd.DataFrame({col: [np.random.normal(0, 1)] for col in engine.feature_cols})
    dummy_input["unit_id"] = 1
    dummy_input["time_cycles"] = 1
    
    out = engine.transform(dummy_input)
    assert "is_anomaly" in out.columns
    assert "anomaly_score" in out.columns
    assert "cluster_name" in out.columns

def test_visualizers():
    fig_gauge = plot_health_gauge(65.0)
    assert fig_gauge is not None
    
    cycles = np.array([1, 2, 3, 4])
    pred = np.array([100, 95, 90, 85])
    low = pred - 10
    upp = pred + 10
    fig_fc = plot_rul_forecast(cycles, pred, low, upp)
    assert fig_fc is not None

def test_user_csv_parser():
    sample_file = SAMPLES_DATA_DIR / "sample_turbofan_upload.csv"
    assert sample_file.exists()
    df, meta = parse_user_csv(sample_file)
    assert not df.empty
    assert meta["unit_count"] >= 1
    assert len(meta["sensor_cols"]) > 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])

