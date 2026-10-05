"""
Configuration and constants for MaintAI Predictive Maintenance System.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# CMAPSS Raw File Paths
CMAPSS_TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
CMAPSS_TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
CMAPSS_RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"

# CMAPSS Column Definitions
SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = ["unit_id", "time_cycles"] + SETTING_COLUMNS + SENSOR_COLUMNS

# CMAPSS Sensors with negligible/zero variance in FD001
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
ACTIVE_SENSORS = [s for s in SENSOR_COLUMNS if s not in CONSTANT_SENSORS]

# Engineering descriptions for CMAPSS Sensors
SENSOR_DESCRIPTIONS = {
    "sensor_1": "Fan Inlet Temperature (T2) [°R]",
    "sensor_2": "LPC Outlet Temperature (T24) [°R]",
    "sensor_3": "HPC Outlet Temperature (T30) [°R]",
    "sensor_4": "LPT Outlet Temperature (T50) [°R]",
    "sensor_5": "Fan Inlet Pressure (P2) [psia]",
    "sensor_6": "Bypass Duct Pressure (P15) [psia]",
    "sensor_7": "HPC Outlet Pressure (P30) [psia]",
    "sensor_8": "Physical Fan Speed (Nf) [rpm]",
    "sensor_9": "Physical Core Speed (Nc) [rpm]",
    "sensor_10": "Engine Pressure Ratio (epr)",
    "sensor_11": "HPC Static Pressure (Ps30) [psia]",
    "sensor_12": "Fuel Flow Ratio (phi) [pps/psia]",
    "sensor_13": "Corrected Fan Speed (NRf) [rpm]",
    "sensor_14": "Corrected Core Speed (NRc) [rpm]",
    "sensor_15": "Bypass Ratio (BPR)",
    "sensor_16": "Burner Fuel-Air Ratio (farB)",
    "sensor_17": "Bleed Enthalpy (htBleed)",
    "sensor_18": "Demanded Fan Speed (Nf_dmd) [rpm]",
    "sensor_19": "Demanded Corrected Fan Speed (PCNfR_dmd)",
    "sensor_20": "HPT Coolant Bleed (W31) [lbm/s]",
    "sensor_21": "LPT Coolant Bleed (W32) [lbm/s]",
}

# RUL and Risk Thresholds
MAX_RUL_CLIP = 125  # Standard CMAPSS threshold for piecewise linear degradation
HEALTH_THRESHOLDS = {
    "CRITICAL": 30,  # RUL <= 30 cycles: High risk of imminent failure
    "WARNING": 75,   # 30 < RUL <= 75 cycles: Maintenance advisory / inspection
    "HEALTHY": 75    # RUL > 75 cycles: Normal operational state
}

# Model Filenames
XGB_MODEL_FILE = MODELS_DIR / "xgboost_rul_model.joblib"
RF_MODEL_FILE = MODELS_DIR / "random_forest_rul_model.joblib"
ISOLATION_FOREST_FILE = MODELS_DIR / "isolation_forest_model.joblib"
KMEANS_FILE = MODELS_DIR / "kmeans_clusterer.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
PCA_FILE = MODELS_DIR / "pca_transformer.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

