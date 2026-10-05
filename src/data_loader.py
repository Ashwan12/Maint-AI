"""
Data Loading and Generation Module for MaintAI.
Handles NASA CMAPSS Turbofan, CWRU Bearing, UCI Hydraulic, and custom uploaded CSV datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

from src.config import (
    RAW_COLUMNS,
    CMAPSS_TRAIN_FILE,
    CMAPSS_TEST_FILE,
    CMAPSS_RUL_FILE,
    SAMPLES_DATA_DIR
)

def load_cmapss_train(filepath: Path = CMAPSS_TRAIN_FILE) -> pd.DataFrame:
    """
    Loads NASA CMAPSS FD001 training dataset.
    26 columns: unit_id, time_cycles, 3 op settings, 21 sensors.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CMAPSS train file not found at {filepath}")
        
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=RAW_COLUMNS)
    df["unit_id"] = df["unit_id"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    return df

def load_cmapss_test(
    test_filepath: Path = CMAPSS_TEST_FILE, 
    rul_filepath: Path = CMAPSS_RUL_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads NASA CMAPSS FD001 test data and ground truth final RUL values.
    Calculates true RUL for each cycle in the test set.
    """
    if not test_filepath.exists() or not rul_filepath.exists():
        raise FileNotFoundError("CMAPSS test or RUL file not found.")
        
    df_test = pd.read_csv(test_filepath, sep=r"\s+", header=None, names=RAW_COLUMNS)
    df_test["unit_id"] = df_test["unit_id"].astype(int)
    df_test["time_cycles"] = df_test["time_cycles"].astype(int)
    
    df_rul = pd.read_csv(rul_filepath, sep=r"\s+", header=None, names=["true_rul_end"])
    df_rul["unit_id"] = range(1, len(df_rul) + 1)
    
    # Calculate ground truth RUL for each cycle in test data
    max_cycles_test = df_test.groupby("unit_id")["time_cycles"].max().reset_index()
    max_cycles_test = max_cycles_test.merge(df_rul, on="unit_id")
    
    df_test = df_test.merge(max_cycles_test, on="unit_id")
    df_test["RUL_true"] = (df_test["time_cycles_y"] - df_test["time_cycles_x"]) + df_test["true_rul_end"]
    df_test = df_test.rename(columns={"time_cycles_x": "time_cycles"}).drop(columns=["time_cycles_y"])
    
    return df_test, df_rul

def generate_bearing_dataset(save_path: Optional[Path] = None, seed: int = 42) -> pd.DataFrame:
    """
    Generates high-fidelity Case Western Reserve University (CWRU) style bearing degradation time series.
    Includes vibration RMS, Kurtosis, Crest Factor, Bearing Temp, RPM, and Motor Load.
    """
    np.random.seed(seed)
    records = []
    
    bearings = [
        {"id": 1, "load_hp": 1.0, "rpm": 1772, "fault": "Inner Race Defect", "max_hours": 320},
        {"id": 2, "load_hp": 2.0, "rpm": 1750, "fault": "Ball Defect", "max_hours": 280},
        {"id": 3, "load_hp": 3.0, "rpm": 1730, "fault": "Outer Race Defect", "max_hours": 240},
        {"id": 4, "load_hp": 1.0, "rpm": 1772, "fault": "Inner Race Defect", "max_hours": 360},
        {"id": 5, "load_hp": 0.0, "rpm": 1797, "fault": "Baseline / Normal", "max_hours": 400}
    ]
    
    for b in bearings:
        b_id = b["id"]
        load = b["load_hp"]
        rpm = b["rpm"]
        fault = b["fault"]
        max_h = b["max_hours"]
        
        for h in range(1, max_h + 1):
            rul = max_h - h
            pct_life = h / max_h
            
            # Baseline vibration plus exponential spike as failure approaches
            if fault == "Baseline / Normal":
                exp_deg = 0.05 * (pct_life ** 2)
            else:
                exp_deg = np.exp(3.5 * (pct_life - 0.6)) if pct_life > 0.6 else 0.02 * pct_life
                
            vib_de = 0.08 + (0.02 * load) + exp_deg + np.random.normal(0, 0.015)
            vib_fe = 0.06 + (0.015 * load) + (0.7 * exp_deg) + np.random.normal(0, 0.012)
            
            kurt = 3.0 + (8.5 * exp_deg) + np.random.normal(0, 0.2)
            crest = 3.2 + (3.8 * exp_deg) + np.random.normal(0, 0.15)
            temp_c = 42.0 + (3.0 * load) + (25.0 * exp_deg) + np.random.normal(0, 0.6)
            jitter_rpm = rpm - (15.0 * exp_deg) + np.random.normal(0, 2.0)
            
            records.append({
                "unit_id": b_id,
                "time_cycles": h,  # hours as time step
                "motor_load_hp": load,
                "motor_rpm": round(jitter_rpm, 1),
                "vibration_de_g": round(max(0.01, vib_de), 4),
                "vibration_fe_g": round(max(0.01, vib_fe), 4),
                "kurtosis": round(max(1.5, kurt), 3),
                "crest_factor": round(max(1.8, crest), 3),
                "bearing_temp_c": round(temp_c, 2),
                "fault_category": fault,
                "RUL": rul
            })
            
    df = pd.DataFrame(records)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df

def generate_hydraulic_dataset(save_path: Optional[Path] = None, seed: int = 101) -> pd.DataFrame:
    """
    Generates UCI Hydraulic Condition Monitoring Dataset representation.
    Monitors cooler condition, valve response, pump leakage, and accumulator pressure.
    """
    np.random.seed(seed)
    records = []
    
    systems = [
        {"id": 1, "valve_wear": "High", "cooling": "Degraded", "max_cycles": 220},
        {"id": 2, "valve_wear": "Moderate", "cooling": "Normal", "max_cycles": 310},
        {"id": 3, "valve_wear": "Low", "cooling": "Normal", "max_cycles": 450},
        {"id": 4, "valve_wear": "Severe", "cooling": "Severe Leak", "max_cycles": 180}
    ]
    
    for s in systems:
        sys_id = s["id"]
        max_c = s["max_cycles"]
        
        for c in range(1, max_c + 1):
            rul = max_c - c
            deg = c / max_c
            
            pump_pressure = 185.0 - (25.0 * (deg ** 1.8)) + np.random.normal(0, 1.2)
            accum_pressure = 105.0 - (18.0 * (deg ** 2)) + np.random.normal(0, 0.8)
            oil_temp = 42.0 + (28.0 * deg) + np.random.normal(0, 0.5)
            flow_rate = 9.8 - (2.5 * deg) + np.random.normal(0, 0.1)
            motor_power = 2250.0 + (320.0 * deg) + np.random.normal(0, 15.0)
            vibration = 0.65 + (1.8 * (deg ** 2)) + np.random.normal(0, 0.05)
            cooler_eff = max(5.0, 100.0 - (90.0 * deg) + np.random.normal(0, 2.0))
            
            records.append({
                "unit_id": sys_id,
                "time_cycles": c,
                "pump_pressure_bar": round(pump_pressure, 2),
                "accum_pressure_bar": round(accum_pressure, 2),
                "oil_temperature_c": round(oil_temp, 2),
                "flow_rate_lmin": round(max(0.5, flow_rate), 2),
                "motor_power_w": round(motor_power, 1),
                "vibration_mms": round(max(0.1, vibration), 3),
                "cooler_efficiency_pct": round(cooler_eff, 1),
                "valve_status": s["valve_wear"],
                "RUL": rul
            })
            
    df = pd.DataFrame(records)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df

def generate_sample_turbofan_upload(save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Extracts a representative unit from CMAPSS test data to serve as a ready-to-use CSV upload sample.
    """
    df_train = load_cmapss_train()
    # Take unit 10, which ran for 166 cycles to failure
    sample_df = df_train[df_train["unit_id"] == 10].copy().reset_index(drop=True)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(save_path, index=False)
    return sample_df

def load_or_create_all_sample_datasets():
    """
    Ensures all sample datasets (Bearing, Hydraulic, Turbofan Sample) exist in data/samples/.
    """
    SAMPLES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    bearing_file = SAMPLES_DATA_DIR / "case_western_bearing.csv"
    if not bearing_file.exists():
        generate_bearing_dataset(save_path=bearing_file)
        
    hydraulic_file = SAMPLES_DATA_DIR / "uci_hydraulic_condition.csv"
    if not hydraulic_file.exists():
        generate_hydraulic_dataset(save_path=hydraulic_file)
        
    sample_turbofan_file = SAMPLES_DATA_DIR / "sample_turbofan_upload.csv"
    if not sample_turbofan_file.exists() and CMAPSS_TRAIN_FILE.exists():
        generate_sample_turbofan_upload(save_path=sample_turbofan_file)

def parse_user_csv(file_obj_or_path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses user uploaded CSV file, intelligently identifying cycle, unit, and sensor columns.
    """
    df = pd.read_csv(file_obj_or_path)
    
    cols_lower = {col: col.lower().strip() for col in df.columns}
    
    # 1. Identify Unit / ID column
    unit_col = None
    for orig_col, low in cols_lower.items():
        if any(term in low for term in ["unit", "engine", "machine", "asset", "bearing", "device", "id"]):
            unit_col = orig_col
            break
            
    # 2. Identify Time / Cycle column
    time_col = None
    for orig_col, low in cols_lower.items():
        if any(term in low for term in ["cycle", "time", "hour", "step", "timestamp", "date", "index"]):
            time_col = orig_col
            break
            
    # Normalize unit and time columns if found, else default
    if unit_col:
        df["unit_id"] = df[unit_col]
    else:
        df["unit_id"] = 1
        
    if time_col:
        df["time_cycles"] = df[time_col]
    else:
        df["time_cycles"] = range(1, len(df) + 1)
        
    # Numerical sensor columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sensor_cols = [c for c in numeric_cols if c not in ["unit_id", "time_cycles", unit_col, time_col, "RUL", "RUL_raw"]]
    
    metadata = {
        "unit_col": unit_col or "Synthetic Unit",
        "time_col": time_col or "Row Index",
        "sensor_cols": sensor_cols,
        "row_count": len(df),
        "unit_count": df["unit_id"].nunique()
    }
    
    return df, metadata

