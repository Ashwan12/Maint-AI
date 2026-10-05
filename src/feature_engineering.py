"""
Feature Engineering Module for MaintAI.
Calculates Remaining Useful Life (RUL), rolling aggregations, degradation trends,
and handles data preprocessing and scaling.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from src.config import (
    RAW_COLUMNS,
    ACTIVE_SENSORS,
    SETTING_COLUMNS,
    CONSTANT_SENSORS,
    MAX_RUL_CLIP,
    SENSOR_COLUMNS
)

def add_remaining_useful_life(df: pd.DataFrame, clip_threshold: Optional[int] = MAX_RUL_CLIP) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for run-to-failure training data.
    RUL(t) = max_cycle_for_unit - current_cycle.
    Optionally clips RUL to a maximum threshold (piecewise linear degradation).
    """
    df_out = df.copy()
    max_cycles = df_out.groupby("unit_id")["time_cycles"].transform("max")
    df_out["RUL_raw"] = max_cycles - df_out["time_cycles"]
    
    if clip_threshold is not None:
        df_out["RUL"] = df_out["RUL_raw"].clip(upper=clip_threshold)
    else:
        df_out["RUL"] = df_out["RUL_raw"]
        
    return df_out

def compute_rolling_features(
    df: pd.DataFrame, 
    sensors: List[str] = ACTIVE_SENSORS, 
    windows: List[int] = [5, 15]
) -> pd.DataFrame:
    """
    Computes rolling mean and standard deviation for active sensors per unit.
    Captures temporal degradation rate and sensor fluctuation.
    """
    df_out = df.copy()
    
    for w in windows:
        for s in sensors:
            if s in df_out.columns:
                mean_col = f"{s}_roll_mean_{w}"
                std_col = f"{s}_roll_std_{w}"
                
                # Rolling mean grouped by unit
                df_out[mean_col] = df_out.groupby("unit_id")[s].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
                # Rolling std grouped by unit
                df_out[std_col] = df_out.groupby("unit_id")[s].transform(
                    lambda x: x.rolling(window=w, min_periods=1).std().fillna(0.0)
                )
                
    return df_out

def compute_lag_features(
    df: pd.DataFrame, 
    sensors: List[str] = ACTIVE_SENSORS, 
    lags: List[int] = [5]
) -> pd.DataFrame:
    """
    Computes rate-of-change (difference) features over time lags.
    """
    df_out = df.copy()
    for lag in lags:
        for s in sensors:
            if s in df_out.columns:
                diff_col = f"{s}_diff_{lag}"
                df_out[diff_col] = df_out.groupby("unit_id")[s].transform(
                    lambda x: x.diff(periods=lag).fillna(0.0)
                )
    return df_out

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns list of feature column names excluding metadata and target columns.
    """
    excluded = ["unit_id", "time_cycles", "RUL", "RUL_raw", "failure_risk", "anomaly_label", "cluster"]
    features = [c for c in df.columns if c not in excluded and c not in CONSTANT_SENSORS]
    return features

def prepare_features(
    df: pd.DataFrame, 
    is_train: bool = True,
    sensors: List[str] = ACTIVE_SENSORS,
    clip_threshold: Optional[int] = MAX_RUL_CLIP
) -> pd.DataFrame:
    """
    Full feature pipeline:
    1. RUL calculation (if training data)
    2. Rolling statistics (mean, std)
    3. Lag differences
    """
    df_proc = df.copy()
    if is_train and "unit_id" in df_proc.columns and "time_cycles" in df_proc.columns:
        df_proc = add_remaining_useful_life(df_proc, clip_threshold=clip_threshold)
        
    df_proc = compute_rolling_features(df_proc, sensors=sensors, windows=[5, 15])
    df_proc = compute_lag_features(df_proc, sensors=sensors, lags=[5])
    
    return df_proc

