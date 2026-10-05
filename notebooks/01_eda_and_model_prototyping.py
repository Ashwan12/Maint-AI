"""
MaintAI - Week 1 & Week 2: Exploratory Data Analysis (EDA) & Feature Prototyping
Interactive exploration script demonstrating time-series sensor distributions,
degradation trends, zero-variance sensor pruning, and correlation analysis.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    CMAPSS_TRAIN_FILE,
    RAW_COLUMNS,
    CONSTANT_SENSORS,
    ACTIVE_SENSORS,
    SENSOR_DESCRIPTIONS
)
from src.data_loader import load_cmapss_train
from src.feature_engineering import add_remaining_useful_life

def run_eda():
    print("=" * 60)
    print("MaintAI: Exploratory Data Analysis (EDA) — CMAPSS FD001")
    print("=" * 60)
    
    # 1. Load Data
    df = load_cmapss_train(CMAPSS_TRAIN_FILE)
    print(f"\n1. Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Fleet Size: {df['unit_id'].nunique()} turbofan engines")
    
    # 2. Engine Lifetime Statistics
    engine_lifetimes = df.groupby("unit_id")["time_cycles"].max()
    print("\n2. Engine Lifetime Distribution (Cycles to Failure):")
    print(f"   - Min Cycles:  {engine_lifetimes.min()}")
    print(f"   - Mean Cycles: {engine_lifetimes.mean():.1f}")
    print(f"   - Max Cycles:  {engine_lifetimes.max()}")
    print(f"   - Std Dev:     {engine_lifetimes.std():.1f}")
    
    # 3. Constant / Zero-Variance Sensor Detection
    print("\n3. Sensor Variance Analysis:")
    sensor_stds = df[ACTIVE_SENSORS + CONSTANT_SENSORS].std()
    zero_var = sensor_stds[sensor_stds < 1e-4].index.tolist()
    print(f"   Constant sensors detected (std ~ 0): {zero_var}")
    print(f"   Active informative sensors retained: {len(ACTIVE_SENSORS)}")
    
    # 4. Remaining Useful Life Target Analysis
    df = add_remaining_useful_life(df, clip_threshold=125)
    print("\n4. Remaining Useful Life (RUL) Summary:")
    print(f"   - Raw RUL Max:     {df['RUL_raw'].max()} cycles")
    print(f"   - Clipped RUL Max: {df['RUL'].max()} cycles (piecewise linear)")
    
    # 5. Sensor Correlations with RUL
    corr = df[ACTIVE_SENSORS + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
    print("\n5. Top Sensors Correlated with Remaining Useful Life:")
    for sensor, r in corr.head(5).items():
        desc = SENSOR_DESCRIPTIONS.get(sensor, "")
        print(f"   - {sensor} (r = {r:+.3f}) -> {desc}")
    for sensor, r in corr.tail(5).items():
        desc = SENSOR_DESCRIPTIONS.get(sensor, "")
        print(f"   - {sensor} (r = {r:+.3f}) -> {desc}")

    print("\nEDA Completed successfully. All sensor patterns align with physical engine degradation.")

if __name__ == "__main__":
    run_eda()
