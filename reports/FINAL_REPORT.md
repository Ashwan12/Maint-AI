# MaintAI — Predictive Maintenance Analytics & Remaining Useful Life (RUL) Forecasting
## Comprehensive Internship Project Final Report

---

### Executive Summary

In modern manufacturing, aviation, and heavy machinery operations, unscheduled equipment downtime costs industries billions of dollars annually, while premature scheduled maintenance incurs wasteful expenditure and unnecessary asset replacement. **MaintAI** is an end-to-end, enterprise-grade predictive maintenance analytics and forecasting platform designed to shift maintenance paradigms from *reactive* or *fixed-schedule* to **condition-based, predictive intelligence**.

MaintAI ingests multi-channel time-series telemetry from industrial sensors (temperature, pressure, vibration, motor RPM, and fluid flow), extracts temporal degradation patterns via rolling statistics and lag transformations, and deploys high-performance machine learning models (**XGBoost** and **Random Forest**) to forecast the **Remaining Useful Life (RUL)** with rigorous **95% confidence intervals**. Additionally, MaintAI integrates unsupervised **Isolation Forest** and **K-Means clustering** to flag abnormal sensor drift and operational regime transitions before catastrophic failure occurs.

---

### 1. Problem Statement & Industrial Motivation

Traditional industrial maintenance follows two classical strategies:
1. **Reactive Maintenance (Run-to-Failure)**: Machinery runs until component breakdown. This introduces severe safety hazards, collateral asset damage, emergency shipping costs, and catastrophic plant outages.
2. **Scheduled Preventative Maintenance**: Overhauls are executed at fixed calendar intervals or operating hours, regardless of true physical condition. This results in the premature retirement of components that possess significant remaining lifespan.

**Predictive Maintenance (PdM)** continuously assesses degradation through sensory data streams, forecasting failure before it occurs and optimizing maintenance scheduling to minimize Total Cost of Ownership (TCO).

```
[Raw Industrial Sensors] -> [Feature Engineering (Rolling/Lags)] -> [ML Models (XGBoost/RF)] -> [Confidence Bounds] -> [Interactive Streamlit UI]
```

---

### 2. Datasets Benchmarked

MaintAI evaluates and supports three premier benchmark datasets:

| Dataset | Domain | Dimensions / Sensors | Primary Failure Mechanism |
| :--- | :--- | :--- | :--- |
| **NASA C-MAPSS (FD001)** | Commercial Turbofan Jet Engines | 21 sensors, 3 op settings, 100 train / 100 test fleets | High-Pressure Compressor (HPC) degradation under sea-level conditions |
| **Case Western Reserve University (CWRU)** | Industrial Electric Motor Bearings | Vibration DE/FE, Kurtosis, Crest Factor, Temp, RPM, Load | Inner race, outer race, and ball element fatigue defects |
| **UCI Hydraulic Condition Monitoring** | Heavy Fluid Power Systems | Pressure (bar), Flow Rate (L/min), Temperature (°C), Motor Power (W) | Valve degradation, accumulator gas leakage, cooler clogging |

---

### 3. Weekly Development Progression (6-Week Internship Timeline)

```
+---------------------------------------------------------------------------------------+
|  Week 1: Dataset Exploration, Ingestion Pipelines & Sensor Variance EDA               |
|  Week 2: RUL Formulation (Piecewise Linear Capping) & Temporal Feature Engineering   |
|  Week 3: ML Model Training & Hyperparameter Tuning (XGBoost vs Random Forest)         |
|  Week 4: Unsupervised Anomaly Engine (Isolation Forest, KMeans, PCA 2D/3D Projection) |
|  Week 5: Interactive Streamlit Dashboard Development & Real-Time Stream Simulator    |
|  Week 6: Evaluation on NASA Benchmark, Confidence Bounds, Polish, Tests & Deployment  |
+---------------------------------------------------------------------------------------+
```

#### Week 1: Dataset Exploration & Exploratory Data Analysis (EDA)
- Analyzed NASA CMAPSS FD001 run-to-failure telemetry comprising 20,631 cycles across 100 training engines.
- Conducted variance analysis, discovering that 7 sensors (`sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) exhibit zero variance in FD001 due to fixed ambient sea-level conditions.
- Pruned zero-variance channels to reduce dimensionality and mitigate noise.

#### Week 2: Label Formulation & Feature Engineering
- Formulated the run-to-failure Remaining Useful Life (RUL) target:
  $$\text{RUL}(t) = \min(125, \max(\text{cycles}) - t)$$
- Implemented piecewise linear degradation (capping at 125 cycles) reflecting the physical reality that early operational cycles experience virtually no measurable wear.
- Engineered 87 rolling features (rolling mean, rolling standard deviation over 5- and 15-cycle windows) and 5-cycle difference lag features to capture degradation velocity.

#### Week 3: Supervised Model Architecture & Training
- Trained an **XGBoost Regressor** with gradient-boosted trees and regularization.
- Trained a **Random Forest Regressor** to provide ensemble dispersion metrics.
- Validated performance against the NASA test benchmark of 100 unseen engines using official ground truth final RUL targets.

#### Week 4: Unsupervised Anomaly Clustering
- Implemented an **Isolation Forest** to compute anomaly severity scores without needing ground truth failure labels.
- Implemented **K-Means clustering** to partition machinery operational health into three distinct regimes:
  - *Healthy (Nominal)*
  - *Degrading (Maintenance Advisory)*
  - *Critical (Failure Imminent)*
- Deployed **Principal Component Analysis (PCA)** to project multi-dimensional sensor space into 2D and 3D visual trajectories.

#### Week 5: Streamlit Interface & Real-Time Simulator
- Created an interactive glassmorphic web dashboard with responsive tabs:
  1. Fleet Overview & Urgency Ranking
  2. Sensor Trend Analytics & Threshold Excursions
  3. Predictive RUL Forecast & 95% Confidence Intervals
  4. Unsupervised Anomaly Clustering & Trajectory Trails
  5. Model Studio & Explainability
  6. Custom CSV Diagnostics & Report Exporter
- Built a real-time cycle simulator allowing engineers to advance operating cycles and observe predictive health adjustments dynamically.

#### Week 6: Testing, Polish & Deployment Preparation
- Implemented automated pytest test suite (`tests/test_pipeline.py`) with 100% pass rate.
- Packaged configuration, pre-trained model weights, and sample datasets for instant zero-configuration deployment on Streamlit Cloud or Render.

---

### 4. Technical Architecture & Methodology

#### 4.1 Piecewise Linear Degradation Target
In CMAPSS data, engines run for hundreds of cycles before degradation manifests. Training a model to distinguish cycle 200 from cycle 250 on a brand-new engine forces the model to fit sensor noise. Capping the target at $RUL_{max} = 125$ cycles stabilizes the model and aligns with physical fatigue mechanics.

#### 4.2 Uncertainty Quantification (95% Confidence Intervals)
Industrial operators require confidence intervals before scheduling multi-million dollar plant shutdowns:
- **Random Forest**: Leverages the empirical variance across individual decision trees in the ensemble:
  $$\hat{\sigma}^2(x) = \frac{1}{B} \sum_{b=1}^B (T_b(x) - \bar{T}(x))^2 + \sigma_{res}^2$$
  $$\text{CI}_{95\%} = [\hat{y} - 1.96 \hat{\sigma}, \hat{y} + 1.96 \hat{\sigma}]$$
- **XGBoost**: Employs empirical residual error dispersion bounds.

#### 4.3 NASA CMAPSS Scoring Function
Standard RMSE weights early and late predictions equally. However, in aviation and manufacturing, **a late prediction ($y_{pred} > y_{true}$) causes catastrophic in-flight failure**, whereas an early prediction only triggers slightly premature inspection. NASA's asymmetric scoring function reflects this penalty:
$$d_i = \hat{y}_i - y_i$$
$$s_i = \begin{cases} e^{-d_i / 13} - 1 & \text{if } d_i < 0 \text{ (Early Warning)} \\ e^{d_i / 10} - 1 & \text{if } d_i \ge 0 \text{ (Late Risk)} \end{cases}$$
$$S = \sum_{i=1}^N s_i$$

---

### 5. Experimental Results & Model Benchmark

Models were trained on 20,631 training cycles and evaluated against the 100 unseen test units of NASA CMAPSS FD001 at their final observed cycles:

| Model | Test RMSE (Cycles) | Test MAE (Cycles) | $R^2$ Score | Accuracy (%) | NASA CMAPSS Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor (Champion)** | **18.78** | **13.89** | **0.7958** | **86.4%** | **854.9** |
| **Random Forest Regressor** | **19.89** | **14.37** | **0.7710** | **85.1%** | **1370.7** |

#### Top Predictive Sensors (Feature Importance)
1. **Sensor 11 (Ps30 - HPC Static Pressure)**: Strongest indicator of compressor surge and pressure ratio loss ($r = -0.775$).
2. **Sensor 4 (T50 - LPT Outlet Temperature)**: Exhaust gas temperature rises as turbine efficiency degrades ($r = -0.757$).
3. **Sensor 12 (phi - Fuel Flow Ratio)**: Required fuel flow rises to compensate for stage pressure drops ($r = +0.749$).
4. **Sensor 7 (P30 - HPC Outlet Pressure)**: Pressure drops as internal blade clearances expand ($r = +0.733$).
5. **Sensor 15 (BPR - Bypass Ratio)**: Diverts more flow as core flow resistance climbs ($r = -0.721$).

---

### 6. Deployment & Deliverables Summary

1. **Dashboard Interface**: Full Streamlit application (`app.py`) featuring dark industrial aesthetic, interactive Plotly visualizations, live streaming simulator, and exportable reports.
2. **Modular Codebase**: Clean OOP architecture separating configuration, data ingestion, feature engineering, modeling, anomaly detection, and visualization.
3. **Trained Weights**: Pre-trained XGBoost, Random Forest, Scaler, and Isolation Forest models saved in `models/`.
4. **Sample Datasets**: Built-in benchmark datasets for NASA Turbofan, CWRU Bearings, and UCI Hydraulics in `data/samples/`.
5. **Automated Test Suite**: 7 unit/integration tests running via `pytest` covering all system components.

---

### 7. Resume & Professional Value

> **"Architected and deployed MaintAI, an end-to-end industrial predictive maintenance platform using NASA CMAPSS sensor telemetry, engineering 80+ temporal rolling features and training XGBoost & Random Forest models achieving >86% accuracy (18.78 RMSE) in Remaining Useful Life (RUL) forecasting with 95% confidence intervals, coupled with an interactive Streamlit operational dashboard."**

