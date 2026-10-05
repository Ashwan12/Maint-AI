# MaintAI: Predictive Maintenance Analytics Dashboard
## Final Project Presentation Deck & Demonstration Script

---

### Slide 1: Title & Overview
- **Title**: MaintAI: Industrial Predictive Maintenance & RUL Forecasting Platform
- **Subtitle**: Transforming Reactive Machinery Repair into Condition-Based Intelligence
- **Domain**: Machine Learning • Time-Series Sensor Analytics • Industrial IoT • Streamlit Dashboard
- **Presenter**: Predictive Maintenance Engineering Intern
- **Key Takeaway**: Accurately forecast machinery Remaining Useful Life (RUL) with 95% confidence intervals and flag abnormal sensor degradation before catastrophic failures occur.

---

### Slide 2: The Industrial Challenge
- **The Problem**:
  - Unscheduled downtime costs manufacturing and aviation plants billions annually.
  - Reactive maintenance leads to secondary machine damage and safety hazards.
  - Fixed-schedule overhauls waste up to 40% of useful equipment life by replacing healthy parts.
- **The Solution**:
  - Continuous telemetry ingestion from temperatures, pressures, vibration, and flow rates.
  - Dynamic RUL forecasting and unsupervised anomaly detection.

---

### Slide 3: Industrial Benchmarks Used
1. **NASA C-MAPSS Turbofan Engine Dataset (FD001)**:
   - 100 jet engines run to failure; 20,631 operational flight cycles.
   - 21 high-frequency thermodynamic sensors + 3 operational settings.
2. **Case Western Reserve University (CWRU) Bearing Benchmark**:
   - Drive-end and fan-end vibration acceleration, kurtosis, crest factor, and motor load.
3. **UCI Hydraulic Condition Monitoring Dataset**:
   - Multi-sensor pressure, fluid temperature, valve efficiency, and accumulator health.

---

### Slide 4: Feature Engineering & Degradation Physics
- **Target Formulation**:
  - Piecewise linear RUL target capped at 125 cycles.
  - Mirrors physical wear mechanics: engines exhibit no measurable degradation during early cycles.
- **Temporal Transformations**:
  - Multi-window rolling statistics (5-cycle and 15-cycle rolling means and standard deviations).
  - 5-cycle difference lags capturing rate-of-change degradation velocity.
  - Pruned 7 zero-variance sensor channels operating under invariant ambient conditions.

---

### Slide 5: Machine Learning Models & Results
- **XGBoost Regressor (Champion)**:
  - **RMSE**: **18.78 cycles** on 100 unseen test engines.
  - **MAE**: **13.89 cycles**.
  - **Accuracy**: **>86%**.
  - **NASA CMAPSS Score**: **854.9** (official asymmetric evaluation metric).
- **Random Forest Regressor**:
  - **RMSE**: 19.89 cycles.
  - Provides native 95% confidence intervals via tree ensemble variance.
- **Top Failure Drivers**:
  - HPC Static Pressure (Ps30 - Sensor 11).
  - LPT Outlet Temperature (T50 - Sensor 4).
  - Fuel Flow Ratio (phi - Sensor 12).

---

### Slide 6: Unsupervised Anomaly Clustering
- **Isolation Forest**:
  - Flags abnormal excursions and generates continuous anomaly severity scores without requiring ground truth labels.
- **K-Means Clustering & 2D PCA**:
  - Maps multi-sensor states into three distinct operational regimes:
    1. *Healthy (Nominal)*
    2. *Degrading (Maintenance Advisory)*
    3. *Critical (Failure Imminent)*
  - Visualizes individual asset degradation trajectories traveling across clusters over time.

---

### Slide 7: Live Dashboard Demonstration
- **Tab 1: Fleet Overview**: High-level KPIs, fleet health distribution, and Top 10 critical assets.
- **Tab 2: Sensor Trend Analytics**: Interactive multi-sensor inspection with abnormal drop highlights and threshold breach alerts.
- **Tab 3: Predictive RUL Forecast**: Real-time RUL trajectory, shaded 95% confidence bands, asset health gauge, and "What-If" stress simulator.
- **Tab 4: Anomaly Clustering**: 2D PCA cluster space and anomaly severity timeline.
- **Tab 5: Model Studio**: Benchmark comparison table, feature importance charts, and architecture details.
- **Tab 6: Custom CSV Diagnostics**: Drag-and-drop file upload, automatic column mapping, automated inference, and CSV report export.

---

### Slide 8: Business Value & Return on Investment (ROI)
- **Reduced Unscheduled Downtime**: Eliminates surprise catastrophic breakdowns via early advisory warnings.
- **Optimized Inventory & Spare Parts**: Orders replacements precisely when RUL enters the 30–75 cycle window.
- **Maximized Asset Utilization**: Safely extracts the remaining lifespan of healthy equipment rather than adhering to rigid calendar schedules.
- **Deployment-Ready**: Clean modular Python architecture, containerizable, and instantly runnable on Streamlit Cloud or Render.

---

### Slide 9: Conclusion & Q&A
- **Summary**: MaintAI demonstrates an end-to-end data analytics, machine learning, and interactive dashboarding workflow for modern industrial maintenance.
- **Questions & Discussion**: Open for technical discussion and live demonstration.

