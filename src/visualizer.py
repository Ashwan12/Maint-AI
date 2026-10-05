"""
Visualization Module for MaintAI.
Creates interactive Plotly and Seaborn charts for sensor trends, degradation curves,
RUL forecasts with confidence intervals, fleet health diagnostics, and anomaly clusters.
"""
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import SENSOR_DESCRIPTIONS, HEALTH_THRESHOLDS

# Consistent Industrial Theme Palette
COLORS = {
    "primary": "#00d2ff",
    "secondary": "#3a7bd5",
    "accent": "#ff007f",
    "healthy": "#00e676",
    "warning": "#ffab00",
    "critical": "#ff1744",
    "background": "#0e1117",
    "card_bg": "#1a1f2c",
    "text": "#e0e6ed",
    "grid": "#2d3748"
}

def plot_health_gauge(rul_value: float, max_rul: float = 125.0) -> go.Figure:
    """
    Renders a semi-circular health gauge indicator (0% - 100%).
    """
    health_pct = min(100.0, max(0.0, (rul_value / max_rul) * 100.0))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_pct,
        number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
        title={"text": "Asset Health Index", "font": {"size": 18, "color": COLORS["primary"]}},
        delta={"reference": 75, "increasing": {"color": COLORS["healthy"]}, "decreasing": {"color": COLORS["critical"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text"]},
            "bar": {"color": COLORS["primary"], "thickness": 0.3},
            "bgcolor": COLORS["card_bg"],
            "borderwidth": 2,
            "bordercolor": COLORS["grid"],
            "steps": [
                {"range": [0, 30], "color": "rgba(255, 23, 68, 0.45)"},
                {"range": [30, 70], "color": "rgba(255, 171, 0, 0.45)"},
                {"range": [70, 100], "color": "rgba(0, 230, 118, 0.45)"}
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 4},
                "thickness": 0.8,
                "value": health_pct
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=220
    )
    return fig

def plot_sensor_trends(
    df: pd.DataFrame, 
    unit_id: int, 
    selected_sensors: List[str],
    highlight_anomalies: bool = True
) -> go.Figure:
    """
    Interactive time-series plot of selected sensor channels for a specific unit.
    Highlights abnormal readings and anomaly points.
    """
    unit_df = df[df["unit_id"] == unit_id].sort_values("time_cycles")
    
    n_sensors = len(selected_sensors)
    if n_sensors == 0:
        fig = go.Figure()
        fig.update_layout(title="No sensors selected")
        return fig
        
    fig = make_subplots(
        rows=n_sensors, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            f"{s}: {SENSOR_DESCRIPTIONS.get(s, s.replace('_', ' ').title())}" 
            for s in selected_sensors
        ]
    )
    
    palette = px.colors.qualitative.Plotly
    
    for idx, sensor in enumerate(selected_sensors, start=1):
        color = palette[(idx - 1) % len(palette)]
        # Base sensor trend
        fig.add_trace(
            go.Scatter(
                x=unit_df["time_cycles"],
                y=unit_df[sensor],
                mode="lines",
                name=sensor,
                line=dict(color=color, width=2.2),
                hovertemplate="Cycle: %{x}<br>Value: %{y:.3f}<extra></extra>"
            ),
            row=idx, 
            col=1
        )
        
        # Overlay anomalies if present in data
        if highlight_anomalies and "is_anomaly" in unit_df.columns:
            anom_points = unit_df[unit_df["is_anomaly"]]
            if not anom_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anom_points["time_cycles"],
                        y=anom_points[sensor],
                        mode="markers",
                        name=f"{sensor} Anomaly",
                        marker=dict(color=COLORS["critical"], size=7, symbol="diamond"),
                        hovertemplate="Anomaly at Cycle %{x}<br>Value: %{y:.3f}<extra></extra>",
                        showlegend=(idx == 1)
                    ),
                    row=idx,
                    col=1
                )
                
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 31, 44, 0.7)",
        height=max(320, n_sensors * 180),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Operational Cycles / Time", row=n_sensors, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

def plot_rul_forecast(
    cycles: np.ndarray,
    rul_pred: np.ndarray,
    rul_lower: np.ndarray,
    rul_upper: np.ndarray,
    rul_true: Optional[np.ndarray] = None,
    unit_id: Optional[int] = None
) -> go.Figure:
    """
    RUL trajectory forecast with 95% Confidence Interval band and alert threshold lines.
    """
    fig = go.Figure()
    
    # Upper confidence boundary (invisible)
    fig.add_trace(go.Scatter(
        x=cycles,
        y=rul_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper CI"
    ))
    
    # Shaded confidence band (filled to upper bound)
    fig.add_trace(go.Scatter(
        x=cycles,
        y=rul_lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 210, 255, 0.18)",
        name="95% Confidence Interval",
        hovertemplate="Cycle: %{x}<br>Lower CI: %{y:.1f}<extra></extra>"
    ))
    
    # Predicted RUL line
    fig.add_trace(go.Scatter(
        x=cycles,
        y=rul_pred,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=4),
        name="Predicted RUL",
        hovertemplate="Cycle: %{x}<br>Predicted RUL: %{y:.1f} cycles<extra></extra>"
    ))
    
    # Ground Truth RUL (if available)
    if rul_true is not None:
        fig.add_trace(go.Scatter(
            x=cycles,
            y=rul_true,
            mode="lines",
            line=dict(color="#ffffff", width=2, dash="dash"),
            name="Ground Truth RUL",
            hovertemplate="Cycle: %{x}<br>True RUL: %{y:.1f} cycles<extra></extra>"
        ))
        
    # Critical and Warning Threshold lines
    max_c = max(cycles) if len(cycles) > 0 else 100
    fig.add_hline(
        y=HEALTH_THRESHOLDS["CRITICAL"], 
        line_dash="dot", 
        line_color=COLORS["critical"], 
        line_width=2,
        annotation_text="Critical Danger Threshold (RUL ≤ 30)", 
        annotation_position="bottom right"
    )
    fig.add_hline(
        y=HEALTH_THRESHOLDS["WARNING"], 
        line_dash="dot", 
        line_color=COLORS["warning"], 
        line_width=2,
        annotation_text="Maintenance Warning Threshold (RUL ≤ 75)", 
        annotation_position="top right"
    )
    
    title_text = f"Remaining Useful Life (RUL) Forecast & Degradation Curve"
    if unit_id is not None:
        title_text += f" - Equipment Unit #{unit_id}"
        
    fig.update_layout(
        title=title_text,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 31, 44, 0.7)",
        xaxis_title="Operational Cycles",
        yaxis_title="Remaining Useful Life (Cycles)",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

def plot_fleet_health_distribution(df_summary: pd.DataFrame) -> go.Figure:
    """
    Fleet Health Status distribution donut chart.
    """
    counts = df_summary["health_status"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    
    color_map = {
        "Healthy": COLORS["healthy"],
        "Warning": COLORS["warning"],
        "Critical": COLORS["critical"]
    }
    marker_colors = [color_map.get(lbl, "#718096") for lbl in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=marker_colors),
        textinfo="label+percent+value",
        hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        title="Fleet Health Status Distribution",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_urgent_units_bar(df_summary: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Bar chart of the top N units with lowest remaining useful life.
    """
    df_sorted = df_summary.sort_values(by="latest_predicted_rul", ascending=True).head(top_n)
    
    colors = [
        COLORS["critical"] if r <= HEALTH_THRESHOLDS["CRITICAL"] 
        else COLORS["warning"] if r <= HEALTH_THRESHOLDS["WARNING"] 
        else COLORS["healthy"] 
        for r in df_sorted["latest_predicted_rul"]
    ]
    
    fig = go.Figure(go.Bar(
        x=[f"Unit #{u}" for u in df_sorted["unit_id"]],
        y=df_sorted["latest_predicted_rul"],
        marker_color=colors,
        text=[f"{r:.0f} cyc" for r in df_sorted["latest_predicted_rul"]],
        textposition="auto",
        hovertemplate="Unit: %{x}<br>Est. RUL: %{y:.1f} cycles<extra></extra>"
    ))
    fig.update_layout(
        title=f"Top {top_n} Equipment Units Requiring Immediate Service",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 31, 44, 0.7)",
        xaxis_title="Machine Unit ID",
        yaxis_title="Estimated RUL (Cycles)",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

def plot_anomaly_clusters_2d(
    df: pd.DataFrame, 
    unit_id: Optional[int] = None
) -> go.Figure:
    """
    2D PCA Scatter plot showing unsupervised clusters and individual unit trajectory.
    """
    if "pca_1" not in df.columns or "pca_2" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="PCA features not computed.")
        return fig
        
    color_col = "cluster_name" if "cluster_name" in df.columns else "cluster_id"
    
    # Sample background fleet for responsiveness
    fleet_sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
    
    fig = px.scatter(
        fleet_sample,
        x="pca_1",
        y="pca_2",
        color=color_col,
        color_discrete_map={
            "Healthy (Nominal)": COLORS["healthy"],
            "Degrading (Advisory)": COLORS["warning"],
            "Critical (Failure Imminent)": COLORS["critical"]
        },
        opacity=0.45,
        title="Unsupervised Operational Clustering (2D PCA Sensor Projection)"
    )
    
    # Overlay selected unit degradation trajectory
    if unit_id is not None:
        unit_data = df[df["unit_id"] == unit_id].sort_values("time_cycles")
        if not unit_data.empty:
            fig.add_trace(go.Scatter(
                x=unit_data["pca_1"],
                y=unit_data["pca_2"],
                mode="lines+markers",
                marker=dict(
                    size=7, 
                    color=unit_data["time_cycles"], 
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cycle", x=1.1)
                ),
                line=dict(color="#ffffff", width=2),
                name=f"Unit #{unit_id} Aging Path",
                hovertemplate="Unit %{text}<br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>",
                text=[f"Cycle {c}" for c in unit_data["time_cycles"]]
            ))
            
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 31, 44, 0.7)",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], title="Principal Component 1 (Primary Variance)")
    fig.update_yaxes(gridcolor=COLORS["grid"], title="Principal Component 2 (Secondary Variance)")
    return fig

def plot_feature_importances(fi_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Horizontal bar chart of top predictive sensors.
    """
    plot_df = fi_df.head(top_n).sort_values("importance", ascending=True)
    
    # Map friendly sensor names
    friendly_names = [
        f"{row['feature']} ({SENSOR_DESCRIPTIONS.get(row['feature'], '').split('[')[0].strip()})"
        if row['feature'] in SENSOR_DESCRIPTIONS else row['feature']
        for _, row in plot_df.iterrows()
    ]
    
    fig = go.Figure(go.Bar(
        x=plot_df["importance_pct"],
        y=friendly_names,
        orientation="h",
        marker=dict(
            color=plot_df["importance_pct"],
            colorscale="Tealgrn"
        ),
        text=[f"{val:.1f}%" for val in plot_df["importance_pct"]],
        textposition="auto",
        hovertemplate="Feature: %{y}<br>Relative Contribution: %{x:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title="Key Sensor Predictors Influencing Equipment Failure (Feature Importance)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 31, 44, 0.7)",
        xaxis_title="Relative Predictive Importance (%)",
        height=440,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, sensors: List[str]) -> go.Figure:
    """
    Correlation heatmap of selected sensors and RUL.
    """
    cols = [s for s in sensors if s in df.columns]
    if "RUL" in df.columns:
        cols.append("RUL")
        
    corr = df[cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Pearson r")
    ))
    fig.update_layout(
        title="Sensor Cross-Correlation & Degradation Association Matrix",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

