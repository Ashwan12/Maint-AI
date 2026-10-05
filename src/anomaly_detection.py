"""
Unsupervised Anomaly Detection & Clustering Module for MaintAI.
Implements Isolation Forest, K-Means Clustering, and PCA 2D/3D decomposition
to identify abnormal sensor trends, degradation regimes, and health states.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    ISOLATION_FOREST_FILE,
    KMEANS_FILE,
    PCA_FILE,
    ACTIVE_SENSORS
)

class AnomalyClusterEngine:
    """
    Engine for unsupervised sensor anomaly detection, clustering, and PCA visualization.
    """
    def __init__(self, n_clusters: int = 3, contamination: float = 0.05):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.feature_cols = []
        self.cluster_names = {}
        
    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        """
        Fits Scaler, Isolation Forest, KMeans, and PCA on training data.
        """
        self.feature_cols = feature_cols or [c for c in ACTIVE_SENSORS if c in df.columns]
        X = df[self.feature_cols].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit models
        self.iso_forest.fit(X_scaled)
        self.kmeans.fit(X_scaled)
        self.pca_2d.fit(X_scaled)
        self.pca_3d.fit(X_scaled)
        
        # Map clusters logically based on average RUL if available, else cluster centers
        if "RUL" in df.columns:
            cluster_rul = df.groupby(self.kmeans.labels_)["RUL"].mean().to_dict()
            # Sort clusters by descending RUL (highest RUL = Healthy)
            sorted_clusters = sorted(cluster_rul.items(), key=lambda x: x[1], reverse=True)
            labels = ["Healthy (Nominal)", "Degrading (Advisory)", "Critical (Failure Imminent)"]
            self.cluster_names = {cid: labels[min(i, len(labels)-1)] for i, (cid, _) in enumerate(sorted_clusters)}
        else:
            self.cluster_names = {i: f"Operational Regime {i+1}" for i in range(self.n_clusters)}
            
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infers anomaly labels, anomaly scores, clusters, and PCA coordinates.
        """
        df_out = df.copy()
        available_cols = [c for c in self.feature_cols if c in df_out.columns]
        if len(available_cols) != len(self.feature_cols):
            # Fallback if some features are missing
            sub_scaler = StandardScaler()
            X_scaled = sub_scaler.fit_transform(df_out[available_cols])
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            df_out["pca_1"] = coords[:, 0]
            df_out["pca_2"] = coords[:, 1]
            df_out["anomaly_score"] = 0.5
            df_out["is_anomaly"] = False
            df_out["cluster_id"] = 0
            df_out["cluster_name"] = "Default Regime"
            return df_out
            
        X_scaled = self.scaler.transform(df_out[self.feature_cols])
        
        # Anomaly scoring: negative score is anomalous, positive is normal
        raw_scores = self.iso_forest.decision_function(X_scaled)
        # Normalize score between 0 (very anomalous) and 1 (completely normal)
        anomaly_prob = 1.0 / (1.0 + np.exp(-raw_scores))
        
        iso_preds = self.iso_forest.predict(X_scaled)  # -1 for anomaly, 1 for normal
        
        clusters = self.kmeans.predict(X_scaled)
        pca_2d_coords = self.pca_2d.transform(X_scaled)
        pca_3d_coords = self.pca_3d.transform(X_scaled)
        
        df_out["anomaly_score"] = np.round(1.0 - anomaly_prob, 4)  # 0 to 1, higher = more anomalous
        df_out["is_anomaly"] = iso_preds == -1
        df_out["cluster_id"] = clusters
        df_out["cluster_name"] = [self.cluster_names.get(c, f"Cluster {c}") for c in clusters]
        df_out["pca_1"] = np.round(pca_2d_coords[:, 0], 3)
        df_out["pca_2"] = np.round(pca_2d_coords[:, 1], 3)
        df_out["pca_3"] = np.round(pca_3d_coords[:, 2], 3)
        
        return df_out
        
    def detect_sensor_excursions(
        self, 
        df: pd.DataFrame, 
        sensors: List[str], 
        z_threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Identifies specific sensors exceeding nominal operating threshold limits (Z-score).
        """
        excursions = []
        for s in sensors:
            if s in df.columns:
                mean_val = df[s].mean()
                std_val = df[s].std()
                if std_val > 1e-6:
                    z_scores = np.abs((df[s] - mean_val) / std_val)
                    breaches = np.sum(z_scores > z_threshold)
                    max_z = np.max(z_scores)
                    excursions.append({
                        "sensor": s,
                        "breach_count": int(breaches),
                        "max_z_score": round(float(max_z), 2),
                        "status": "Alert" if breaches > 0 else "Normal"
                    })
        return pd.DataFrame(excursions)

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "iso_forest": self.iso_forest,
            "kmeans": self.kmeans,
            "pca_2d": self.pca_2d,
            "pca_3d": self.pca_3d,
            "feature_cols": self.feature_cols,
            "cluster_names": self.cluster_names,
            "n_clusters": self.n_clusters,
            "contamination": self.contamination
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "AnomalyClusterEngine":
        data = joblib.load(filepath)
        instance = cls(n_clusters=data["n_clusters"], contamination=data["contamination"])
        instance.scaler = data["scaler"]
        instance.iso_forest = data["iso_forest"]
        instance.kmeans = data["kmeans"]
        instance.pca_2d = data["pca_2d"]
        instance.pca_3d = data["pca_3d"]
        instance.feature_cols = data["feature_cols"]
        instance.cluster_names = data["cluster_names"]
        return instance

