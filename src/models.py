"""
Machine Learning Models and Inference Pipeline for MaintAI.
Implements XGBoost and Random Forest models for Remaining Useful Life (RUL) prediction,
confidence intervals, performance evaluation, and feature importance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.config import (
    XGB_MODEL_FILE,
    RF_MODEL_FILE,
    SCALER_FILE,
    METADATA_FILE
)

def cmapss_score_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA CMAPSS asymmetric score function.
    Penalizes late predictions (where y_pred > y_true) more heavily than early predictions.
    d = y_pred - y_true
    s = exp(-d/13) - 1 for d < 0
    s = exp(d/10) - 1 for d >= 0
    """
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: RMSE, MAE, R^2, and NASA Score.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    cmapss_score = cmapss_score_function(y_true, y_pred)
    
    # Accuracy percentage based on mean relative error
    non_zero = y_true > 0
    if np.sum(non_zero) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])))
        accuracy_pct = max(0.0, (1.0 - mape) * 100.0)
    else:
        accuracy_pct = max(0.0, r2 * 100.0)
        
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "cmapss_score": round(cmapss_score, 1),
        "accuracy_pct": round(accuracy_pct, 2)
    }

class PredictiveMaintenanceModel:
    """
    Unified interface for XGBoost and Random Forest RUL forecasting.
    """
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.model = None
        self.feature_names = []
        self.rmse_residual = 14.5  # default residual standard deviation for confidence bounds
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Trains the selected model (XGBoost or Random Forest).
        """
        self.feature_names = feature_names or list(X_train.columns)
        
        if self.model_type == "xgboost":
            params = {
                "n_estimators": kwargs.get("n_estimators", 250),
                "max_depth": kwargs.get("max_depth", 5),
                "learning_rate": kwargs.get("learning_rate", 0.05),
                "subsample": kwargs.get("subsample", 0.8),
                "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
                "random_state": 42,
                "n_jobs": -1
            }
            self.model = xgb.XGBRegressor(**params)
        elif self.model_type in ["random_forest", "rf"]:
            params = {
                "n_estimators": kwargs.get("n_estimators", 150),
                "max_depth": kwargs.get("max_depth", 14),
                "min_samples_split": kwargs.get("min_samples_split", 5),
                "min_samples_leaf": kwargs.get("min_samples_leaf", 2),
                "random_state": 42,
                "n_jobs": -1
            }
            self.model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        self.model.fit(X_train[self.feature_names], y_train)
        
        # Calculate in-sample residual error for XGBoost confidence intervals
        y_pred = self.model.predict(X_train[self.feature_names])
        self.rmse_residual = float(np.sqrt(mean_squared_error(y_train, y_pred)))
        
    def predict_with_confidence(
        self, 
        X: pd.DataFrame, 
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns predictions along with lower and upper confidence bounds.
        - For Random Forest: Uses ensemble tree variance across estimators.
        - For XGBoost: Uses empirical residual dispersion.
        """
        X_eval = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        z_multiplier = 1.96 if confidence_level == 0.95 else 1.645
        
        if self.model_type in ["random_forest", "rf"] and hasattr(self.model, "estimators_"):
            # Tree-based variance estimation
            tree_preds = np.array([tree.predict(X_eval.values if isinstance(X_eval, pd.DataFrame) else X_eval) 
                                   for tree in self.model.estimators_])
            pred_mean = np.mean(tree_preds, axis=0)
            pred_std = np.std(tree_preds, axis=0)
            
            # Incorporate both tree variance and intrinsic residual variance
            total_std = np.sqrt(pred_std**2 + (0.5 * self.rmse_residual)**2)
            lower = np.maximum(0.0, pred_mean - (z_multiplier * total_std))
            upper = pred_mean + (z_multiplier * total_std)
            return pred_mean, lower, upper
        else:
            # XGBoost or general regressor
            pred = self.model.predict(X_eval)
            pred = np.maximum(0.0, pred)
            margin = z_multiplier * self.rmse_residual
            lower = np.maximum(0.0, pred - margin)
            upper = pred + margin
            return pred, lower, upper
            
    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts feature importances and returns a sorted DataFrame.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            fi_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False).head(top_n)
            fi_df["importance_pct"] = (fi_df["importance"] / fi_df["importance"].sum()) * 100.0
            return fi_df.reset_index(drop=True)
        return pd.DataFrame()

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "rmse_residual": self.rmse_residual
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "PredictiveMaintenanceModel":
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.rmse_residual = data.get("rmse_residual", 14.5)
        return instance

