"""Inference module for 30-day CLV prediction."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_artifacts(models_dir: Path) -> dict:
    """Load model artifacts from disk."""
    models_dir = Path(models_dir)
    return {
        "model1": joblib.load(models_dir / "model1_stack.joblib"),
        "model2": joblib.load(models_dir / "model2_stack.joblib"),
        "scaler": joblib.load(models_dir / "scaler.joblib"),
        "feature_cols": joblib.load(models_dir / "feature_columns.joblib"),
        "metrics": joblib.load(models_dir / "metrics.joblib"),
    }


def predict(
    X: pd.DataFrame,
    artifacts: dict,
) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Make predictions with both models.
    Returns: (pred_model1, pred_model2, metrics)
    """
    feature_cols = artifacts["feature_cols"]
    available = [c for c in feature_cols if c in X.columns]
    missing = [c for c in feature_cols if c not in X.columns]
    
    X_pred = X.reindex(columns=feature_cols, fill_value=0)
    X_scaled = artifacts["scaler"].transform(X_pred[feature_cols])
    
    pred1 = artifacts["model1"].predict(X_scaled)
    pred2 = artifacts["model2"].predict(X_scaled)
    
    pred1_series = pd.Series(pred1, index=X.index)
    pred2_series = pd.Series(pred2, index=X.index)
    
    return pred1_series, pred2_series, artifacts["metrics"]


def estimate_products_from_spend(
    predicted_spend: pd.Series,
    avg_products_per_dollar: float = 0.1,
) -> pd.Series:
    """Estimate number of products from predicted spend (heuristic)."""
    if avg_products_per_dollar <= 0:
        avg_products_per_dollar = 0.1
    return (predicted_spend * avg_products_per_dollar).clip(lower=0).round().astype(int)
