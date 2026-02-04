"""
ML Training for 30-Day CLV Prediction.
Model 1: Random Forest + Ridge + Lasso (stacking)
Model 2: Random Forest + XGBoost + Ridge + Lasso (stacking)
Hyperparameter tuning: RandomizedSearchCV
Evaluation: R2
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

RANDOM_STATE = 42
N_ITER = 40  # RandomizedSearchCV iterations (more for better Model 2)
CV = 3


def get_feature_columns() -> list:
    return [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "days_since_first_purchase",
        "products_per_order",
        "category_diversity",
        "unique_products_purchased",
    ]


def build_model_1() -> StackingRegressor:
    """Model 1: RF + Ridge + Lasso."""
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    ridge = Ridge(random_state=RANDOM_STATE)
    lasso = Lasso(random_state=RANDOM_STATE)
    return StackingRegressor(
        estimators=[
            ("rf", rf),
            ("ridge", ridge),
            ("lasso", lasso),
        ],
        final_estimator=Ridge(random_state=RANDOM_STATE),
        cv=3,
    )


def build_model_2() -> StackingRegressor:
    """Model 2: RF + XGBoost + Ridge + Lasso."""
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    xgb_reg = xgb.XGBRegressor(random_state=RANDOM_STATE)
    ridge = Ridge(random_state=RANDOM_STATE)
    lasso = Lasso(random_state=RANDOM_STATE)
    return StackingRegressor(
        estimators=[
            ("rf", rf),
            ("xgb", xgb_reg),
            ("ridge", ridge),
            ("lasso", lasso),
        ],
        final_estimator=Ridge(random_state=RANDOM_STATE),
        cv=3,
    )


def get_param_grids():
    """Param grids for RandomizedSearchCV - Model 1 (RF+Ridge+Lasso)."""
    return {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [5, 10, 15, None],
        "rf__min_samples_split": [2, 5, 10],
        "ridge__alpha": np.logspace(-2, 2, 10),
        "lasso__alpha": np.logspace(-4, 0, 10),
    }


def get_param_grids_model2():
    """Param grids for RandomizedSearchCV - Model 2 (RF+XGB+Ridge+Lasso).
    XGBoost: add regularization to reduce overfitting and improve R2.
    """
    return {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [5, 10, 15, None],
        "xgb__n_estimators": [100, 200, 300],
        "xgb__max_depth": [3, 5, 6],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__reg_alpha": [0.1, 1.0, 10.0],
        "xgb__reg_lambda": [0.1, 1.0, 10.0],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__subsample": [0.7, 0.8, 1.0],
        "ridge__alpha": np.logspace(-2, 2, 10),
        "lasso__alpha": np.logspace(-4, 0, 10),
    }


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_dir: Path,
) -> dict:
    """
    Train both models with RandomizedSearchCV.
    Returns dict with models, scaler, R2 scores, and metadata.
    """
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    X = X[feature_cols].copy()
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Model 1: RF + Ridge + Lasso
    print("Training Model 1 (RF + Ridge + Lasso)...")
    model1 = build_model_1()
    search1 = RandomizedSearchCV(
        model1,
        get_param_grids(),
        n_iter=N_ITER,
        cv=CV,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search1.fit(X_train_scaled, y_train)
    model1_best = search1.best_estimator_
    y_pred1 = model1_best.predict(X_test_scaled)
    r2_1 = r2_score(y_test, y_pred1)
    results["model1"] = {
        "model": model1_best,
        "r2": r2_1,
        "search": search1,
    }
    print(f"  Model 1 R2: {r2_1:.4f}")
    
    # Model 2: RF + XGBoost + Ridge + Lasso
    print("Training Model 2 (RF + XGBoost + Ridge + Lasso)...")
    model2 = build_model_2()
    search2 = RandomizedSearchCV(
        model2,
        get_param_grids_model2(),
        n_iter=N_ITER,
        cv=CV,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search2.fit(X_train_scaled, y_train)
    model2_best = search2.best_estimator_
    y_pred2 = model2_best.predict(X_test_scaled)
    r2_2 = r2_score(y_test, y_pred2)
    results["model2"] = {
        "model": model2_best,
        "r2": r2_2,
        "search": search2,
    }
    print(f"  Model 2 R2: {r2_2:.4f}")
    
    # Save artifacts
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model1_best, models_dir / "model1_stack.joblib")
    joblib.dump(model2_best, models_dir / "model2_stack.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(feature_cols, models_dir / "feature_columns.joblib")
    joblib.dump({"r2_model1": r2_1, "r2_model2": r2_2}, models_dir / "metrics.joblib")
    
    results["scaler"] = scaler
    results["feature_cols"] = feature_cols
    results["r2_model1"] = r2_1
    results["r2_model2"] = r2_2
    return results
