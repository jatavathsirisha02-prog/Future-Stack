"""
Advanced ML Training - Multiple models, ensemble, auto-train.
Models: RF+Ridge+Lasso, RF+XGB+Ridge+Lasso, GradientBoosting, XGBoost standalone, Ensemble.
Uses RandomizedSearchCV, picks best model, supports auto-retrain.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

RANDOM_STATE = 42
N_ITER = 20  # Per model for faster auto-train
CV = 3

FEATURE_COLS = [
    "recency", "frequency", "monetary", "avg_order_value",
    "days_since_first_purchase", "products_per_order",
    "category_diversity", "unique_products_purchased",
]


def build_models():
    """Build all models for comparison."""
    return {
        "model1_rf_ridge_lasso": StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(random_state=RANDOM_STATE)),
                ("ridge", Ridge(random_state=RANDOM_STATE)),
                ("lasso", Lasso(random_state=RANDOM_STATE)),
            ],
            final_estimator=Ridge(random_state=RANDOM_STATE),
            cv=3,
        ),
        "model2_rf_xgb_ridge_lasso": StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(random_state=RANDOM_STATE)),
                ("xgb", xgb.XGBRegressor(random_state=RANDOM_STATE)),
                ("ridge", Ridge(random_state=RANDOM_STATE)),
                ("lasso", Lasso(random_state=RANDOM_STATE)),
            ],
            final_estimator=Ridge(random_state=RANDOM_STATE),
            cv=3,
        ),
        "model3_gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "model4_xgboost": xgb.XGBRegressor(random_state=RANDOM_STATE),
    }


def get_param_grids():
    """Param grids per model."""
    return {
        "model1_rf_ridge_lasso": {
            "rf__n_estimators": [50, 100, 200],
            "rf__max_depth": [5, 10, 15, None],
            "ridge__alpha": np.logspace(-2, 2, 8),
            "lasso__alpha": np.logspace(-4, 0, 8),
        },
        "model2_rf_xgb_ridge_lasso": {
            "rf__n_estimators": [50, 100, 200],
            "xgb__n_estimators": [100, 200],
            "xgb__max_depth": [3, 5, 6],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__reg_alpha": [0.1, 1.0],
            "xgb__reg_lambda": [0.1, 1.0],
            "ridge__alpha": np.logspace(-2, 2, 5),
            "lasso__alpha": np.logspace(-4, 0, 5),
        },
        "model3_gradient_boosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_samples_split": [2, 5],
        },
        "model4_xgboost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "reg_alpha": [0.1, 1.0, 10.0],
            "reg_lambda": [0.1, 1.0, 10.0],
            "min_child_weight": [1, 3, 5],
        },
    }


def train_all_models(X: pd.DataFrame, y: pd.Series, models_dir: Path) -> dict:
    """Train all models, pick best, save ensemble."""
    feature_cols = [c for c in FEATURE_COLS if c in X.columns]
    X = X.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = build_models()
    param_grids = get_param_grids()
    results = {}
    
    for name, model in models.items():
        try:
            params = param_grids.get(name, {})
            if params:
                search = RandomizedSearchCV(
                    model, params, n_iter=min(N_ITER, 50),
                    cv=CV, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1
                )
                search.fit(X_train_s, y_train)
                best = search.best_estimator_
            else:
                best = model.fit(X_train_s, y_train)
            pred = best.predict(X_test_s)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            results[name] = {"model": best, "r2": r2, "mae": mae}
            print(f"  {name}: R2={r2:.4f}, MAE={mae:.2f}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")
    
    # Pick best by R2
    top2 = sorted(results.keys(), key=lambda k: results[k]["r2"], reverse=True)[:2]
    best_name = top2[0]
    best_r2 = results[best_name]["r2"]
    
    # Ensemble: average top 2 models
    def ensemble_predict(X_s):
        return np.mean([results[n]["model"].predict(X_s) for n in top2], axis=0)
    ensemble_r2 = r2_score(y_test, ensemble_predict(X_test_s))
    if ensemble_r2 > best_r2:
        best_name = "ensemble"
        results["ensemble"] = {"r2": ensemble_r2, "top_models": top2}
        print(f"  Ensemble (top2): R2={ensemble_r2:.4f}")
    
    # Save
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_name]["model"] if best_name != "ensemble" else results[top2[0]]["model"], models_dir / "model_best.joblib")
    joblib.dump(results[top2[1]]["model"] if best_name == "ensemble" else results[best_name]["model"], models_dir / "model_secondary.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(FEATURE_COLS, models_dir / "feature_columns.joblib")
    all_metrics = {k: v["r2"] for k, v in results.items() if isinstance(v, dict) and "r2" in v}
    all_metrics["best_model"] = best_name
    all_metrics["r2_best"] = results[best_name]["r2"]
    joblib.dump(all_metrics, models_dir / "metrics_full.joblib")
    
    # Backward compat for existing API (model1_stack, model2_stack, metrics.joblib)
    m1 = results.get("model1_rf_ridge_lasso", {}).get("model")
    m2 = results.get("model2_rf_xgb_ridge_lasso", {}).get("model")
    if m1 is None:
        m1 = results[list(results.keys())[0]]["model"]
    if m2 is None:
        m2 = results[list(results.keys())[1]]["model"] if len(results) > 1 else m1
    joblib.dump(m1, models_dir / "model1_stack.joblib")
    joblib.dump(m2, models_dir / "model2_stack.joblib")
    r2_1 = results.get("model1_rf_ridge_lasso", {}).get("r2", all_metrics["r2_best"])
    r2_2 = results.get("model2_rf_xgb_ridge_lasso", {}).get("r2", all_metrics["r2_best"])
    joblib.dump({"r2_model1": r2_1, "r2_model2": r2_2}, models_dir / "metrics.joblib")
    
    return results
