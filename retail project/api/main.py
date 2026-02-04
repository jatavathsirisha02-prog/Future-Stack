"""
FastAPI Backend for 30-Day CLV Prediction.
Endpoints: health, predict (accepts Excel/CSV), train (auto-retrain), model info.
Supports: transaction-level, clv_training, customer_summary datasets.
"""
import io
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "retail_clv.db"
sys_path_add = str(PROJECT_ROOT)
if sys_path_add not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_add)

from ml.feature_engineering import create_customer_features
from ml.inference import load_artifacts, predict, estimate_products_from_spend
from ml.predict_features import extract_features_for_predict, _detect_schema
from data_cleaning.cleaner import clean_retail_data
from data_ingestion.loader import load_to_database
from database.init_db import init_sqlite_db

app = FastAPI(
    title="30-Day CLV Prediction API",
    description="Predict short-term customer spend (30-day CLV). Supports multiple dataset types. Auto-train on new data.",
    version="2.0.0",
)

# Lazy-load artifacts
_artifacts = None


def get_artifacts():
    global _artifacts
    if _artifacts is None:
        if not (MODELS_DIR / "model1_stack.joblib").exists():
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Run: python scripts/run_training_all.py or POST /train"
            )
        _artifacts = load_artifacts(MODELS_DIR)
    return _artifacts


def clear_artifacts_cache():
    global _artifacts
    _artifacts = None


@app.get("/")
def root():
    return {"message": "30-Day CLV Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        arts = get_artifacts()
        return {"status": "ok", "models_loaded": True}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/model-info")
def model_info():
    arts = get_artifacts()
    return {
        "r2_model1": arts["metrics"].get("r2_model1"),
        "r2_model2": arts["metrics"].get("r2_model2"),
        "feature_columns": arts["feature_cols"],
    }


async def _parse_upload(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(contents))
    return pd.read_csv(io.BytesIO(contents), encoding="latin-1", low_memory=False)


@app.post("/upload-clean-predict")
async def upload_clean_predict_endpoint(file: UploadFile = File(...)):
    """
    Full pipeline: Upload uncleaned CSV -> Clean with pandas -> Load into schema tables ->
    Store bad data in rejected_data with rejection_reason column -> Run prediction on cleaned data.
    Returns predictions CSV + stats.
    """
    try:
        df = await _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    
    # Clean data (pandas) - segregates bad records with rejection_reason
    try:
        result = clean_retail_data(df=df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cleaning failed: {str(e)}")
    
    # Initialize DB and load clean data into schema tables
    init_sqlite_db(str(DB_PATH))
    load_to_database(str(DB_PATH), result)
    
    # Run prediction on cleaned data
    if result.clean_df.empty:
        raise HTTPException(status_code=400, detail="No valid records after cleaning. Cannot run prediction.")
    
    clean_df = result.clean_df.copy()
    if "product_category" not in clean_df.columns:
        clean_df["product_category"] = "General"
    if "product_id" not in clean_df.columns and "StockCode" in clean_df.columns:
        clean_df["product_id"] = clean_df["StockCode"].astype(str)
    
    try:
        features_df, _ = create_customer_features(clean_df, for_prediction=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")
    
    if features_df.empty:
        raise HTTPException(status_code=400, detail="No customer features generated")
    
    arts = get_artifacts()
    pred1, pred2, metrics = predict(features_df, arts)
    avg_ppd = 0.05
    pred_products = estimate_products_from_spend((pred1 + pred2) / 2, avg_ppd)
    
    # Build output
    n = len(pred1)
    customer_ids = features_df.index.astype(str).values[:n]
    output = pd.DataFrame({
        "customer_id": list(customer_ids),
        "r2_model1": metrics.get("r2_model1", 0),
        "r2_model2": metrics.get("r2_model2", 0),
        "predicted_30d_spend_model1": pred1.values,
        "predicted_30d_spend_model2": pred2.values,
        "predicted_no_of_products": pred_products.values,
    })
    
    # Rejection reasons summary
    rejection_summary = {}
    if not result.rejected_df.empty and "rejection_reason" in result.rejected_df.columns:
        rejection_summary = result.rejected_df["rejection_reason"].value_counts().to_dict()
    
    buffer = io.StringIO()
    output.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv",
            "X-Original-Count": str(result.stats["original_count"]),
            "X-Clean-Count": str(result.stats["clean_count"]),
            "X-Rejected-Count": str(result.stats["rejected_count"]),
            "X-Rejection-Reasons": str(rejection_summary),
        }
    )


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept Excel or CSV. Supports: transaction-level, clv_training, customer_summary.
    Returns predictions CSV.
    """
    try:
        df = await _parse_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    
    schema = _detect_schema(df)
    
    # Customer-level: extract features directly (no cleaning)
    if schema in ("clv_training", "customer_summary"):
        try:
            features_df = extract_features_for_predict(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature extraction failed: {str(e)}")
        features_df.index = range(len(features_df))
        customer_ids = df.get("CustomerID", df.index.astype(str)) if "CustomerID" in df.columns else range(len(df))
        if hasattr(customer_ids, "values"):
            customer_ids = customer_ids.astype(str).values
    else:
        # Transaction-level: clean then create features
        try:
            result = clean_retail_data(df=df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cleaning failed: {str(e)}")
        if result.clean_df.empty:
            raise HTTPException(status_code=400, detail="No valid records after cleaning")
        clean_df = result.clean_df.copy()
        if "product_category" not in clean_df.columns:
            clean_df["product_category"] = "General"
        if "product_id" not in clean_df.columns and "StockCode" in clean_df.columns:
            clean_df["product_id"] = clean_df["StockCode"].astype(str)
        try:
            features_df, _ = create_customer_features(clean_df, for_prediction=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")
        customer_ids = features_df.index.astype(str).values
    
    if features_df.empty:
        raise HTTPException(status_code=400, detail="No features generated")
    
    # Predict
    arts = get_artifacts()
    pred1, pred2, metrics = predict(features_df, arts)
    
    # Estimate products
    avg_ppd = 0.05
    pred_products = estimate_products_from_spend((pred1 + pred2) / 2, avg_ppd)
    
    n = len(pred1)
    cust_ids = (list(customer_ids)[:n] if len(customer_ids) >= n else list(customer_ids) + [f"c{i}" for i in range(len(customer_ids), n)])
    output = pd.DataFrame({
        "customer_id": cust_ids,
        "r2_model1": metrics.get("r2_model1", 0),
        "r2_model2": metrics.get("r2_model2", 0),
        "predicted_30d_spend_model1": pred1.values,
        "predicted_30d_spend_model2": pred2.values,
        "predicted_no_of_products": pred_products.values,
    })
    
    buffer = io.StringIO()
    output.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )


@app.post("/train")
async def train_endpoint(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
):
    """
    Auto-train models. Option A: Upload file to add to data/raw and retrain.
    Option B: No file - retrain on all existing files in data/raw.
    Returns training status. Training runs in background.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if file:
        try:
            contents = await file.read()
            path = RAW_DATA_DIR / (file.filename or "uploaded_data.csv")
            with open(path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save file: {str(e)}")
    
    def _run_training():
        try:
            from ml.multi_dataset_loader import load_all_datasets
            from ml.train_advanced import train_all_models
            X, y = load_all_datasets(RAW_DATA_DIR)
            if X.empty or len(X) < 50:
                return
            train_all_models(X, y, MODELS_DIR)
            clear_artifacts_cache()
        except Exception as e:
            pass
    
    if background_tasks:
        background_tasks.add_task(_run_training)
        return JSONResponse({"status": "training_started", "message": "Training in background. Check /model-info in 1-2 min."})
    else:
        try:
            from ml.multi_dataset_loader import load_all_datasets
            from ml.train_advanced import train_all_models
            X, y = load_all_datasets(RAW_DATA_DIR)
            if X.empty:
                raise HTTPException(status_code=400, detail="No compatible datasets in data/raw")
            if len(X) < 50:
                raise HTTPException(status_code=400, detail="Need at least 50 samples")
            train_all_models(X, y, MODELS_DIR)
            clear_artifacts_cache()
            metrics = __import__("joblib").load(MODELS_DIR / "metrics.joblib")
            return JSONResponse({"status": "ok", "metrics": metrics})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
