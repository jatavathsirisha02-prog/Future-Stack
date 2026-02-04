"""
Unified feature extraction for prediction.
Handles: transaction-level, clv_training (customer features), customer_summary.
"""
import pandas as pd
from typing import Tuple
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_COLS = [
    "recency", "frequency", "monetary", "avg_order_value",
    "days_since_first_purchase", "products_per_order",
    "category_diversity", "unique_products_purchased",
]


def _detect_schema(df: pd.DataFrame) -> str:
    cols = [c.strip().lower() for c in df.columns]
    if "recency" in cols and "frequency" in cols and "monetary" in cols:
        return "clv_training"
    if "customerid" in cols and "totalorders" in cols:
        return "customer_summary"
    if any("invoice" in c for c in cols) and any("quantity" in c or "qty" in c for c in cols):
        return "transaction"
    return "unknown"


def extract_features_for_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract canonical features from uploaded DataFrame.
    Handles transaction, clv_training, customer_summary schemas.
    Returns DataFrame with FEATURE_COLS, index = customer_id or row index.
    """
    schema = _detect_schema(df)
    
    if schema == "clv_training":
        return _extract_clv_training(df)
    if schema == "customer_summary":
        return _extract_customer_summary(df)
    if schema == "transaction":
        return _extract_transaction(df)
    
    raise ValueError(f"Unknown schema. Columns: {list(df.columns)}")


def _extract_clv_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    col_map = {
        "Recency": "recency", "Frequency": "frequency", "Monetary": "monetary",
        "AvgItemsPerOrder": "products_per_order", "UniqueCategories": "category_diversity",
        "DaysBetweenOrders": "days_since_first_purchase",
    }
    for old, new in col_map.items():
        if old in df.columns:
            df[new] = df[old]
    if "monetary" in df.columns and "frequency" in df.columns:
        df["avg_order_value"] = df["monetary"] / df["frequency"].replace(0, 1)
    if "unique_products_purchased" not in df.columns:
        df["unique_products_purchased"] = df.get("products_per_order", 1) * df.get("frequency", 1)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[FEATURE_COLS].fillna(0)


def _extract_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    X = pd.DataFrame(index=range(len(df)))
    X["recency"] = df.get("LastPurchaseDaysAgo", 30)
    X["frequency"] = df.get("TotalOrders", 1)
    X["monetary"] = df.get("TotalSpend", df.get("AvgBasketValue", 0))
    X["avg_order_value"] = df.get("AvgBasketValue", X["monetary"] / X["frequency"].replace(0, 1))
    X["days_since_first_purchase"] = df.get("TenureDays", 365)
    X["products_per_order"] = 5
    X["category_diversity"] = 1
    X["unique_products_purchased"] = X["frequency"] * 3
    return X[FEATURE_COLS].fillna(0)


def _extract_transaction(df: pd.DataFrame) -> pd.DataFrame:
    from ml.feature_engineering import create_customer_features
    
    clean_df = df.copy()
    if "product_category" not in clean_df.columns:
        clean_df["product_category"] = "General"
    if "product_id" not in clean_df.columns and "StockCode" in clean_df.columns:
        clean_df["product_id"] = clean_df["StockCode"].astype(str)
    
    features_df, _ = create_customer_features(clean_df, for_prediction=True)
    if features_df.empty:
        return pd.DataFrame()
    for c in FEATURE_COLS:
        if c not in features_df.columns:
            features_df[c] = 0
    return features_df.reindex(columns=FEATURE_COLS, fill_value=0)
