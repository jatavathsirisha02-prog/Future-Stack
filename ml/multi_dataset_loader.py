"""
Multi-Dataset Loader - Handles different schemas and combines for training.
Supports: transaction-level (Online Retail II, data.csv, synthetic_retail),
          customer-level (clv_training_dataset, customer_summary_dataset).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Canonical feature columns (must align across all datasets)
FEATURE_COLS = [
    "recency", "frequency", "monetary", "avg_order_value",
    "days_since_first_purchase", "products_per_order",
    "category_diversity", "unique_products_purchased",
]

TARGET_COL = "future_spend_30d"


def _detect_schema(df: pd.DataFrame) -> str:
    """Detect schema: 'transaction', 'clv_training', 'customer_summary', or 'unknown'."""
    cols = [c.strip().lower() for c in df.columns]
    # clv_training_dataset: Recency, Frequency, Monetary, TargetSpendNext30d
    if "recency" in cols and "frequency" in cols and "monetary" in cols and "targetspendnext30d" in cols:
        return "clv_training"
    # customer_summary: CustomerID, TotalOrders, TotalSpend, LastPurchaseDaysAgo
    if "customerid" in cols and "totalorders" in cols and ("totalspend" in cols or "totalspend" in str(cols)):
        return "customer_summary"
    # transaction: InvoiceNo/Invoice, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID
    trans_markers = ["invoice", "stockcode", "quantity", "unitprice", "customerid"]
    if sum(1 for m in trans_markers if any(m in c for c in cols)) >= 4:
        return "transaction"
    return "unknown"


def _load_clv_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Map clv_training_dataset to canonical features."""
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
    # avg_order_value = monetary / frequency
    if "monetary" in df.columns and "frequency" in df.columns:
        df["avg_order_value"] = np.where(df["frequency"] > 0, df["monetary"] / df["frequency"], 0)
    if "unique_products_purchased" not in df.columns:
        df["unique_products_purchased"] = df.get("products_per_order", 1) * df.get("frequency", 1)
    # Target
    if "TargetSpendNext30d" in df.columns:
        y = df["TargetSpendNext30d"].copy()
    elif "TargetProductsNext30d" in df.columns:
        y = df["TargetProductsNext30d"].copy() * (df["monetary"].median() / max(df["frequency"].median(), 1))  # estimate spend
    else:
        return pd.DataFrame(), pd.Series()
    # Fill missing features
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    X = df[[c for c in FEATURE_COLS if c in df.columns]].reindex(columns=FEATURE_COLS, fill_value=0)
    X.index = range(len(X))
    y.index = range(len(y))
    return X, y


def _load_customer_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Map customer_summary_dataset to canonical features."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Map: TotalOrders->frequency, TotalSpend->monetary, LastPurchaseDaysAgo->recency
    # TenureDays->days_since_first, AvgBasketValue->avg_order_value
    X = pd.DataFrame(index=range(len(df)))
    X["recency"] = df["LastPurchaseDaysAgo"] if "LastPurchaseDaysAgo" in df.columns else 30
    X["frequency"] = df["TotalOrders"] if "TotalOrders" in df.columns else 1
    X["monetary"] = df["TotalSpend"] if "TotalSpend" in df.columns else df["AvgBasketValue"] if "AvgBasketValue" in df.columns else 0
    X["avg_order_value"] = df["AvgBasketValue"] if "AvgBasketValue" in df.columns else X["monetary"] / np.maximum(X["frequency"], 1)
    X["days_since_first_purchase"] = df["TenureDays"] if "TenureDays" in df.columns else 365
    X["products_per_order"] = 5  # unknown
    X["category_diversity"] = 1  # PreferredCategory exists but single value
    X["unique_products_purchased"] = X["frequency"] * 3  # heuristic
    # Target
    y = df["PredictedNext30dProducts"] if "PredictedNext30dProducts" in df.columns else pd.Series(0, index=df.index)
    if "PredictedNext30dProducts" in df.columns:
        avg_basket = df["AvgBasketValue"].median()
        y = df["PredictedNext30dProducts"] * avg_basket  # convert products to spend estimate
    return X[FEATURE_COLS], y


def _load_transaction(df: pd.DataFrame, from_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load transaction-level data via feature engineering."""
    from ml.feature_engineering import create_customer_features, load_transactions_from_file
    from data_cleaning.cleaner import clean_retail_data
    
    if from_path:
        try:
            result = clean_retail_data(input_path=from_path)
            if not result.clean_df.empty:
                df = result.clean_df.copy()
                df["transaction_date"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
                df["line_amount"] = df["Quantity"] * df["UnitPrice"]
                df["customer_id"] = df["CustomerID"].astype(str)
                df["transaction_id"] = df["InvoiceNo"].astype(str)
                df["product_category"] = "General"
                df["product_id"] = df["StockCode"].astype(str)
            else:
                df = load_transactions_from_file(from_path)
        except Exception:
            df = load_transactions_from_file(from_path)
    else:
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        for old, new in [("Invoice", "InvoiceNo"), ("order_id", "InvoiceNo"), ("product_id", "StockCode")]:
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        if "InvoiceDate" not in df.columns and "Date" in df.columns:
            df.rename(columns={"Date": "InvoiceDate"}, inplace=True)
        df["transaction_date"] = pd.to_datetime(df.get("InvoiceDate", df.get("InvoiceDate")), errors="coerce")
        df["line_amount"] = df.get("Quantity", 1) * df.get("UnitPrice", 0)
        df["customer_id"] = df.get("CustomerID", df.get("Customer_ID", "")).astype(str)
        df["transaction_id"] = df.get("InvoiceNo", df.get("Invoice", "")).astype(str)
        df["product_category"] = df.get("Category", "General")
        df["product_id"] = df.get("StockCode", "").astype(str)
    
    X, y = create_customer_features(df, for_prediction=False)
    if X.empty:
        return pd.DataFrame(), pd.Series()
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = 0
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return X, y


def load_single_file(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load single file and return (X, y)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False, nrows=100000)
    
    schema = _detect_schema(df)
    if schema == "clv_training":
        return _load_clv_training(df)
    if schema == "customer_summary":
        return _load_customer_summary(df)
    if schema == "transaction":
        return _load_transaction(df, from_path=path)
    return pd.DataFrame(), pd.Series()


# Skip datasets that don't fit CLV (product-level, complex JSON)
SKIP_FILES = {"product_daily_sales_dataset.csv", "E-commerce.csv"}


def load_all_datasets(raw_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and combine all compatible datasets from raw_dir."""
    raw_dir = Path(raw_dir)
    all_X, all_y = [], []
    loaded = []
    
    files = sorted(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
    for f in files:
        if f.name in SKIP_FILES:
            continue
        try:
            X, y = load_single_file(f)
            if not X.empty and len(X) >= 30:
                all_X.append(X)
                all_y.append(y)
                loaded.append(f.name)
        except Exception:
            continue
    
    if not all_X:
        return pd.DataFrame(), pd.Series()
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    if len(y_combined) != len(X_combined):
        y_combined = y_combined.reindex(range(len(X_combined)), fill_value=0)
    
    X_combined = X_combined.fillna(0)
    return X_combined, y_combined
