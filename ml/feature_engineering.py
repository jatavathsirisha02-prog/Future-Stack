"""
Feature Engineering for 30-Day CLV Prediction.
Creates customer-level features (RFM, category, etc.) and target future_spend_30d.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sqlite3


def load_transactions_from_db(db_path: str) -> pd.DataFrame:
    """Load transactions from database and merge with header."""
    conn = sqlite3.connect(db_path)
    header = pd.read_sql("SELECT * FROM store_sales_header", conn)
    line_items = pd.read_sql("SELECT * FROM store_sales_line_items", conn)
    products = pd.read_sql("SELECT product_id, product_category FROM products", conn)
    conn.close()
    
    line_items = line_items.merge(
        products, on="product_id", how="left", suffixes=("", "_cat")
    )
    df = line_items.merge(header, on="transaction_id", how="inner")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["line_amount"] = df["line_item_amount"]
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def _infer_col(df: pd.DataFrame, candidates: list, hints: list = None) -> Optional[str]:
    """Find column matching candidates or hints."""
    cols_lower = [c.strip().lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df.columns[cols_lower.index(cand.lower())]
    if hints:
        for hint in hints:
            for col in df.columns:
                if hint in col.lower():
                    return col
    return None


def load_transactions_from_file(file_path: Path) -> pd.DataFrame:
    """Load transactions from Excel/CSV (flexible column names)."""
    if str(file_path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
    
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    for old, new in [
        ("order_id", "InvoiceNo"), ("invoice_no", "InvoiceNo"), ("product_id", "StockCode"),
        ("stock_code", "StockCode"), ("sku", "StockCode"), ("qty", "Quantity"),
        ("quantity", "Quantity"), ("price", "UnitPrice"), ("unit_price", "UnitPrice"),
        ("customer_id", "CustomerID"), ("cust_id", "CustomerID"), ("order_date", "InvoiceDate"),
        ("transaction_date", "InvoiceDate"),
    ]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    if "InvoiceNo" not in df.columns:
        c = _infer_col(df, ["orderid", "invoice"], ["invoice", "order"])
        if c:
            df.rename(columns={c: "InvoiceNo"}, inplace=True)
    if "StockCode" not in df.columns:
        c = _infer_col(df, ["productid", "stockcode"], ["stock", "product", "sku"])
        if c:
            df.rename(columns={c: "StockCode"}, inplace=True)
    if "InvoiceDate" not in df.columns:
        c = _infer_col(df, ["orderdate", "transactiondate"], ["date"])
        if c:
            df.rename(columns={c: "InvoiceDate"}, inplace=True)
    if "Quantity" not in df.columns:
        c = _infer_col(df, ["qty"], ["qty", "quantity"])
        if c:
            df.rename(columns={c: "Quantity"}, inplace=True)
    if "UnitPrice" not in df.columns:
        c = _infer_col(df, ["price", "unitprice"], ["price"])
        if c:
            df.rename(columns={c: "UnitPrice"}, inplace=True)
    if "CustomerID" not in df.columns:
        c = _infer_col(df, ["customerid", "custid", "userid"], ["customer", "user"])
        if c:
            df.rename(columns={c: "CustomerID"}, inplace=True)
    
    df["transaction_date"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["line_amount"] = df["Quantity"] * df["UnitPrice"]
    df["transaction_id"] = df["InvoiceNo"].astype(str)
    df["customer_id"] = df["CustomerID"].astype(str)
    if "product_category" not in df.columns:
        df["product_category"] = "General"
    if "product_id" not in df.columns:
        df["product_id"] = df["StockCode"].astype(str)
    return df


def create_customer_features(
    df: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
    horizon_days: int = 30,
    for_prediction: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create customer-level features and target.
    Target: future_spend_30d = total spend in next 30 days after cutoff.
    Features: RFM, category diversity, etc. from behavior before cutoff.
    For prediction: use ALL data as past (no future split needed).
    """
    if "transaction_date" not in df.columns:
        df["transaction_date"] = pd.to_datetime(df.get("InvoiceDate", df.get("invoice_date")), errors="coerce")
    if "line_amount" not in df.columns:
        q = df.get("Quantity", df.get("quantity", 0))
        p = df.get("UnitPrice", df.get("unit_price", 0))
        df["line_amount"] = q * p
    if "customer_id" not in df.columns and "CustomerID" in df.columns:
        df["customer_id"] = df["CustomerID"].astype(str)
    if "transaction_id" not in df.columns and "InvoiceNo" in df.columns:
        df["transaction_id"] = df["InvoiceNo"].astype(str)
    
    df = df.dropna(subset=["transaction_date", "customer_id"])
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    max_date = df["transaction_date"].max()
    min_date = df["transaction_date"].min()
    
    if for_prediction:
        # For inference: use ALL data as past (we only need features, no target)
        cutoff_date = max_date + pd.Timedelta(days=1)
        past = df.copy()
    else:
        # For training: split past/future for target
        if cutoff_date is None:
            cutoff_date = max_date - pd.Timedelta(days=60)
        if cutoff_date <= min_date:
            cutoff_date = min_date + pd.Timedelta(days=30)
        past = df[df["transaction_date"] < cutoff_date].copy()
    future_start = cutoff_date
    future_end = cutoff_date + pd.Timedelta(days=horizon_days)
    future = df[(df["transaction_date"] >= future_start) & (df["transaction_date"] < future_end)]
    
    # Target: future spend per customer (only for training)
    if not for_prediction:
        target = future.groupby("customer_id")["line_amount"].sum()
        target.name = "future_spend_30d"
    else:
        target = pd.Series(dtype=float)
    
    # Features from past behavior
    past_trans = past.groupby("transaction_id").agg({
        "transaction_date": "first",
        "line_amount": "sum",
        "customer_id": "first",
    }).reset_index()
    
    cust_features = []
    for cid, g in past.groupby("customer_id"):
        trans = past_trans[past_trans["customer_id"] == cid]
        if trans.empty:
            continue
        recency = (cutoff_date - trans["transaction_date"].max()).days
        frequency = len(trans)
        monetary = trans["line_amount"].sum()
        avg_order_value = monetary / frequency if frequency > 0 else 0
        days_since_first = (cutoff_date - trans["transaction_date"].min()).days
        products_per_order = len(g) / frequency if frequency > 0 else 0
        category_count = g["product_category"].nunique() if "product_category" in g.columns else 1
        sp_col = "StockCode" if "StockCode" in g.columns else "product_id"
        unique_products = g[sp_col].nunique() if sp_col in g.columns else 1
        
        cust_features.append({
            "customer_id": str(cid),
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_order_value": avg_order_value,
            "days_since_first_purchase": days_since_first,
            "products_per_order": products_per_order,
            "category_diversity": category_count,
            "unique_products_purchased": unique_products,
        })
    
    features_df = pd.DataFrame(cust_features)
    if features_df.empty:
        return features_df, pd.Series()
    features_df = features_df.set_index("customer_id")
    
    # Align target with features (only for training)
    if not for_prediction and not target.empty:
        common = features_df.index.intersection(target.index)
        features_df = features_df.loc[common].copy()
        target_aligned = target.reindex(common, fill_value=0)
    else:
        target_aligned = pd.Series(dtype=float)
    
    return features_df, target_aligned


def prepare_training_data(
    db_path: Optional[str] = None,
    file_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data and prepare customer-level features + target."""
    if db_path:
        df = load_transactions_from_db(db_path)
    elif file_path:
        df = load_transactions_from_file(file_path)
        if "product_category" not in df.columns:
            df["product_category"] = "General"
        if "product_id" not in df.columns:
            df["product_id"] = df["StockCode"].astype(str)
    else:
        raise ValueError("Provide db_path or file_path")
    
    return create_customer_features(df)
