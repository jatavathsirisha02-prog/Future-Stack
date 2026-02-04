"""
Data Cleaning Module - 30-Day CLV Retail Project
Separate module for data quality validation and cleaning.
Applies data quality rules and segregates bad records.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Online Retail II dataset columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]

# Alternative column names for flexible dataset support
COLUMN_MAPPING = {
    "invoice_no": "InvoiceNo",
    "stock_code": "StockCode",
    "invoice_date": "InvoiceDate",
    "unit_price": "UnitPrice",
    "customer_id": "CustomerID",
    "order_id": "InvoiceNo",
    "orderid": "InvoiceNo",
    "invoice": "InvoiceNo",
    "product_id": "StockCode",
    "productid": "StockCode",
    "stockcode": "StockCode",
    "sku": "StockCode",
    "qty": "Quantity",
    "quantity": "Quantity",
    "price": "UnitPrice",
    "unitprice": "UnitPrice",
    "cust_id": "CustomerID",
    "customerid": "CustomerID",
    "user_id": "CustomerID",
    "userid": "CustomerID",
    "order_date": "InvoiceDate",
    "orderdate": "InvoiceDate",
    "transaction_date": "InvoiceDate",
    "transactiondate": "InvoiceDate",
    "date": "InvoiceDate",
}


def _infer_column(df: pd.DataFrame, candidates: list, fallback_hints: list = None) -> Optional[str]:
    """Find column that matches candidates or hints (e.g., 'invoice', 'order')."""
    cols = [c.strip().lower() for c in df.columns]
    for c in candidates:
        if c.lower() in cols:
            return df.columns[cols.index(c.lower())]
    if fallback_hints:
        for hint in fallback_hints:
            for c in df.columns:
                if hint in c.lower():
                    return c
    return None


@dataclass
class CleaningResult:
    """Result of data cleaning."""
    clean_df: pd.DataFrame
    rejected_df: pd.DataFrame
    rejection_log: List[dict]
    stats: dict


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (handle Excel variations and other dataset formats)."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    # Apply direct mappings
    for old, new in COLUMN_MAPPING.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    # Map common variants to expected
    col_map = {c: c for c in df.columns}
    for k, v in {
        "Invoice": "InvoiceNo",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price": "UnitPrice",
        "UnitPrice": "UnitPrice",
        "CustomerID": "CustomerID",
        "Customer_ID": "CustomerID",
        "Country": "Country",
    }.items():
        if k in df.columns:
            col_map[k] = v
    df.rename(columns={k: v for k, v in col_map.items() if k != v}, inplace=True)
    # Auto-infer missing columns from common patterns
    if "InvoiceNo" not in df.columns:
        c = _infer_column(df, ["order_id", "orderid", "invoice_no"], ["invoice", "order"])
        if c:
            df.rename(columns={c: "InvoiceNo"}, inplace=True)
    if "StockCode" not in df.columns:
        c = _infer_column(df, ["product_id", "productid", "sku"], ["stock", "product"])
        if c:
            df.rename(columns={c: "StockCode"}, inplace=True)
    if "InvoiceDate" not in df.columns:
        c = _infer_column(df, ["order_date", "orderdate", "transaction_date"], ["date"])
        if c:
            df.rename(columns={c: "InvoiceDate"}, inplace=True)
    if "Quantity" not in df.columns:
        c = _infer_column(df, ["qty", "quantity"], ["qty", "quantity"])
        if c:
            df.rename(columns={c: "Quantity"}, inplace=True)
    if "UnitPrice" not in df.columns:
        c = _infer_column(df, ["price", "unit_price"], ["price"])
        if c:
            df.rename(columns={c: "UnitPrice"}, inplace=True)
    if "CustomerID" not in df.columns:
        c = _infer_column(df, ["customer_id", "cust_id", "user_id"], ["customer", "user"])
        if c:
            df.rename(columns={c: "CustomerID"}, inplace=True)
    return df


def apply_quality_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """
    Apply data quality rules using pandas. Segregate bad records with rejection_reason.
    Returns: (clean_df, rejected_df with rejection_reason column, rejection_log)
    """
    rejected_rows = []
    rejection_log = []
    
    # Rule 1: Drop rows with null CustomerID (required for CLV)
    null_customer = df["CustomerID"].isna()
    if null_customer.any():
        rejected = df[null_customer].copy()
        rejected["rejection_reason"] = "Null CustomerID"
        rejected_rows.append(rejected)
        rejection_log.extend([
            {"row_idx": idx, "reason": "Null CustomerID", "invoice": row.get("InvoiceNo", "")}
            for idx, row in df[null_customer].iterrows()
        ])
    
    df = df[~null_customer].copy()
    
    # Rule 2: Drop cancellation invoices (InvoiceNo starting with 'C' or 'c')
    if "InvoiceNo" in df.columns:
        cancellation = df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
        if cancellation.any():
            rejected = df[cancellation].copy()
            rejected["rejection_reason"] = "Cancellation invoice"
            rejected_rows.append(rejected)
            rejection_log.extend([
                {"row_idx": idx, "reason": "Cancellation invoice", "invoice": str(row.get("InvoiceNo", ""))}
                for idx, row in df[cancellation].iterrows()
            ])
        df = df[~cancellation]
    
    # Rule 3: Quantity must be positive
    if "Quantity" in df.columns:
        invalid_qty = df["Quantity"] <= 0
        if invalid_qty.any():
            rejected = df[invalid_qty].copy()
            rejected["rejection_reason"] = "Quantity <= 0"
            rejected_rows.append(rejected)
            rejection_log.extend([
                {"row_idx": idx, "reason": "Quantity <= 0", "invoice": row.get("InvoiceNo", "")}
                for idx, row in df[invalid_qty].iterrows()
            ])
        df = df[~invalid_qty]
    
    # Rule 4: UnitPrice must be non-negative
    if "UnitPrice" in df.columns:
        invalid_price = df["UnitPrice"] < 0
        if invalid_price.any():
            rejected = df[invalid_price].copy()
            rejected["rejection_reason"] = "Negative UnitPrice"
            rejected_rows.append(rejected)
            rejection_log.extend([
                {"row_idx": idx, "reason": "Negative UnitPrice", "invoice": row.get("InvoiceNo", "")}
                for idx, row in df[invalid_price].iterrows()
            ])
        df = df[~invalid_price]
    
    # Rule 5: Valid InvoiceDate
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        invalid_date = df["InvoiceDate"].isna()
        if invalid_date.any():
            rejected = df[invalid_date].copy()
            rejected["rejection_reason"] = "Invalid InvoiceDate"
            rejected_rows.append(rejected)
            rejection_log.extend([
                {"row_idx": idx, "reason": "Invalid InvoiceDate", "invoice": row.get("InvoiceNo", "")}
                for idx, row in df[invalid_date].iterrows()
            ])
        df = df[~invalid_date]
    
    # Rule 6: Drop duplicate rows (duplicates not stored in rejected - just logged)
    before_dedup = len(df)
    df = df.drop_duplicates()
    if len(df) < before_dedup:
        rejection_log.append({
            "reason": "Duplicate rows removed", "count": before_dedup - len(df)
        })
    
    rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    
    return df, rejected_df, rejection_log


def clean_retail_data(
    input_path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
) -> CleaningResult:
    """
    Main cleaning function. Accepts file path or DataFrame.
    Returns CleaningResult with clean_df, rejected_df, log, and stats.
    """
    if df is None and input_path is None:
        raise ValueError("Provide either input_path or df")
    
    if df is None:
        input_path = Path(input_path)
        if input_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, encoding="latin-1", low_memory=False)
    
    original_count = len(df)
    
    # Normalize columns
    df = normalize_columns(df)
    
    # Ensure required columns exist (transaction-level format expected)
    required = ["CustomerID", "Quantity", "UnitPrice", "InvoiceDate", "InvoiceNo", "StockCode"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}. "
            "Expected transaction-level data with: InvoiceNo/OrderID, StockCode/ProductID, Quantity, "
            "InvoiceDate/OrderDate, UnitPrice/Price, CustomerID."
        )
    
    # Apply quality rules
    clean_df, rejected_df, rejection_log = apply_quality_rules(df)
    
    stats = {
        "original_count": original_count,
        "clean_count": len(clean_df),
        "rejected_count": len(rejected_df),
        "rejection_rate": len(rejected_df) / original_count if original_count > 0 else 0,
    }
    
    return CleaningResult(
        clean_df=clean_df,
        rejected_df=rejected_df,
        rejection_log=rejection_log,
        stats=stats,
    )
