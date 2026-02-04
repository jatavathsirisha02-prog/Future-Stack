"""
Data Ingestion - Load cleaned data into database.
Maps Online Retail II columns to schema tables.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
from data_cleaning.cleaner import clean_retail_data, CleaningResult


def load_to_database(
    db_path: str,
    clean_result: CleaningResult,
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Load clean data into database tables.
    Maps Online Retail II -> store_sales_header, store_sales_line_items, products, stores, customer_details.
    Online Retail II has no stores - we create synthetic store from Country.
    """
    df = clean_result.clean_df.copy()
    df["line_item_amount"] = df["Quantity"] * df["UnitPrice"]
    
    # Create store_id from Country (Online Retail is online, use Country as region/store)
    df["store_id"] = df["Country"].astype(str).str[:10].str.upper().str.replace(" ", "_")
    
    # Unique stores
    stores = df[["store_id", "Country"]].drop_duplicates()
    stores_df = pd.DataFrame({
        "store_id": stores["store_id"],
        "store_name": stores["Country"],
        "store_city": stores["Country"],
        "store_region": stores["Country"],
        "opening_date": pd.NaT,
    })
    
    # Unique products
    prod_agg = df.groupby("StockCode").agg({
        "Description": "first",
        "UnitPrice": "first",
    }).reset_index()
    products_df = pd.DataFrame({
        "product_id": prod_agg["StockCode"].astype(str),
        "product_name": prod_agg["Description"].fillna("Unknown"),
        "product_category": "General",
        "unit_price": prod_agg["UnitPrice"],
        "current_stock_level": 0,
    })
    
    # Customer details (basic - we'll enrich later)
    cust_agg = df.groupby("CustomerID").agg({"InvoiceDate": "max"}).reset_index()
    customers_df = pd.DataFrame({
        "customer_id": cust_agg["CustomerID"].astype(int).astype(str),
        "first_name": "",
        "email": "",
        "loyalty_status": "Standard",
        "total_loyalty_points": 0,
        "last_purchase_date": cust_agg["InvoiceDate"],
        "segment_id": "",
        "customer_phone": "",
        "customer_since": pd.NaT,
    })
    
    # Transaction headers
    header = df.groupby("InvoiceNo").agg({
        "store_id": "first",
        "CustomerID": "first",
        "InvoiceDate": "first",
        "line_item_amount": "sum",
    }).reset_index()
    header_df = pd.DataFrame({
        "transaction_id": header["InvoiceNo"].astype(str),
        "store_id": header["store_id"],
        "customer_id": header["CustomerID"].astype(int).astype(str),
        "transaction_date": header["InvoiceDate"],
        "total_amount": header["line_item_amount"],
        "customer_phone": "",
    })
    
    # Line items
    df["line_item_id"] = df.groupby("InvoiceNo").cumcount() + 1
    line_items_df = pd.DataFrame({
        "line_item_id": df["line_item_id"],
        "transaction_id": df["InvoiceNo"].astype(str),
        "product_id": df["StockCode"].astype(str),
        "promotion_id": "",
        "quantity": df["Quantity"],
        "line_item_amount": df["line_item_amount"],
    })
    
    # Rejected data (with rejection_reason per row)
    rejected_df = clean_result.rejected_df.copy()
    if not rejected_df.empty:
        # Exclude rejection_reason from raw_data (we store it in its own column)
        cols_for_raw = [c for c in rejected_df.columns if c != "rejection_reason"]
        rejected_df["raw_data"] = rejected_df[cols_for_raw].apply(lambda r: "|".join(str(v) for v in r.values), axis=1)
    
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(db_path)
        close_conn = True
    
    try:
        # Load stores (chunk to avoid SQLite variable limit)
        stores_df.to_sql("stores", conn, if_exists="replace", index=False, chunksize=500)
        # Load products
        products_df.to_sql("products", conn, if_exists="replace", index=False, chunksize=500)
        # Load customers
        customers_df.to_sql("customer_details", conn, if_exists="replace", index=False, chunksize=500)
        # Load headers
        header_df.to_sql("store_sales_header", conn, if_exists="replace", index=False, chunksize=500)
        # Load line items
        line_items_df.to_sql("store_sales_line_items", conn, if_exists="replace", index=False, chunksize=1000)
        
        # Load rejected (with per-row rejection_reason)
        if not rejected_df.empty and "raw_data" in rejected_df.columns:
            rej = rejected_df[["raw_data", "rejection_reason"]].copy()
            rej.to_sql("rejected_data", conn, if_exists="append", index=False, chunksize=500)
        
        # Seed loyalty_rules and promotion_details if empty
        cur = conn.execute("SELECT COUNT(*) FROM loyalty_rules")
        if cur.fetchone()[0] == 0:
            conn.execute("""
                INSERT INTO loyalty_rules (rule_id, rule_name, points_per_unit_spend, min_spend_threshold, bonus_points)
                VALUES (1, 'Standard Earning', 1.0, 0, 0),
                       (2, 'Weekend Bonus', 1.5, 50, 10)
            """)
        
        cur = conn.execute("SELECT COUNT(*) FROM promotion_details")
        if cur.fetchone()[0] == 0:
            conn.execute("""
                INSERT INTO promotion_details (promotion_id, promotion_name, start_date, end_date, discount_percentage, applicable_category)
                VALUES ('PROMO0', 'No Promo', '2000-01-01', '2099-12-31', 0, 'ALL')
            """)
        
        conn.commit()
    finally:
        if close_conn:
            conn.close()
    
    return {
        "stores": len(stores_df),
        "products": len(products_df),
        "customers": len(customers_df),
        "transactions": len(header_df),
        "line_items": len(line_items_df),
    }
