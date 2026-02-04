"""
Full Pipeline: Upload CSV -> Clean with pandas -> Load into schema tables ->
Store rejected data in rejected_data with rejection_reason -> Run prediction.
Usage: python scripts/run_full_pipeline.py path/to/uncleaned.csv
"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.init_db import init_sqlite_db
from data_cleaning.cleaner import clean_retail_data
from data_ingestion.loader import load_to_database
from ml.feature_engineering import create_customer_features
from ml.inference import load_artifacts, predict, estimate_products_from_spend

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <path_to_uncleaned.csv>")
        print("Example: python run_full_pipeline.py data/raw/online_retail_II.csv")
        return
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    
    db_path = project_root / "retail_clv.db"
    output_path = project_root / "output" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("1. Cleaning data (pandas)...")
    result = clean_retail_data(input_path=csv_path)
    print(f"   Original: {result.stats['original_count']}")
    print(f"   Clean: {result.stats['clean_count']}")
    print(f"   Rejected: {result.stats['rejected_count']}")
    
    if not result.rejected_df.empty and "rejection_reason" in result.rejected_df.columns:
        print("   Rejection reasons:")
        for reason, count in result.rejected_df["rejection_reason"].value_counts().items():
            print(f"     - {reason}: {count}")
    
    print("2. Loading into schema tables (stores, products, customer_details, store_sales_header, store_sales_line_items, rejected_data)...")
    init_sqlite_db(str(db_path))
    counts = load_to_database(str(db_path), result)
    print(f"   Loaded: {counts}")
    
    if result.clean_df.empty:
        print("No clean data. Cannot run prediction.")
        return
    
    print("3. Running prediction on cleaned data...")
    clean_df = result.clean_df.copy()
    clean_df["product_category"] = "General"
    clean_df["product_id"] = clean_df["StockCode"].astype(str)
    
    features_df, _ = create_customer_features(clean_df, for_prediction=True)
    if features_df.empty:
        print("No features generated.")
        return
    
    artifacts = load_artifacts(project_root / "models")
    pred1, pred2, metrics = predict(features_df, artifacts)
    pred_products = estimate_products_from_spend((pred1 + pred2) / 2, 0.05)
    
    output = pd.DataFrame({
        "customer_id": features_df.index.astype(str),
        "r2_model1": metrics.get("r2_model1", 0),
        "r2_model2": metrics.get("r2_model2", 0),
        "predicted_30d_spend_model1": pred1.values,
        "predicted_30d_spend_model2": pred2.values,
        "predicted_no_of_products": pred_products.values,
    })
    output.to_csv(output_path, index=False)
    print(f"   Predictions saved to {output_path}")
    print("   Done.")


if __name__ == "__main__":
    main()
