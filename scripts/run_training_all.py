"""
Train on ALL datasets in data/raw.
Combines clv_training_dataset, customer_summary_dataset, transaction datasets.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.multi_dataset_loader import load_all_datasets, FEATURE_COLS
from ml.train_advanced import train_all_models


def main():
    raw_dir = project_root / "data" / "raw"
    models_dir = project_root / "models"
    
    print("Loading all datasets from data/raw...")
    X, y = load_all_datasets(raw_dir)
    
    if X.empty:
        print("No compatible datasets found. Ensure data/raw has:")
        print("  - Transaction: InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID")
        print("  - CLV training: Recency, Frequency, Monetary, TargetSpendNext30d")
        print("  - Customer summary: CustomerID, TotalOrders, TotalSpend, LastPurchaseDaysAgo, PredictedNext30dProducts")
        return
    
    print(f"Combined: {len(X)} samples, {list(X.columns)}")
    
    if len(X) < 50:
        print("Too few samples. Need at least 50.")
        return
    
    print("Training all models...")
    train_all_models(X, y, models_dir)
    print("Done. Models saved in models/")


if __name__ == "__main__":
    main()
