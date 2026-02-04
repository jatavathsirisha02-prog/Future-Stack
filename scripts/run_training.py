"""
Run full training pipeline: load data -> features -> train models -> save.
Requires data in data/raw/ (Online Retail II Excel/CSV).
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.init_db import init_sqlite_db
from data_cleaning.cleaner import clean_retail_data
from data_ingestion.loader import load_to_database
from ml.feature_engineering import prepare_training_data, load_transactions_from_db
from ml.train import train_models


def main():
    db_path = project_root / "retail_clv.db"
    raw_dir = project_root / "data" / "raw"
    models_dir = project_root / "models"
    
    # Find data file (prefer online_retail_II for transaction-level format)
    candidates = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls")) + list(raw_dir.glob("*.csv"))
    preferred = raw_dir / "online_retail_II.csv"
    if preferred.exists():
        candidates = [preferred] + [c for c in candidates if c != preferred]
    if not candidates:
        print("No data file in data/raw/")
        print("Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
        print("Place Excel or CSV in data/raw/")
        return
    
    data_file = candidates[0]
    print(f"Using data: {data_file}")
    
    # Initialize DB and load
    init_sqlite_db(str(db_path))
    result = clean_retail_data(input_path=data_file)
    load_to_database(str(db_path), result)
    
    # Prepare training data from DB
    print("Preparing features...")
    X, y = prepare_training_data(db_path=str(db_path))
    print(f"  Samples: {len(X)}, Features: {list(X.columns)}")
    
    if len(X) < 50:
        print("Too few samples for training. Need at least 50 customers.")
        return
    
    # Train
    print("Training models...")
    train_models(X, y, models_dir)
    print("Done. Models saved in models/")


if __name__ == "__main__":
    main()
