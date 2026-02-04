"""
Script: Download data (user places in data/raw) and load into database.
Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
Place downloaded Excel/CSV in data/raw/
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.init_db import init_sqlite_db
from data_cleaning.cleaner import clean_retail_data
from data_ingestion.loader import load_to_database


def main():
    db_path = project_root / "retail_clv.db"
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Find data file
    candidates = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls")) + list(raw_dir.glob("*.csv"))
    if not candidates:
        print("No data file found. Please download from:")
        print("https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
        print(f"Place the Excel or CSV file in: {raw_dir}")
        return
    
    data_file = candidates[0]
    print(f"Using data file: {data_file}")
    
    # Initialize database
    init_sqlite_db(str(db_path))
    
    # Clean data
    print("Cleaning data...")
    result = clean_retail_data(input_path=data_file)
    print(f"  Original: {result.stats['original_count']}")
    print(f"  Clean: {result.stats['clean_count']}")
    print(f"  Rejected: {result.stats['rejected_count']}")
    
    # Load to database
    print("Loading to database...")
    counts = load_to_database(str(db_path), result)
    print(f"  Loaded: {counts}")
    print(f"Database ready at {db_path}")


if __name__ == "__main__":
    main()
