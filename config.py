"""Configuration for 30-day CLV Retail Project."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Database
# Use SQLite by default; set DATABASE_URL for PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT / 'retail_clv.db'}"
)

# Kaggle dataset paths (user downloads and places here)
RAW_DATA_PATH = DATA_DIR / "raw"
CLEAN_DATA_PATH = DATA_DIR / "clean"

# ML settings
RANDOM_STATE = 42
TARGET_COLUMN = "future_spend_30d"
CUTOFF_DAYS = 30

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
