"""Initialize database and create all tables."""
import sqlite3
from pathlib import Path

# SQLite-friendly schema (no REFERENCES for SQLite compatibility with CSV import)
SCHEMA_SQLITE = """
-- 1. stores (Store Information)
CREATE TABLE IF NOT EXISTS stores (
    store_id VARCHAR(10) PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    store_city VARCHAR(50),
    store_region VARCHAR(50),
    opening_date DATE
);

-- 2. products (Product Catalog)
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(20) PRIMARY KEY,
    product_name VARCHAR(100),
    product_category VARCHAR(50),
    unit_price DECIMAL(10, 2),
    current_stock_level INT DEFAULT 0
);

-- 3. customer_details (Customer and Loyalty Data)
CREATE TABLE IF NOT EXISTS customer_details (
    customer_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50),
    email VARCHAR(100),
    loyalty_status VARCHAR(20),
    total_loyalty_points INT DEFAULT 0,
    last_purchase_date DATE,
    segment_id VARCHAR(10),
    customer_phone VARCHAR(20),
    customer_since DATE
);

-- 4. promotion_details (Promotion Rules)
CREATE TABLE IF NOT EXISTS promotion_details (
    promotion_id VARCHAR(10) PRIMARY KEY,
    promotion_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    discount_percentage DECIMAL(5, 2),
    applicable_category VARCHAR(50)
);

-- 5. loyalty_rules (Point Earning Logic)
CREATE TABLE IF NOT EXISTS loyalty_rules (
    rule_id INTEGER PRIMARY KEY,
    rule_name VARCHAR(100),
    points_per_unit_spend DECIMAL(5, 2),
    min_spend_threshold DECIMAL(10, 2),
    bonus_points INT
);

-- 6. store_sales_header (Transaction Header)
CREATE TABLE IF NOT EXISTS store_sales_header (
    transaction_id VARCHAR(30) PRIMARY KEY,
    store_id VARCHAR(10),
    customer_id VARCHAR(20),
    transaction_date TEXT NOT NULL,
    total_amount DECIMAL(10, 2),
    customer_phone VARCHAR(20)
);

-- 7. store_sales_line_items (Transaction Details)
CREATE TABLE IF NOT EXISTS store_sales_line_items (
    line_item_id INTEGER,
    transaction_id VARCHAR(30),
    product_id VARCHAR(20),
    promotion_id VARCHAR(10),
    quantity INTEGER NOT NULL,
    line_item_amount DECIMAL(10, 2),
    PRIMARY KEY (line_item_id, transaction_id)
);

-- 8. rejected_data (Bad/Invalid Records)
CREATE TABLE IF NOT EXISTS rejected_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_data TEXT,
    rejection_reason VARCHAR(255),
    rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_header_customer ON store_sales_header(customer_id);
CREATE INDEX IF NOT EXISTS idx_header_date ON store_sales_header(transaction_date);
CREATE INDEX IF NOT EXISTS idx_line_items_transaction ON store_sales_line_items(transaction_id);
CREATE INDEX IF NOT EXISTS idx_line_items_product ON store_sales_line_items(product_id);
"""


def init_sqlite_db(db_path: str) -> None:
    """Create SQLite database and all tables."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQLITE)
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    from pathlib import Path
    db_path = Path(__file__).parent.parent / "retail_clv.db"
    init_sqlite_db(str(db_path))
