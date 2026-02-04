"""Show top 5 rows from each SQL table."""
import sqlite3

conn = sqlite3.connect("retail_clv.db")
cursor = conn.cursor()
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()

for (t,) in tables:
    print(f"\n=== {t} (top 5) ===")
    try:
        cursor.execute(f"SELECT * FROM [{t}] LIMIT 5")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        print("Columns:", cols)
        for r in rows:
            print(r)
    except Exception as e:
        print("Error:", e)
conn.close()
