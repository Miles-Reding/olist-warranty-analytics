"""
Build the raw DuckDB database from Olist CSVs.

Loads each source CSV verbatim into a `raw` schema table — no cleaning,
no renaming, no type coercion. This keeps an honest mirror of the source
data. All cleaning/reshaping happens downstream in separate scripts that
read from `raw` and write to `main` (or a `clean` schema).

Usage:
    python sql/build_db.py
"""

import duckdb
from pathlib import Path

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/olist.duckdb")

# maps table name -> source filename
TABLES = {
    "raw_customers": "olist_customers_dataset.csv",
    "raw_geolocation": "olist_geolocation_dataset.csv",
    "raw_order_items": "olist_order_items_dataset.csv",
    "raw_order_payments": "olist_order_payments_dataset.csv",
    "raw_order_reviews": "olist_order_reviews_dataset.csv",
    "raw_orders": "olist_orders_dataset.csv",
    "raw_products": "olist_products_dataset.csv",
    "raw_sellers": "olist_sellers_dataset.csv",
    "raw_category_translation": "product_category_name_translation.csv",
}


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # fresh build each time — delete existing db so this script is idempotent
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))

    print(f"Building {DB_PATH} from {RAW_DIR}/\n")

    for table_name, filename in TABLES.items():
        csv_path = RAW_DIR / filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected {csv_path} — did you download the data? "
                f"See README Setup section."
            )

        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv('{csv_path}', header=True, sample_size=-1)
        """)

        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        col_count = len(con.execute(f"DESCRIBE {table_name}").fetchall())
        print(f"  {table_name:<28} {row_count:>8,} rows   {col_count:>3} cols")

    print("\nDone. Running basic sanity checks...\n")
    sanity_checks(con)

    con.close()


def sanity_checks(con):
    """Quick checks against known facts about this dataset, so a broken
    download or bad parse fails loudly here instead of silently downstream."""

    checks = [
        ("raw_orders", 99441, "orders"),
        ("raw_order_items", 112650, "order items"),
        ("raw_customers", 99441, "customers"),
        ("raw_order_reviews", 99224, "reviews"),
        ("raw_products", 32951, "products"),
        ("raw_sellers", 3095, "sellers"),
    ]

    for table, expected, label in checks:
        actual = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        flag = "OK" if actual == expected else "CHECK THIS"
        print(f"  [{flag:>10}] {label}: expected ~{expected:,}, got {actual:,}")

    # customer_id vs customer_unique_id — flagged earlier as a common trap
    n_customer_id = con.execute(
        "SELECT COUNT(DISTINCT customer_id) FROM raw_customers"
    ).fetchone()[0]
    n_unique_customer = con.execute(
        "SELECT COUNT(DISTINCT customer_unique_id) FROM raw_customers"
    ).fetchone()[0]
    print(f"\n  distinct customer_id:        {n_customer_id:,}")
    print(f"  distinct customer_unique_id: {n_unique_customer:,}")
    print(f"  (unique_id should be lower — that's the real repeat-customer signal)")

    # null category check
    n_null_cat = con.execute(
        "SELECT COUNT(*) FROM raw_products WHERE product_category_name IS NULL"
    ).fetchone()[0]
    print(f"\n  products with null category: {n_null_cat:,} (expect several hundred)")

    # date range
    date_range = con.execute("""
        SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp)
        FROM raw_orders
    """).fetchone()
    print(f"\n  order date range: {date_range[0]} to {date_range[1]}")
    print(f"  (plan to trim to ~Jan 2017 – Aug 2018 for time-based analysis)")


if __name__ == "__main__":
    main()

