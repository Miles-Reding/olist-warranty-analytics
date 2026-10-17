"""
Export dim_warranty_plans to data/processed/dim_warranty_plans.csv.

Usage:
    python src/export_rate_card_csv.py
"""

from pathlib import Path

import duckdb

DB_PATH = Path("data/processed/olist.duckdb")
CSV_PATH = Path("data/processed/dim_warranty_plans.csv")


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"COPY dim_warranty_plans TO '{CSV_PATH}' (HEADER, DELIMITER ',')")
    con.close()
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
