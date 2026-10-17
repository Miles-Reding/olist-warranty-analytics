"""
Build analysis-ready tables from the raw_* tables loaded by sql/build_db.py.

Reads:  raw_* tables in data/processed/olist.duckdb
Writes: dim_products, fact_order_items, fact_orders

Design notes
------------
1. order_items and order_payments are aggregated to ORDER GRAIN *before*
   any join to orders. Joining them directly fans out rows and inflates
   every revenue figure. This is the single most common error in public
   analyses of this dataset.
2. geolocation is collapsed to one centroid per zip prefix before joining.
   Raw it is ~1M rows with many points per prefix.
3. customer_unique_id is the real customer. customer_id is a per-order
   surrogate key, 1:1 with orders.
4. Source column misspellings (product_name_lenght) are corrected here,
   not in the raw layer.
5. All money is BRL. No FX conversion applied.

Usage:
    python src/clean.py
"""

import duckdb
import yaml
from pathlib import Path

DB_PATH = Path("data/processed/olist.duckdb")
MAPPING_PATH = Path("src/category_mapping.yaml")

# analysis window — 2016 is sparse, Oct 2018 tapers off mid-month
WINDOW_START = "2017-01-01"
WINDOW_END = "2018-08-31"


def load_mapping(con):
    """Load the category YAML into a DuckDB lookup table."""
    with open(MAPPING_PATH) as f:
        cfg = yaml.safe_load(f)

    rows = []
    for bucket, spec in cfg["buckets"].items():
        eligible = spec.get("warranty_eligible", False)
        for cat in spec["categories"]:
            rows.append((cat, bucket, eligible))

    con.execute("DROP TABLE IF EXISTS map_category")
    con.execute("""
        CREATE TABLE map_category (
            category_english VARCHAR,
            category_l1      VARCHAR,
            warranty_eligible BOOLEAN
        )
    """)
    con.executemany("INSERT INTO map_category VALUES (?, ?, ?)", rows)

    print(f"  loaded {len(rows)} category mappings into "
          f"{len(cfg['buckets'])} L1 buckets")
    return cfg


def check_mapping_coverage(con):
    """Report any category present in the data but missing from the YAML.

    This matters: the mapping was written by hand, and the source
    translation file has misspellings. Unmapped categories fall to
    'unmapped' rather than being silently dropped.
    """
    missing = con.execute("""
        SELECT t.product_category_name_english AS category,
               COUNT(*) AS n_products
        FROM raw_products p
        JOIN raw_category_translation t
          ON p.product_category_name = t.product_category_name
        LEFT JOIN map_category m
          ON t.product_category_name_english = m.category_english
        WHERE m.category_english IS NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """).fetchall()

    if missing:
        print("\n  !! UNMAPPED CATEGORIES — add these to category_mapping.yaml:")
        for cat, n in missing:
            print(f"       {cat:<50} {n:>6,} products")
    else:
        print("  all categories mapped")

    # products whose Portuguese name has no translation row at all
    untranslated = con.execute("""
        SELECT COUNT(*) FROM raw_products p
        LEFT JOIN raw_category_translation t
          ON p.product_category_name = t.product_category_name
        WHERE p.product_category_name IS NOT NULL
          AND t.product_category_name_english IS NULL
    """).fetchone()[0]
    if untranslated:
        print(f"  note: {untranslated:,} products have a category with no "
              f"translation row (known gap in the source file)")


def build_dim_products(con):
    """Product dimension: translated category, L1 bucket, corrected columns."""
    con.execute("DROP TABLE IF EXISTS dim_products")
    con.execute("""
        CREATE TABLE dim_products AS
        SELECT
            p.product_id,
            p.product_category_name                      AS category_pt,
            t.product_category_name_english              AS category_english,
            COALESCE(m.category_l1, 'unmapped')          AS category_l1,
            COALESCE(m.warranty_eligible, FALSE)         AS warranty_eligible,
            -- source misspellings corrected here
            p.product_name_lenght                        AS product_name_length,
            p.product_description_lenght                 AS product_description_length,
            p.product_photos_qty,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            -- volume proxy for shipping/damage exposure
            (p.product_length_cm * p.product_height_cm * p.product_width_cm)
                                                         AS product_volume_cm3
        FROM raw_products p
        LEFT JOIN raw_category_translation t
               ON p.product_category_name = t.product_category_name
        LEFT JOIN map_category m
               ON t.product_category_name_english = m.category_english
    """)
    n = con.execute("SELECT COUNT(*) FROM dim_products").fetchone()[0]
    print(f"  dim_products: {n:,} rows")


def build_geo(con):
    """Collapse ~1M geolocation rows to one centroid per zip prefix."""
    con.execute("DROP TABLE IF EXISTS dim_geo")
    con.execute("""
        CREATE TABLE dim_geo AS
        SELECT
            geolocation_zip_code_prefix AS zip_prefix,
            AVG(geolocation_lat)        AS lat,
            AVG(geolocation_lng)        AS lng
        FROM raw_geolocation
        GROUP BY 1
    """)
    n = con.execute("SELECT COUNT(*) FROM dim_geo").fetchone()[0]
    print(f"  dim_geo: {n:,} zip prefixes (collapsed from ~1M raw rows)")


def build_order_aggregates(con):
    """Aggregate items and payments to ORDER GRAIN before any join."""

    con.execute("DROP TABLE IF EXISTS agg_items")
    con.execute("""
        CREATE TABLE agg_items AS
        SELECT
            order_id,
            COUNT(*)                        AS n_items,
            COUNT(DISTINCT product_id)      AS n_distinct_products,
            COUNT(DISTINCT seller_id)       AS n_sellers,
            SUM(price)                      AS items_total,
            SUM(freight_value)              AS freight_total,
            SUM(price) + SUM(freight_value) AS order_value,
            MAX(price)                      AS max_item_price,
            -- seller of the highest-priced item, used as the order's
            -- primary seller for geography and performance cuts
            ARG_MAX(seller_id, price)       AS primary_seller_id,
            ARG_MAX(product_id, price)      AS primary_product_id
        FROM raw_order_items
        GROUP BY 1
    """)

    con.execute("DROP TABLE IF EXISTS agg_payments")
    con.execute("""
        CREATE TABLE agg_payments AS
        SELECT
            order_id,
            SUM(payment_value)                  AS payment_total,
            MAX(payment_installments)           AS max_installments,
            COUNT(*)                            AS n_payment_records,
            -- payment type carrying the largest value
            ARG_MAX(payment_type, payment_value) AS primary_payment_type
        FROM raw_order_payments
        GROUP BY 1
    """)

    # reviews: a small number of orders carry more than one review.
    # keep the earliest per order so the grain is clean.
    con.execute("DROP TABLE IF EXISTS agg_reviews")
    con.execute("""
        CREATE TABLE agg_reviews AS
        SELECT
            order_id,
            ARG_MIN(review_score, review_creation_date)   AS review_score,
            MIN(review_creation_date)                     AS review_creation_date,
            COUNT(*)                                      AS n_reviews,
            ARG_MIN(review_comment_message, review_creation_date)
                                                          AS review_comment
        FROM raw_order_reviews
        GROUP BY 1
    """)

    dupes = con.execute(
        "SELECT COUNT(*) FROM agg_reviews WHERE n_reviews > 1"
    ).fetchone()[0]
    print(f"  agg_items / agg_payments / agg_reviews built "
          f"({dupes:,} orders had multiple reviews, kept earliest)")


def build_fact_orders(con):
    """Order-grain fact table with delivery metrics and geography."""
    con.execute("DROP TABLE IF EXISTS fact_orders")
    con.execute(f"""
        CREATE TABLE fact_orders AS
        WITH base AS (
            SELECT
                o.order_id,
                o.order_status,
                c.customer_unique_id,
                c.customer_state,
                c.customer_zip_code_prefix,
                s.seller_state,
                s.seller_zip_code_prefix,

                o.order_purchase_timestamp,
                o.order_approved_at,
                o.order_delivered_carrier_date,
                o.order_delivered_customer_date,
                o.order_estimated_delivery_date,

                i.n_items, i.n_sellers, i.items_total, i.freight_total,
                i.order_value, i.max_item_price,
                i.primary_seller_id, i.primary_product_id,

                p.payment_total, p.max_installments, p.primary_payment_type,

                r.review_score, r.review_creation_date, r.review_comment,

                gc.lat AS cust_lat, gc.lng AS cust_lng,
                gs.lat AS sell_lat, gs.lng AS sell_lng

            FROM raw_orders o
            LEFT JOIN raw_customers c ON o.customer_id = c.customer_id
            LEFT JOIN agg_items     i ON o.order_id = i.order_id
            LEFT JOIN agg_payments  p ON o.order_id = p.order_id
            LEFT JOIN agg_reviews   r ON o.order_id = r.order_id
            LEFT JOIN raw_sellers   s ON i.primary_seller_id = s.seller_id
            LEFT JOIN dim_geo      gc ON c.customer_zip_code_prefix = gc.zip_prefix
            LEFT JOIN dim_geo      gs ON s.seller_zip_code_prefix = gs.zip_prefix
        )
        SELECT
            *,

            -- delivery timing
            date_diff('day', order_purchase_timestamp,
                             order_delivered_customer_date) AS transit_days,
            date_diff('hour', order_purchase_timestamp,
                              order_approved_at) / 24.0     AS approval_lag_days,
            date_diff('day', order_approved_at,
                             order_delivered_carrier_date)  AS carrier_lag_days,

            -- promise vs actual. positive = LATE.
            -- estimate is a date, delivery is a timestamp; cast both to date
            -- so an 11pm delivery on the promised day is not counted late.
            date_diff('day', CAST(order_estimated_delivery_date AS DATE),
                             CAST(order_delivered_customer_date AS DATE))
                                                            AS days_late,

            -- how much padding Olist built into the promise
            date_diff('day', order_purchase_timestamp,
                             order_estimated_delivery_date) AS promise_window_days,

            CASE WHEN order_delivered_customer_date IS NULL THEN NULL
                 WHEN CAST(order_delivered_customer_date AS DATE)
                      > CAST(order_estimated_delivery_date AS DATE)
                 THEN TRUE ELSE FALSE END                   AS is_late,

            -- geography
            (customer_state = seller_state)                 AS same_state,
            CASE WHEN cust_lat IS NULL OR sell_lat IS NULL THEN NULL ELSE
                6371 * 2 * ASIN(SQRT(
                    POWER(SIN(RADIANS(sell_lat - cust_lat) / 2), 2) +
                    COS(RADIANS(cust_lat)) * COS(RADIANS(sell_lat)) *
                    POWER(SIN(RADIANS(sell_lng - cust_lng) / 2), 2)
                )) END                                      AS ship_distance_km,

            -- analysis window flag (do not drop rows here — flag them,
            -- so the exclusion is visible and reversible downstream)
            (order_purchase_timestamp >= '{WINDOW_START}'
             AND order_purchase_timestamp < '{WINDOW_END}')  AS in_window

        FROM base
    """)

    n = con.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0]
    print(f"  fact_orders: {n:,} rows (order grain)")


def build_fact_order_items(con):
    """Item-grain fact table for pricing and category work."""
    con.execute("DROP TABLE IF EXISTS fact_order_items")
    con.execute("""
        CREATE TABLE fact_order_items AS
        SELECT
            i.order_id,
            i.order_item_id,
            i.product_id,
            i.seller_id,
            i.price,
            i.freight_value,
            i.price + i.freight_value            AS landed_price,
            i.freight_value / NULLIF(i.price, 0) AS freight_ratio,

            d.category_english,
            d.category_l1,
            d.warranty_eligible,
            d.product_weight_g,
            d.product_volume_cm3,
            d.product_photos_qty,

            o.order_purchase_timestamp,
            date_trunc('month', o.order_purchase_timestamp) AS purchase_month,
            o.order_status,
            o.review_score,
            o.days_late,
            o.is_late,
            o.customer_state,
            o.in_window,

            CASE
                WHEN i.price <   50 THEN '0-50'
                WHEN i.price <  100 THEN '50-100'
                WHEN i.price <  250 THEN '100-250'
                WHEN i.price <  500 THEN '250-500'
                WHEN i.price < 1000 THEN '500-1000'
                ELSE '1000+'
            END                                  AS price_band

        FROM raw_order_items i
        LEFT JOIN dim_products d ON i.product_id = d.product_id
        LEFT JOIN fact_orders  o ON i.order_id = o.order_id
    """)
    n = con.execute("SELECT COUNT(*) FROM fact_order_items").fetchone()[0]
    print(f"  fact_order_items: {n:,} rows (item grain)")


def validate(con):
    """Checks that catch the failure modes this script exists to prevent."""
    print("\nValidation:")

    # 1. grain integrity — fact_orders must be 1 row per order
    total, distinct = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT order_id) FROM fact_orders
    """).fetchone()
    flag = "OK" if total == distinct else "FAN-OUT BUG"
    print(f"  [{flag:>12}] fact_orders grain: {total:,} rows / "
          f"{distinct:,} distinct order_ids")

    # 2. revenue sanity — items_total must not have been multiplied
    rev = con.execute("""
        SELECT SUM(items_total) FROM fact_orders WHERE in_window
    """).fetchone()[0]
    raw_rev = con.execute("""
        SELECT SUM(price) FROM raw_order_items
    """).fetchone()[0]
    print(f"  [{'INFO':>12}] in-window item revenue: R$ {rev:,.0f} "
          f"(all-time raw: R$ {raw_rev:,.0f})")

    # 3. the headline relationship should already be visible
    print(f"\n  Review score by delivery outcome (in-window, delivered):")
    rows = con.execute("""
        SELECT
            CASE WHEN is_late THEN 'late' ELSE 'on time / early' END AS outcome,
            COUNT(*)                AS n_orders,
            ROUND(AVG(review_score), 2) AS avg_review,
            ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END), 1)
                                    AS pct_1_2_star
        FROM fact_orders
        WHERE in_window
          AND order_status = 'delivered'
          AND review_score IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    for outcome, n, avg, pct in rows:
        print(f"       {outcome:<18} {n:>7,} orders   "
              f"avg {avg}   {pct}% rated 1-2 stars")

    # 4. warranty-eligible share, for Phase 3 scoping
    elig = con.execute("""
        SELECT
            ROUND(100.0 * SUM(CASE WHEN warranty_eligible THEN 1 ELSE 0 END)
                  / COUNT(*), 1),
            ROUND(100.0 * SUM(CASE WHEN warranty_eligible THEN price ELSE 0 END)
                  / SUM(price), 1)
        FROM fact_order_items WHERE in_window
    """).fetchone()
    print(f"\n  [{'INFO':>12}] warranty-eligible: {elig[0]}% of items, "
          f"{elig[1]}% of item revenue")

    # 5. unmapped leakage
    unmapped = con.execute("""
        SELECT COUNT(*) FROM fact_order_items WHERE category_l1 = 'unmapped'
    """).fetchone()[0]
    pct = con.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN category_l1 = 'unmapped' THEN 1 ELSE 0 END)
               / COUNT(*), 2) FROM fact_order_items
    """).fetchone()[0]
    flag = "OK" if pct < 2 else "REVIEW"
    print(f"  [{flag:>12}] unmapped items: {unmapped:,} ({pct}%)")


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run `python sql/build_db.py` first."
        )

    con = duckdb.connect(str(DB_PATH))
    print(f"Cleaning {DB_PATH}\n")

    load_mapping(con)
    check_mapping_coverage(con)
    print()
    build_dim_products(con)
    build_geo(con)
    build_order_aggregates(con)
    build_fact_orders(con)
    build_fact_order_items(con)
    validate(con)

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
