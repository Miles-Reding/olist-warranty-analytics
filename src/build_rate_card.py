"""
Generate the warranty rate card: one priced plan per
(category_l1 x price_band x term_years).

Reads:  fact_order_items in data/processed/olist.duckdb
        src/warranty_params.yaml
Writes: dim_warranty_plans

Pricing chain
-------------
    Weibull hazard
      -> P(claim) over the COVERED WINDOW (not the marketed term)
      -> expected severity (declining as a share of price)
      -> pure premium
      -> / target loss ratio
      -> modeled price
      -> commercial guardrails (floor, cap, rounding)
      -> retail price, and the loss ratio that price actually implies

The percentage-of-product-price that a merchant sees is an OUTPUT of this
chain, never an input.

Key modeling decision
---------------------
Coverage begins when the manufacturer warranty ends, not at purchase.
A 3-year plan sold against a 1-year manufacturer warranty carries two
years of claims exposure. Pricing the full term would overstate the
premium substantially, and by different amounts across categories.

Reference prices come from the actual Olist item distribution (median
price within each category x band), not band midpoints, so the rate card
is anchored to real transacted prices.

Usage:
    python src/build_rate_card.py
"""

import math
from pathlib import Path

import duckdb
import yaml

DB_PATH = Path("data/processed/olist.duckdb")
PARAMS_PATH = Path("src/warranty_params.yaml")


# ---------------------------------------------------------------------------
# Core actuarial functions
# ---------------------------------------------------------------------------

def weibull_survival(t, shape, scale):
    """S(t) = exp(-(t/eta)^beta). Share of units still working at time t."""
    if t <= 0:
        return 1.0
    return math.exp(-((t / scale) ** shape))


def claim_probability(cover_start, cover_end, shape, scale):
    """P(first failure lands inside the covered window), measured from sale.

    Unconditional at point of sale: the customer buys at t=0 and the
    program is exposed only between cover_start and cover_end. Failures
    before cover_start are the manufacturer's problem and cost the
    program nothing.
    """
    if cover_end <= cover_start:
        return 0.0
    return (weibull_survival(cover_start, shape, scale)
            - weibull_survival(cover_end, shape, scale))


def severity_ratio(price, base, decay, ref_price):
    """Expected claim cost as a share of product price.

    Falls as price rises: repair labor is largely fixed, so a cheap item
    costs proportionally more to fix and is more often written off at
    full replacement value. Capped at 1.0 — a claim never costs the
    program more than replacing the unit.
    """
    if price <= 0:
        return base
    ratio = base * ((ref_price / price) ** decay)
    return min(ratio, 1.0)


def apply_guardrails(modeled_price, product_price, g):
    """Floor, cap, and round the modeled price into a sellable one.

    Returns (retail_price, was_capped). A capped plan is one where the
    risk-based price exceeds what the market will bear — a finding in
    itself, not just a formatting step.
    """
    price = modeled_price
    was_capped = False

    ceiling = product_price * g["max_pct_of_product"]
    if price > ceiling:
        price = ceiling
        was_capped = True

    price = max(price, g["min_retail_brl"])

    # round UP to the next X4.90 / X9.90 style price point
    mult = g["round_to_multiple"]
    price = math.ceil(price / mult) * mult - g["round_offset"]

    return round(price, 2), was_capped


def attach_rate(category, band, price_pct, params, n_terms=1):
    """Expected share of buyers in a category x band who take THIS plan.

    Demand model, not an estimate: see the `attach` block in
    warranty_params.yaml for the assumptions and their sources. Returns 0
    where the category or band has no attach parameters.
    """
    a = params["attach"]
    cat = a["categories"].get(category)
    band_mult = a["band_multiplier"].get(band)
    if cat is None or band_mult is None or price_pct <= 0:
        return 0.0
    # logistic in price ratio: 1 at the norm, elasticity e there,
    # -> 2 as price -> 0, -> 0 as price rises (see yaml)
    k = 2 * abs(a["price_elasticity"])
    price_response = 2 / (1 + (price_pct / cat["ref_price_pct"]) ** k)
    attach = (cat["base"] * band_mult * a["channel_multiplier"]
              * price_response / max(n_terms, 1))
    return min(attach, a["max_attach"] / max(n_terms, 1))


# ---------------------------------------------------------------------------
# Reference prices from actual transacted data
# ---------------------------------------------------------------------------

def get_reference_prices(con):
    """Median transacted item price per category_l1 x price_band.

    Anchors the rate card to what customers actually paid rather than to
    arbitrary band midpoints, which would misprice skewed bands badly
    (the 1000+ band especially).
    """
    rows = con.execute("""
        SELECT
            category_l1,
            price_band,
            -- cast explicitly: DuckDB returns DECIMAL for MEDIAN over a
            -- DECIMAL column, which will not mix with float math below
            CAST(MEDIAN(price) AS DOUBLE) AS ref_price,
            COUNT(*)                      AS n_items,
            CAST(SUM(price) AS DOUBLE)    AS band_revenue
        FROM fact_order_items
        WHERE in_window
          AND warranty_eligible
          AND price IS NOT NULL
        GROUP BY 1, 2
    """).fetchall()

    return {
        (r[0], r[1]): {"ref_price": r[2], "n_items": r[3], "revenue": r[4]}
        for r in rows
    }


# ---------------------------------------------------------------------------
# Rate card construction
# ---------------------------------------------------------------------------

def build_rate_card(con, params):
    g = params["global"]
    cats = params["categories"]
    band_rules = params["band_rules"]

    refs = get_reference_prices(con)
    if not refs:
        raise RuntimeError(
            "No warranty-eligible items found. Did src/clean.py run, and does "
            "category_mapping.yaml mark any bucket warranty_eligible: true?"
        )

    plans = []
    skipped = []

    for (category, band), ref in sorted(refs.items()):
        if category not in cats:
            skipped.append((category, band, "no risk parameters in YAML"))
            continue

        rule = band_rules.get(band)
        if rule is None or not rule.get("offer", False):
            skipped.append((category, band, "band below offer floor"))
            continue

        ref_price = ref["ref_price"]
        if ref_price < g["min_product_price_brl"]:
            skipped.append((category, band, "median price below offer floor"))
            continue

        c = cats[category]
        shape = c["weibull_shape"]
        scale = c["weibull_scale"]
        loss_ratio = c.get("target_loss_ratio", g["default_target_loss_ratio"])

        # coverage starts at manufacturer warranty end, floored at the
        # CDC legal minimum
        cover_start = max(c["mfr_warranty_years"], g["legal_warranty_years"])

        for term in c["terms_years"]:
            if term > rule["max_term_years"]:
                continue
            if term <= cover_start:
                # plan would carry no exposure at all
                continue

            p_claim = claim_probability(cover_start, term, shape, scale)

            sev_ratio = severity_ratio(
                ref_price, c["severity_base"], c["severity_decay"],
                g["severity_reference_price_brl"],
            )
            expected_severity = sev_ratio * ref_price

            pure_premium = p_claim * expected_severity
            modeled_price = pure_premium / loss_ratio

            retail, was_capped = apply_guardrails(modeled_price, ref_price, g)

            # what loss ratio does the price we can ACTUALLY charge imply?
            # diverges from target wherever a guardrail bound.
            implied_lr = pure_premium / retail if retail > 0 else None

            plans.append({
                "plan_id": f"{category}|{band}|{term}y",
                "category_l1": category,
                "price_band": band,
                "term_years": term,
                "ref_price_brl": round(ref_price, 2),
                "n_items_in_band": ref["n_items"],
                "mfr_warranty_years": c["mfr_warranty_years"],
                "cover_start_years": cover_start,
                "cover_end_years": float(term),
                "exposure_years": round(term - cover_start, 2),
                "p_claim": round(p_claim, 5),
                "severity_ratio": round(sev_ratio, 4),
                "expected_severity_brl": round(expected_severity, 2),
                "pure_premium_brl": round(pure_premium, 2),
                "target_loss_ratio": loss_ratio,
                "modeled_price_brl": round(modeled_price, 2),
                "retail_price_brl": retail,
                "price_pct_of_product": round(retail / ref_price, 4),
                "implied_loss_ratio": round(implied_lr, 4) if implied_lr else None,
                "price_capped": was_capped,
            })

    # buyers pick one term per cell: split cell attach across offered terms
    n_terms = {}
    for p in plans:
        key = (p["category_l1"], p["price_band"])
        n_terms[key] = n_terms.get(key, 0) + 1
    for p in plans:
        n = n_terms[(p["category_l1"], p["price_band"])]
        att = attach_rate(p["category_l1"], p["price_band"],
                          p["price_pct_of_product"], params, n)
        sold = p["n_items_in_band"] * att
        p["expected_attach"] = round(att, 5)
        p["expected_plans_sold"] = round(sold, 2)
        p["expected_premium_brl"] = round(sold * p["retail_price_brl"], 2)
        p["expected_profit_brl"] = round(
            sold * (p["retail_price_brl"] * (1 - g["admin_cost_pct_of_premium"])
                    - p["pure_premium_brl"]), 2)

    return plans, skipped


def write_rate_card(con, plans):
    con.execute("DROP TABLE IF EXISTS dim_warranty_plans")
    con.execute("""
        CREATE TABLE dim_warranty_plans (
            plan_id               VARCHAR,
            category_l1           VARCHAR,
            price_band            VARCHAR,
            term_years            INTEGER,
            ref_price_brl         DOUBLE,
            n_items_in_band       BIGINT,
            mfr_warranty_years    DOUBLE,
            cover_start_years     DOUBLE,
            cover_end_years       DOUBLE,
            exposure_years        DOUBLE,
            p_claim               DOUBLE,
            severity_ratio        DOUBLE,
            expected_severity_brl DOUBLE,
            pure_premium_brl      DOUBLE,
            target_loss_ratio     DOUBLE,
            modeled_price_brl     DOUBLE,
            retail_price_brl      DOUBLE,
            price_pct_of_product  DOUBLE,
            implied_loss_ratio    DOUBLE,
            price_capped          BOOLEAN,
            expected_attach       DOUBLE,
            expected_plans_sold   DOUBLE,
            expected_premium_brl  DOUBLE,
            expected_profit_brl   DOUBLE
        )
    """)
    cols = [
        "plan_id", "category_l1", "price_band", "term_years", "ref_price_brl",
        "n_items_in_band", "mfr_warranty_years", "cover_start_years",
        "cover_end_years", "exposure_years", "p_claim", "severity_ratio",
        "expected_severity_brl", "pure_premium_brl", "target_loss_ratio",
        "modeled_price_brl", "retail_price_brl", "price_pct_of_product",
        "implied_loss_ratio", "price_capped", "expected_attach",
        "expected_plans_sold", "expected_premium_brl", "expected_profit_brl",
    ]
    con.executemany(
        f"INSERT INTO dim_warranty_plans VALUES ({','.join(['?'] * len(cols))})",
        [[p[c] for c in cols] for p in plans],
    )


def report(con, plans, skipped):
    print(f"  {len(plans)} plans generated\n")

    if skipped:
        print("  Skipped combinations:")
        for cat, band, why in skipped:
            print(f"    {cat:<20} {band:<10} {why}")
        print()

    print("  Rate card (% of product price):\n")
    df = con.execute("""
        SELECT category_l1, price_band, term_years,
               ROUND(ref_price_brl, 0)              AS ref_price,
               retail_price_brl                     AS price,
               ROUND(100 * price_pct_of_product, 1) AS pct_of_product,
               ROUND(100 * p_claim, 1)              AS pct_claim_prob,
               exposure_years,
               ROUND(100 * expected_attach, 1)      AS pct_attach,
               price_capped
        FROM dim_warranty_plans
        ORDER BY category_l1, ref_price_brl, term_years
    """).df()
    print(df.to_string(index=False))

    tot = con.execute("""
        SELECT SUM(expected_plans_sold), SUM(expected_premium_brl),
               SUM(expected_profit_brl)
        FROM dim_warranty_plans
    """).fetchone()
    print(f"\n  Scenario totals (assumed demand): {tot[0]:,.0f} plans, "
          f"R$ {tot[1]:,.0f} premium, R$ {tot[2]:,.0f} profit "
          f"(premium - expected claims - admin cost)")

    capped = con.execute(
        "SELECT COUNT(*) FROM dim_warranty_plans WHERE price_capped"
    ).fetchone()[0]
    if capped:
        print(f"\n  !! {capped} plans hit the {'price ceiling'}. Their implied loss")
        print("     ratio exceeds target — the risk-based price is above what")
        print("     the market bears. These are candidates to drop, shorten,")
        print("     or restrict to higher price bands.")
        rows = con.execute("""
            SELECT plan_id, ROUND(modeled_price_brl, 2), retail_price_brl,
                   ROUND(100 * implied_loss_ratio, 1),
                   ROUND(100 * target_loss_ratio, 1)
            FROM dim_warranty_plans WHERE price_capped
            ORDER BY implied_loss_ratio DESC
        """).fetchall()
        print(f"\n     {'plan':<30} {'modeled':>9} {'retail':>9} "
              f"{'implied LR':>11} {'target LR':>10}")
        for pid, modeled, retail, ilr, tlr in rows:
            print(f"     {pid:<30} {modeled:>9.2f} {retail:>9.2f} "
                  f"{ilr:>10.1f}% {tlr:>9.1f}%")


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run sql/build_db.py then src/clean.py first."
        )

    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    con = duckdb.connect(str(DB_PATH))
    print(f"Building rate card from {PARAMS_PATH}\n")

    plans, skipped = build_rate_card(con, params)
    write_rate_card(con, plans)
    report(con, plans, skipped)

    con.close()
    print("\nDone. Table: dim_warranty_plans")


if __name__ == "__main__":
    main()
