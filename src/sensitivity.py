"""
Sensitivity analysis for the warranty rate card.

Every risk parameter in src/warranty_params.yaml is ASSUMED — Olist has no
claims data — so no rate-card result is reportable until it has been swept.
This script answers two different questions:

  1. REPRICE:  if an assumption were different and we re-priced from
     scratch, how far would the card move?  (price % of product, how many
     plans hit the price ceiling)

  2. HELD CARD:  if we ship the BASE card and an assumption turns out to be
     wrong, what loss ratio do we actually realize?  This is the one that
     measures downside risk. It holds base retail prices fixed and swaps in
     the perturbed expected claim cost.

Pricing levers (target loss ratio, price ceiling) change question 1 but not
question 2: they are decisions, not facts about the world, so the held-card
loss ratio is unchanged by construction.

Portfolio weighting: each plan is weighted by expected plans sold =
items in its category x band x modeled attach rate (see the `attach` block
in warranty_params.yaml). Attach is an ASSUMED demand model, so volumes,
premium and profit are scenario figures conditional on those assumptions,
not forecasts. In the HELD CARD view, prices are fixed at the base card, so
attach moves only when an attach assumption changes; in the REPRICE view,
attach also responds to the new price (price elasticity).

Profit = premium - expected claims - admin/commission cost (a flat share of
premium, global.admin_cost_pct_of_premium in the yaml).

Reads:  dim_* via build_rate_card.build_rate_card (read-only DB access)
Writes: outputs/sensitivity_oat.csv, outputs/sensitivity_grid.csv,
        outputs/sensitivity_profit_grid.csv,
        outputs/sensitivity_tornado.png, outputs/sensitivity_profit_tornado.png,
        outputs/sensitivity_grid.png, outputs/sensitivity_profit_grid.png

Usage:
    python src/sensitivity.py
"""

import copy
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from build_rate_card import DB_PATH, PARAMS_PATH, attach_rate, build_rate_card

OUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Perturbations — each takes a params dict and a magnitude, returns a copy
# ---------------------------------------------------------------------------

def _scale_cat(params, key, factor):
    p = copy.deepcopy(params)
    for c in p["categories"].values():
        c[key] *= factor
    return p


def mult_weibull_scale(params, f):
    return _scale_cat(params, "weibull_scale", f)


def mult_weibull_shape(params, f):
    return _scale_cat(params, "weibull_shape", f)


def mult_severity_base(params, f):
    return _scale_cat(params, "severity_base", f)


def shift_mfr_warranty(params, d):
    p = copy.deepcopy(params)
    for c in p["categories"].values():
        c["mfr_warranty_years"] = max(0.0, c["mfr_warranty_years"] + d)
    return p


def shift_target_lr(params, d):
    p = copy.deepcopy(params)
    p["global"]["default_target_loss_ratio"] += d
    for c in p["categories"].values():
        if "target_loss_ratio" in c:
            c["target_loss_ratio"] += d
    return p


def set_price_ceiling(params, v):
    p = copy.deepcopy(params)
    p["global"]["max_pct_of_product"] = v
    return p


def mult_base_attach(params, f):
    p = copy.deepcopy(params)
    for c in p["attach"]["categories"].values():
        c["base"] *= f
    return p


def set_elasticity(params, v):
    p = copy.deepcopy(params)
    p["attach"]["price_elasticity"] = v
    return p


def set_channel_multiplier(params, v):
    p = copy.deepcopy(params)
    p["attach"]["channel_multiplier"] = v
    return p


def set_admin_cost(params, v):
    p = copy.deepcopy(params)
    p["global"]["admin_cost_pct_of_premium"] = v
    return p


def set_all_target_lr(params, v):
    """Absolute override: one target loss ratio for every category."""
    p = copy.deepcopy(params)
    p["global"]["default_target_loss_ratio"] = v
    for c in p["categories"].values():
        c["target_loss_ratio"] = v
    return p


# name, function, (low, high), label for low/high, is_pricing_lever
SCENARIOS = [
    ("Weibull scale (durability)", mult_weibull_scale, (0.75, 1.25), "x0.75 / x1.25", False),
    ("Weibull shape (wear-out)",   mult_weibull_shape, (0.80, 1.20), "x0.80 / x1.20", False),
    ("Severity base",              mult_severity_base, (0.75, 1.25), "x0.75 / x1.25", False),
    ("Mfr warranty length",        shift_mfr_warranty, (-0.5, 0.5),  "-0.5y / +0.5y", False),
    ("Base attach rate",           mult_base_attach,   (0.70, 1.30), "x0.70 / x1.30", False),
    ("Price elasticity",           set_elasticity,     (-4.0, -1.0), "-4.0 / -1.0",   False),
    ("Online channel multiplier",  set_channel_multiplier, (0.15, 1.0), "0.15 / 1.0", False),
    ("Admin cost (% of premium)",  set_admin_cost,     (0.05, 0.30), "5% / 30%",      False),
    ("Target loss ratio",          shift_target_lr,    (-0.10, 0.10), "-10pt / +10pt", True),
    ("Price ceiling (% of product)", set_price_ceiling, (0.20, 0.40), "20% / 40%",    True),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def to_frame(plans):
    cols = ["plan_id", "category_l1", "price_band", "n_items_in_band",
            "ref_price_brl", "pure_premium_brl", "retail_price_brl",
            "price_pct_of_product", "price_capped", "expected_plans_sold",
            "expected_premium_brl", "expected_profit_brl"]
    return pd.DataFrame(plans, columns=cols) if plans else pd.DataFrame(columns=cols)


def held_volumes(base, params):
    """Plans sold if the BASE card (prices fixed) met the world described by
    `params`. Only attach assumptions change this; risk assumptions don't."""
    cell = base.groupby(["category_l1", "price_band"])["plan_id"].transform("count")
    att = [attach_rate(r.category_l1, r.price_band, r.price_pct_of_product,
                       params, n)
           for r, n in zip(base.itertuples(), cell)]
    return base["n_items_in_band"] * pd.Series(att, index=base.index)


def summarize(base, scen, scen_params):
    """Compare a scenario card against the base card."""
    # REPRICE: new prices, attach responds to them
    sold = scen["expected_plans_sold"]
    avg_pct = ((scen["retail_price_brl"] * sold).sum()
               / (scen["ref_price_brl"] * sold).sum()) if sold.sum() else float("nan")

    # HELD CARD: base retail price and volume-at-base-price under the
    # scenario's world. A base plan with no scenario counterpart carries no
    # exposure there, so its claims are 0.
    w = held_volumes(base, scen_params)
    claims = base[["plan_id"]].merge(
        scen[["plan_id", "pure_premium_brl"]], on="plan_id", how="left"
    )["pure_premium_brl"].fillna(0.0)
    premium = (w * base["retail_price_brl"]).sum()
    held_lr = (w * claims).sum() / premium

    return {
        "n_plans": len(scen),
        "n_capped": int(scen["price_capped"].sum()),
        "avg_price_pct_of_product": avg_pct,
        "held_card_loss_ratio": held_lr,
        "held_card_plans_sold": w.sum(),
        "held_card_profit_brl": (premium
                                 * (1 - scen_params["global"]["admin_cost_pct_of_premium"])
                                 - (w * claims).sum()),
        "reprice_plans_sold": sold.sum(),
        "reprice_premium_brl": scen["expected_premium_brl"].sum(),
        "reprice_profit_brl": scen["expected_profit_brl"].sum(),
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_oat(con, params, base):
    rows = [{"scenario": "BASE", "setting": "-", "pricing_lever": False,
             **summarize(base, base, params)}]
    for name, fn, (lo, hi), label, is_lever in SCENARIOS:
        for side, v in (("low", lo), ("high", hi)):
            p = fn(params, v)
            plans, _ = build_rate_card(con, p)
            rows.append({
                "scenario": name,
                "setting": f"{side} ({v:+g})" if fn is shift_mfr_warranty
                           or fn is shift_target_lr else f"{side} ({v:g})",
                "pricing_lever": is_lever,
                **summarize(base, to_frame(plans), p),
            })
    return pd.DataFrame(rows)


def run_grid(con, params, base):
    """Joint stress of the two assumptions that matter most for claim cost:
    characteristic life and severity. Shows whether they compound."""
    scale_f = [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50]
    sev_f = [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50]
    rows = []
    for sf in scale_f:
        for vf in sev_f:
            p = mult_severity_base(mult_weibull_scale(params, sf), vf)
            plans, _ = build_rate_card(con, p)
            rows.append({"weibull_scale_x": sf, "severity_base_x": vf,
                         **summarize(base, to_frame(plans), p)})
    return pd.DataFrame(rows)


def run_profit_grid(con, params, base):
    """Target loss ratio x price elasticity x admin cost -> repriced profit.

    Pricing for a lower loss ratio raises price and cuts volume. How far to
    push depends on how price-sensitive buyers are and what each plan costs
    to administer, both uncertain, so the profit-maximizing target moves.
    """
    lrs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    elast = [-1.0, -1.5, -2.2, -3.0, -4.0]
    admins = [0.05, params["global"]["admin_cost_pct_of_premium"], 0.30]
    admins = sorted(set(admins))
    rows = []
    for a in admins:
        for e in elast:
            for lr in lrs:
                p = set_all_target_lr(set_elasticity(set_admin_cost(params, a), e), lr)
                plans, _ = build_rate_card(con, p)
                rows.append({"admin_cost": a, "elasticity": e,
                             "target_loss_ratio": lr,
                             **summarize(base, to_frame(plans), p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_tornado(oat, metric, base_val, ref_val, ref_label, xlabel, title,
                 fmt, path):
    risk = oat[(oat["scenario"] != "BASE") & (~oat["pricing_lever"])]
    names = list(dict.fromkeys(risk["scenario"]))
    # drop scenarios that don't move this metric (e.g. admin cost vs loss ratio)
    names = [n for n in names
             if risk[risk.scenario == n][metric].sub(base_val).abs().max() > 1e-9]
    lows = [risk[(risk.scenario == n) & risk.setting.str.startswith("low")]
            [metric].iloc[0] for n in names]
    highs = [risk[(risk.scenario == n) & risk.setting.str.startswith("high")]
             [metric].iloc[0] for n in names]
    order = sorted(range(len(names)),
                   key=lambda i: abs(highs[i] - lows[i]))
    both = lows + highs
    pad = 0.06 * (max(both + [base_val]) - min(both + [base_val]))

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.6))
    for y, i in enumerate(order):
        for val in (lows[i], highs[i]):
            ax.barh(y, val - base_val, left=base_val, height=0.55,
                    color="#c0504d" if val < base_val else "#4f81bd")
            ax.text(val + (pad / 4 if val > base_val else -pad / 4), y,
                    fmt(val), va="center",
                    ha="left" if val > base_val else "right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.axvline(base_val, color="black", lw=1)
    if ref_val is not None:
        ax.axvline(ref_val, color="gray", ls="--", lw=1)
        ax.text(ref_val, len(order) - 0.45, f" {ref_label}", fontsize=8,
                color="gray", va="bottom")
    ax.set_xlim(min(both + [base_val]) - 2 * pad, max(both + [base_val]) + 2 * pad)
    ax.set_ylim(-0.5, len(order) - 0.1)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
    ax.set_title(title, fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profit_grid(pg, base_admin, path):
    pg = pg[pg["admin_cost"] == base_admin]
    pv = (pg.pivot(index="elasticity", columns="target_loss_ratio",
                   values="reprice_profit_brl") / 1000).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in pv.columns])
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([f"{i:g}" for i in pv.index])
    vmax, vmin = pv.values.max(), pv.values.min()
    for r in range(pv.shape[0]):
        best = pv.values[r].argmax()
        for c in range(pv.shape[1]):
            v = pv.values[r, c]
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if c == best else "normal",
                    color="white" if v < (vmin + vmax) / 2 else "black")
    ax.set_xlabel("Target loss ratio (all categories)")
    ax.set_ylabel("Price elasticity of attach")
    ax.set_title(f"Repriced profit, R$ thousand, admin {base_admin:.0%} of premium "
                 "(bold = best per row)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(grid, path):
    pv = grid.pivot(index="severity_base_x", columns="weibull_scale_x",
                    values="held_card_loss_ratio").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pv.values, cmap="RdBu_r", aspect="auto",
                   vmin=0.0, vmax=max(pv.values.max(), 0.8))
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([f"x{c:g}" for c in pv.columns])
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([f"x{i:g}" for i in pv.index])
    for r in range(pv.shape[0]):
        for c in range(pv.shape[1]):
            ax.text(c, r, f"{pv.values[r, c]:.0%}", ha="center", va="center",
                    fontsize=8)
    ax.set_xlabel("Weibull scale multiplier (lower = less durable)")
    ax.set_ylabel("Severity multiplier")
    ax.set_title("Held-card loss ratio: durability x severity", fontsize=10)
    fig.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run sql/build_db.py, src/clean.py and "
            f"src/build_rate_card.py first."
        )
    OUT_DIR.mkdir(exist_ok=True)

    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    con = duckdb.connect(str(DB_PATH), read_only=True)

    base_plans, _ = build_rate_card(con, params)
    base = to_frame(base_plans)
    target_lr = params["global"]["default_target_loss_ratio"]
    print(f"Base card: {len(base)} plans\n")

    oat = run_oat(con, params, base)
    grid = run_grid(con, params, base)
    profit_grid = run_profit_grid(con, params, base)
    con.close()

    oat.to_csv(OUT_DIR / "sensitivity_oat.csv", index=False)
    grid.to_csv(OUT_DIR / "sensitivity_grid.csv", index=False)
    profit_grid.to_csv(OUT_DIR / "sensitivity_profit_grid.csv", index=False)

    base_row = oat.loc[oat.scenario == "BASE"].iloc[0]
    plot_tornado(oat, "held_card_loss_ratio", base_row["held_card_loss_ratio"],
                 target_lr, "target", "Realized loss ratio if base card is shipped",
                 "Held-card loss ratio under one-at-a-time assumption changes",
                 lambda v: f"{v:.0%}", OUT_DIR / "sensitivity_tornado.png")
    plot_tornado(oat, "held_card_profit_brl", base_row["held_card_profit_brl"],
                 None, "", "Profit if base card is shipped, R$ thousand",
                 "Held-card profit under one-at-a-time assumption changes\n"
                 "(assumed demand and admin cost)",
                 lambda v: f"{v / 1000:.0f}", OUT_DIR / "sensitivity_profit_tornado.png")
    plot_grid(grid, OUT_DIR / "sensitivity_grid.png")
    plot_profit_grid(profit_grid, params["global"]["admin_cost_pct_of_premium"], OUT_DIR / "sensitivity_profit_grid.png")

    show = oat.copy()
    show["avg_price_pct_of_product"] = (100 * show["avg_price_pct_of_product"]).round(1)
    show["held_card_loss_ratio"] = (100 * show["held_card_loss_ratio"]).round(1)
    for c in ("held_card_plans_sold", "held_card_profit_brl",
              "reprice_plans_sold", "reprice_premium_brl", "reprice_profit_brl"):
        show[c] = show[c].round(0)
    print(show.drop(columns="pricing_lever").to_string(index=False))
    best = profit_grid.loc[profit_grid.groupby(["admin_cost", "elasticity"])
                           ["reprice_profit_brl"].idxmax()]
    print("\nProfit-maximizing target loss ratio (admin cost x elasticity):")
    print(best[["admin_cost", "elasticity", "target_loss_ratio",
                "reprice_profit_brl"]]
          .round({"reprice_profit_brl": 0}).to_string(index=False))
    print(f"\nWrote {OUT_DIR}/sensitivity_*.csv and .png")


if __name__ == "__main__":
    main()
