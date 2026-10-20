# Extended Warranty Revenue Analytics Model 

## Overview

This project is intended to illustrate some of the work I did as a revenue analyst at an extended warranty company. Working primarily in e-commerce, but increasingly in brick and mortar spaces as well, our job as revenue analysts involved evaluating the economic potential behind a partnership with a prospective merchant. In order to determine a given program's profitability, multiple factors needed to be considered: 

1. The merchant's revenue share.
2. The risk level and claim severity associated with given product categories.
3. Administrative costs associated with program management (labor, underwriting, reimbursing claims, regulatory compliance, mandatory financial reserves). 
4. Logical attach rate assumptions for a given product category and at a given product price.
5. Furthermore, assumptions for optimizing attach rate. Are customers more likely to buy product protection if the product is more expensive? At what percentage of the product purchase price do warranty sales begin to drop off? 

## Data
Source: Olist Brazilian E-Commerce(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
Not included in this repo (see Setup). Currency: BRL, no FX conversion applied unless noted.

The benefit of using the Olist dataset is that it not only includes full product data (titles, categories, descriptions, and pricing), but it also includes sales data. With the former, we can demonstrate how detailed product data allows us to build a warranty program for a prospective merchant. It also allows us to apply weighting to product categories and price bands, which is critical for reliable revenue forecasting. With the sales data, we can apply our attach rate assumptions to the merchant's overall sales performance, which further cements the reliability of the revenue forecast. 

For attach rate assumptions, I drew upon industry research: Abito, J. M., & Salant, Y. “The Effect of Product Misperception on Economic Outcomes: Evidence from the Extended Warranty Market.” and All Shield. “2026 Warranty Program ROI Benchmark Report.” https://allshield.co/2026-warranty-program-roi-benchmark-report/ (fetched October 2026). 

## Setup
1. Download data from Kaggle, place CSVs in `data/raw/`
2. `pip install -r requirements.txt`
3. Run `sql/build_db.py` to construct the DuckDB database. Once the DB is constructed, you're free to explore the data to get a feel for the structures of the tables, how the tables relate, and what kinds of products are included in this database. 
4. Run src/clean.py, which reads the raw_* tables constructed above and builds dim_products, fact_order_items and fact_orders
5. Run src/build_rate_card.py, which reads fact_order_items from step 2 and writes dim_warranty_plans. It will not work without clean.py having run, because the items table doesn’t exist until then.
6. src/sensitivity.py reads the rate-card inputs and writes the files in outputs/
7. src/export_rate_card_csv.py is an optional run. It exports dim_warranty_plans to a CSV, which is helpful if you want to dig into the warranty plans and make some modifications.

**Note on changing inputs**: Edit only warranty_params.yaml and category_mapping.yaml. Assumptions live in those files, not in the code, and the attach block cites its sources. Rerun from build_rate_card.py onward.

## Key limitations

**Fan-out joins.** order_items and order_payments both have several rows per order. Joining them directly to orders, or to each other, multiplies rows, so every revenue and item count would be inflated. clean.py avoids this by aggregating items and payments to order grain before any join. The consequence is that any figure that didn’t go through that path may be inflated. Order-level payment fields can’t be assigned to individual items.

**customer_id vs customer_unique_id.** In Olist, customer_id is a surrogate key generated per order, so it maps one-to-one to orders. The real customer is customer_unique_id. Counting customer_id makes every buyer look like a one-time customer and hides repeat purchasing. This project works at item and order level, so it doesn’t use customer behavior at all. It makes no repeat-buyer or customer-lifetime claims.

**Currency.** All amounts are in Brazilian reais (BRL), with no conversion to USD or any other currency and no inflation adjustment. Attach-rate benchmarks come from US-based sources quoted in dollars and percentages. The model only borrows their percentages and price ratios, not dollar values. The price bands (BRL 100–250, 250–500 and so on) are mapped to the paper’s TV price tiers by order, not by exchange rate. The BRL 150 offer floor is an assumption, not a derived figure.

**Date range trimming.** The analysis window is 2017-01-01 to 2018-08-31. Orders from 2016 are too sparse to trust, and October 2018 tapers off mid-month, so including either would distort volumes. The trade-off is that everything is a total over about 20 months, not an annual rate. There is no seasonality, no growth adjustment, and no per-month view, so the volume figures can’t be quoted as yearly numbers.

### Limitations and other considerations specific to the warranty model

1. **No claims data.** Every risk parameter (Weibull durability, severity, manufacturer coverage) is assumed based on my past experience. Results are only meaningful alongside the sensitivity analysis.

2. The online channel multiplier (0.5) is a big uncertainty. It moves profit from about R$9k to R$57k. 

3. Admin cost is a placeholder (15% of premium), with no source behind it.

4. **Median pricing.** Each category × band cell is priced and valued at its median item price, not at each item’s actual price.

5. **Partial coverage.** Only warranty-eligible categories and bands with a median price of at least R$150 are modeled. The totals describe that slice of Olist, not the whole marketplace.

6. **Sample data.** Olist is a sample of one marketplace’s sellers, and its mix of products and customers may not match other retailers. Ideally, as an extended warranty business grows, it would fine-tune its analytics and program models based on real data from its merchant partners. 

## Findings

