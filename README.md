# Olist Warranty Pricing & Delivery Analytics

## Overview


## Data
Source: Olist Brazilian E-Commerce(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
Not included in this repo (see Setup). Currency: BRL, no FX conversion applied unless noted.

## Setup
1. Download data from Kaggle, place CSVs in `data/raw/`
2. `pip install -r requirements.txt`
3. Run `sql/build_db.py` to construct the DuckDB database
4. Notebooks run in numbered order

## Key limitations
[fan-out joins, customer_id vs customer_unique_id, currency, date range trimming]

## Findings

