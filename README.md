# 💸 Micro Leakage Detection — Personal Finance Analysis

An end-to-end data science project detecting micro financial leakages from real anonymized PNB bank transaction data using Python, PostgreSQL, and Power BI.

---

## 📌 Project Overview

**Micro leakage** refers to small, recurring, often unnoticed financial outflows — discretionary spending on food, shopping, entertainment, and subscriptions — that accumulate silently over time.

This project builds a complete detection pipeline:
- ETL and anonymization of real bank statement data
- Feature engineering and leakage scoring (0–100 risk score)
- ML classification (Logistic Regression, Random Forest, XGBoost)
- PostgreSQL analytical layer with 17 business queries
- Power BI dashboard for visualization

---

## 📊 Key Findings

| Metric | Value |
|--------|-------|
| Total Transactions | 500 (Jan–Jul 2026) |
| Leakage Rate | **4.2%** (21 transactions) |
| Potential Savings | **₹19,372** (cutting top-risk spend in half) |
| Highest Risk Month | June (28% leakage rate) |
| Riskiest Day | Saturday (avg risk score: 51.7/100) |
| Sole Leakage Channel | UPI (4.31% leakage rate) |
| March Anomaly | ₹75,000 single transaction drove 3× normal spend |
| First-visit merchants | 2.6× riskier than repeat visits (7.94% vs 3.00%) |

---

## 🗂 Project Structure

```
finance-leakage/
│
├── data/
│   ├── raw/          ← raw bank statement (gitignored, never committed)
│   ├── processed/    ← structured transaction data
│   ├── clean/        ← anonymized, PII-free data
│   ├── features/     ← engineered features + leakage scores
│   └── lookup/       ← merchant category mapping tables
│
├── scripts/
│   ├── 00_load_pnb_statement.py   ← parse raw PNB CSV
│   ├── 01_anonymize.py            ← strip PII, extract categories
│   ├── 02_feature_engineering.py  ← build 56 features
│   ├── 03_leakage_scoring.py      ← assign 0-100 risk score
│   ├── 04_train_model.py          ← train & evaluate ML models
│   ├── 05_load_to_postgres.py     ← load into PostgreSQL
│   └── config.py                  ← path configuration
│
├── sql/
│   ├── 01_schema.sql              ← PostgreSQL table schema + indexes
│   └── 02_analysis_queries.sql    ← 17 analytical SQL queries
│
├── outputs/
│   ├── model_results.csv          ← precision, recall, F1, AUC per model
│   └── feature_importance.csv     ← top features from Random Forest
│
├── dashboard/
│   └── finance_dashboard.pbix     ← Power BI dashboard (3 pages)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

```
Raw PNB Bank Statement (.csv)
        ↓
00_load_pnb_statement.py   — parse, clean, structure
        ↓
01_anonymize.py            — strip account numbers, names, UPI IDs
                             extract payment mode + category
        ↓
02_feature_engineering.py  — 56 features: time, amount, entity
                             behaviour, recency, monthly aggregates
                             Define leakage: debit in Food/Shopping/
                             Entertainment/Subscription = potential leak
        ↓
03_leakage_scoring.py      — 5-factor risk score (0-100):
                             category risk + amount percentile +
                             frequency + timing + recency
        ↓
04_train_model.py          — LogReg / RF / XGBoost
                             SMOTE + class weighting for 21/500 imbalance
                             Evaluated on F1 + PR-AUC (not accuracy)
        ↓
05_load_to_postgres.py     — load into finance_leakage database
        ↓
02_analysis_queries.sql    — 17 queries: LAG, RANK, NTILE,
                             PERCENTILE_CONT, CTEs, window functions
        ↓
finance_dashboard.pbix     — Power BI: 3-page interactive dashboard
```

---

## 🤖 ML Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Random Forest | 0.36 | 0.43 | **0.39** | **0.89** | 0.33 |
| XGBoost | 0.29 | 0.57 | 0.38 | 0.86 | 0.35 |
| Logistic Regression | 0.21 | 0.57 | 0.31 | 0.88 | 0.30 |

**Why F1=0.39 is honest and expected:**
- 21 positive samples out of 500 (4.2% class)
- A model predicting 0 for everything gets 95.8% accuracy — useless
- F1 and PR-AUC penalize that; ROC-AUC 0.89 shows the model genuinely ranks leakage transactions higher than normal ones
- The continuous leakage score (0–100) is more actionable than the binary prediction for this use case

**Imbalance handling:**
- SMOTE inside pipeline (never applied to test folds)
- `class_weight='balanced'` for LogReg and Random Forest
- `scale_pos_weight=22.81` for XGBoost
- StratifiedKFold (5 folds) to preserve class ratio

---

## 🧮 SQL Techniques Demonstrated

| Query | Technique |
|-------|-----------|
| Monthly savings rate | `NULLIF`, aggregation |
| Monthly leakage trend | `LAG()` window function |
| Risk band distribution | `SUM() OVER ()` window |
| Savings potential | `NTILE(10)` window function |
| Category rank per month | `RANK() PARTITION BY month` |
| High value alerts | `PERCENTILE_CONT`, cross join |
| ML confusion matrix | `CASE WHEN` multi-condition counting |
| Running cash flow | `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` |
| Payment mode leakage | CTE + leakage rate calculation |

---

## 📈 Power BI Dashboard

**Page 1 — Financial Overview**
- KPI cards: Total Income, Total Expenses, Net Savings, Leakage Rate
- Line chart: Monthly income vs expenses (Jan–Jul 2026)
- Donut chart: Spending by category

**Page 2 — Leakage Analysis**
- Bar chart: Leakage count by month
- Donut chart: Spending by risk band
- Bar chart: Leakage by category
- Table: High risk transactions (filtered to High Risk only)

**Page 3 — ML Results**
- Bar chart: Top 10 feature importance
- Table: Model comparison (all 3 models, all metrics)

---

## 🔒 Privacy & Data Handling

- Raw bank statement is **gitignored** — never committed
- Anonymization runs locally before any analysis:
  - Account numbers, IFSC, branch details stripped
  - Balance column dropped
  - Real names replaced with `PERSON_001`, `PERSON_002` tokens
  - Merchant names replaced with `MERCHANT_001`, `MERCHANT_002` tokens
  - UPI IDs and phone numbers removed via regex
- Only anonymized data enters PostgreSQL and Power BI

---

## 🛠 Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.10 | ETL, feature engineering, ML |
| pandas, numpy | Data processing |
| scikit-learn | LogReg, Random Forest, preprocessing |
| XGBoost | Gradient boosting model |
| imbalanced-learn | SMOTE oversampling |
| psycopg2 | PostgreSQL connection |
| PostgreSQL 18 | Analytical data warehouse |
| Power BI Desktop | Interactive dashboard |

---

## 🚀 How to Run

```bash
# 1. Create environment
conda create -n finance_env python=3.10
conda activate finance_env
pip install -r requirements.txt

# 2. Place raw bank statement in data/raw/

# 3. Run pipeline in order
python scripts/00_load_pnb_statement.py
python scripts/01_anonymize.py
python scripts/02_feature_engineering.py
python scripts/03_leakage_scoring.py
python scripts/04_train_model.py
python scripts/05_load_to_postgres.py

# 4. Run SQL analysis
psql -U postgres -d finance_leakage -f sql/01_schema.sql
psql -U postgres -d finance_leakage -f sql/02_analysis_queries.sql

# 5. Open dashboard/finance_dashboard.pbix in Power BI Desktop
```

---

## ⚠️ Honest Limitations

- **Sample size**: 500 transactions over 6 months — methodology scales, but findings are personal, not generalizable
- **ML performance**: F1=0.39 is modest but correct given 21 positive samples; documented rather than papered over
- **Merchant categorization**: 85% of transactions are person-to-person UPI transfers with no merchant name — a known constraint of Indian UPI description format
- **No time-of-day data**: Bank statements only provide dates, not timestamps — time-block analysis not possible

---

*Built on real anonymized personal finance data — findings reflect actual spending patterns, not synthetic examples.*
