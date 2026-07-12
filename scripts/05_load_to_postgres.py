"""
05_load_to_postgres.py

Reads:
    outputs/predictions.csv

Loads into:
    PostgreSQL database: finance_leakage
    Table: transactions

Run:
    python scripts/05_load_to_postgres.py
"""

from pathlib import Path
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ----------------------------------------------------------
# Database config
# ----------------------------------------------------------

DB_CONFIG = {
    "dbname"  : "finance_leakage",
    "user"    : "postgres",
    "password": "",
    "host"    : "localhost",
    "port"    : "5432"
}

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# ----------------------------------------------------------
# Columns to load — must match schema exactly
# ----------------------------------------------------------

COLUMNS = [
    "txn_no", "txn_date", "payment_mode", "direction",
    "category", "entity_type", "entity_id", "description",
    "dr_amount", "cr_amount", "amount",
    "year", "quarter", "month", "month_name", "week",
    "day", "weekday", "day_type", "is_weekend",
    "is_small_transaction", "is_medium_transaction",
    "is_high_value_transaction", "transaction_sign",
    "is_debit", "is_credit",
    "entity_transaction_count", "entity_total_amount",
    "entity_avg_amount", "entity_days_active",
    "entity_transactions_per_day", "entity_debit_count",
    "entity_credit_count", "entity_avg_debit", "entity_avg_credit",
    "entity_frequency",
    "days_since_last_transaction", "is_first_transaction",
    "is_recent_transaction",
    "is_regular_entity", "is_long_term_entity",
    "is_weekend_spending", "is_recurring_entity", "is_subscription",
    "is_micro_leakage", "is_large_shopping", "is_food_habit",
    "potential_leakage",
    "leakage_score", "score_category", "score_amount",
    "score_frequency", "score_timing", "score_recency", "risk_band",
    "running_income", "running_expense", "net_cash_flow",
    "monthly_spend", "monthly_income", "monthly_savings",
    "predicted_leakage", "predicted_proba", "correctly_identified"
]


def load_csv():
    path = OUTPUT_DIR / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"predictions.csv not found at {path}\n"
            f"Run 04_train_model.py first."
        )

    df = pd.read_csv(path)

    # Keep only columns that exist in both schema and CSV
    cols = [c for c in COLUMNS if c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        logging.warning(f"Columns in schema but not in CSV (will be NULL): {missing}")

    df = df[cols]

    # Fix boolean columns — postgres needs True/False not 1/0
    bool_cols = [
        "is_weekend", "is_small_transaction", "is_medium_transaction",
        "is_high_value_transaction", "is_debit", "is_credit",
        "is_first_transaction", "is_recent_transaction",
        "is_regular_entity", "is_long_term_entity", "is_weekend_spending",
        "is_recurring_entity", "is_subscription", "is_micro_leakage",
        "is_large_shopping", "is_food_habit", "correctly_identified"
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Fix date column
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce").dt.date

    # Replace NaN with None for postgres
    df = df.where(pd.notnull(df), None)

    logging.info(f"Loaded {len(df)} rows from predictions.csv")
    logging.info(f"Columns: {len(cols)}")
    return df, cols


def create_schema(conn):
    schema_path = BASE_DIR / "sql" / "01_schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        sql = f.read()

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logging.info("Schema created successfully.")


def insert_data(conn, df, cols):
    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

    placeholders = ",".join(["%s"] * len(cols))
    col_names    = ",".join(cols)
    insert_sql   = f"INSERT INTO transactions ({col_names}) VALUES %s ON CONFLICT (txn_no) DO NOTHING"

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows, page_size=100)
    conn.commit()
    logging.info(f"Inserted {len(rows)} rows into transactions table.")


def verify(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM transactions;")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM transactions WHERE potential_leakage = 1;")
        leakage = cur.fetchone()[0]

        cur.execute("SELECT MIN(txn_date), MAX(txn_date) FROM transactions;")
        date_range = cur.fetchone()

    logging.info("")
    logging.info("=" * 60)
    logging.info("DATABASE LOAD COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total transactions : {total}")
    logging.info(f"Leakage flagged    : {leakage}")
    logging.info(f"Date range         : {date_range[0]} to {date_range[1]}")
    logging.info("=" * 60)


def main():
    logging.info("=" * 60)
    logging.info("LOADING DATA INTO POSTGRESQL")
    logging.info("=" * 60)

    # Load CSV
    df, cols = load_csv()

    # Connect to PostgreSQL
    logging.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    logging.info("Connected.")

    # Create schema
    create_schema(conn)

    # Insert data
    insert_data(conn, df, cols)

    # Verify
    verify(conn)

    conn.close()


if __name__ == "__main__":
    main()
