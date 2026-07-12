"""
02_feature_engineering.py

Reads:
    data/clean/transactions_clean.csv

Produces:
    data/features/transactions_features.csv

Feature groups built:
    1. Time features        — year, quarter, month, week, day, weekday, is_weekend
    2. Amount features      — amount, transaction_sign, is_debit, is_credit,
                              is_small/medium/high_value
    3. Entity features      — per-entity aggregates (count, total, avg, frequency)
    4. Behavioural flags    — is_recurring, is_subscription, is_weekend_spending,
                              is_regular_entity, is_long_term_entity
    5. Leakage indicators   — is_micro_leakage, is_large_shopping, is_food_habit,
                              potential_leakage (the target for ML)
    6. Running totals       — running_income, running_expense, net_cash_flow
    7. Monthly aggregates   — monthly_spend, monthly_income, monthly_savings

Run:
    python scripts/02_feature_engineering.py
"""

from pathlib import Path
import logging

import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent.parent
CLEAN_DIR     = BASE_DIR / "data" / "clean"
FEATURES_DIR  = BASE_DIR / "data" / "features"

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# ----------------------------------------------------------
# Thresholds (define what counts as a leakage)
# ----------------------------------------------------------

MICRO_LEAKAGE_MAX   = 500     # transactions under ₹500 in leakage categories
LARGE_SHOPPING_MIN  = 2000    # shopping transactions above ₹2000
FOOD_HABIT_MIN      = 200     # food transactions above ₹200
RECURRING_MIN_COUNT = 3       # entity must appear 3+ times to be "recurring"
REGULAR_MIN_COUNT   = 5       # entity must appear 5+ times to be "regular"
LONG_TERM_DAYS      = 30      # entity seen across 30+ days = long term

LEAKAGE_CATEGORIES = {
    "Food", "Shopping", "Entertainment",
    "Subscription"
}

# ----------------------------------------------------------
# 1. Load
# ----------------------------------------------------------

def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    df = df.dropna(subset=["txn_date"])
    df = df.sort_values("txn_date").reset_index(drop=True)
    logging.info(f"Loaded {len(df)} transactions from {path.name}")
    return df


# ----------------------------------------------------------
# 2. Time features
# ----------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]       = df["txn_date"].dt.year
    df["quarter"]    = df["txn_date"].dt.quarter
    df["month"]      = df["txn_date"].dt.month
    df["month_name"] = df["txn_date"].dt.strftime("%B")
    df["week"]       = df["txn_date"].dt.isocalendar().week.astype(int)
    df["day"]        = df["txn_date"].dt.day
    df["weekday"]    = df["txn_date"].dt.strftime("%A")
    df["is_weekend"] = df["txn_date"].dt.dayofweek >= 5
    logging.info("Time features added.")
    return df

def refine_transfer_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sub-categorize Transfer transactions by amount and frequency.
    Transfer is too broad when it covers 85% of transactions.
    """
    # Count how many times each entity appears
    entity_counts = df.groupby("entity_id")["txn_no"].count().to_dict()
    
    def sub_categorize(row):
        if row["category"] != "Transfer":
            return row["category"]
        
        count = entity_counts.get(row["entity_id"], 1)
        amount = row["dr_amount"]
        
        if row["direction"] == "CREDIT":
            return "Income_Transfer"
        
        if count >= 3:
            if amount < 500:
                return "Recurring_Micro"      # bill splits, regular small payments
            elif amount < 2000:
                return "Recurring_Transfer"   # regular payments (could be rent share etc)
            else:
                return "Recurring_Large"      # rent, tuition, regular large
        else:
            if amount < 500:
                return "One_off_Micro"        # impulse small payments
            elif amount < 5000:
                return "One_off_Transfer"     # occasional payments
            else:
                return "One_off_Large"        # large one-time payments
    
    df["category"] = df.apply(sub_categorize, axis=1)
    
    logging.info("Transfer sub-categorization applied.")
    logging.info(df["category"].value_counts().to_string())
    return df


# ----------------------------------------------------------
# 3. Amount features
# ----------------------------------------------------------

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df["amount"]           = df["dr_amount"] + df["cr_amount"]
    df["transaction_sign"] = df["direction"].map(
        {"DEBIT": -1, "CREDIT": 1}
    ).fillna(0).astype(int)
    df["is_debit"]         = df["direction"] == "DEBIT"
    df["is_credit"]        = df["direction"] == "CREDIT"

    # Size buckets based on debit amount
    df["is_small_transaction"]  = (df["dr_amount"] > 0) & (df["dr_amount"] <= 500)
    df["is_medium_transaction"] = (df["dr_amount"] > 500) & (df["dr_amount"] <= 5000)
    df["is_high_value_transaction"] = df["dr_amount"] > 5000

    logging.info("Amount features added.")
    return df


# ----------------------------------------------------------
# 4. Entity-level aggregates
# ----------------------------------------------------------

def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each entity_id, compute aggregates across ALL its transactions,
    then join back to the main dataframe. This gives each row the context
    of how "typical" or "unusual" that entity is.
    """
    grp = df.groupby("entity_id")

    entity_stats = pd.DataFrame({
        "entity_transaction_count"  : grp["amount"].count(),
        "entity_total_amount"       : grp["dr_amount"].sum(),
        "entity_avg_amount"         : grp["dr_amount"].mean().round(2),
        "entity_first_seen"         : grp["txn_date"].min(),
        "entity_last_seen"          : grp["txn_date"].max(),
        "entity_debit_count"        : grp["is_debit"].sum(),
        "entity_credit_count"       : grp["is_credit"].sum(),
        "entity_avg_debit"          : grp["dr_amount"].mean().round(2),
        "entity_avg_credit"         : grp["cr_amount"].mean().round(2),
    }).reset_index()

    entity_stats["entity_days_active"] = (
        (entity_stats["entity_last_seen"] - entity_stats["entity_first_seen"])
        .dt.days + 1
    )
    entity_stats["entity_transactions_per_day"] = (
        entity_stats["entity_transaction_count"] /
        entity_stats["entity_days_active"]
    ).round(3)

    # Frequency label
    def label_frequency(count):
        if count >= 10:
            return "Frequent"
        elif count >= 5:
            return "Regular"
        elif count >= 3:
            return "Occasional"
        else:
            return "Rare"

    entity_stats["entity_frequency"] = entity_stats[
        "entity_transaction_count"
    ].apply(label_frequency)

    df = df.merge(entity_stats, on="entity_id", how="left")
    logging.info("Entity aggregate features added.")
    return df


# ----------------------------------------------------------
# 5. Recency features
# ----------------------------------------------------------

def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["entity_id", "txn_date"]).reset_index(drop=True)

    df["previous_transaction_date"] = df.groupby("entity_id")["txn_date"].shift(1)
    df["days_since_last_transaction"] = (
        df["txn_date"] - df["previous_transaction_date"]
    ).dt.days.fillna(-1).astype(int)

    df["is_first_transaction"] = df["days_since_last_transaction"] == -1

    max_date = df["txn_date"].max()
    df["is_recent_transaction"] = (max_date - df["txn_date"]).dt.days <= 30

    df = df.sort_values("txn_date").reset_index(drop=True)
    logging.info("Recency features added.")
    return df


# ----------------------------------------------------------
# 6. Behavioural flags
# ----------------------------------------------------------

def add_behavioural_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_regular_entity"]   = df["entity_transaction_count"] >= REGULAR_MIN_COUNT
    df["is_long_term_entity"] = df["entity_days_active"] >= LONG_TERM_DAYS
    df["is_weekend_spending"] = df["is_weekend"] & df["is_debit"]
    df["is_recurring_entity"] = df["entity_transaction_count"] >= RECURRING_MIN_COUNT

    df["is_subscription"] = df["category"] == "Subscription"

    logging.info("Behavioural flags added.")
    return df


# ----------------------------------------------------------
# 7. Leakage indicators (target variable construction)
# ----------------------------------------------------------

def add_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    potential_leakage is the ML target variable.
    A transaction is flagged as potential leakage if:
      - It's a debit
      - It falls in a leakage category (Food, Shopping, Entertainment,
        Subscription, or Transfer to persons)
      - AND at least one of:
          * It's a small transaction (under ₹500) — micro leakage
          * It's a recurring entity (same payee 3+ times) — habit spend
          * It's a subscription — silent recurring drain
          * It's weekend spending in a discretionary category
    This is an explicit definition, not a leaked label — we can defend
    every flag with a business reason.
    """
    in_leakage_category = df["category"].isin(LEAKAGE_CATEGORIES)
    is_debit            = df["is_debit"]

    df["is_micro_leakage"] = (
        is_debit &
        in_leakage_category &
        df["is_small_transaction"]
    )

    df["is_large_shopping"] = (
        is_debit &
        (df["category"] == "Shopping") &
        (df["dr_amount"] >= LARGE_SHOPPING_MIN)
    )

    df["is_food_habit"] = (
        is_debit &
        (df["category"] == "Food") &
        (df["dr_amount"] >= FOOD_HABIT_MIN)
    )

    df["potential_leakage"] = (
        is_debit &
        in_leakage_category
        
    ).astype(int)

    leakage_count = df["potential_leakage"].sum()
    leakage_pct   = round(leakage_count / len(df) * 100, 2)
    logging.info(f"Leakage transactions: {leakage_count} ({leakage_pct}% of total)")
    logging.info("Leakage features added.")
    return df


# ----------------------------------------------------------
# 8. Running totals
# ----------------------------------------------------------

def add_running_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("txn_date").reset_index(drop=True)
    df["running_income"]  = df["cr_amount"].cumsum().round(2)
    df["running_expense"] = df["dr_amount"].cumsum().round(2)
    df["net_cash_flow"]   = (df["running_income"] - df["running_expense"]).round(2)
    logging.info("Running totals added.")
    return df


# ----------------------------------------------------------
# 9. Monthly aggregates (joined back per row)
# ----------------------------------------------------------

def add_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby(["year", "month"]).agg(
        monthly_spend  = ("dr_amount", "sum"),
        monthly_income = ("cr_amount", "sum"),
    ).reset_index()
    monthly["monthly_savings"] = (
        monthly["monthly_income"] - monthly["monthly_spend"]
    ).round(2)
    monthly["monthly_spend"]  = monthly["monthly_spend"].round(2)
    monthly["monthly_income"] = monthly["monthly_income"].round(2)

    df = df.merge(monthly, on=["year", "month"], how="left")
    logging.info("Monthly aggregates added.")
    return df


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    logging.info("=" * 60)
    logging.info("STARTING FEATURE ENGINEERING")
    logging.info("=" * 60)

    input_file  = CLEAN_DIR  / "transactions_clean.csv"
    output_file = FEATURES_DIR / "transactions_features.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input not found: {input_file}\n"
            f"Run 01_anonymize.py first."
        )

    df = load_clean(input_file)

    df = add_time_features(df)
    df = refine_transfer_categories(df)
    df = add_amount_features(df)
    df = add_entity_features(df)
    df = add_recency_features(df)
    df = add_behavioural_flags(df)
    df = add_leakage_features(df)
    df = add_running_totals(df)
    df = add_monthly_aggregates(df)

    # Confirm balance is NOT in the output
    assert "balance" not in df.columns, "balance column leaked in — check input file"

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    logging.info("")
    logging.info("=" * 60)
    logging.info("FEATURE ENGINEERING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Transactions  : {len(df)}")
    logging.info(f"Features      : {len(df.columns)}")
    logging.info(f"Output        : {output_file}")
    logging.info("")
    logging.info("Leakage breakdown:")
    print(df["potential_leakage"].value_counts().to_string())
    logging.info("")
    logging.info("Category distribution:")
    print(df["category"].value_counts().to_string())
    logging.info("")
    logging.info("Monthly summary:")
    print(df.groupby("month_name")[["monthly_spend","monthly_income","monthly_savings"]]
          .first().to_string())
    logging.info("=" * 60)


if __name__ == "__main__":
    main()