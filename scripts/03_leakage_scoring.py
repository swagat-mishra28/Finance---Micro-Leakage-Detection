"""
03_leakage_scoring.py

Reads:
    data/features/transactions_features.csv

Produces:
    data/features/transactions_scored.csv

What this script does:
    Assigns every transaction a leakage risk SCORE from 0 to 100.
    This is different from the binary potential_leakage flag:
        - potential_leakage = yes/no (is this a leak?)
        - leakage_score     = 0-100 (HOW RISKY is this transaction?)

    The score is built from 5 independent factors, each worth 0-20 points:
        1. Category Risk     — how leakage-prone is this spending category?
        2. Amount Risk       — how large is the spend relative to your average?
        3. Frequency Risk    — how often do you spend at this merchant/person?
        4. Timing Risk       — did this happen on a weekend or at month-start?
        5. Recency Risk      — is this a recent new merchant you haven't used before?

    Final score = sum of all 5 factors, capped at 100.
    Score bands:
        0-25   → Low Risk
        26-50  → Moderate Risk
        51-75  → High Risk
        76-100 → Critical Risk

Run:
    python scripts/03_leakage_scoring.py
"""

from pathlib import Path
import logging
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# ----------------------------------------------------------
# Factor 1: Category Risk Score (0-20)
# How inherently risky is this spending category?
# Higher = more discretionary / leakage-prone
# ----------------------------------------------------------

CATEGORY_RISK = {
    # High risk — pure discretionary
    "Subscription"    : 20,
    "Entertainment"   : 18,
    "Food"            : 16,
    "Shopping"        : 15,

    # Medium risk — sometimes necessary
    "Recurring_Micro" : 10,
    "One_off_Micro"   : 8,
    "Electronics"     : 12,
    "Travel"          : 8,

    # Low risk — mostly necessary
    "Recurring_Transfer" : 5,
    "One_off_Transfer"   : 4,
    "Recurring_Large"    : 3,
    "One_off_Large"      : 3,
    "Utilities"          : 2,
    "Healthcare"         : 2,
    "Grocery"            : 2,
    "Finance"            : 1,

    # No risk — not leakage
    "Income_Transfer" : 0,
    "Interest"        : 0,
    "Bank Charges"    : 0,
    "Bank Transfer"   : 0,
    "Cash Withdrawal" : 0,
}


def score_category(category: str) -> int:
    """
    Look up the category in the risk table.
    Unknown categories get a neutral score of 5.
    """
    return CATEGORY_RISK.get(category, 5)


# ----------------------------------------------------------
# Factor 2: Amount Risk Score (0-20)
# How large is this transaction compared to your
# average debit transaction?
# We use percentile rank — a transaction in the top 25%
# of your spending gets a higher score.
# ----------------------------------------------------------

def score_amount(df: pd.DataFrame) -> pd.Series:
    """
    Rank each debit amount within all debit transactions.
    Non-debit transactions (credits) get 0.
    """
    # Only score debit transactions
    debit_mask = df["direction"] == "DEBIT"

    # Percentile rank of amount among ALL debits
    # rank(pct=True) gives a value 0-1 for each row
    amount_pct = df.loc[debit_mask, "dr_amount"].rank(pct=True)

    # Scale to 0-20
    amount_score = (amount_pct * 20).round(0).astype(int)

    # Credits get 0
    result = pd.Series(0, index=df.index)
    result.loc[debit_mask] = amount_score

    return result


# ----------------------------------------------------------
# Factor 3: Frequency Risk Score (0-20)
# How often do you transact with this entity?
# Counter-intuitive: HIGH frequency = HIGH risk
# because recurring spend at the same merchant
# is a habit leak (e.g. ordering food from the
# same place every week)
# ----------------------------------------------------------

def score_frequency(df: pd.DataFrame) -> pd.Series:
    """
    Map entity_frequency label to a risk score.
    Frequent entities in discretionary categories
    are the definition of a spending habit = leakage.
    """
    frequency_risk = {
        "Frequent"   : 20,   # 10+ transactions — clear habit
        "Regular"    : 15,   # 5-9 transactions — developing habit
        "Occasional" : 8,    # 3-4 transactions — possible pattern
        "Rare"       : 3,    # 1-2 transactions — one-off
    }

    return df["entity_frequency"].map(frequency_risk).fillna(5).astype(int)


# ----------------------------------------------------------
# Factor 4: Timing Risk Score (0-20)
# When did this transaction happen?
# Weekend spending + early-month spending (post-salary)
# are classic leakage patterns.
# ----------------------------------------------------------

def score_timing(df: pd.DataFrame) -> pd.Series:
    """
    Weekend transactions score higher (impulse buying).
    Transactions in the first 5 days of the month score
    higher (post-salary spending surge).
    Both together = maximum timing risk.
    """
    score = pd.Series(0, index=df.index)

    # Weekend = +10
    score += df["is_weekend"].astype(int) * 10

    # First 5 days of month = +10 (post-salary impulse window)
    score += (df["day"] <= 5).astype(int) * 10

    return score.clip(0, 20)


# ----------------------------------------------------------
# Factor 5: Recency Risk Score (0-20)
# Is this a NEW merchant you haven't used before?
# First-time transactions with a merchant are riskier
# because there's no established pattern — could be
# an impulse buy at a new place.
# ----------------------------------------------------------

def score_recency(df: pd.DataFrame) -> pd.Series:
    """
    First transaction with an entity = higher risk (new merchant).
    Long-term entities (seen 30+ days) = lower risk (established).
    Recent transactions with known entities = medium risk.
    """
    score = pd.Series(10, index=df.index)  # default: medium

    # New merchant — first time seeing this entity
    score[df["is_first_transaction"]] = 20

    # Long-term known entity — lower risk
    score[df["is_long_term_entity"]] = 5

    # Very established entity (seen 5+ times) — lowest risk
    score[df["entity_transaction_count"] >= 5] = 2

    return score.clip(0, 20)


# ----------------------------------------------------------
# Score band labeling
# ----------------------------------------------------------

def assign_band(score: int) -> str:
    if score <= 25:
        return "Low Risk"
    elif score <= 50:
        return "Moderate Risk"
    elif score <= 75:
        return "High Risk"
    else:
        return "Critical Risk"


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    logging.info("=" * 60)
    logging.info("STARTING LEAKAGE SCORING")
    logging.info("=" * 60)

    input_file  = FEATURES_DIR / "transactions_features.csv"
    output_file = FEATURES_DIR / "transactions_scored.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input not found: {input_file}\n"
            f"Run 02_feature_engineering.py first."
        )

    df = pd.read_csv(input_file)
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    logging.info(f"Loaded {len(df)} transactions.")

    # --------------------------------------------------
    # Compute each factor score
    # --------------------------------------------------

    logging.info("Computing factor scores...")

    df["score_category"]  = df["category"].apply(score_category)
    df["score_amount"]    = score_amount(df)
    df["score_frequency"] = score_frequency(df)
    df["score_timing"]    = score_timing(df)
    df["score_recency"]   = score_recency(df)

    # --------------------------------------------------
    # Final leakage score = sum of all factors, cap at 100
    # --------------------------------------------------

    df["leakage_score"] = (
        df["score_category"] +
        df["score_amount"]   +
        df["score_frequency"]+
        df["score_timing"]   +
        df["score_recency"]
    ).clip(0, 100)

    # --------------------------------------------------
    # Score band
    # --------------------------------------------------

    df["risk_band"] = df["leakage_score"].apply(assign_band)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(output_file, index=False)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    logging.info("")
    logging.info("=" * 60)
    logging.info("SCORING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Output : {output_file}")
    logging.info("")
    logging.info("Score distribution (leakage transactions only):")
    leaks = df[df["potential_leakage"] == 1]
    print(leaks[["category", "dr_amount", "leakage_score",
                 "risk_band"]].sort_values(
                 "leakage_score", ascending=False).to_string())

    logging.info("")
    logging.info("Risk band distribution (ALL transactions):")
    print(df["risk_band"].value_counts().to_string())

    logging.info("")
    logging.info("Average leakage score by category:")
    print(df[df["direction"] == "DEBIT"].groupby("category")["leakage_score"]
          .mean().round(1).sort_values(ascending=False).to_string())

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
