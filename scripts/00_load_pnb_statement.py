"""
00_load_pnb_statement.py

Reads the raw Punjab National Bank statement and converts it into a
clean structured CSV.

Input:
    data/raw/*.csv

Output:
    data/processed/transactions_processed.csv
"""

from pathlib import Path
import csv
import logging

import pandas as pd

from config import RAW_DIR, PROCESSED_DIR

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# -------------------------------------------------------
# Expected Header
# -------------------------------------------------------

EXPECTED_HEADER = [
    "Txn No.",
    "Txn Date",
    "Description",
    "Branch Name",
    "Cheque No.",
    "Dr Amount",
    "Cr Amount",
    "Balance"
]

# -------------------------------------------------------
# Locate CSV
# -------------------------------------------------------

def get_raw_file():

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No raw CSV found.")

    return csv_files[0]

# -------------------------------------------------------
# Detect Header Row
# -------------------------------------------------------

def find_header(filepath):

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:

        reader = csv.reader(file)

        for idx, row in enumerate(reader):

            if len(row) >= 8:

                row = [cell.strip() for cell in row[:8]]

                if row == EXPECTED_HEADER:
                    return idx

    raise RuntimeError("Transaction header not found.")

# -------------------------------------------------------
# Read Transactions
# -------------------------------------------------------

def read_transactions(filepath, header_row):

    rows = []

    loaded = 0
    skipped = 0
    corrected = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:

        reader = csv.reader(file)

        for i, row in enumerate(reader):

            if i <= header_row:
                continue

            # remove trailing empty fields
            while len(row) > 8 and row[-1] == "":
                row.pop()
                corrected += 1

            if len(row) < 8:
                skipped += 1
                continue

            row = row[:8]

            txn_no = row[0].strip()

            if txn_no == "":
                skipped += 1
                continue

            if not txn_no.startswith(("T", "S", "U")):
                skipped += 1
                continue

            rows.append(row)
            loaded += 1

    logging.info(f"Rows Loaded    : {loaded}")
    logging.info(f"Rows Corrected : {corrected}")
    logging.info(f"Rows Skipped   : {skipped}")

    return rows

# -------------------------------------------------------
# DataFrame
# -------------------------------------------------------

def build_dataframe(rows):

    df = pd.DataFrame(
        rows,
        columns=[
            "txn_no",
            "txn_date",
            "description",
            "branch_name",
            "cheque_no",
            "dr_amount",
            "cr_amount",
            "balance"
        ]
    )

    return df

# -------------------------------------------------------
# Clean Columns
# -------------------------------------------------------

def clean_dataframe(df):

    # ----------------------------------------------------
    # Standardize missing values
    # ----------------------------------------------------

    df = df.replace("-", "")
    df = df.fillna("")

    # ----------------------------------------------------
    # Amount columns
    # ----------------------------------------------------

    for col in ["dr_amount", "cr_amount"]:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", "0")
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0.0)

    # ----------------------------------------------------
    # Balance
    # ----------------------------------------------------

    df["balance"] = (
        df["balance"]
        .astype(str)
        .str.replace("Cr.", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df["balance"] = pd.to_numeric(
        df["balance"],
        errors="coerce"
    )

    # ----------------------------------------------------
    # Date
    # ----------------------------------------------------

    df["txn_date"] = pd.to_datetime(
        df["txn_date"],
        dayfirst=True,
        errors="coerce"
    )

    # ----------------------------------------------------
    # Remove invalid rows
    # ----------------------------------------------------

    df = df[df["txn_date"].notna()]

    # ----------------------------------------------------
    # Sort chronologically
    # ----------------------------------------------------

    df = (
        df.sort_values("txn_date")
          .reset_index(drop=True)
    )

    return df

# -------------------------------------------------------
# Save
# -------------------------------------------------------

def save_dataframe(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = PROCESSED_DIR / "transactions_processed.csv"

    df.to_csv(
        output,
        index=False
    )

    logging.info("")
    logging.info("=" * 60)
    logging.info("PNB STATEMENT LOADED")
    logging.info("=" * 60)

    logging.info(f"Transactions : {len(df)}")

    logging.info(f"Output File  : {output}")

    logging.info("")

    print(df.head())

# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    raw_file = get_raw_file()

    logging.info(f"Reading : {raw_file.name}")

    header = find_header(raw_file)

    logging.info(f"Header Row : {header}")

    rows = read_transactions(
        raw_file,
        header
    )

    df = build_dataframe(rows)

    df = clean_dataframe(df)

    save_dataframe(df)

if __name__ == "__main__":
    main()