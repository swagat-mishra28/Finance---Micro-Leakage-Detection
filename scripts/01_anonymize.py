"""
01_anonymize.py

Reads:
    data/processed/transactions_processed.csv
    data/lookup/merchant_mapping.csv

Produces:
    data/clean/transactions_clean.csv

What this script does:
    - Parses every transaction description (UPI/NEFT/IMPS/ATM/BANK)
    - Extracts payment mode and direction from description
    - Maps merchant names to categories using merchant_mapping.csv
    - Replaces real names with PERSON_001, PERSON_002 tokens
    - Replaces merchant names with MERCHANT_001, MERCHANT_002 tokens
    - Drops: balance, branch_name, cheque_no (sensitive fields)
    - Keeps: txn_no, txn_date, payment_mode, direction, category,
             entity_type, entity_id, description (anonymized),
             dr_amount, cr_amount

Run:
    python scripts/01_anonymize.py
"""

from pathlib import Path
import logging
import re

import pandas as pd

# ----------------------------------------------------------
# Paths — all relative to this script's parent directory
# ----------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_DIR     = BASE_DIR / "data" / "clean"
LOOKUP_DIR    = BASE_DIR / "data" / "lookup"

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# ----------------------------------------------------------
# Global anonymization state
# ----------------------------------------------------------

person_map     = {}
merchant_map   = {}
person_counter  = 1
merchant_counter = 1


# ----------------------------------------------------------
# Load merchant lookup
# ----------------------------------------------------------

def load_merchant_lookup():
    """
    merchant_mapping.csv has duplicate header rows scattered through it
    (the file was built by appending multiple CSVs together).
    We filter those out before building the lookup dict.
    """
    path = LOOKUP_DIR / "merchant_mapping.csv"

    if not path.exists():
        logging.warning(f"merchant_mapping.csv not found at {path}. Category detection will be limited.")
        return {}

    df = pd.read_csv(path, dtype=str)

    # Drop any row where merchant_name literally says "merchant_name"
    # (these are the duplicate header rows)
    df = df[df["merchant_name"].str.strip().str.lower() != "merchant_name"]
    df = df.dropna(subset=["merchant_name", "category"])
    df["merchant_name"] = df["merchant_name"].str.strip().str.upper()
    df["category"]      = df["category"].str.strip()

    lookup = dict(zip(df["merchant_name"], df["category"]))
    logging.info(f"Loaded {len(lookup)} merchant mappings from lookup table.")
    return lookup


# ----------------------------------------------------------
# Anonymization helpers
# ----------------------------------------------------------

def get_person_token(name: str) -> str:
    global person_counter
    key = name.upper().strip()
    if key not in person_map:
        person_map[key] = f"PERSON_{person_counter:03d}"
        person_counter += 1
    return person_map[key]


def get_merchant_token(name: str) -> str:
    global merchant_counter
    key = name.upper().strip()
    if key not in merchant_map:
        merchant_map[key] = f"MERCHANT_{merchant_counter:03d}"
        merchant_counter += 1
    return merchant_map[key]


# ----------------------------------------------------------
# Category detection
# ----------------------------------------------------------

def detect_category(text: str, merchant_lookup: dict) -> str:
    """
    Try merchant lookup first (exact and partial match),
    then fall back to keyword rules.
    """
    text_upper = str(text).upper().strip()

    # Exact match first
    if text_upper in merchant_lookup:
        return merchant_lookup[text_upper]

    # Partial match — check if any known merchant name appears in the text
    for merchant_name, category in merchant_lookup.items():
        if merchant_name in text_upper:
            return category

    # Keyword fallback
    if any(k in text_upper for k in ["SALARY", "SAL ", "STIPEND", "PAYROLL"]):
        return "Income"
    if any(k in text_upper for k in ["INT.PD", "INTEREST PAID", "INT CREDIT"]):
        return "Interest"
    if any(k in text_upper for k in ["CHRG", "CHARGE", "FEE", "PENALTY", "MIN BAL", "SMS"]):
        return "Bank Charges"
    if any(k in text_upper for k in ["ATM", "CASH WITH"]):
        return "Cash Withdrawal"
    if any(k in text_upper for k in ["NEFT", "IMPS", "RTGS"]):
        return "Bank Transfer"

    return "Transfer"


# ----------------------------------------------------------
# Description parser
# ----------------------------------------------------------

def parse_description(description: str, merchant_lookup: dict) -> dict:
    """
    Parse a raw PNB transaction description and return a dict of:
        payment_mode, direction, category,
        entity_type, entity_id, clean_description
    """
    desc = str(description).strip()
    desc_upper = desc.upper()

    result = {
        "payment_mode"     : "OTHER",
        "direction"        : "OTHER",
        "category"         : "Transfer",
        "entity_type"      : "UNKNOWN",
        "entity_id"        : "UNKNOWN",
        "clean_description": "UNKNOWN",
    }

    # --------------------------------------------------
    # Bank-generated entries: SMS charges, interest
    # --------------------------------------------------
    if desc_upper.startswith("SMS CHRG") or "SMS CHARGE" in desc_upper:
        result.update({
            "payment_mode"     : "BANK",
            "direction"        : "DEBIT",
            "category"         : "Bank Charges",
            "entity_type"      : "BANK",
            "entity_id"        : "BANK_CHARGES",
            "clean_description": "SMS_CHARGES",
        })
        return result

    if "INT.PD" in desc_upper or "INTEREST PAID" in desc_upper:
        result.update({
            "payment_mode"     : "BANK",
            "direction"        : "CREDIT",
            "category"         : "Interest",
            "entity_type"      : "BANK",
            "entity_id"        : "BANK_INTEREST",
            "clean_description": "INTEREST_CREDIT",
        })
        return result

    # --------------------------------------------------
    # ATM
    # --------------------------------------------------
    if "ATM" in desc_upper:
        result.update({
            "payment_mode"     : "ATM",
            "direction"        : "DEBIT",
            "category"         : "Cash Withdrawal",
            "entity_type"      : "BANK",
            "entity_id"        : "ATM_WITHDRAWAL",
            "clean_description": "ATM_CASH",
        })
        return result

    # --------------------------------------------------
    # NEFT
    # --------------------------------------------------
    if desc_upper.startswith("NEFT"):
        direction = "CREDIT" if "/IN/" in desc_upper or "-IN-" in desc_upper else "DEBIT"
        result.update({
            "payment_mode"     : "NEFT",
            "direction"        : direction,
            "category"         : "Bank Transfer",
            "entity_type"      : "BANK",
            "entity_id"        : "BANK_TRANSFER",
            "clean_description": "NEFT_TRANSFER",
        })
        return result

    # --------------------------------------------------
    # IMPS
    # --------------------------------------------------
    if desc_upper.startswith("IMPS"):
        result.update({
            "payment_mode"     : "IMPS",
            "direction"        : "DEBIT",
            "category"         : "Bank Transfer",
            "entity_type"      : "BANK",
            "entity_id"        : "BANK_TRANSFER",
            "clean_description": "IMPS_TRANSFER",
        })
        return result

    # --------------------------------------------------
    # UPI — main branch
    # Format: UPI/DR or CR / txn_ref / entity_name / bank / upi_id / remarks
    # --------------------------------------------------
    if desc_upper.startswith("UPI"):
        parts = desc.split("/")

        # Direction
        direction = "OTHER"
        if len(parts) > 1:
            if parts[1].upper() == "DR":
                direction = "DEBIT"
            elif parts[1].upper() == "CR":
                direction = "CREDIT"

        # Entity name is at index 3 in standard PNB UPI format
        entity_name = ""
        if len(parts) >= 4:
            entity_name = parts[3].strip()

        # Strip phone numbers and UPI IDs from entity name
        entity_name = re.sub(r"\b\d{10}\b", "", entity_name)
        entity_name = re.sub(r"[\w.\-]+@[\w]+", "", entity_name)
        entity_name = entity_name.strip()

        if not entity_name:
            entity_name = "UNKNOWN"

        # Detect category from entity name using lookup
        category = detect_category(entity_name, merchant_lookup)

        # Assign entity type and token
        if category in ("Transfer", "Bank Transfer", "Income"):
            entity_type = "PERSON"
            entity_id   = get_person_token(entity_name)
        else:
            entity_type = "MERCHANT"
            entity_id   = get_merchant_token(entity_name)

        result.update({
            "payment_mode"     : "UPI",
            "direction"        : direction,
            "category"         : category,
            "entity_type"      : entity_type,
            "entity_id"        : entity_id,
            "clean_description": f"UPI_{direction}_{entity_type}_{entity_id}",
        })
        return result

    # --------------------------------------------------
    # POS / card purchase
    # --------------------------------------------------
    if "POS" in desc_upper:
        result.update({
            "payment_mode"     : "CARD",
            "direction"        : "DEBIT",
            "category"         : "Card Purchase",
            "entity_type"      : "MERCHANT",
            "entity_id"        : "CARD_PURCHASE",
            "clean_description": "CARD_PURCHASE",
        })
        return result

    return result


# ----------------------------------------------------------
# Enrich dataframe
# ----------------------------------------------------------

def enrich_dataframe(df: pd.DataFrame, merchant_lookup: dict) -> pd.DataFrame:
    logging.info("Parsing descriptions...")

    parsed = df["description"].apply(
        lambda d: parse_description(d, merchant_lookup)
    )
    parsed_df = pd.DataFrame(parsed.tolist())

    df = pd.concat(
        [df.reset_index(drop=True), parsed_df.reset_index(drop=True)],
        axis=1
    )
    return df


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    logging.info("=" * 60)
    logging.info("STARTING ANONYMIZATION")
    logging.info("=" * 60)

    input_file  = PROCESSED_DIR / "transactions_processed.csv"
    output_file = CLEAN_DIR / "transactions_clean.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input not found: {input_file}\n"
            f"Run 00_load_pnb_statement.py first."
        )

    # Load
    df = pd.read_csv(input_file)
    logging.info(f"Loaded {len(df)} transactions from {input_file.name}")

    # Load merchant lookup
    merchant_lookup = load_merchant_lookup()

    # Parse and enrich
    df = enrich_dataframe(df, merchant_lookup)

    # Drop original raw description (replaced by clean_description)
    df = df.drop(columns=["description"], errors="ignore")
    df = df.rename(columns={"clean_description": "description"})

    # Drop ALL sensitive columns
    # balance: most sensitive — shows your exact money at every point
    # branch_name, cheque_no: identifying metadata
    sensitive_cols = ["balance", "branch_name", "cheque_no"]
    df = df.drop(columns=sensitive_cols, errors="ignore")

    # Final column order
    column_order = [
        "txn_no",
        "txn_date",
        "payment_mode",
        "direction",
        "category",
        "entity_type",
        "entity_id",
        "description",
        "dr_amount",
        "cr_amount",
    ]
    # Only keep columns that actually exist
    df = df[[c for c in column_order if c in df.columns]]

    # Save
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("ANONYMIZATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Transactions   : {len(df)}")
    logging.info(f"Unique Persons : {len(person_map)}")
    logging.info(f"Unique Merchants: {len(merchant_map)}")
    logging.info(f"Output         : {output_file}")
    logging.info("")
    logging.info("Category Distribution:")
    print(df["category"].value_counts().to_string())
    logging.info("")
    logging.info("First 5 rows (no sensitive data):")
    print(df[["txn_no", "txn_date", "payment_mode", "direction",
              "category", "entity_id", "dr_amount", "cr_amount"]].head().to_string())
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
