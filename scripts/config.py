# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_DIR     = BASE_DIR / "data" / "clean"
FEATURES_DIR  = BASE_DIR / "data" / "features"
LOOKUP_DIR    = BASE_DIR / "data" / "lookup"
OUTPUT_DIR    = BASE_DIR / "outputs"