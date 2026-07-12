"""
04_train_model.py

Reads:
    data/features/transactions_scored.csv

Produces:
    outputs/model_results.csv        — per-model metrics
    outputs/feature_importance.csv   — top features from Random Forest
    outputs/predictions.csv          — full dataset with predictions

Models trained:
    1. Logistic Regression  (baseline)
    2. Random Forest        (ensemble)
    3. XGBoost              (gradient boosting)

Imbalance handling:
    - SMOTE oversampling on training set only (never on test set)
    - class_weight='balanced' for LogReg and RF
    - scale_pos_weight for XGBoost
    - StratifiedKFold to preserve class ratio in every fold

Evaluation metrics:
    - Precision, Recall, F1  (primary)
    - PR-AUC                 (better than ROC-AUC for imbalanced data)
    - ROC-AUC                (reported but not the primary metric)
    - Confusion matrix

Why NOT accuracy:
    With 21/500 positives, a model predicting 0 for everything
    gets 95.8% accuracy. That's useless. F1 and PR-AUC penalize
    that behavior — they require the model to actually find leakage.

Run:
    python scripts/04_train_model.py
"""

from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
OUTPUT_DIR   = BASE_DIR / "outputs"

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# ----------------------------------------------------------
# Features to use for ML
# These are chosen deliberately:
# - No raw description (text, not useful here)
# - No entity_id (too many categories, causes overfitting)
# - No txn_date (temporal leakage risk)
# - No txn_no (identifier, not a feature)
# - Include engineered features + leakage score
# ----------------------------------------------------------

FEATURE_COLS = [
    # Amount features
    "dr_amount",
    "cr_amount",
    "amount",
    "is_small_transaction",
    "is_medium_transaction",
    "is_high_value_transaction",

    # Time features
    "month",
    "day",
    "is_weekend",
    "quarter",
    "week",

    # Entity features
    "entity_transaction_count",
    "entity_total_amount",
    "entity_avg_amount",
    "entity_days_active",
    "entity_transactions_per_day",
    "entity_debit_count",
    "entity_credit_count",
    "entity_avg_debit",
    "entity_avg_credit",

    # Behavioural flags
    "is_regular_entity",
    "is_long_term_entity",
    "is_weekend_spending",
    "is_recurring_entity",
    "is_subscription",
    "is_first_transaction",
    "is_recent_transaction",

    # Running totals
    "running_expense",
    "net_cash_flow",
    "monthly_spend",
    "monthly_savings",
]

TARGET_COL = "potential_leakage"


# ----------------------------------------------------------
# Load and prepare data
# ----------------------------------------------------------

def load_data():
    path = FEATURES_DIR / "transactions_scored.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}\n"
            f"Run 03_leakage_scoring.py first."
        )

    df = pd.read_csv(path)

    # Convert boolean columns to int (ML models need numeric)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Keep only columns that exist in the dataframe
    features = [c for c in FEATURE_COLS if c in df.columns]
    missing  = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        logging.warning(f"Missing feature columns (skipped): {missing}")

    X = df[features].fillna(0)
    y = df[TARGET_COL]

    logging.info(f"Loaded {len(df)} transactions.")
    logging.info(f"Features used: {len(features)}")
    logging.info(f"Positive class (leakage): {y.sum()} ({round(y.mean()*100,1)}%)")
    logging.info(f"Negative class (normal):  {(y==0).sum()}")

    return df, X, y, features


# ----------------------------------------------------------
# Evaluate a model using StratifiedKFold cross-validation
# We use cross_val_predict so every sample gets a prediction
# from a fold where it was in the TEST set — no data leakage.
# ----------------------------------------------------------

def evaluate_model(name, pipeline, X, y, cv):
    logging.info(f"  Training {name}...")

    # Get predictions on test folds
    y_pred  = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    # Metrics
    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    f1        = f1_score(y, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y, y_proba)
    pr_auc    = average_precision_score(y, y_proba)
    cm        = confusion_matrix(y, y_pred)

    results = {
        "model"    : name,
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1_score" : round(f1,        4),
        "roc_auc"  : round(roc_auc,   4),
        "pr_auc"   : round(pr_auc,    4),
        "tp"       : int(cm[1, 1]),
        "fp"       : int(cm[0, 1]),
        "tn"       : int(cm[0, 0]),
        "fn"       : int(cm[1, 0]),
    }

    logging.info(f"    Precision : {precision:.4f}")
    logging.info(f"    Recall    : {recall:.4f}")
    logging.info(f"    F1        : {f1:.4f}")
    logging.info(f"    PR-AUC    : {pr_auc:.4f}")
    logging.info(f"    ROC-AUC   : {roc_auc:.4f}")
    logging.info(f"    Confusion : TP={cm[1,1]} FP={cm[0,1]} TN={cm[0,0]} FN={cm[1,0]}")

    return results, y_pred, y_proba


# ----------------------------------------------------------
# Feature importance from Random Forest
# ----------------------------------------------------------

def get_feature_importance(X, y, features):
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
    # Fit on full data just for importance (not for evaluation)
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_res, y_res = smote.fit_resample(X, y)
    rf.fit(X_res, y_res)

    importance_df = pd.DataFrame({
        "feature"   : features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    return importance_df


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    logging.info("=" * 60)
    logging.info("STARTING MODEL TRAINING")
    logging.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, X, y, features = load_data()

    # Stratified K-Fold — preserves class ratio in every fold
    # 5 folds with 21 positives = ~4 positives per fold in test
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------
    # Define pipelines
    # SMOTE is inside the pipeline so it only applies
    # to TRAINING folds — never to test folds.
    # This is the correct way to handle imbalance.
    # --------------------------------------------------

    # Positive class ratio for XGBoost
    neg_count = int((y == 0).sum())
    pos_count = int(y.sum())
    scale_pos = round(neg_count / pos_count, 2)
    logging.info(f"XGBoost scale_pos_weight: {scale_pos}")

    pipelines = {
        "Logistic Regression": ImbPipeline([
            ("smote",     SMOTE(random_state=42, k_neighbors=2)),
            ("scaler",    StandardScaler()),
            ("model",     LogisticRegression(
                              class_weight="balanced",
                              max_iter=1000,
                              random_state=42
                          ))
        ]),

        "Random Forest": ImbPipeline([
            ("smote",  SMOTE(random_state=42, k_neighbors=2)),
            ("model",  RandomForestClassifier(
                           n_estimators=100,
                           class_weight="balanced",
                           random_state=42
                       ))
        ]),

        "XGBoost": ImbPipeline([
            ("smote",  SMOTE(random_state=42, k_neighbors=2)),
            ("model",  XGBClassifier(
                           scale_pos_weight=scale_pos,
                           n_estimators=100,
                           learning_rate=0.1,
                           max_depth=4,
                           random_state=42,
                           eval_metric="logloss",
                           verbosity=0
                       ))
        ]),
    }

    # --------------------------------------------------
    # Train and evaluate all models
    # --------------------------------------------------

    logging.info("")
    logging.info("Training models with 5-fold StratifiedKFold CV...")
    logging.info("")

    all_results  = []
    all_preds    = {}
    all_probas   = {}

    for name, pipeline in pipelines.items():
        results, y_pred, y_proba = evaluate_model(name, pipeline, X, y, cv)
        all_results.append(results)
        all_preds[name]  = y_pred
        all_probas[name] = y_proba
        logging.info("")

    # --------------------------------------------------
    # Results table
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("f1_score", ascending=False)

    logging.info("=" * 60)
    logging.info("MODEL COMPARISON")
    logging.info("=" * 60)
    print(results_df[["model", "precision", "recall",
                       "f1_score", "pr_auc", "roc_auc"]].to_string(index=False))

    best_model = results_df.iloc[0]["model"]
    logging.info(f"\nBest model by F1: {best_model}")

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    logging.info("")
    logging.info("Computing feature importance (Random Forest)...")
    importance_df = get_feature_importance(X, y, features)

    logging.info("Top 10 most important features:")
    print(importance_df.head(10).to_string(index=False))

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Add best model predictions to full dataframe
    best_pred  = all_preds[best_model]
    best_proba = all_probas[best_model]

    df["predicted_leakage"] = best_pred
    df["predicted_proba"]   = best_proba.round(4)

    # Flag correctly identified leakages
    df["correctly_identified"] = (
        (df["potential_leakage"] == 1) &
        (df["predicted_leakage"] == 1)
    )

    df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    logging.info("")
    logging.info("=" * 60)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"model_results.csv     -> {OUTPUT_DIR}")
    logging.info(f"feature_importance.csv-> {OUTPUT_DIR}")
    logging.info(f"predictions.csv       -> {OUTPUT_DIR}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
