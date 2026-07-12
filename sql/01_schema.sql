-- =============================================================
-- 01_schema.sql
-- Creates the main transactions table in finance_leakage DB
-- =============================================================

-- Drop if exists (safe to re-run)
DROP TABLE IF EXISTS transactions CASCADE;

-- Main transactions table
CREATE TABLE transactions (
    -- Identity
    txn_no              VARCHAR(20) PRIMARY KEY,
    txn_date            DATE NOT NULL,

    -- Transaction details
    payment_mode        VARCHAR(20),
    direction           VARCHAR(10),
    category            VARCHAR(50),
    entity_type         VARCHAR(20),
    entity_id           VARCHAR(20),
    description         TEXT,

    -- Amounts
    dr_amount           NUMERIC(12,2) DEFAULT 0,
    cr_amount           NUMERIC(12,2) DEFAULT 0,
    amount              NUMERIC(12,2) DEFAULT 0,

    -- Time features
    year                INT,
    quarter             INT,
    month               INT,
    month_name          VARCHAR(15),
    week                INT,
    day                 INT,
    weekday             VARCHAR(15),
    day_type            VARCHAR(10),
    is_weekend          BOOLEAN,

    -- Amount flags
    is_small_transaction        BOOLEAN,
    is_medium_transaction       BOOLEAN,
    is_high_value_transaction   BOOLEAN,
    transaction_sign            INT,
    is_debit                    BOOLEAN,
    is_credit                   BOOLEAN,

    -- Entity aggregates
    entity_transaction_count    INT,
    entity_total_amount         NUMERIC(12,2),
    entity_avg_amount           NUMERIC(12,2),
    entity_days_active          INT,
    entity_transactions_per_day NUMERIC(8,3),
    entity_debit_count          INT,
    entity_credit_count         INT,
    entity_avg_debit            NUMERIC(12,2),
    entity_avg_credit           NUMERIC(12,2),
    entity_frequency            VARCHAR(15),

    -- Recency
    days_since_last_transaction INT,
    is_first_transaction        BOOLEAN,
    is_recent_transaction       BOOLEAN,

    -- Behavioural flags
    is_regular_entity           BOOLEAN,
    is_long_term_entity         BOOLEAN,
    is_weekend_spending         BOOLEAN,
    is_recurring_entity         BOOLEAN,
    is_subscription             BOOLEAN,

    -- Leakage features
    is_micro_leakage            BOOLEAN,
    is_large_shopping           BOOLEAN,
    is_food_habit               BOOLEAN,
    potential_leakage           INT,

    -- Leakage scores
    leakage_score               NUMERIC(6,2),
    score_category              INT,
    score_amount                INT,
    score_frequency             INT,
    score_timing                INT,
    score_recency               INT,
    risk_band                   VARCHAR(20),

    -- Running totals
    running_income              NUMERIC(12,2),
    running_expense             NUMERIC(12,2),
    net_cash_flow               NUMERIC(12,2),

    -- Monthly aggregates
    monthly_spend               NUMERIC(12,2),
    monthly_income              NUMERIC(12,2),
    monthly_savings             NUMERIC(12,2),

    -- ML predictions
    predicted_leakage           INT,
    predicted_proba             NUMERIC(6,4),
    correctly_identified        BOOLEAN
);

-- Indexes for query performance
CREATE INDEX idx_txn_date        ON transactions (txn_date);
CREATE INDEX idx_category        ON transactions (category);
CREATE INDEX idx_direction       ON transactions (direction);
CREATE INDEX idx_entity_id       ON transactions (entity_id);
CREATE INDEX idx_potential_leak  ON transactions (potential_leakage);
CREATE INDEX idx_risk_band       ON transactions (risk_band);
CREATE INDEX idx_month           ON transactions (month);
CREATE INDEX idx_payment_mode    ON transactions (payment_mode);

-- Confirm
SELECT 'Schema created successfully.' AS status;
