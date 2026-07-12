-- =============================================================
-- 02_analysis_queries.sql
-- 15+ analytical queries for Micro Leakage Detection Project
-- Database: finance_leakage | Table: transactions
-- =============================================================


-- =============================================================
-- QUERY 1: Total income vs expenses vs savings by month
-- Basic financial health overview
-- =============================================================

SELECT
    month_name,
    month,
    ROUND(SUM(cr_amount), 2)                          AS total_income,
    ROUND(SUM(dr_amount), 2)                          AS total_expenses,
    ROUND(SUM(cr_amount) - SUM(dr_amount), 2)         AS net_savings,
    ROUND(
        (SUM(cr_amount) - SUM(dr_amount))
        / NULLIF(SUM(cr_amount), 0) * 100, 2
    )                                                  AS savings_rate_pct
FROM transactions
GROUP BY month_name, month
ORDER BY month;


-- =============================================================
-- QUERY 2: Top 5 spending categories by total amount
-- Where is money actually going?
-- =============================================================

SELECT
    category,
    COUNT(*)                        AS transaction_count,
    ROUND(SUM(dr_amount), 2)        AS total_spent,
    ROUND(AVG(dr_amount), 2)        AS avg_per_transaction,
    ROUND(MAX(dr_amount), 2)        AS largest_transaction
FROM transactions
WHERE direction = 'DEBIT'
  AND dr_amount > 0
GROUP BY category
ORDER BY total_spent DESC
LIMIT 5;


-- =============================================================
-- QUERY 3: Leakage transactions with risk band breakdown
-- Core leakage analysis
-- =============================================================

SELECT
    risk_band,
    COUNT(*)                                    AS transaction_count,
    ROUND(SUM(dr_amount), 2)                    AS total_amount,
    ROUND(AVG(leakage_score), 2)                AS avg_leakage_score,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*))
          OVER (), 2)                           AS pct_of_all_transactions
FROM transactions
GROUP BY risk_band
ORDER BY avg_leakage_score DESC;


-- =============================================================
-- QUERY 4: Monthly leakage trend
-- Is leakage increasing or decreasing over time?
-- =============================================================

SELECT
    month_name,
    month,
    COUNT(*)                            AS total_transactions,
    SUM(potential_leakage)              AS leakage_count,
    ROUND(SUM(CASE WHEN potential_leakage = 1
                   THEN dr_amount ELSE 0 END), 2)   AS leakage_amount,
    ROUND(SUM(potential_leakage) * 100.0
          / COUNT(*), 2)                AS leakage_rate_pct,
    -- Month over month change in leakage amount
    ROUND(SUM(CASE WHEN potential_leakage = 1
                   THEN dr_amount ELSE 0 END)
        - LAG(SUM(CASE WHEN potential_leakage = 1
                       THEN dr_amount ELSE 0 END))
          OVER (ORDER BY month), 2)     AS mom_leakage_change
FROM transactions
GROUP BY month_name, month
ORDER BY month;


-- =============================================================
-- QUERY 5: Payment mode leakage analysis
-- Which payment channel has highest leakage rate?
-- =============================================================

WITH mode_stats AS (
    SELECT
        payment_mode,
        COUNT(*)                                        AS total_txn,
        SUM(potential_leakage)                          AS leakage_txn,
        ROUND(SUM(dr_amount), 2)                        AS total_spent,
        ROUND(AVG(leakage_score), 2)                    AS avg_leakage_score
    FROM transactions
    GROUP BY payment_mode
)
SELECT
    payment_mode,
    total_txn,
    leakage_txn,
    total_spent,
    avg_leakage_score,
    ROUND(leakage_txn * 100.0 / NULLIF(total_txn, 0), 2) AS leakage_rate_pct
FROM mode_stats
ORDER BY leakage_rate_pct DESC;


-- =============================================================
-- QUERY 6: Weekend vs weekday spending behaviour
-- Do you spend more impulsively on weekends?
-- =============================================================

SELECT
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END  AS day_type,
    COUNT(*)                                                  AS transaction_count,
    ROUND(SUM(dr_amount), 2)                                 AS total_spent,
    ROUND(AVG(dr_amount), 2)                                 AS avg_transaction,
    SUM(potential_leakage)                                    AS leakage_count,
    ROUND(AVG(leakage_score), 2)                             AS avg_risk_score
FROM transactions
WHERE direction = 'DEBIT'
GROUP BY is_weekend
ORDER BY avg_risk_score DESC;


-- =============================================================
-- QUERY 7: Top 10 highest risk transactions
-- Which specific transactions should you worry about?
-- =============================================================

SELECT
    txn_no,
    txn_date,
    category,
    entity_id,
    ROUND(dr_amount, 2)         AS amount,
    leakage_score,
    risk_band,
    ROUND(predicted_proba, 4)   AS ml_leakage_probability
FROM transactions
WHERE direction = 'DEBIT'
ORDER BY leakage_score DESC
LIMIT 10;


-- =============================================================
-- QUERY 8: Recurring entity analysis
-- Who do you keep paying repeatedly?
-- =============================================================

SELECT
    entity_id,
    entity_type,
    category,
    entity_transaction_count                        AS times_paid,
    ROUND(entity_total_amount, 2)                   AS total_paid,
    ROUND(entity_avg_debit, 2)                      AS avg_payment,
    entity_days_active                              AS days_span,
    ROUND(entity_transactions_per_day, 3)           AS payments_per_day,
    MAX(leakage_score)                              AS max_risk_score
FROM transactions
WHERE entity_type IN ('PERSON', 'MERCHANT')
  AND direction = 'DEBIT'
GROUP BY entity_id, entity_type, category,
         entity_transaction_count, entity_total_amount,
         entity_avg_debit, entity_days_active,
         entity_transactions_per_day
HAVING entity_transaction_count >= 3
ORDER BY total_paid DESC
LIMIT 10;


-- =============================================================
-- QUERY 9: Monthly savings potential
-- How much could you save by cutting top leakage in half?
-- =============================================================

WITH ranked AS (
    SELECT
        month,
        month_name,
        dr_amount,
        leakage_score,
        NTILE(10) OVER (ORDER BY leakage_score DESC) AS score_decile
    FROM transactions
    WHERE direction = 'DEBIT'
)
SELECT
    month_name,
    month,
    ROUND(SUM(dr_amount), 2)            AS total_high_risk_spend,
    ROUND(SUM(dr_amount) * 0.5, 2)      AS potential_savings_if_halved,
    COUNT(*)                             AS high_risk_transactions
FROM ranked
WHERE score_decile = 1
GROUP BY month_name, month
ORDER BY month;


-- =============================================================
-- QUERY 10: ML model performance summary
-- How well did the model identify leakage?
-- =============================================================

SELECT
    COUNT(*)                                                AS total_transactions,
    SUM(potential_leakage)                                  AS actual_leakage,
    SUM(predicted_leakage)                                  AS predicted_leakage,
    SUM(CASE WHEN potential_leakage = 1
             AND predicted_leakage = 1 THEN 1 ELSE 0 END)  AS true_positives,
    SUM(CASE WHEN potential_leakage = 0
             AND predicted_leakage = 1 THEN 1 ELSE 0 END)  AS false_positives,
    SUM(CASE WHEN potential_leakage = 1
             AND predicted_leakage = 0 THEN 1 ELSE 0 END)  AS false_negatives,
    SUM(CASE WHEN potential_leakage = 0
             AND predicted_leakage = 0 THEN 1 ELSE 0 END)  AS true_negatives,
    ROUND(
        SUM(CASE WHEN potential_leakage = 1
                 AND predicted_leakage = 1 THEN 1 ELSE 0 END) * 100.0
        / NULLIF(SUM(potential_leakage), 0), 2
    )                                                       AS recall_pct
FROM transactions;


-- =============================================================
-- QUERY 11: Running cash flow trend
-- Window function: cumulative income vs expense over time
-- =============================================================

SELECT
    txn_date,
    txn_no,
    direction,
    ROUND(dr_amount, 2)             AS debit,
    ROUND(cr_amount, 2)             AS credit,
    ROUND(running_expense, 2)       AS cumulative_expense,
    ROUND(running_income, 2)        AS cumulative_income,
    ROUND(net_cash_flow, 2)         AS net_cash_flow,
    -- Daily running balance trend
    ROUND(SUM(cr_amount - dr_amount)
          OVER (ORDER BY txn_date, txn_no
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ), 2)                     AS running_balance
FROM transactions
ORDER BY txn_date, txn_no
LIMIT 20;


-- =============================================================
-- QUERY 12: Category leakage ranking by month (window function)
-- Which category is the biggest leak each month?
-- =============================================================

WITH monthly_category AS (
    SELECT
        month_name,
        month,
        category,
        ROUND(SUM(dr_amount), 2)    AS category_spend,
        SUM(potential_leakage)      AS leakage_count
    FROM transactions
    WHERE direction = 'DEBIT'
    GROUP BY month_name, month, category
),
ranked AS (
    SELECT
        month_name,
        month,
        category,
        category_spend,
        leakage_count,
        RANK() OVER (
            PARTITION BY month
            ORDER BY category_spend DESC
        ) AS spend_rank
    FROM monthly_category
)
SELECT *
FROM ranked
WHERE spend_rank <= 3
ORDER BY month, spend_rank;


-- =============================================================
-- QUERY 13: Subscription drain detection
-- Silent recurring costs adding up
-- =============================================================

SELECT
    entity_id,
    COUNT(*)                            AS charge_count,
    ROUND(MIN(dr_amount), 2)            AS min_charge,
    ROUND(MAX(dr_amount), 2)            AS max_charge,
    ROUND(SUM(dr_amount), 2)            AS total_charged,
    MIN(txn_date)                       AS first_charge,
    MAX(txn_date)                       AS last_charge,
    MAX(txn_date) - MIN(txn_date)       AS days_subscribed,
    ROUND(AVG(leakage_score), 2)        AS avg_risk_score
FROM transactions
WHERE category = 'Subscription'
  AND direction = 'DEBIT'
GROUP BY entity_id
ORDER BY total_charged DESC;


-- =============================================================
-- QUERY 14: Day of week spending pattern
-- Which day do you spend most?
-- =============================================================

SELECT
    weekday,
    COUNT(*)                            AS transaction_count,
    ROUND(SUM(dr_amount), 2)            AS total_spent,
    ROUND(AVG(dr_amount), 2)            AS avg_spent,
    SUM(potential_leakage)              AS leakage_transactions,
    ROUND(AVG(leakage_score), 2)        AS avg_risk_score,
    -- Rank days by total spend
    RANK() OVER (
        ORDER BY SUM(dr_amount) DESC
    )                                   AS spend_rank
FROM transactions
WHERE direction = 'DEBIT'
GROUP BY weekday
ORDER BY total_spent DESC;


-- =============================================================
-- QUERY 15: High value transaction alert
-- Transactions above 75th percentile of your spending
-- =============================================================

WITH percentiles AS (
    SELECT
        PERCENTILE_CONT(0.75) WITHIN GROUP
            (ORDER BY dr_amount)        AS p75,
        PERCENTILE_CONT(0.90) WITHIN GROUP
            (ORDER BY dr_amount)        AS p90
    FROM transactions
    WHERE direction = 'DEBIT'
      AND dr_amount > 0
)
SELECT
    t.txn_date,
    t.txn_no,
    t.category,
    t.entity_id,
    ROUND(t.dr_amount, 2)               AS amount,
    ROUND(p.p75::NUMERIC, 2)            AS p75_threshold,
    ROUND(p.p90::NUMERIC, 2)            AS p90_threshold,
    CASE
        WHEN t.dr_amount >= p.p90 THEN 'Top 10% Spend'
        WHEN t.dr_amount >= p.p75 THEN 'Top 25% Spend'
    END                                 AS alert_level,
    t.leakage_score,
    t.risk_band
FROM transactions t, percentiles p
WHERE t.direction = 'DEBIT'
  AND t.dr_amount >= p.p75
ORDER BY t.dr_amount DESC;


-- =============================================================
-- QUERY 16: First vs repeat merchant behaviour
-- Are first-time merchants riskier?
-- =============================================================

SELECT
    CASE WHEN is_first_transaction THEN 'First Visit'
         ELSE 'Repeat Visit' END            AS visit_type,
    COUNT(*)                                AS transaction_count,
    ROUND(AVG(dr_amount), 2)               AS avg_amount,
    ROUND(AVG(leakage_score), 2)           AS avg_risk_score,
    SUM(potential_leakage)                  AS leakage_count,
    ROUND(SUM(potential_leakage) * 100.0
          / NULLIF(COUNT(*), 0), 2)         AS leakage_rate_pct
FROM transactions
WHERE direction = 'DEBIT'
GROUP BY is_first_transaction
ORDER BY avg_risk_score DESC;


-- =============================================================
-- QUERY 17: March anomaly investigation
-- March had ₹1,36,537 in spending — 3x any other month
-- What drove it?
-- =============================================================

SELECT
    category,
    COUNT(*)                        AS transactions,
    ROUND(SUM(dr_amount), 2)        AS total_spent,
    ROUND(AVG(dr_amount), 2)        AS avg_amount,
    ROUND(MAX(dr_amount), 2)        AS largest_transaction,
    SUM(potential_leakage)          AS leakage_count
FROM transactions
WHERE month = 3
  AND direction = 'DEBIT'
GROUP BY category
ORDER BY total_spent DESC;
