import pandas as pd

df = pd.read_csv('data/features/transactions_scored.csv')

print("=== OVERALL SCORE SUMMARY ===")
print(df['leakage_score'].describe().round(2))

print("\n=== RISK BAND BREAKDOWN ===")
print(df['risk_band'].value_counts())

print("\n=== LEAKAGE vs NON-LEAKAGE SCORES ===")
print(df.groupby('potential_leakage')['leakage_score'].describe().round(2))

print("\n=== TOP 10 HIGHEST SCORED TRANSACTIONS ===")
print(df.nlargest(10, 'leakage_score')[
    ['txn_date','category','dr_amount','leakage_score','risk_band']
].to_string())

print("\n=== AVERAGE SCORE BY MONTH ===")
print(df.groupby('month_name')['leakage_score'].mean().round(1).sort_values(ascending=False))

print("\n=== SCORE BREAKDOWN BY FACTOR ===")
factors = ['score_category','score_amount','score_frequency',
           'score_timing','score_recency']
print(df[factors].mean().round(2))