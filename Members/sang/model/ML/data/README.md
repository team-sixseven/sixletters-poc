# Clinical ML data

The shared source of truth is `../../basic.yaml`. Clinical features come from `../../datas/clincal/clinical_clean.csv` and split assignment comes only from `../../datas/clincal/split_team_v1.csv`. IDs are read as strings so leading zeros and `P` prefixes are preserved.

Column names are whitespace-trimmed in memory. Duplicate names created by trimming are rejected. The explicit 23-feature allowlist and seven categorical columns live in `basic.yaml`; patient ID, `SCD`, and split values are excluded from model features. Imputation, scaling, and one-hot encoding for XGBoost and Random Forest are fit on train only. CatBoost receives the declared categorical columns as string categories.
