# ML Development Summary

Predicting probability of default on Klarna Pay Later loans using a calibrated XGBoost model. This document traces the development from target definition through deployment.

## 1. Problem Framing & Target Definition

The dataset contains ~100k loans issued over several months in 2023, each with outstanding balances at 14 and 21 days after issuance.

We defined default as `amount_outstanding_21d > 0`, yielding a ~5% positive rate. The 21-day threshold was preferred over 14 days because:

- It represents the final repayment deadline before Klarna declares a default.
- On average, nearly half the outstanding balance at 14 days gets recovered in the extra week, meaning a 14-day target would overcount defaults.

The EDA also revealed a decreasing default rate trend over time, but the observation window was too short to distinguish trend from seasonality — so the data is treated as i.i.d. for modeling purposes.

**References:**
- [notebooks/01_eda_and_quality.py](../notebooks/01_eda_and_quality.py) — full exploratory analysis
- [reports/01_eda_and_quality.pdf](01_eda_and_quality.pdf) — rendered report

## 2. Data Processing

A reproducible pipeline (`just process`) loads the raw CSV, validates it against a pandera schema, and produces a clean parquet file. Steps:

1. Load with schema enforcement and parse dates
2. Deduplicate (exact rows and repeated `loan_id` values)
3. Derive target flags (`default_14d`, `default_21d`)
4. Impute card expiry (mode for month, median for year) for missing/invalid rows
5. Clamp `existing_klarna_debt` negatives and `days_since_first_loan` sentinels to zero
6. Collapse rare merchant categories (< threshold count) into an "Other" bucket
7. Cast final columns and validate against the processed schema

**References:**
- [src/klarna_ds_case_study/data_processing/pipeline.py](../src/klarna_ds_case_study/data_processing/pipeline.py) — pipeline logic
- [src/klarna_ds_case_study/data_processing/schemas.py](../src/klarna_ds_case_study/data_processing/schemas.py) — pandera schemas
- [src/klarna_ds_case_study/data_processing/config.py](../src/klarna_ds_case_study/data_processing/config.py) — paths and constants

## 3. Modeling

The modeling notebook iterates through progressively stronger approaches:

1. **Logistic regression** (numeric features only, StandardScaler) — baseline for discrimination and calibration.
2. **XGBoost with default hyperparameters** — adds categorical features natively via `enable_categorical`, uses `scale_pos_weight` for class imbalance.
3. **Optuna-tuned XGBoost** — 50-trial search over 8 hyperparameters, optimizing mean CV ROC-AUC with 5-fold stratified splits.
4. **Isotonic calibration** — `CalibratedClassifierCV` wrapping the tuned model to correct probability estimates.

Metrics tracked: ROC-AUC, PR-AUC, log loss, Brier score, and Expected Calibration Error (ECE).

The production training pipeline (`just train`) reproduces this flow with MLflow tracking and model registration.

**References:**
- [notebooks/02_modeling_experiments.py](../notebooks/02_modeling_experiments.py) — interactive experimentation
- [reports/02_modeling_experiments.pdf](02_modeling_experiments.pdf) — rendered report
- [src/klarna_ds_case_study/training/pipeline.py](../src/klarna_ds_case_study/training/pipeline.py) — production training
- [src/klarna_ds_case_study/training/config.py](../src/klarna_ds_case_study/training/config.py) — search space and constants
- [src/klarna_ds_case_study/training/tracking.py](../src/klarna_ds_case_study/training/tracking.py) — MLflow helpers

## 4. Evaluation & Business Impact

Statistical metrics alone don't capture business value. The portfolio analysis evaluates models at the decision level:

- **Cross-validated predictions** (`cross_val_predict`) give out-of-fold probabilities for every loan without leakage.
- **Threshold sweep** — for each probability cutoff, compute approval rate, default rate among approved loans, and fraction of defaults caught.
- **Profit simulation** — models revenue as a merchant fee on approved volume and loss as the actual outstanding balance at 21 days (adjustable recovery rate). Sweeps thresholds to find the profit-maximizing operating point.

Both logistic regression and XGBoost are compared against a perfect-information baseline.

**References:**
- [notebooks/03_portfolio_analysis.py](../notebooks/03_portfolio_analysis.py) — threshold and profit analysis
- [reports/03_portfolio_analysis.pdf](03_portfolio_analysis.pdf) — rendered report

## 5. Deployment

The final calibrated model is saved to `data/models/serving_model/` in MLflow sklearn format and committed to the repo. A FastAPI application loads it at startup and exposes a `/predict` endpoint returning `default_probability`.

Run with `just serve`, test with `just predict` or `just test`.

**References:**
- [src/klarna_ds_case_study/serving/app.py](../src/klarna_ds_case_study/serving/app.py) — prediction API
- [tests/](../tests/) — unit and integration tests

## 6. Future Directions

Two additional analyses explored improvements that were not incorporated into the serving model but inform next steps:

### Feature selection ([notebooks/04_feature_analysis.py](../notebooks/04_feature_analysis.py))

- XGBoost gain, mean |SHAP|, and permutation importance all agree on ~10 low-value features (mostly shorter-window duplicates of longer-window signals).
- Correlation analysis confirms heavy redundancy among `amount_repaid_*`, `new_exposure_*`, and `num_confirmed_payments_*` families.
- Dropping these features causes no measurable performance loss — a minimal 10-feature set is viable.
- Engineered features (debt-to-loan ratio, failed-to-confirmed ratio, repayment deltas, loan intensity) showed no improvement, confirming raw features already capture the available signal.

### Interpretability & monotone constraints ([notebooks/05_shap_and_monotonicity.py](../notebooks/05_shap_and_monotonicity.py))

- SHAP dependence plots revealed some features used in counter-intuitive directions by the unconstrained model.
- Enforcing monotone constraints (e.g. more failed payments → higher risk, more confirmed payments → lower risk) preserves performance while guaranteeing alignment with domain expectations.
- If regulatory compliance or explainability requirements arise, monotone constraints are a zero-cost improvement.

**References:**
- [reports/04_feature_analysis.pdf](04_feature_analysis.pdf)
- [reports/05_shap_and_monotonicity.pdf](05_shap_and_monotonicity.pdf)

