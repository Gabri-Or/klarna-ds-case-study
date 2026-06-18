from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mlcasestudy Final.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "loans_processed.parquet"

MIN_MERCHANT_CATEGORY_COUNT = 100
UNKNOWN_CATEGORY = "Unknown"
OTHER_CATEGORY = "Other"


RAW_COLUMNS = [
    "loan_id",
    "loan_issue_date",
    "loan_amount",
    "amount_outstanding_14d",
    "amount_outstanding_21d",
    "card_expiry_month",
    "card_expiry_year",
    "existing_klarna_debt",
    "num_active_loans",
    "days_since_first_loan",
    "new_exposure_7d",
    "new_exposure_14d",
    "num_confirmed_payments_3m",
    "num_confirmed_payments_6m",
    "num_failed_payments_3m",
    "num_failed_payments_6m",
    "num_failed_payments_1y",
    "amount_repaid_14d",
    "amount_repaid_1m",
    "amount_repaid_3m",
    "amount_repaid_6m",
    "amount_repaid_1y",
    "merchant_group",
    "merchant_category",
]

RAW_COLUMN_DTYPES = {
    "loan_id": pl.String,
    "loan_issue_date": pl.Date,
    "loan_amount": pl.Int64,
    "amount_outstanding_14d": pl.Int64,
    "amount_outstanding_21d": pl.Int64,
    "card_expiry_month": pl.Int64,
    "card_expiry_year": pl.Int64,
    "existing_klarna_debt": pl.Int64,
    "num_active_loans": pl.Int64,
    "days_since_first_loan": pl.Int64,
    "new_exposure_7d": pl.Int64,
    "new_exposure_14d": pl.Int64,
    "num_confirmed_payments_3m": pl.Int64,
    "num_confirmed_payments_6m": pl.Int64,
    "num_failed_payments_3m": pl.Int64,
    "num_failed_payments_6m": pl.Int64,
    "num_failed_payments_1y": pl.Int64,
    "amount_repaid_14d": pl.Int64,
    "amount_repaid_1m": pl.Int64,
    "amount_repaid_3m": pl.Int64,
    "amount_repaid_6m": pl.Int64,
    "amount_repaid_1y": pl.Int64,
    "merchant_group": pl.String,
    "merchant_category": pl.String,
}

PROCESSED_COLUMNS = [
    "loan_id",
    "loan_issue_date",
    "loan_amount",
    "card_expiry_month",
    "card_expiry_year",
    "card_expiry_missing",
    "existing_klarna_debt",
    "existing_klarna_debt_missing",
    "existing_klarna_debt_was_negative",
    "num_active_loans",
    "days_since_first_loan",
    "days_since_first_loan_was_negative",
    "has_prior_loan",
    "new_exposure_7d",
    "new_exposure_14d",
    "num_confirmed_payments_3m",
    "num_confirmed_payments_6m",
    "num_failed_payments_3m",
    "num_failed_payments_6m",
    "num_failed_payments_1y",
    "amount_repaid_14d",
    "amount_repaid_1m",
    "amount_repaid_3m",
    "amount_repaid_6m",
    "amount_repaid_1y",
    "merchant_group",
    "merchant_category",
    "loan_issue_year",
    "loan_issue_month",
    "loan_issue_quarter",
    "loan_issue_weekday",
    "existing_debt_to_loan_amount",
    "new_exposure_14d_to_loan_amount",
    "amount_repaid_1y_to_loan_amount",
    "default_14d",
    "default_21d",
]
