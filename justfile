set dotenv-load

default:
    @just --list

setup:
    uv sync
    uv run pre-commit install

process:
    uv run python -m klarna_ds_case_study.data_processing.main

train:
    uv run python -m klarna_ds_case_study.training.main

serve:
    uv run uvicorn klarna_ds_case_study.serving.app:app --reload

predict:
    @curl -s -X POST http://127.0.0.1:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"loan_amount":5000,"card_expiry_month":6,"card_expiry_year":2027,"existing_klarna_debt":0,"num_active_loans":1,"days_since_first_loan":120,"new_exposure_7d":0,"new_exposure_14d":0,"num_confirmed_payments_3m":3,"num_confirmed_payments_6m":5,"num_failed_payments_3m":0,"num_failed_payments_6m":0,"num_failed_payments_1y":0,"amount_repaid_14d":2000,"amount_repaid_1m":4000,"amount_repaid_3m":8000,"amount_repaid_6m":12000,"amount_repaid_1y":15000,"merchant_group":"Entertainment","merchant_category":"Event Tickets"}' | python -m json.tool

test:
    uv run pytest

mlflow-ui:
    uv run mlflow ui

notebook file="01_eda_and_quality.py":
    uv run marimo edit notebooks/{{file}}

