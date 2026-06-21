# Klarna Data Science Classification Case Study

Predicts the probability of default on Pay Later loans using a calibrated XGBoost model.

## Quick Start

Prerequisites: Python ≥3.12 and [uv](https://docs.astral.sh/uv/).

```zsh
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install just (command runner)
uv tool install just

# Install all dependencies (including dev tools) and the project itself
uv sync

# Start the prediction API
just serve
```

The API is now running at http://127.0.0.1:8000. Interactive docs at http://127.0.0.1:8000/docs.

## Making Predictions

In another terminal:

```zsh
just predict
```

Or manually:

```zsh
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 5000,
    "card_expiry_month": 6,
    "card_expiry_year": 2027,
    "existing_klarna_debt": 0,
    "num_active_loans": 1,
    "days_since_first_loan": 120,
    "new_exposure_7d": 0,
    "new_exposure_14d": 0,
    "num_confirmed_payments_3m": 3,
    "num_confirmed_payments_6m": 5,
    "num_failed_payments_3m": 0,
    "num_failed_payments_6m": 0,
    "num_failed_payments_1y": 0,
    "amount_repaid_14d": 2000,
    "amount_repaid_1m": 4000,
    "amount_repaid_3m": 8000,
    "amount_repaid_6m": 12000,
    "amount_repaid_1y": 15000,
    "merchant_group": "Entertainment",
    "merchant_category": "Event Tickets"
  }'
```

Response: `{"default_probability": 0.0399...}`

## Available Commands

All commands are defined in the `justfile`. Run `just` to see the full list.

| Command | Description |
| --- | --- |
| `just setup` | Install dependencies and pre-commit hooks |
| `just serve` | Start the prediction API with hot-reload |
| `just predict` | Send a sample request to the running API |
| `just test` | Run the test suite |
| `just process` | Run the data processing pipeline |
| `just train` | Run model training (Optuna + MLflow) |
| `just mlflow-ui` | Launch the MLflow experiment browser |
| `just notebook` | Open a marimo notebook |

## How It Works

The trained model is saved to `data/models/serving_model/` and committed to the repo. The API loads it directly at startup — no MLflow server required.

To use a model from the MLflow registry instead:

```zsh
MODEL_URI="models:/loan-default-prediction/latest" just serve
```

## Tests

```zsh
just test
```

## Notebooks

Interactive marimo notebooks for EDA, modeling experiments, and portfolio analysis:

```zsh
just notebook                              # opens 01_eda_and_quality.py
just notebook file=02_modeling_experiments.py
just notebook file=03_portfolio_analysis.py
```

## Data Dictionary

| Feature | Definition |
| --- | --- |
| `loan_id` | A random, distinct ID for the loan |
| `loan_issue_date` | The date the loan was issued |
| `loan_amount` | The value of the loan being underwritten |
| `amount_outstanding_14d` | How much of the loan remained unpaid 14 days after the loan was issued |
| `amount_outstanding_21d` | How much of the loan remained unpaid 21 days after the loan was issued |
| `card_expiry_month` | The month the consumer's payment card will expire |
| `card_expiry_year` | The year the consumer's payment card will expire |
| `existing_klarna_debt` | How much the consumer already owed to Klarna at the time the loan was issued |
| `num_active_loans` | The number of loans the consumer needed to repay at the time the loan was issued |
| `days_since_first_loan` | How many days had passed since the consumer's first loan, as of the time the current loan was issued |
| `new_exposure_7d` | How much Klarna had lent the consumer 7 days before the loan was issued |
| `new_exposure_14d` | How much Klarna had lent the consumer 14 days before the loan was issued |
| `num_confirmed_payments_3m` | How many repayments towards other loans the consumer had made 3 months before the loan was issued |
| `num_confirmed_payments_6m` | How many repayments towards other loans the consumer had made 6 months before the loan was issued |
| `num_failed_payments_3m` | How many repayments towards other loans the consumer had missed 3 months before the loan was issued |
| `num_failed_payments_6m` | How many repayments towards other loans the consumer had missed 6 months before the loan was issued |
| `num_failed_payments_1y` | How many repayments towards other loans the consumer had missed 1 year before the loan was issued |
| `amount_repaid_14d` | How much the consumer had repaid in the 14 days before the loan was issued |
| `amount_repaid_1m` | How much the consumer had repaid in the month before the loan was issued |
| `amount_repaid_3m` | How much the consumer had repaid in the 3 months before the loan was issued |
| `amount_repaid_6m` | How much the consumer had repaid in the 6 months before the loan was issued |
| `amount_repaid_1y` | How much the consumer had repaid the year before the loan was issued |
| `merchant_group` | These features describe the merchant where the consumer is shopping |
| `merchant_category` | These features describe the merchant where the consumer is shopping |
