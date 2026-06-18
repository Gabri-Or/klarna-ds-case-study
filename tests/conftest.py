from collections.abc import Callable
from datetime import date

import polars as pl
import pytest


@pytest.fixture
def make_raw_df() -> Callable[[bool], pl.DataFrame]:
    def _make_raw_df(include_invalid_card: bool = False) -> pl.DataFrame:
        b_card_month = 13 if include_invalid_card else 12
        b_card_year = 2022 if include_invalid_card else 2027

        return pl.DataFrame(
            [
                {
                    "loan_id": "loan_a",
                    "loan_issue_date": date(2023, 6, 1),
                    "loan_amount": 100,
                    "amount_outstanding_14d": 0,
                    "amount_outstanding_21d": 0,
                    "card_expiry_month": None,
                    "card_expiry_year": None,
                    "existing_klarna_debt": None,
                    "num_active_loans": 0,
                    "days_since_first_loan": -1,
                    "new_exposure_7d": 10,
                    "new_exposure_14d": 20,
                    "num_confirmed_payments_3m": 0,
                    "num_confirmed_payments_6m": 1,
                    "num_failed_payments_3m": 0,
                    "num_failed_payments_6m": 0,
                    "num_failed_payments_1y": 0,
                    "amount_repaid_14d": 0,
                    "amount_repaid_1m": 0,
                    "amount_repaid_3m": 10,
                    "amount_repaid_6m": 20,
                    "amount_repaid_1y": 30,
                    "merchant_group": "Group A",
                    "merchant_category": "Common",
                },
                {
                    "loan_id": "loan_b",
                    "loan_issue_date": date(2023, 7, 15),
                    "loan_amount": 200,
                    "amount_outstanding_14d": 50,
                    "amount_outstanding_21d": 25,
                    "card_expiry_month": b_card_month,
                    "card_expiry_year": b_card_year,
                    "existing_klarna_debt": -5,
                    "num_active_loans": 0,
                    "days_since_first_loan": 0,
                    "new_exposure_7d": 0,
                    "new_exposure_14d": 0,
                    "num_confirmed_payments_3m": 0,
                    "num_confirmed_payments_6m": 0,
                    "num_failed_payments_3m": 0,
                    "num_failed_payments_6m": 1,
                    "num_failed_payments_1y": 1,
                    "amount_repaid_14d": 0,
                    "amount_repaid_1m": 5,
                    "amount_repaid_3m": 5,
                    "amount_repaid_6m": 10,
                    "amount_repaid_1y": 10,
                    "merchant_group": "Group B",
                    "merchant_category": "Rare",
                },
                {
                    "loan_id": "loan_c",
                    "loan_issue_date": date(2023, 8, 20),
                    "loan_amount": 300,
                    "amount_outstanding_14d": 10,
                    "amount_outstanding_21d": 0,
                    "card_expiry_month": 6,
                    "card_expiry_year": 2026,
                    "existing_klarna_debt": 10,
                    "num_active_loans": 1,
                    "days_since_first_loan": 100,
                    "new_exposure_7d": 5,
                    "new_exposure_14d": 5,
                    "num_confirmed_payments_3m": 1,
                    "num_confirmed_payments_6m": 2,
                    "num_failed_payments_3m": 0,
                    "num_failed_payments_6m": 0,
                    "num_failed_payments_1y": 1,
                    "amount_repaid_14d": 1,
                    "amount_repaid_1m": 2,
                    "amount_repaid_3m": 3,
                    "amount_repaid_6m": 4,
                    "amount_repaid_1y": 5,
                    "merchant_group": "Group A",
                    "merchant_category": "Common",
                },
            ]
        )

    return _make_raw_df
