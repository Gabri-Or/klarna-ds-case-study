from collections.abc import Callable

import polars as pl
import pytest

from klarna_ds_case_study.data_processing.config import OTHER_CATEGORY
from klarna_ds_case_study.data_processing import pipeline
from klarna_ds_case_study.data_processing.pipeline import (
    clean_and_prepare_data,
)
from klarna_ds_case_study.data_processing.utils import (
    count_where,
    mode_value,
    median_value,
)
from klarna_ds_case_study.data_processing.schemas import (
    ProcessedLoans,
    schema_column_names,
)


def test_helper_functions_treat_nulls_and_empty_values() -> None:
    df = pl.DataFrame({"value": [None, 2, 2, 1], "flag": [True, None, False, True]})

    assert count_where(df, pl.col("flag")) == 2
    assert mode_value(df, "value") == 2
    assert median_value(pl.DataFrame({"empty": [None, None]}), "empty") is None


def test_clean_and_prepare_data_applies_eda_cleaning_rules(
    make_raw_df: Callable[[bool], pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "MIN_MERCHANT_CATEGORY_COUNT", 2)

    processed_df = clean_and_prepare_data(make_raw_df(True))

    assert processed_df.columns == schema_column_names(ProcessedLoans)
    assert processed_df.height == 3
    assert processed_df.null_count().sum_horizontal().item() == 0
    assert "amount_outstanding_14d" not in processed_df.columns
    assert "amount_outstanding_21d" not in processed_df.columns
    assert processed_df.schema["merchant_group"] == pl.Categorical
    assert processed_df.schema["merchant_category"] == pl.Categorical

    loan_a = processed_df.filter(pl.col("loan_id") == "loan_a").row(0, named=True)
    assert loan_a["default_14d"] is False
    assert loan_a["default_21d"] is False
    assert loan_a["existing_klarna_debt"] == 0
    assert loan_a["days_since_first_loan"] == 0

    loan_b = processed_df.filter(pl.col("loan_id") == "loan_b").row(0, named=True)
    assert loan_b["default_14d"] is True
    assert loan_b["default_21d"] is True
    assert loan_b["existing_klarna_debt"] == 0
    assert loan_b["card_expiry_month"] == 6
    assert loan_b["card_expiry_year"] >= 2023
    assert loan_b["merchant_category"] == OTHER_CATEGORY

    loan_c = processed_df.filter(pl.col("loan_id") == "loan_c").row(0, named=True)
    assert loan_c["default_14d"] is True
    assert loan_c["default_21d"] is False
    assert loan_c["merchant_category"] == "Common"
