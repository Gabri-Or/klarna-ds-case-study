from typing import Any

import pandera.polars as pa
import polars as pl
from pandera.typing import polars as pt


def _all_rows(data: Any, expr: pl.Expr) -> bool:
    return bool(data.lazyframe.select(expr.fill_null(False).all()).collect().item())


def schema_column_names(schema_model) -> list[str]:
    return list(schema_model.to_schema().columns)


def schema_column_dtypes(schema_model):
    return {
        name: column.dtype.type
        for name, column in schema_model.to_schema().columns.items()
    }


class RawLoans(pa.DataFrameModel):
    loan_id: pt.Series[pl.String]
    loan_issue_date: pt.Series[pl.Date]
    loan_amount: pt.Series[pl.Int64] = pa.Field(gt=0)
    amount_outstanding_14d: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_outstanding_21d: pt.Series[pl.Int64] = pa.Field(ge=0)
    card_expiry_month: pt.Series[pl.Int64] = pa.Field(nullable=True)
    card_expiry_year: pt.Series[pl.Int64] = pa.Field(nullable=True)
    existing_klarna_debt: pt.Series[pl.Int64] = pa.Field(nullable=True)
    num_active_loans: pt.Series[pl.Int64] = pa.Field(ge=0)
    days_since_first_loan: pt.Series[pl.Int64] = pa.Field(ge=-1)
    new_exposure_7d: pt.Series[pl.Int64] = pa.Field(ge=0)
    new_exposure_14d: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_confirmed_payments_3m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_confirmed_payments_6m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_failed_payments_3m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_failed_payments_6m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_failed_payments_1y: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_14d: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_1m: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_3m: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_6m: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_1y: pt.Series[pl.Int64] = pa.Field(ge=0)
    merchant_group: pt.Series[pl.String]
    merchant_category: pt.Series[pl.String]

    @pa.dataframe_check
    def amount_outstanding_14d_not_above_loan_amount(cls, data: Any) -> bool:
        return _all_rows(
            data, pl.col("amount_outstanding_14d") <= pl.col("loan_amount")
        )

    @pa.dataframe_check
    def amount_outstanding_21d_not_above_loan_amount(cls, data: Any) -> bool:
        return _all_rows(
            data, pl.col("amount_outstanding_21d") <= pl.col("loan_amount")
        )

    @pa.dataframe_check
    def amount_outstanding_monotonic_14d_to_21d(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("amount_outstanding_21d") <= pl.col("amount_outstanding_14d"),
        )

    @pa.dataframe_check
    def new_exposure_monotonic_7d_to_14d(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("new_exposure_14d") >= pl.col("new_exposure_7d"))

    @pa.dataframe_check
    def confirmed_payments_monotonic_3m_to_6m(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("num_confirmed_payments_6m") >= pl.col("num_confirmed_payments_3m"),
        )

    @pa.dataframe_check
    def failed_payments_monotonic_3m_to_6m(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("num_failed_payments_6m") >= pl.col("num_failed_payments_3m"),
        )

    @pa.dataframe_check
    def failed_payments_monotonic_6m_to_1y(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("num_failed_payments_1y") >= pl.col("num_failed_payments_6m"),
        )

    @pa.dataframe_check
    def amount_repaid_monotonic_14d_to_1m(cls, data: Any) -> bool:
        return _all_rows(
            data, pl.col("amount_repaid_1m") >= pl.col("amount_repaid_14d")
        )

    @pa.dataframe_check
    def amount_repaid_monotonic_1m_to_3m(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("amount_repaid_3m") >= pl.col("amount_repaid_1m"))

    @pa.dataframe_check
    def amount_repaid_monotonic_3m_to_6m(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("amount_repaid_6m") >= pl.col("amount_repaid_3m"))

    @pa.dataframe_check
    def amount_repaid_monotonic_6m_to_1y(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("amount_repaid_1y") >= pl.col("amount_repaid_6m"))

    class Config:
        strict = True
        ordered = True


class ProcessedLoans(pa.DataFrameModel):
    loan_id: pt.Series[pl.String] = pa.Field(unique=True)
    loan_issue_date: pt.Series[pl.Date]
    loan_amount: pt.Series[pl.Int64] = pa.Field(gt=0)
    card_expiry_month: pt.Series[pl.UInt8] = pa.Field(ge=1, le=12)
    card_expiry_year: pt.Series[pl.Int32] = pa.Field(ge=2023)
    existing_klarna_debt: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_active_loans: pt.Series[pl.Int64] = pa.Field(ge=0)
    days_since_first_loan: pt.Series[pl.Int64] = pa.Field(ge=0)
    has_prior_loan: pt.Series[pl.Boolean]
    new_exposure_7d: pt.Series[pl.Int64] = pa.Field(ge=0)
    new_exposure_14d: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_confirmed_payments_3m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_confirmed_payments_6m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_failed_payments_3m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_failed_payments_6m: pt.Series[pl.Int64] = pa.Field(ge=0)
    num_failed_payments_1y: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_14d: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_1m: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_3m: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_6m: pt.Series[pl.Int64] = pa.Field(ge=0)
    amount_repaid_1y: pt.Series[pl.Int64] = pa.Field(ge=0)
    merchant_group: pt.Series[pl.Categorical]
    merchant_category: pt.Series[pl.Categorical]
    loan_issue_year: pt.Series[pl.Int32]
    loan_issue_month: pt.Series[pl.UInt8] = pa.Field(ge=1, le=12)
    loan_issue_quarter: pt.Series[pl.UInt8] = pa.Field(ge=1, le=4)
    loan_issue_weekday: pt.Series[pl.UInt8] = pa.Field(ge=1, le=7)
    existing_debt_to_loan_amount: pt.Series[pl.Float64] = pa.Field(ge=0)
    new_exposure_14d_to_loan_amount: pt.Series[pl.Float64] = pa.Field(ge=0)
    amount_repaid_1y_to_loan_amount: pt.Series[pl.Float64] = pa.Field(ge=0)
    default_14d: pt.Series[pl.Boolean]
    default_21d: pt.Series[pl.Boolean]

    @pa.dataframe_check
    def new_exposure_monotonic_7d_to_14d(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("new_exposure_14d") >= pl.col("new_exposure_7d"))

    @pa.dataframe_check
    def confirmed_payments_monotonic_3m_to_6m(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("num_confirmed_payments_6m") >= pl.col("num_confirmed_payments_3m"),
        )

    @pa.dataframe_check
    def failed_payments_monotonic_3m_to_6m(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("num_failed_payments_6m") >= pl.col("num_failed_payments_3m"),
        )

    @pa.dataframe_check
    def failed_payments_monotonic_6m_to_1y(cls, data: Any) -> bool:
        return _all_rows(
            data,
            pl.col("num_failed_payments_1y") >= pl.col("num_failed_payments_6m"),
        )

    @pa.dataframe_check
    def amount_repaid_monotonic_14d_to_1m(cls, data: Any) -> bool:
        return _all_rows(
            data, pl.col("amount_repaid_1m") >= pl.col("amount_repaid_14d")
        )

    @pa.dataframe_check
    def amount_repaid_monotonic_1m_to_3m(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("amount_repaid_3m") >= pl.col("amount_repaid_1m"))

    @pa.dataframe_check
    def amount_repaid_monotonic_3m_to_6m(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("amount_repaid_6m") >= pl.col("amount_repaid_3m"))

    @pa.dataframe_check
    def amount_repaid_monotonic_6m_to_1y(cls, data: Any) -> bool:
        return _all_rows(data, pl.col("amount_repaid_1y") >= pl.col("amount_repaid_6m"))

    class Config:
        strict = True
        ordered = True

