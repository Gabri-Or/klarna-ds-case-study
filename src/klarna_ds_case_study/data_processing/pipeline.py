import logging

import pandera.polars as pa
import polars as pl
from pandera.typing import polars as pt

from klarna_ds_case_study.data_processing.config import (
    MIN_MERCHANT_CATEGORY_COUNT,
    OTHER_CATEGORY,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    UNKNOWN_CATEGORY,
)
from klarna_ds_case_study.data_processing.schemas import (
    ProcessedLoans,
    RawLoans,
    schema_column_dtypes,
    schema_column_names,
)
from klarna_ds_case_study.data_processing.utils import (
    count_where,
    median_value,
    mode_value,
)

logger = logging.getLogger(__name__)


def log_missingness(df: pl.DataFrame, name: str) -> None:
    missing_counts = {
        column: int(df.select(pl.col(column).null_count()).item())
        for column in df.columns
    }
    non_zero_missing = {
        key: value for key, value in missing_counts.items() if value > 0
    }
    if non_zero_missing:
        logger.info(f"{name} missing values: {non_zero_missing}")
    else:
        logger.info(f"{name} has no missing values")


@pa.check_types(lazy=True)
def load_raw_data() -> pt.DataFrame[RawLoans]:
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    df = pl.read_csv(
        RAW_DATA_PATH,
        try_parse_dates=True,
        schema_overrides=schema_column_dtypes(RawLoans),
        null_values=["", "NA", "N/A", "null", "None"],
    )
    logger.info(f"Loaded raw data with {df.height} rows and {df.width} columns")
    log_missingness(df, "raw data")
    return df


def log_quality_findings(df: pt.DataFrame[RawLoans]) -> None:
    duplicate_row_count = int(df.is_duplicated().sum())
    duplicate_loan_id_rows = (
        df.height
        - df.unique(subset=["loan_id"], keep="first", maintain_order=True).height
    )
    negative_debt_count = count_where(df, pl.col("existing_klarna_debt") < 0)
    debt_missing_count = count_where(df, pl.col("existing_klarna_debt").is_null())
    days_sentinel_count = count_where(df, pl.col("days_since_first_loan") < 0)
    card_missing_or_invalid_count = count_where(
        df,
        pl.col("card_expiry_month").is_null()
        | pl.col("card_expiry_year").is_null()
        | ~pl.col("card_expiry_month").is_between(1, 12)
        | (pl.col("card_expiry_year") < 2023),
    )

    logger.info(f"Duplicate full rows: {duplicate_row_count}")
    logger.info(f"Rows removed if loan_id is de-duplicated: {duplicate_loan_id_rows}")
    logger.info(
        f"Missing existing_klarna_debt values to impute as 0: {debt_missing_count}"
    )
    logger.info(
        f"Negative existing_klarna_debt values to clamp to 0: {negative_debt_count}"
    )
    logger.info(
        f"Negative days_since_first_loan sentinel values to map to 0: {days_sentinel_count}"
    )
    logger.info(
        f"Missing/invalid card expiry values to impute: {card_missing_or_invalid_count}"
    )


def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    deduped = df.unique(maintain_order=True)
    if deduped.height != df.height:
        logger.info(f"Removed {df.height - deduped.height} exact duplicate rows")
    by_loan_id = deduped.unique(subset=["loan_id"], keep="first", maintain_order=True)
    if by_loan_id.height != deduped.height:
        logger.info(
            f"Removed {deduped.height - by_loan_id.height} duplicated loan_id rows"
        )
    return by_loan_id


def add_default_flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("amount_outstanding_14d") > 0).alias("default_14d"),
        (pl.col("amount_outstanding_21d") > 0).alias("default_21d"),
    )


def impute_card_expiry(df: pl.DataFrame) -> pl.DataFrame:
    month = mode_value(df, "card_expiry_month")
    year = median_value(df, "card_expiry_year")
    return df.with_columns(
        pl.when(pl.col("card_expiry_month").is_between(1, 12))
        .then(pl.col("card_expiry_month"))
        .otherwise(pl.lit(month))
        .cast(pl.UInt8)
        .alias("card_expiry_month"),
        pl.when(pl.col("card_expiry_year") >= 2023)
        .then(pl.col("card_expiry_year"))
        .otherwise(pl.lit(year))
        .cast(pl.Int32)
        .alias("card_expiry_year"),
    )


def clamp_debt_and_days(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.max_horizontal(pl.col("existing_klarna_debt").fill_null(0), pl.lit(0)).alias(
            "existing_klarna_debt"
        ),
        pl.max_horizontal(pl.col("days_since_first_loan"), pl.lit(0)).alias(
            "days_since_first_loan"
        ),
    )


def collapse_rare_merchant_categories(df: pl.DataFrame) -> pl.DataFrame:
    frequent = (
        df.filter(pl.col("merchant_category").is_not_null())
        .group_by("merchant_category")
        .len()
        .filter(pl.col("len") >= MIN_MERCHANT_CATEGORY_COUNT)
        .select("merchant_category")
        .to_series()
        .to_list()
    )
    logger.info(
        f"Keeping {len(frequent)} merchant_category values with >= "
        f"{MIN_MERCHANT_CATEGORY_COUNT} rows; rest collapsed to {OTHER_CATEGORY!r}"
    )
    return df.with_columns(
        pl.col("merchant_group").fill_null(UNKNOWN_CATEGORY),
        pl.when(pl.col("merchant_category").is_in(frequent))
        .then(pl.col("merchant_category"))
        .when(pl.col("merchant_category").is_null())
        .then(pl.lit(UNKNOWN_CATEGORY))
        .otherwise(pl.lit(OTHER_CATEGORY))
        .alias("merchant_category"),
    )


def select_and_cast_final_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(schema_column_names(ProcessedLoans)).with_columns(
        pl.col("merchant_group").cast(pl.Categorical),
        pl.col("merchant_category").cast(pl.Categorical),
    )


@pa.check_types(lazy=True)
def clean_and_prepare_data(
    raw_df: pt.DataFrame[RawLoans],
) -> pt.DataFrame[ProcessedLoans]:
    logger.info("Cleaning and preparing data")
    log_quality_findings(raw_df)
    df = deduplicate(raw_df)
    df = add_default_flags(df)
    df = impute_card_expiry(df)
    df = clamp_debt_and_days(df)
    df = collapse_rare_merchant_categories(df)
    processed_df = select_and_cast_final_columns(df)

    logger.info(
        f"Processed: {processed_df.height} rows, {processed_df.width} columns, "
        f"{100 * float(processed_df.select(pl.mean('default_21d')).item()):.2f}% "
        "21-day default rate"
    )
    log_missingness(processed_df, "processed data")
    return processed_df


@pa.check_types(lazy=True)
def write_processed_data(df: pt.DataFrame[ProcessedLoans]) -> None:
    logger.info(f"Writing processed data to {PROCESSED_DATA_PATH}")
    df.write_parquet(PROCESSED_DATA_PATH)
    logger.info(
        f"Wrote {df.height} rows and {df.width} columns to {PROCESSED_DATA_PATH}"
    )


def run_pipeline() -> None:
    raw_df = load_raw_data()
    processed_df = clean_and_prepare_data(raw_df)
    write_processed_data(processed_df)
