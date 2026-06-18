from collections.abc import Callable

import polars as pl
import pytest

from klarna_ds_case_study.data_processing import pipeline
from klarna_ds_case_study.data_processing.pipeline import run_pipeline
from klarna_ds_case_study.data_processing.schemas import (
    ProcessedLoans,
    schema_column_names,
)


def test_run_pipeline_validates_and_writes_parquet(
    tmp_path,
    make_raw_df: Callable[[bool], pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "loans_raw.csv"
    output_path = tmp_path / "loans_processed.parquet"
    make_raw_df(False).write_csv(input_path)
    monkeypatch.setattr(pipeline, "RAW_DATA_PATH", input_path)
    monkeypatch.setattr(pipeline, "PROCESSED_DATA_PATH", output_path)
    monkeypatch.setattr(pipeline, "MIN_MERCHANT_CATEGORY_COUNT", 2)

    run_pipeline()
    processed_columns = schema_column_names(ProcessedLoans)

    assert output_path.exists()

    processed_df = pl.read_parquet(output_path)
    assert processed_df.columns == processed_columns
    assert processed_df.shape == (3, len(processed_columns))
    assert processed_df.null_count().sum_horizontal().item() == 0
    assert processed_df.select(pl.col("loan_id").is_unique().all()).item() is True
    assert processed_df.schema["merchant_group"] == pl.Categorical
    assert processed_df.schema["merchant_category"] == pl.Categorical
