import polars as pl


def count_where(df: pl.DataFrame, predicate: pl.Expr) -> int:
    return int(df.select(predicate.fill_null(False).sum()).item())


def mode_value(df: pl.DataFrame, column: str) -> int | str | None:
    value = (
        df.filter(pl.col(column).is_not_null())
        .group_by(column)
        .len()
        .sort(["len", column], descending=[True, False])
        .select(column)
        .head(1)
    )
    if value.is_empty():
        return None
    return value.item(0, 0)


def median_value(df: pl.DataFrame, column: str) -> int | None:
    value = df.select(pl.col(column).median()).item()
    return None if value is None else int(value)
