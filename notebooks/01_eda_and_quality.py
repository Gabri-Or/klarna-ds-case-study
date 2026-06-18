import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl
    from pathlib import Path

    return Path, go, mo, pl


@app.cell
def _(mo):
    mo.md(f"""
    ## 1. Data loading
    """)
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mlcasestudy Final.csv"
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, pl):
    df = pl.scan_csv(DATA_PATH, try_parse_dates=True)

    row_count = df.height
    column_count = df.width
    return column_count, df, row_count


@app.cell
def _(column_count, default_rate, mo, row_count):
    mo.md(
        f"The dataset consists of {row_count:,} rows and {column_count:,} columns, with a 21-day default rate of {default_rate:.1%}."
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each row represents a loan, identified by `loan_id` and dated by `loan_issue_date`; we know the initial amount of the loan and the outstanding after 14 and 21 days, which are the two deadlines for repayment.

    Next we have information on the current card associated with the loan, and on the credit history of the consumer.

    Finally, the last two columns describe the merchant the financed purchase is associated to.
    """)
    return


@app.cell
def _(df):
    categorical_columns = ["merchant_group", "merchant_category"]
    numeric_columns = [
        column for column, dtype in df.schema.items() if dtype.is_numeric()
    ]
    return categorical_columns, numeric_columns


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Data understanding
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Date coverage
    """)
    return


@app.cell
def _(df, pl):
    date_range = df.select(
        pl.min("loan_issue_date").alias("min_loan_issue_date"),
        pl.max("loan_issue_date").alias("max_loan_issue_date"),
        pl.col("loan_issue_date").n_unique().alias("distinct_issue_dates"),
    )
    return (date_range,)


@app.cell
def _(date_range):
    date_range
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Target definition and prevalence
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the moment, let's focus on the 21-days target which is the hardest deadline for repayment.
    """)
    return


@app.cell
def _(df, pl):
    target_df = df.with_columns(
        (pl.col("amount_outstanding_14d") > 0).alias("default_14d"),
        (pl.col("amount_outstanding_21d") > 0).alias("default_21d"),
    )

    target_df.select(
        pl.mean("default_14d").alias("default_rate_14d"),
        pl.mean("default_21d").alias("default_rate_21d"),
    )
    return (target_df,)


@app.cell
def _(pl, target_df):
    target_summary = (
        target_df.group_by("default_21d")
        .agg(
            pl.len().alias("loans"),
            pl.sum("loan_amount").alias("loan_amount_total"),
            pl.mean("loan_amount").round(2).alias("avg_loan_amount"),
            pl.mean("amount_outstanding_21d").round(2).alias("avg_outstanding_21d"),
        )
        .with_columns(
            pl.when(pl.col("default_21d"))
            .then(pl.lit("Default at 21 days"))
            .otherwise(pl.lit("Repaid by 21 days"))
            .alias("target_label"),
            (pl.col("loans") / pl.sum("loans")).round(4).alias("loan_share"),
        )
        .select(
            "target_label",
            "default_21d",
            "loans",
            "loan_share",
            "loan_amount_total",
            "avg_loan_amount",
            "avg_outstanding_21d",
        )
        .sort("default_21d")
    )
    return (target_summary,)


@app.cell
def _(target_summary):
    target_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Defaulted loans amount to about 5% and do not seem to differ significantly by their amount; on average, the outstanding at default time is a bit less than 50% the initial amount.
    """)
    return


@app.cell
def _(go, mo, target_summary):
    _fig = go.Figure(
        go.Bar(
            x=target_summary["target_label"].to_list(),
            y=target_summary["loans"].to_list(),
            text=[f"{value:.1%}" for value in target_summary["loan_share"].to_list()],
            textposition="outside",
            marker_color=["#2E86AB", "#C73E1D"],
        )
    )
    _fig.update_layout(
        title="Loan count by 21-day default status",
        xaxis_title="Target class",
        yaxis_title="Loans",
        yaxis_tickformat=",",
        showlegend=False,
        height=500,
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Numeric feature profile
    """)
    return


@app.cell
def _(df, numeric_columns, pl):
    numeric_profile = pl.concat(
        [
            df.select(
                pl.lit(column).alias("column"),
                pl.col(column).count().alias("non_null_count"),
                pl.col(column).null_count().alias("missing_count"),
                pl.col(column).min().alias("min"),
                pl.col(column).quantile(0.25, interpolation="nearest").alias("p25"),
                pl.col(column).median().alias("median"),
                pl.col(column).mean().round(2).alias("mean"),
                pl.col(column).quantile(0.75, interpolation="nearest").alias("p75"),
                pl.col(column).quantile(0.95, interpolation="nearest").alias("p95"),
                pl.col(column).max().alias("max"),
                pl.col(column).std().round(2).alias("std"),
            )
            for column in numeric_columns
        ],
        how="vertical",
    )
    return (numeric_profile,)


@app.cell
def _(numeric_profile):
    numeric_profile
    return


@app.cell
def _(mo, numeric_columns):
    numeric_feature_dropdown = mo.ui.dropdown(
        options=numeric_columns,
        value=numeric_columns[0],
        label="Select numeric feature",
    )
    return (numeric_feature_dropdown,)


@app.cell
def _(df, go, mo, numeric_feature_dropdown, pl):
    _selected_numeric_feature = numeric_feature_dropdown.value
    _distribution_plot_df = df.select(
        pl.col(_selected_numeric_feature).alias("value")
    ).drop_nulls()

    _distribution_values = _distribution_plot_df["value"].to_list()

    _distribution_fig = go.Figure()

    _distribution_fig.add_trace(
        go.Histogram(
            x=_distribution_values,
            name=_selected_numeric_feature,
            nbinsx=60,
            marker_color="#2E86AB",
            opacity=0.85,
            histnorm="percent",
            hovertemplate=(
                f"{_selected_numeric_feature}: %{{x}}<br>Count: %{{y:,}}<extra></extra>"
            ),
        )
    )

    _distribution_fig.update_layout(
        title=f"Distribution of {_selected_numeric_feature}",
        xaxis_title=_selected_numeric_feature,
        yaxis_title="Loans",
        yaxis_tickformat=",",
        yaxis2={
            "visible": False,
            "overlaying": "y",
            "range": [0, 1],
        },
        bargap=0.03,
        height=520,
        showlegend=False,
        template="plotly_white",
    )

    mo.vstack(
        [
            numeric_feature_dropdown,
            mo.ui.plotly(_distribution_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some considerations on the distributions of the numeric features:
    - `loan_amount` shows a skewed distribution akin to an inverse power law, with a thin long tail of high-value loans; `amount_outstanding_14d` and `amount_outstanding_21d` are even more skewed, but they contain a more complex mixture of zeroes (for fully repaid loans) and positive values with a similar long-tail pattern, which we will study later in more detail.
    - `card_expiry_month` and `card_expiry_year` display what we would expect: uniform distribution for month, and a bell-like curve centered around 2-4 years after the loan issue dates in the dataset (all from 2023) and with a small tail.
    - `existing_klarna_debt` is also right-skewed like the "amount" columns, but it also has some negative values. Since we don't have information on their meaning, we will probably clamp them to zero.
    - `num_active_loans` shows that most loans in the dataset occur as first loans (zero active loans at the time of issue), but there are also a few customers with high counts that might be worth investigating, since having so many active loans could be a sign of financial distress or reckless behavior.
    - `days_since_first_loan` has a surprising 32% of "-1" values, all associated with "0" or "null" `existing_klarna_debt`. To be investigated further.
    - `new_exposure_7d`, `new_exposure_14d`, `num_confirmed_payments_3m` and `num_confirmed_payments_6m` show once again the typical right-skewed shape of other numeric columns.
    - `num_failed_payments_3m`, `num_failed_payments_6m` and `num_failed_payments_1y` all have more than 95% of zeroes, which suggest that failed payments are relatively rare events in the customer history, but they might still be important predictors of default risk.
    - finally, all the `amount_repaid` columns are once again right-skewed with a large share of zeroes, which is consistent with the fact that many customers do not have any repayment history with Klarna, but for those who do, the amounts can vary widely and reach high values.
    """)
    return


@app.cell
def _(df, go, mo):
    _repaid = df.filter(~df["default_21d"])["loan_amount"].to_list()
    _defaulted = df.filter(df["default_21d"])["loan_amount"].to_list()

    _fig = go.Figure()
    _fig.add_trace(
        go.Histogram(
            x=_repaid,
            name="Repaid by 21 days",
            opacity=0.7,
            marker_color="#2E86AB",
            nbinsx=60,
        )
    )
    _fig.add_trace(
        go.Histogram(
            x=_defaulted,
            name="Default at 21 days",
            opacity=0.7,
            marker_color="#C73E1D",
            nbinsx=60,
        )
    )
    _fig.update_layout(
        title="Loan amount distribution by 21-day default status",
        xaxis_title="Loan amount",
        yaxis_title="Loans",
        barmode="overlay",
        height=460,
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Outstanding balances
    """)
    return


@app.cell
def _(df, pl):
    outstanding_profile = df.select(
        pl.col("amount_outstanding_14d").mean().round(2).alias("avg_outstanding_14d"),
        pl.col("amount_outstanding_21d").mean().round(2).alias("avg_outstanding_21d"),
        (pl.col("amount_outstanding_14d") > 0)
        .mean()
        .round(4)
        .alias("share_outstanding_14d"),
        (pl.col("amount_outstanding_21d") > 0)
        .mean()
        .round(4)
        .alias("share_outstanding_21d"),
        (pl.col("amount_outstanding_14d") - pl.col("amount_outstanding_21d"))
        .mean()
        .round(2)
        .alias("avg_balance_reduction_14d_to_21d"),
    )
    return (outstanding_profile,)


@app.cell
def _(outstanding_profile):
    outstanding_profile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We defined the target at 21 days, preferring it over the one at 14 days because it represented the last deadline before default.

    Nonetheless, it may be also useful to predict 14-days default, as an extra signal that may prompt some action towards the customer.

    The higher default rate at 14 days may amount to an easier prediction task, too.

    From the data aggregated above, though, we can see that on average almost half of the outstanding gets recovered in the extra week granted by klarna before declaring a default; this suggests that the extra week is indeed useful and that leaving the threshold at 21 days may be the right choice.

    Of course an appropriate assessment of this threshold should include the economic effects that this extra time has on klarna.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 Timing effect
    """)
    return


@app.cell
def _(df, pl):
    monthly_profile = (
        df.group_by(pl.col("loan_issue_date").dt.truncate("1mo").alias("month"))
        .agg(
            pl.len().alias("loans"),
            pl.mean("default_21d").round(4).alias("default_rate"),
            pl.mean("loan_amount").round(2).alias("avg_loan_amount"),
            pl.sum("loan_amount").alias("loan_amount_total"),
        )
        .sort("month")
    )
    return (monthly_profile,)


@app.cell
def _(go, mo, monthly_profile):
    _fig = go.Figure()
    _fig.add_trace(
        go.Bar(
            x=monthly_profile["month"].to_list(),
            y=monthly_profile["loans"].to_list(),
            name="Loans",
            marker_color="#2E86AB",
            yaxis="y",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=monthly_profile["month"].to_list(),
            y=monthly_profile["default_rate"].to_list(),
            name="Default rate",
            mode="lines+markers",
            marker_color="#C73E1D",
            yaxis="y2",
        )
    )
    _fig.update_layout(
        title="Monthly loan volume and 21-day default rate",
        xaxis_title="Loan issue month",
        yaxis={"title": "Loans", "tickformat": ","},
        yaxis2={
            "title": "Default rate",
            "tickformat": ".1%",
            "overlaying": "y",
            "side": "right",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        height=460,
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There seems to be a significant trend in the default rate, which decreases as time passes while the sample size is mostly constant if not increasing.

    Is this actually a trend? a seasonal effect? random fluctuations? The time span of provided data is too short to tell, but it would definitely merit further investigations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6 Merchant dimensions
    """)
    return


@app.cell
def _(df, pl):
    merchant_group_profile = (
        df.group_by("merchant_group")
        .agg(
            pl.len().alias("loans"),
            pl.mean("default_21d").round(4).alias("default_rate"),
            pl.mean("loan_amount").round(2).alias("avg_loan_amount"),
            pl.sum("loan_amount").alias("loan_amount_total"),
        )
        .sort("loans", descending=True)
    )

    merchant_category_profile = (
        df.group_by("merchant_category")
        .agg(
            pl.len().alias("loans"),
            pl.mean("default_21d").round(4).alias("default_rate"),
            pl.mean("loan_amount").round(2).alias("avg_loan_amount"),
        )
        .sort("loans", descending=True)
    )
    return merchant_category_profile, merchant_group_profile


@app.cell
def _(merchant_group_profile):
    merchant_group_profile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Merchant groups seem to be quite heterogeneous in number of loans, default rate and avg loan amounts.

    Each merchant group showcasing different patterns in consumer habits would be typical; as a predictor, it's definitely a promising feature that can be encoded as categorical or possibly with some clever numerical embedding.

    If this was not enough, a possible strategy would be dealing with at least some of the main groups separately, to make sure they get tailored predictions.
    """)
    return


@app.cell
def _(go, merchant_group_profile, mo):
    _plot_df = merchant_group_profile.head(15).sort("loans")
    _fig = go.Figure(
        go.Bar(
            x=_plot_df["default_rate"].to_list(),
            y=_plot_df["merchant_group"].to_list(),
            orientation="h",
            marker_color="#C73E1D",
            text=[f"{value:.1%}" for value in _plot_df["default_rate"].to_list()],
            customdata=_plot_df["loans"].to_list(),
            hovertemplate="%{y}<br>Default rate: %{x:.2%}<br>Loans: %{customdata:,}<extra></extra>",
        )
    )
    _fig.update_layout(
        title="21-day default rate for the largest merchant groups",
        xaxis_title="Default rate",
        xaxis_tickformat=".1%",
        yaxis_title="Merchant group",
        height=520,
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(categorical_columns, df, pl):
    categorical_profile = pl.DataFrame(
        {
            "column": categorical_columns,
            "distinct_values": [
                df.select(pl.col(column).n_unique()).item()
                for column in categorical_columns
            ],
            "missing_count": [
                df.select(pl.col(column).null_count()).item()
                for column in categorical_columns
            ],
            "top_value": [
                df.group_by(column)
                .len()
                .sort("len", descending=True)
                .select(column)
                .item(0, 0)
                for column in categorical_columns
            ],
            "top_value_count": [
                df.group_by(column)
                .len()
                .sort("len", descending=True)
                .select("len")
                .item(0, 0)
                for column in categorical_columns
            ],
        }
    ).with_columns(
        (pl.col("top_value_count") / df.height).round(4).alias("top_value_share")
    )
    return (categorical_profile,)


@app.cell
def _(categorical_profile):
    categorical_profile
    return


@app.cell
def _(merchant_category_profile):
    merchant_category_profile.head(25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The sub-division, "merchant category", starts having way too many unique values to be dealt properly as a categorical; moreover, there's way too many categories that are very small and would have to be clumped together.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Data quality

    This section checks whether the dataset is internally consistent enough to support modelling.

    The checks focus on missingness, uniqueness, valid ranges, monotonic time windows, and target leakage considerations.
    """)
    return


@app.cell
def _(df, pl):
    missingness = (
        pl.DataFrame(
            {
                "column": df.columns,
                "missing_count": [
                    df.select(pl.col(column).null_count()).item()
                    for column in df.columns
                ],
            }
        )
        .with_columns(
            (pl.col("missing_count") / df.height).round(4).alias("missing_share")
        )
        .sort(["missing_count", "column"], descending=[True, False])
    )
    return (missingness,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Missing values
    """)
    return


@app.cell
def _(missingness):
    missingness
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we already noticed earlier, `existing_klarna_debt` is the only column having a substantial amount of missing values.

    About `card_expiry_month` and `card_expiry_year`, the fact that the share of missing values is so small and that these columns appeared to be among the least interesting ones, makes us conclude that we can confidently ignore them for now.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Uniqueness
    """)
    return


@app.cell
def _(df, pl):
    duplicate_row_count = int(df.is_duplicated().sum())
    duplicate_loan_ids = (
        df.group_by("loan_id")
        .len()
        .filter(pl.col("len") > 1)
        .sort("len", descending=True)
    )
    duplicate_loan_id_rows = duplicate_loan_ids.select(pl.sum("len")).item() or 0

    uniqueness_summary = pl.DataFrame(
        {
            "check": [
                "duplicate full rows",
                "loan_id values appearing more than once",
                "rows with duplicated loan_id",
            ],
            "count": [
                duplicate_row_count,
                duplicate_loan_ids.height,
                duplicate_loan_id_rows,
            ],
        }
    )
    return (uniqueness_summary,)


@app.cell
def _(uniqueness_summary):
    uniqueness_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No duplicate rows, that's good.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Range and consistency checks

    Based off our understanding of data, we can enforce some domain-driven checks on ranges, signs, consistency among columns, etc.
    """)
    return


@app.cell
def _(df, pl):
    def count_where(predicate: pl.Expr) -> int:
        return int(df.select(predicate.fill_null(False).sum()).item())

    return (count_where,)


@app.cell
def _(count_where, pl):
    # negative loan amount
    count_where(pl.col("loan_amount") <= 0)
    return


@app.cell
def _(count_where, pl):
    # outstanding at 14 days negative or higher than initial amount
    count_where(
        (pl.col("amount_outstanding_14d") < 0)
        | (pl.col("amount_outstanding_14d") > pl.col("loan_amount"))
    )
    return


@app.cell
def _(count_where, pl):
    # outstanding at 21 days negative or higher than initial amount
    count_where(
        (pl.col("amount_outstanding_21d") < 0)
        | (pl.col("amount_outstanding_21d") > pl.col("loan_amount"))
    )
    return


@app.cell
def _(count_where, pl):
    # outstanding at 21 days higher than outstanding at 14 days
    count_where(pl.col("amount_outstanding_21d") > pl.col("amount_outstanding_14d"))
    return


@app.cell
def _(count_where, pl):
    # card expiry date not a valid future date
    count_where(
        ~pl.col("card_expiry_month").is_between(1, 12, closed="both")
        | ~pl.col("card_expiry_month")
        > 2022
    )
    return


@app.cell
def _(count_where, pl):
    # negative existing debt
    count_where(pl.col("existing_klarna_debt") < 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we already noticed when looking at distributions, negative values for debt don't have a clear cut explanation and should be investigated further. since they are so few, we can safely clamp them to zero for our purposes.
    """)
    return


@app.cell
def _(count_where, pl):
    # negative number of loans
    count_where(pl.col("num_active_loans") < 0)
    return


@app.cell
def _(count_where, pl):
    # negative number of days since first loan
    count_where(pl.col("days_since_first_loan") < 0)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("days_since_first_loan") < 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks like `days_since_first_loan` is set to -1 whenever `num_active_loans` is zero. Is this true?
    """)
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("days_since_first_loan") == 0) & (pl.col("num_active_loans") == 0)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Actually, sometimes both columns are zeroed out; but most of the times, no active loans is reflected in a "-1" in number of days. we can probably just identify the two cases in one.
    """)
    return


@app.cell
def _(count_where, pl):
    # Numbers of payments < 0
    count_where(
        (pl.col("num_confirmed_payments_3m") < 0)
        | (pl.col("num_confirmed_payments_6m") < 0)
        | (pl.col("num_failed_payments_3m") < 0)
        | (pl.col("num_failed_payments_6m") < 0)
        | (pl.col("num_failed_payments_1y") < 0)
    )
    return


@app.cell
def _(count_where, pl):
    # Monetary amounts < 0
    count_where(
        (pl.col("new_exposure_7d") < 0)
        | (pl.col("new_exposure_14d") < 0)
        | (pl.col("amount_repaid_14d") < 0)
        | (pl.col("amount_repaid_1m") < 0)
        | (pl.col("amount_repaid_3m") < 0)
        | (pl.col("amount_repaid_6m") < 0)
        | (pl.col("amount_repaid_1y") < 0)
    )
    return


@app.cell
def _(count_where, pl):
    # Exposure at 14 days lower than exposure at 7 days
    count_where(pl.col("new_exposure_14d") < pl.col("new_exposure_7d"))
    return


@app.cell
def _(count_where, pl):
    # Number of payments lower at an earlier date
    count_where(
        (pl.col("num_confirmed_payments_6m") < pl.col("num_confirmed_payments_3m"))
        | (pl.col("num_failed_payments_6m") < pl.col("num_failed_payments_3m"))
        | (pl.col("num_failed_payments_1y") < pl.col("num_failed_payments_6m"))
    )
    return


@app.cell
def _(count_where, pl):
    # Amount repaid higher at an earlier date
    count_where(
        (pl.col("amount_repaid_1m") < pl.col("amount_repaid_14d"))
        | (pl.col("amount_repaid_3m") < pl.col("amount_repaid_1m"))
        | (pl.col("amount_repaid_6m") < pl.col("amount_repaid_3m"))
        | (pl.col("amount_repaid_1y") < pl.col("amount_repaid_6m"))
    )
    return


if __name__ == "__main__":
    app.run()
