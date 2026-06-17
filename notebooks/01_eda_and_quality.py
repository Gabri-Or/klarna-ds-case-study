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
def _(Path):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "mlcasestudy Final.csv"
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, pl):
    raw_df = pl.scan_csv(DATA_PATH, try_parse_dates=True)

    df = raw_df.with_columns(
    	(pl.col("amount_outstanding_21d") > 0).alias("default_21d")
    ).collect()

    row_count = df.height
    column_count = df.width
    return (df,)


@app.cell
def _(mo):
    mo.md(f"""
    ## 1. Data loading
    """)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    categorical_columns = ["merchant_group", "merchant_category"]
    numeric_columns = [
    	column
    	for column, dtype in df.schema.items()
    	if dtype.is_numeric()
    ]
    return categorical_columns, numeric_columns


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Data understanding
    """)
    return


@app.cell
def _(df, pl):
    target_summary = (
    	df.group_by("default_21d")
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

    default_rate = df.select(pl.mean("default_21d")).item()
    return (target_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Target definition and prevalence
    """)
    return


@app.cell
def _(target_summary):
    target_summary
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Numeric feature profile
    """)
    return


@app.cell
def _(numeric_profile):
    numeric_profile
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


@app.cell
def _(df, pl):
    outstanding_profile = (
    	df.select(
    		pl.col("amount_outstanding_14d").mean().round(2).alias("avg_outstanding_14d"),
    		pl.col("amount_outstanding_21d").mean().round(2).alias("avg_outstanding_21d"),
    		(pl.col("amount_outstanding_14d") > 0).mean().round(4).alias("share_outstanding_14d"),
    		(pl.col("amount_outstanding_21d") > 0).mean().round(4).alias("share_outstanding_21d"),
    		(pl.col("amount_outstanding_14d") - pl.col("amount_outstanding_21d"))
    		.mean()
    		.round(2)
    		.alias("avg_balance_reduction_14d_to_21d"),
    	)
    )
    return (outstanding_profile,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Outstanding balances
    """)
    return


@app.cell
def _(outstanding_profile):
    outstanding_profile
    return


@app.cell
def _(df, go, mo, pl):
    _sample_n = min(10_000, df.height)
    _plot_df = df.sample(n=_sample_n, seed=7) if df.height > _sample_n else df

    _fig = go.Figure(
    	go.Scattergl(
    		x=_plot_df["amount_outstanding_14d"].to_list(),
    		y=_plot_df["amount_outstanding_21d"].to_list(),
    		mode="markers",
    		marker={
    			"size": 5,
    			"opacity": 0.35,
    			"color": _plot_df.select(pl.col("default_21d").cast(pl.Int8))["default_21d"].to_list(),
    			"colorscale": [[0, "#2E86AB"], [1, "#C73E1D"]],
    			"showscale": False,
    		},
    		text=_plot_df["loan_id"].to_list(),
    		name="Loans",
    	)
    )
    _fig.update_layout(
    	title=f"Outstanding balance at 14 vs 21 days ({_sample_n:,} sampled loans)",
    	xaxis_title="Amount outstanding after 14 days",
    	yaxis_title="Amount outstanding after 21 days",
    	height=520,
    )
    mo.ui.plotly(_fig)
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
    	legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    	height=460,
    )
    mo.ui.plotly(_fig)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Merchant dimensions
    """)
    return


@app.cell
def _(merchant_group_profile):
    merchant_group_profile
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
    		"distinct_values": [df.select(pl.col(column).n_unique()).item() for column in categorical_columns],
    		"missing_count": [df.select(pl.col(column).null_count()).item() for column in categorical_columns],
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
    ).with_columns((pl.col("top_value_count") / df.height).round(4).alias("top_value_share"))
    return (categorical_profile,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 Categorical cardinality
    """)
    return


@app.cell
def _(categorical_profile):
    categorical_profile
    return


@app.cell
def _(merchant_category_profile):
    merchant_category_profile.head(25)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Data quality

    This section checks whether the dataset is internally consistent enough to
    support modelling. The checks focus on missingness, uniqueness, valid ranges,
    monotonic time windows, and target leakage considerations.
    """)
    return


@app.cell
def _(df, pl):
    missingness = (
    	pl.DataFrame(
    		{
    			"column": df.columns,
    			"missing_count": [df.select(pl.col(column).null_count()).item() for column in df.columns],
    		}
    	)
    	.with_columns((pl.col("missing_count") / df.height).round(4).alias("missing_share"))
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
    		"check": ["duplicate full rows", "loan_id values appearing more than once", "rows with duplicated loan_id"],
    		"count": [duplicate_row_count, duplicate_loan_ids.height, duplicate_loan_id_rows],
    	}
    )
    return duplicate_loan_ids, uniqueness_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Uniqueness
    """)
    return


@app.cell
def _(uniqueness_summary):
    uniqueness_summary
    return


@app.cell
def _(duplicate_loan_ids):
    duplicate_loan_ids.head(20)
    return


@app.cell
def _(df, pl):
    def count_where(predicate: pl.Expr) -> int:
    	return int(df.select(predicate.fill_null(False).sum()).item())

    quality_checks = pl.DataFrame(
    	[
    		{
    			"area": "identity",
    			"check": "loan_id is missing",
    			"issue_count": count_where(pl.col("loan_id").is_null()),
    		},
    		{
    			"area": "dates",
    			"check": "loan_issue_date is missing",
    			"issue_count": count_where(pl.col("loan_issue_date").is_null()),
    		},
    		{
    			"area": "amounts",
    			"check": "loan_amount <= 0",
    			"issue_count": count_where(pl.col("loan_amount") <= 0),
    		},
    		{
    			"area": "amounts",
    			"check": "amount_outstanding_14d outside [0, loan_amount]",
    			"issue_count": count_where(
    				(pl.col("amount_outstanding_14d") < 0)
    				| (pl.col("amount_outstanding_14d") > pl.col("loan_amount"))
    			),
    		},
    		{
    			"area": "amounts",
    			"check": "amount_outstanding_21d outside [0, loan_amount]",
    			"issue_count": count_where(
    				(pl.col("amount_outstanding_21d") < 0)
    				| (pl.col("amount_outstanding_21d") > pl.col("loan_amount"))
    			),
    		},
    		{
    			"area": "amounts",
    			"check": "amount_outstanding_21d > amount_outstanding_14d",
    			"issue_count": count_where(
    				pl.col("amount_outstanding_21d") > pl.col("amount_outstanding_14d")
    			),
    		},
    		{
    			"area": "card",
    			"check": "card_expiry_month outside 1-12",
    			"issue_count": count_where(
    				~pl.col("card_expiry_month").is_between(1, 12, closed="both")
    			),
    		},
    		{
    			"area": "customer history",
    			"check": "negative existing debt, exposure, repayment, count, or tenure values",
    			"issue_count": count_where(
    				(pl.col("existing_klarna_debt") < 0)
    				| (pl.col("num_active_loans") < 0)
    				| (pl.col("days_since_first_loan") < 0)
    				| (pl.col("new_exposure_7d") < 0)
    				| (pl.col("new_exposure_14d") < 0)
    				| (pl.col("num_confirmed_payments_3m") < 0)
    				| (pl.col("num_confirmed_payments_6m") < 0)
    				| (pl.col("num_failed_payments_3m") < 0)
    				| (pl.col("num_failed_payments_6m") < 0)
    				| (pl.col("num_failed_payments_1y") < 0)
    				| (pl.col("amount_repaid_14d") < 0)
    				| (pl.col("amount_repaid_1m") < 0)
    				| (pl.col("amount_repaid_3m") < 0)
    				| (pl.col("amount_repaid_6m") < 0)
    				| (pl.col("amount_repaid_1y") < 0)
    			),
    		},
    		{
    			"area": "customer history",
    			"check": "new_exposure_14d < new_exposure_7d",
    			"issue_count": count_where(pl.col("new_exposure_14d") < pl.col("new_exposure_7d")),
    		},
    		{
    			"area": "customer history",
    			"check": "num_confirmed_payments_6m < num_confirmed_payments_3m",
    			"issue_count": count_where(
    				pl.col("num_confirmed_payments_6m") < pl.col("num_confirmed_payments_3m")
    			),
    		},
    		{
    			"area": "customer history",
    			"check": "num_failed_payments_6m < num_failed_payments_3m",
    			"issue_count": count_where(
    				pl.col("num_failed_payments_6m") < pl.col("num_failed_payments_3m")
    			),
    		},
    		{
    			"area": "customer history",
    			"check": "num_failed_payments_1y < num_failed_payments_6m",
    			"issue_count": count_where(
    				pl.col("num_failed_payments_1y") < pl.col("num_failed_payments_6m")
    			),
    		},
    		{
    			"area": "customer history",
    			"check": "repayment windows are not monotonic",
    			"issue_count": count_where(
    				(pl.col("amount_repaid_1m") < pl.col("amount_repaid_14d"))
    				| (pl.col("amount_repaid_3m") < pl.col("amount_repaid_1m"))
    				| (pl.col("amount_repaid_6m") < pl.col("amount_repaid_3m"))
    				| (pl.col("amount_repaid_1y") < pl.col("amount_repaid_6m"))
    			),
    		},
    	]
    ).with_columns((pl.col("issue_count") / df.height).round(4).alias("issue_share"))
    return (quality_checks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Range and consistency checks
    """)
    return


@app.cell
def _(quality_checks):
    quality_checks.sort("issue_count", descending=True)
    return


@app.cell
def _(df, pl):
    date_range = df.select(
    	pl.min("loan_issue_date").alias("min_loan_issue_date"),
    	pl.max("loan_issue_date").alias("max_loan_issue_date"),
    	pl.col("loan_issue_date").n_unique().alias("distinct_issue_dates"),
    )
    return (date_range,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Date coverage and leakage notes
    """)
    return


@app.cell
def _(date_range):
    date_range
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Follow-up analysis ideas

    - Choose a train/validation split that respects `loan_issue_date` to avoid
      optimistic estimates from temporal leakage.
    - Exclude `amount_outstanding_21d` from predictors because it defines the
      target; treat `amount_outstanding_14d` as unavailable for underwriting unless
      the intended scoring moment is after day 14.
    - Investigate any non-zero quality-check counts before feature engineering.
    - Compare default rates across loan amount bands, merchant dimensions, and
      customer history features before moving into model development.
    """)
    return


if __name__ == "__main__":
    app.run()
