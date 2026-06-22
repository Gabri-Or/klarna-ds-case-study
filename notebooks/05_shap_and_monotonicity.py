import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import pandas as pd
    import shap
    import xgboost as xgb
    from pathlib import Path
    from sklearn.calibration import CalibratedClassifierCV

    from klarna_ds_case_study.training.config import CATEGORICAL_FEATURES
    from klarna_ds_case_study.training.pipeline import (
        load_training_data,
        split_data,
        compute_metrics,
    )

    return (
        CalibratedClassifierCV,
        Path,
        compute_metrics,
        json,
        load_training_data,
        mo,
        np,
        pd,
        shap,
        split_data,
        xgb,
    )


@app.cell
def _(mo):
    mo.md("""
    # SHAP Analysis & Monotone Constraints

    Goals:
    1. Inspect SHAP values on the minimal feature set to verify features are used as expected
    2. Identify features with counter-intuitive directions
    3. Enforce monotone constraints where needed and compare performance
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load data and train model on minimal feature set
    """)
    return


@app.cell
def _(load_training_data, split_data):
    FEATURES_TO_REMOVE = [
        "num_failed_payments_6m",
        "amount_repaid_1m",
        "num_confirmed_payments_3m",
        "amount_repaid_3m",
        "amount_repaid_6m",
        "new_exposure_7d",
        "amount_repaid_14d",
        "card_expiry_month",
        "new_exposure_14d",
        "num_failed_payments_3m",
    ]

    X, y = load_training_data()
    X = X.drop(columns=FEATURES_TO_REMOVE)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_test, X_train, y_test, y_train


@app.cell
def _(Path, X_train, json, np, xgb, y_train):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    best_params = json.loads(
        (PROJECT_ROOT / "data" / "models" / "best_xgb_params.json").read_text()
    )

    spw = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    model = xgb.XGBClassifier(**{**best_params, "scale_pos_weight": spw})
    model.fit(X_train, y_train)
    return best_params, model, spw


@app.cell
def _(
    CalibratedClassifierCV,
    X_test,
    X_train,
    compute_metrics,
    mo,
    model,
    y_test,
    y_train,
):
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated_model.fit(X_train, y_train)

    baseline_probs = calibrated_model.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test, baseline_probs)

    mo.md(f"**Baseline metrics (minimal feature set):** {baseline_metrics}")
    return (baseline_metrics,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. SHAP values
    """)
    return


@app.cell
def _(X_train, model, shap):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return (shap_values,)


@app.cell
def _(mo):
    mo.md("""
    ### Summary plot (bar) — global feature importance
    """)
    return


@app.cell
def _(X_train, shap, shap_values):
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Summary plot (beeswarm) — feature value vs SHAP direction
    """)
    return


@app.cell
def _(X_train, shap, shap_values):
    shap.summary_plot(shap_values, X_train)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Dependence plots for key features

    These show how the SHAP value of a feature varies with the feature's actual value.
    """)
    return


@app.cell
def _(X_train, mo):
    shap_dependence_feature_dropdown = mo.ui.dropdown(
        options=list(X_train.columns),
        value=list(X_train.columns)[0],
        label="Feature for SHAP dependence plot",
    )
    return (shap_dependence_feature_dropdown,)


@app.cell
def _(X_train, mo, shap, shap_dependence_feature_dropdown, shap_values):
    import matplotlib.pyplot as plt

    _selected_feature = shap_dependence_feature_dropdown.value

    plt.figure()
    shap.dependence_plot(
        _selected_feature, shap_values, X_train, show=False, interaction_index=None
    )
    _ax = plt.gca()
    _ax.set_title(f"SHAP Dependence Plot: {_selected_feature}")
    _ax.set_xlabel(_selected_feature)
    _ax.set_ylabel("SHAP value")

    mo.vstack(
        [
            shap_dependence_feature_dropdown,
            mo.ui.matplotlib(_ax),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Expected monotonic directions

    Based on (limited) domain knowledge, we define the expected direction for each numeric feature:
    - **+1**: higher value → higher default risk (positive SHAP)
    - **-1**: higher value → lower default risk (negative SHAP)
    - **0**: no constraint (categorical or ambiguous)

    We can already see that not all features are used monotonically, but we will enforce the expected directions for the next part of the analysis.
    """)
    return


@app.cell
def _(X_train, mo):
    # Expected monotone directions for the minimal feature set
    # +1 = more → more default risk, -1 = more → less default risk, 0 = no constraint
    monotone_directions = {
        "loan_amount": 1,
        "num_active_loans": 1,
        "existing_klarna_debt": 1,
        "num_failed_payments_1y": 1,
        "num_confirmed_payments_6m": -1,
        "days_since_first_loan": -1,
        "amount_repaid_1y": -1,
        "card_expiry_year": 0,
        "merchant_group": 0,
        "merchant_category": 0,
    }

    monotone_constraints_tuple = tuple(
        monotone_directions.get(col, 0) for col in X_train.columns
    )

    mo.md(f"""
    Features and their constraints:

    {chr(10).join(f"- `{col}`: {monotone_directions.get(col, 0)}" for col in X_train.columns)}
    """)
    return (monotone_constraints_tuple,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Retrain with monotone constraints
    """)
    return


@app.cell
def _(X_train, best_params, monotone_constraints_tuple, spw, xgb, y_train):
    constrained_params = {
        **best_params,
        "scale_pos_weight": spw,
        "monotone_constraints": monotone_constraints_tuple,
    }

    model_constrained = xgb.XGBClassifier(**constrained_params)
    model_constrained.fit(X_train, y_train)

    return (model_constrained,)


@app.cell
def _(
    CalibratedClassifierCV,
    X_test,
    X_train,
    baseline_metrics,
    compute_metrics,
    mo,
    model_constrained,
    pd,
    y_test,
    y_train,
):
    calibrated_constrained = CalibratedClassifierCV(
        model_constrained, method="isotonic", cv=5
    )
    calibrated_constrained.fit(X_train, y_train)

    constrained_probs = calibrated_constrained.predict_proba(X_test)[:, 1]
    constrained_metrics = compute_metrics(y_test, constrained_probs)

    comparison_df = pd.DataFrame(
        [
            {"model": "baseline (no constraints)", **baseline_metrics},
            {"model": "monotone constrained", **constrained_metrics},
        ]
    )

    mo.md(f"""
    **Performance comparison:**

    {mo.as_html(comparison_df)}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. SHAP values with monotone constraints
    """)
    return


@app.cell
def _(X_train, model_constrained, shap):
    explainer_constrained = shap.TreeExplainer(model_constrained)
    shap_values_constrained = explainer_constrained.shap_values(X_train)
    return (shap_values_constrained,)


@app.cell
def _(X_train, shap, shap_values_constrained):
    shap.summary_plot(shap_values_constrained, X_train, plot_type="bar", show=False)
    return


@app.cell
def _(X_train, shap, shap_values_constrained):
    shap.summary_plot(shap_values_constrained, X_train, show=False)
    return


@app.cell
def _(
    X_train,
    mo,
    shap,
    shap_dependence_feature_dropdown,
    shap_values_constrained,
):
    def _():
        import matplotlib.pyplot as plt

        _selected_feature = shap_dependence_feature_dropdown.value

        plt.figure()
        shap.dependence_plot(
            _selected_feature,
            shap_values_constrained,
            X_train,
            show=False,
            interaction_index=None,
        )
        _ax = plt.gca()
        _ax.set_title(f"SHAP Dependence Plot: {_selected_feature}")
        _ax.set_xlabel(_selected_feature)
        _ax.set_ylabel("SHAP value")
        return mo.vstack(
            [
                shap_dependence_feature_dropdown,
                mo.ui.matplotlib(_ax),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Conclusion

    The performance is once again very similar once we constrain our features.

    If interpretability, alignment with domain knowledge and regulatory compliance are important, enforcing monotone constraints can be a valuable tool.
    """)
    return


if __name__ == "__main__":
    app.run()
