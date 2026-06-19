import json
import logging
import tempfile

import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from klarna_ds_case_study.training.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    N_CV_FOLDS,
    N_TRIALS,
    NON_FEATURE_COLUMNS,
    OPTUNA_SEARCH_SPACE,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from klarna_ds_case_study.training.tracking import (
    log_artifact,
    log_metrics,
    log_params,
    register_model,
    start_run,
)

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_training_data(path=PROCESSED_DATA_PATH):
    logger.info(f"Loading processed data from {path}")
    df = pl.read_parquet(path)
    logger.info(f"Loaded {df.height} rows, {df.width} columns")

    X = df.drop(NON_FEATURE_COLUMNS).to_pandas()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    y = df[TARGET].to_numpy().astype(int)
    return X, y


def split_data(X, y):
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"Split: {len(train_idx)} train, {len(test_idx)} test, "
        f"train default rate {y[train_idx].mean():.4f}, "
        f"test default rate {y[test_idx].mean():.4f}"
    )
    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]


def compute_scale_pos_weight(y_train):
    return float(np.sum(y_train == 0) / np.sum(y_train == 1))


def compute_metrics(y_true, y_prob):
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "ece": round(_compute_ece(y_true, y_prob), 4),
    }


def _compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (
            (y_prob >= lo) & (y_prob <= hi)
            if i == n_bins - 1
            else (y_prob >= lo) & (y_prob < hi)
        )
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)


def _build_xgb_params(trial, scale_pos_weight):
    space = OPTUNA_SEARCH_SPACE
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True
        ),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *space["colsample_bytree"]
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *space["min_child_weight"]
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
        "scale_pos_weight": scale_pos_weight,
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }


def tune_hyperparameters(X_train, y_train, n_trials=N_TRIALS):
    spw = compute_scale_pos_weight(y_train)
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = _build_xgb_params(trial, spw)
        scores = []
        for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train.iloc[fold_train_idx], y_train[fold_train_idx])
            probs = model.predict_proba(X_train.iloc[fold_val_idx])[:, 1]
            scores.append(roc_auc_score(y_train[fold_val_idx], probs))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study


def log_optuna_plots(study):
    plots = {
        "optimization_history": plot_optimization_history(study),
        "param_importances": plot_param_importances(study),
        "parallel_coordinate": plot_parallel_coordinate(study),
        "slice_plot": plot_slice(study),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, fig in plots.items():
            path = f"{tmpdir}/{name}.html"
            fig.write_html(path)
            log_artifact(path)
    logger.info("Logged Optuna visualization artifacts")


def train_model(X_train, y_train, tuned_params):
    spw = compute_scale_pos_weight(y_train)
    params = {
        **tuned_params,
        "scale_pos_weight": spw,
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Trained XGBoost model")
    return model, params


def calibrate_model(base_model, X_train, y_train):
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=N_CV_FOLDS)
    calibrated.fit(X_train, y_train)
    logger.info("Calibrated model with isotonic regression")
    return calibrated


def _save_params_json(params):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "best_xgb_params.json"
    path.write_text(json.dumps(params, indent=2))
    logger.info(f"Saved params to {path}")
    return str(path)


def run_pipeline():
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    with start_run(run_name="xgb-tuned-calibrated"):
        study = tune_hyperparameters(X_train, y_train)
        log_optuna_plots(study)
        model, full_params = train_model(X_train, y_train, study.best_params)

        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        log_params(full_params)
        log_metrics(compute_metrics(y_train, train_probs), prefix="train_")
        log_metrics(compute_metrics(y_test, test_probs), prefix="test_")

        calibrated = calibrate_model(model, X_train, y_train)
        cal_test_probs = calibrated.predict_proba(X_test)[:, 1]
        log_metrics(compute_metrics(y_test, cal_test_probs), prefix="calibrated_test_")

        params_path = _save_params_json(full_params)
        log_artifact(params_path)
        register_model(calibrated)

    logger.info("Training pipeline complete")
