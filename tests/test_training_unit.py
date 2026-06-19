from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from klarna_ds_case_study.training.pipeline import (
    _compute_ece,
    calibrate_model,
    compute_metrics,
    compute_scale_pos_weight,
    split_data,
    train_model,
    tune_hyperparameters,
)


@pytest.fixture
def binary_predictions():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
    return y_true, y_prob


@pytest.fixture
def sample_train_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "cat": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        }
    )
    y = rng.binomial(1, 0.1, n)
    return X, y


def test_compute_metrics_returns_expected_keys(binary_predictions):
    y_true, y_prob = binary_predictions
    metrics = compute_metrics(y_true, y_prob)
    assert set(metrics.keys()) == {
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "ece",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier_score"] == 0.0


def test_compute_scale_pos_weight():
    y = np.array([0, 0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == 4.0


def test_compute_ece_perfect_calibration():
    y_true = np.array([0, 1])
    y_prob = np.array([0.0, 1.0])
    assert _compute_ece(y_true, y_prob) == 0.0


def test_split_data_preserves_sizes(sample_train_data):
    X, y = sample_train_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_data_stratifies(sample_train_data):
    X, y = sample_train_data
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


@patch("klarna_ds_case_study.training.pipeline.N_CV_FOLDS", 2)
def test_tune_hyperparameters_returns_params(sample_train_data):
    X, y = sample_train_data
    study = tune_hyperparameters(X, y, n_trials=2)
    params = study.best_params
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
    }
    assert expected_keys.issubset(params.keys())


@patch("klarna_ds_case_study.training.pipeline.N_CV_FOLDS", 2)
def test_train_and_calibrate_produces_probabilities(sample_train_data):
    X, y = sample_train_data
    X_train, X_test, y_train, y_test = split_data(X, y)

    study = tune_hyperparameters(X_train, y_train, n_trials=2)
    model, full_params = train_model(X_train, y_train, study.best_params)
    probs = model.predict_proba(X_test)[:, 1]
    assert len(probs) == len(y_test)
    assert all(0 <= p <= 1 for p in probs)

    calibrated = calibrate_model(model, X_train, y_train)
    cal_probs = calibrated.predict_proba(X_test)[:, 1]
    assert len(cal_probs) == len(y_test)
    assert all(0 <= p <= 1 for p in cal_probs)
