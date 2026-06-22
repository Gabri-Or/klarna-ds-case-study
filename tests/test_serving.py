from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


VALID_REQUEST = {
    "loan_amount": 5000,
    "card_expiry_month": 6,
    "card_expiry_year": 2027,
    "existing_klarna_debt": 0,
    "num_active_loans": 1,
    "days_since_first_loan": 120,
    "new_exposure_7d": 0,
    "new_exposure_14d": 0,
    "num_confirmed_payments_3m": 3,
    "num_confirmed_payments_6m": 5,
    "num_failed_payments_3m": 0,
    "num_failed_payments_6m": 0,
    "num_failed_payments_1y": 0,
    "amount_repaid_14d": 2000,
    "amount_repaid_1m": 4000,
    "amount_repaid_3m": 8000,
    "amount_repaid_6m": 12000,
    "amount_repaid_1y": 15000,
    "merchant_group": "Entertainment",
    "merchant_category": "Event Tickets",
}


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.95, 0.05]])
    return model


@pytest.fixture
def client(mock_model):
    import pandas as pd

    fake_dtypes = {
        "merchant_group": pd.CategoricalDtype(
            categories=["Entertainment", "Electronics", "Unknown"]
        ),
        "merchant_category": pd.CategoricalDtype(
            categories=["Event Tickets", "Digital Services", "Other"]
        ),
    }

    with (
        patch(
            "klarna_ds_case_study.serving.app._load_categorical_dtypes",
            return_value=fake_dtypes,
        ),
        patch(
            "klarna_ds_case_study.serving.app.mlflow.sklearn.load_model",
            return_value=mock_model,
        ),
    ):
        import klarna_ds_case_study.serving.app as serving_module

        serving_module.model = None
        serving_module.categorical_dtypes = {}

        from klarna_ds_case_study.serving.app import app

        with TestClient(app) as tc:
            yield tc


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_probability(client):
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert "default_probability" in body
    assert 0.0 <= body["default_probability"] <= 1.0


def test_predict_calls_model_with_correct_shape(client, mock_model):
    client.post("/predict", json=VALID_REQUEST)
    mock_model.predict_proba.assert_called_once()
    df = mock_model.predict_proba.call_args[0][0]
    assert len(df) == 1
    assert "merchant_group" in df.columns
    assert df["merchant_group"].dtype.name == "category"


def test_predict_rejects_unknown_category(client):
    bad_request = {**VALID_REQUEST, "merchant_group": "NonExistent"}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422
    assert "Unknown merchant_group" in response.json()["detail"]


def test_predict_rejects_invalid_loan_amount(client):
    bad_request = {**VALID_REQUEST, "loan_amount": -1}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_REQUEST.items() if k != "loan_amount"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_invalid_card_month(client):
    bad_request = {**VALID_REQUEST, "card_expiry_month": 13}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422
