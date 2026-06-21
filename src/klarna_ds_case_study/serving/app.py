import os
import logging
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from klarna_ds_case_study.training.config import (
    CATEGORICAL_FEATURES,
    PROCESSED_DATA_PATH,
    SERVING_MODEL_DIR,
)

logger = logging.getLogger(__name__)

MODEL_URI = os.environ.get("MODEL_URI", str(SERVING_MODEL_DIR))

model = None
categorical_dtypes: dict[str, pd.CategoricalDtype] = {}


def _load_categorical_dtypes():
    df = pl.read_parquet(PROCESSED_DATA_PATH, columns=CATEGORICAL_FEATURES)
    return {
        col: pd.CategoricalDtype(categories=sorted(df[col].unique().to_list()))
        for col in CATEGORICAL_FEATURES
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, categorical_dtypes
    categorical_dtypes = _load_categorical_dtypes()
    logger.info(f"Loading model from {MODEL_URI}")
    model = mlflow.sklearn.load_model(MODEL_URI)
    logger.info("Model loaded successfully")
    yield


app = FastAPI(title="Loan Default Prediction API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    loan_amount: int = Field(gt=0)
    card_expiry_month: int = Field(ge=1, le=12)
    card_expiry_year: int = Field(ge=2023)
    existing_klarna_debt: int = Field(ge=0)
    num_active_loans: int = Field(ge=0)
    days_since_first_loan: int = Field(ge=0)
    new_exposure_7d: int = Field(ge=0)
    new_exposure_14d: int = Field(ge=0)
    num_confirmed_payments_3m: int = Field(ge=0)
    num_confirmed_payments_6m: int = Field(ge=0)
    num_failed_payments_3m: int = Field(ge=0)
    num_failed_payments_6m: int = Field(ge=0)
    num_failed_payments_1y: int = Field(ge=0)
    amount_repaid_14d: int = Field(ge=0)
    amount_repaid_1m: int = Field(ge=0)
    amount_repaid_3m: int = Field(ge=0)
    amount_repaid_6m: int = Field(ge=0)
    amount_repaid_1y: int = Field(ge=0)
    merchant_group: str
    merchant_category: str


class PredictionResponse(BaseModel):
    default_probability: float


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([request.model_dump()])
    for col in CATEGORICAL_FEATURES:
        value = df[col].iloc[0]
        if value not in categorical_dtypes[col].categories:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown {col}: '{value}'. Must be one of {categorical_dtypes[col].categories.tolist()}",
            )
        df[col] = df[col].astype(categorical_dtypes[col])

    probability = float(model.predict_proba(df)[:, 1][0])
    return PredictionResponse(default_probability=probability)
