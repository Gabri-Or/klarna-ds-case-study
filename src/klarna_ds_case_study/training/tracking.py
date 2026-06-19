import logging

import mlflow
import mlflow.sklearn

from klarna_ds_case_study.training.config import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def start_run(run_name: str | None = None) -> mlflow.ActiveRun:
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} parameters to MLflow")


def log_metrics(metrics: dict, prefix: str = "") -> None:
    prefixed = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics
    mlflow.log_metrics(prefixed)
    logger.info(f"Logged metrics to MLflow: {prefixed}")


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)
    logger.info(f"Logged artifact: {path}")


def register_model(model, artifact_path: str = "model") -> None:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=[
            "sklearn.calibration._CalibratedClassifier",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )
    logger.info(
        f"Registered model as '{REGISTERED_MODEL_NAME}' under artifact path '{artifact_path}'"
    )
