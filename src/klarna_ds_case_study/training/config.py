from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "loans_processed.parquet"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
SERVING_MODEL_DIR = MODELS_DIR / "serving_model"
CATEGORICAL_CATEGORIES_PATH = SERVING_MODEL_DIR / "categorical_categories.json"

TARGET = "default_21d"
NON_FEATURE_COLUMNS = ["loan_id", "loan_issue_date", "default_14d", TARGET]
CATEGORICAL_FEATURES = ["merchant_group", "merchant_category"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

N_TRIALS = 50
N_CV_FOLDS = 5

EXPERIMENT_NAME = "loan-default-training"
REGISTERED_MODEL_NAME = "loan-default-prediction"

OPTUNA_SEARCH_SPACE = {
    "n_estimators": (100, 800),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 20),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 10.0),
}
