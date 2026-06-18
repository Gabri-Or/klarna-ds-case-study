from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mlcasestudy Final.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "loans_processed.parquet"

MIN_MERCHANT_CATEGORY_COUNT = 100
UNKNOWN_CATEGORY = "Unknown"
OTHER_CATEGORY = "Other"
