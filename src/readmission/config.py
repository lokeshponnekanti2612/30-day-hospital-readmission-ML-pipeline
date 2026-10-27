from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
MODEL_PATH = ROOT / "models" / "readmission_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

TARGET_RAW = "readmitted"
TARGET = "readmit_binary"
ID_COLS = ["encounter_id", "patient_nbr"]

# Columns dropped for being >39% missing (encoded as "?") with no reliable
# imputation strategy, same call made in the original exploratory notebook.
HIGH_MISSING_COLS = ["weight", "medical_specialty", "payer_code"]

# Constant across the full training set (confirmed in the EDA notebook,
# e.g. "citoglipton" and "examide" have exactly 1 unique value in all
# 101,766 rows) - a fixed list, not detected per-batch. Detecting
# zero-variance columns dynamically would be wrong at inference time: any
# column looks "constant" in a batch of 1.
CONSTANT_COLS = ["examide", "citoglipton"]

# IDs that are numeric-looking but are actually categorical codes.
FORCE_CATEGORICAL = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

RANDOM_STATE = 42
