"""Data loading, cleaning, and feature engineering for the readmission
pipeline. Logic is carried over from the exploratory work in
notebooks/eda.ipynb, with one correctness fix: patient_nbr is kept around
until the train/test split so the same patient's encounters never end up on
both sides of it (the original notebook dropped patient_nbr before
splitting, which would have let a patient leak between train and test).
"""
import numpy as np
import pandas as pd

from . import config

ICD9_CHAPTERS = [
    (1, 139, "Infectious"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine"),
    (280, 289, "Blood"),
    (290, 319, "Mental"),
    (320, 389, "Nervous"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms"),
    (800, 999, "Injury"),
]


def load_raw(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.DATA_RAW)


def age_to_num(age_range) -> float:
    """'[40-50)' -> 45.0 (midpoint of the age bucket)."""
    if pd.isna(age_range):
        return np.nan
    s = str(age_range).strip("[]()")
    if "-" not in s:
        return np.nan
    lo, hi = s.split("-")
    return (int(lo) + int(hi)) / 2


def group_icd9(code) -> str:
    """Map a raw ICD-9 diagnosis code to its diagnostic chapter."""
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        x = float(code)
    except ValueError:
        return "Other"
    for lo, hi, label in ICD9_CHAPTERS:
        if lo <= x <= hi:
            return label
    return "Other"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.mask(df == "?", np.nan).drop_duplicates()
    if "race" in df.columns:
        df["race"] = df["race"].replace("0", np.nan)
    df = df.drop(columns=[c for c in config.HIGH_MISSING_COLS if c in df.columns])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if config.TARGET_RAW in df.columns:
        df[config.TARGET] = (df[config.TARGET_RAW] == "<30").astype(int)
    df["age_num"] = df["age"].apply(age_to_num)

    for c in ("diag_1", "diag_2", "diag_3"):
        if c in df.columns:
            df[f"{c}_grp"] = df[c].apply(group_icd9)

    # Prior-utilization signal: total non-inpatient-stay encounters in the
    # year before admission is a stronger readmission signal than any one
    # of the three counts alone.
    df["prior_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    df = df.drop(columns=["diag_1", "diag_2", "diag_3", "age"], errors="ignore")

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    df = df.drop(columns=[c for c in config.CONSTANT_COLS if c in df.columns])
    return df


def build_feature_table(raw_path=None, df: pd.DataFrame = None) -> pd.DataFrame:
    """Full load -> clean -> engineer pass, target + patient_nbr retained
    for the caller to split on. ID columns other than patient_nbr are
    dropped here (leakage risk, no predictive value). Pass `df` directly
    (e.g. for inference on new encounters with no `readmitted` column)
    instead of `raw_path` to skip the CSV load."""
    if df is None:
        df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    df = df.drop(columns=["encounter_id", config.TARGET_RAW], errors="ignore")
    return df
