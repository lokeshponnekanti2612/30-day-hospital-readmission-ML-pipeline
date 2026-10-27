"""Minimal inference API over the trained readmission-risk pipeline.

Run: uvicorn app.main:app --reload
Demonstrates the real-time-scoring half of the deployment pattern described
in the project README (the batch-scoring half is `scripts/predict.py`).
This is a local demo service, not a deployed/hosted endpoint.
"""
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmission import config
from readmission.features import build_feature_table

app = FastAPI(
    title="30-Day Readmission Risk API",
    description="Scores a single hospital encounter for 30-day readmission risk.",
    version="1.0.0",
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, f"Model not found at {config.MODEL_PATH}. Run scripts/train.py first.")
        _bundle = joblib.load(config.MODEL_PATH)
    return _bundle


class Encounter(BaseModel):
    """Raw encounter fields, same schema as diabetic_data.csv minus the
    target and identifiers. Extra/missing drug columns are tolerated -
    anything the model doesn't recognize is imputed as 'Unknown'."""

    model_config = ConfigDict(extra="allow")

    race: Optional[str] = None
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    max_glu_serum: Optional[str] = None
    A1Cresult: Optional[str] = None
    insulin: Optional[str] = None
    change: Optional[str] = None
    diabetesMed: Optional[str] = None


class RiskResponse(BaseModel):
    readmit_risk_prob: float
    readmit_risk_flag: int
    threshold: float
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict", response_model=RiskResponse)
def predict(encounter: Encounter):
    bundle = get_bundle()
    pipeline, threshold, feature_columns = bundle["pipeline"], bundle["threshold"], bundle["feature_columns"]

    raw_df = pd.DataFrame([encounter.model_dump()])
    features = build_feature_table(df=raw_df).reindex(columns=feature_columns, fill_value="Unknown")

    prob = float(pipeline.predict_proba(features)[0, 1])
    return RiskResponse(
        readmit_risk_prob=prob,
        readmit_risk_flag=int(prob >= threshold),
        threshold=threshold,
        model=bundle["pipeline"].named_steps["model"].__class__.__name__,
    )
