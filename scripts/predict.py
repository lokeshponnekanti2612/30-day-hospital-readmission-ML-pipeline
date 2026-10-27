#!/usr/bin/env python3
"""Score new encounters with the trained pipeline.

Usage: python scripts/predict.py path/to/new_encounters.csv [-o out.csv]

Input CSV must have the same raw columns as diabetic_data.csv (readmitted
not required). Adds `readmit_risk_prob` and `readmit_risk_flag` columns.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmission import config
from readmission.features import build_feature_table


def predict(raw_df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(config.MODEL_PATH)
    pipeline, threshold, feature_columns = bundle["pipeline"], bundle["threshold"], bundle["feature_columns"]

    patient_ids = raw_df.get("patient_nbr")
    features = build_feature_table(df=raw_df).drop(columns=["patient_nbr"], errors="ignore")
    features = features.reindex(columns=feature_columns, fill_value="Unknown")

    prob = pipeline.predict_proba(features)[:, 1]
    out = raw_df.copy()
    out["readmit_risk_prob"] = prob
    out["readmit_risk_flag"] = (prob >= threshold).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input_csv)
    scored = predict(raw_df)

    out_path = args.output or str(Path(args.input_csv).with_suffix("")) + "_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"Scored {len(scored):,} encounters -> {out_path}")
    print(f"Flagged high-risk: {scored['readmit_risk_flag'].sum():,} "
          f"({scored['readmit_risk_flag'].mean():.1%})")


if __name__ == "__main__":
    main()
