import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmission.features import age_to_num, build_feature_table, group_icd9


def test_age_to_num_midpoint():
    assert age_to_num("[40-50)") == 45.0
    assert age_to_num("[0-10)") == 5.0


def test_age_to_num_missing():
    assert pd.isna(age_to_num(None))


def test_group_icd9_chapters():
    assert group_icd9("250.83") == "Endocrine"
    assert group_icd9("410") == "Circulatory"
    assert group_icd9("V27") == "Other"


def test_group_icd9_missing():
    assert group_icd9(None) == "Unknown"


def test_build_feature_table_no_target_column_needed_for_inference():
    raw = pd.DataFrame([{
        "encounter_id": 1, "patient_nbr": 1, "race": "Caucasian", "gender": "Female",
        "age": "[40-50)", "weight": "?", "admission_type_id": 1, "discharge_disposition_id": 1,
        "admission_source_id": 1, "time_in_hospital": 2, "payer_code": "?", "medical_specialty": "?",
        "num_lab_procedures": 10, "num_procedures": 0, "num_medications": 5, "number_outpatient": 0,
        "number_emergency": 0, "number_inpatient": 0, "diag_1": "250.83", "diag_2": "?", "diag_3": "?",
        "number_diagnoses": 3, "max_glu_serum": "None", "A1Cresult": "None", "insulin": "No",
        "change": "No", "diabetesMed": "No",
    }])
    features = build_feature_table(df=raw)
    assert "readmit_binary" not in features.columns
    assert "diag_1_grp" in features.columns
    assert features.loc[0, "age_num"] == 45.0
