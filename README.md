# 30-Day Hospital Readmission Pipeline

An end-to-end ML pipeline that predicts whether a hospital encounter will
lead to readmission within 30 days, built on the public UCI **"Diabetes
130-US hospitals for years 1999–2008"** dataset (Strack et al., 2014) —
101,766 encounters across 130 US hospitals, 50 raw fields per encounter.

This is a from-scratch, from-public-data recreation of the methodology
behind a 30-day readmission system I built professionally on proprietary,
multi-hospital EHR data (structured + unstructured clinical notes) — that
code and data aren't mine to publish, so this repo rebuilds the same kind
of pipeline (cleaning → feature engineering → imbalance-aware model
selection → threshold tuning → served predictions) on the standard public
benchmark for this exact problem. Metrics below are this repo's own,
measured on this dataset — not a restatement of results from the
production system.

## The hard part of this dataset

Only **11.2%** of encounters are readmitted within 30 days. A classifier
that always predicts "no" gets 89% accuracy and is useless. The
[original exploratory notebook](notebooks/eda.ipynb) in this repo shows
exactly that failure mode: an unweighted Logistic Regression baseline hits
89% accuracy with **1.6% recall** on the class that actually matters — it
caught 36 of 2,271 true readmissions.

## What the completed pipeline does differently

| | Baseline (notebook) | Completed pipeline (`src/`, `scripts/`) |
|---|---|---|
| Train/test split | Random, by encounter | By **patient** (`GroupShuffleSplit`/`GroupKFold`) — the same patient's repeat encounters can't leak across the split |
| Class imbalance | None | `class_weight="balanced"` / equivalent on every candidate model |
| Model selection | Single Logistic Regression | Logistic Regression, Random Forest, and gradient-boosted trees, each tuned via `RandomizedSearchCV` on **PR-AUC** (the right metric at an 11% positive rate — ROC-AUC alone looks good for almost any classifier here) |
| Decision threshold | Default 0.5 | Picked from out-of-fold **training** predictions (never the test set), by max F1 |
| Zero-variance column drop | Recomputed per batch (breaks on small inference batches — every column looks constant with 1 row) | Fixed list from the full training set |
| Serving | None | CLI batch scorer (`scripts/predict.py`) + FastAPI endpoint (`app/main.py`) |

Gradient boosting here is scikit-learn's `HistGradientBoostingClassifier`,
not XGBoost: same family of algorithm, but no native OpenMP library to
install (XGBoost's macOS wheel needs Homebrew's `libomp`, which wasn't
worth pulling in a new package manager for on this machine).

## Results

Patient-level 80/20 split (81,613 train / 20,153 test encounters).

| Model | CV PR-AUC |
|---|---|
| Logistic Regression | 0.214 |
| Random Forest | 0.214 |
| **Gradient Boosting (selected)** | **0.222** |

Gradient Boosting on the held-out test set, at the F1-optimal threshold (0.529):

| Metric | Value |
|---|---|
| Recall (readmitted class) | **0.505** |
| Precision (readmitted class) | 0.185 |
| F1 (readmitted class) | 0.271 |
| ROC-AUC | 0.673 |
| PR-AUC | 0.210 |
| Accuracy | 0.710 |

Recall went from 1.6% (baseline) to 50.5% — the pipeline now catches about
half of all actual 30-day readmissions, at the cost of flagging a lot of
encounters that don't end up readmitted (precision 18.5%). That trade-off
is tunable: `scripts/train.py` picks the threshold that maximizes F1, but
a clinical deployment would move that threshold toward recall (catch more
at-risk patients, tolerate more false alarms) or precision (fewer alerts,
higher confidence) depending on intervention cost — see `pick_threshold`
in `scripts/train.py`. Full numbers: `models/metrics.json`.

## Repo layout

```
data/raw/diabetic_data.csv   raw UCI dataset
notebooks/eda.ipynb          original exploratory analysis (unweighted baseline)
src/readmission/
  config.py                  paths, fixed column lists
  features.py                clean(), engineer_features(), build_feature_table()
  pipeline.py                preprocessing ColumnTransformer + candidate models
scripts/
  train.py                   trains/tunes/selects the model, saves pipeline + metrics
  predict.py                 CLI batch scorer
app/main.py                  FastAPI real-time scoring endpoint
tests/test_features.py       unit tests for the feature functions
models/                      trained pipeline (.joblib) + metrics.json
Dockerfile
```

## Running it

```bash
pip install -r requirements.txt

python scripts/train.py                 # trains, tunes, saves models/readmission_pipeline.joblib
python scripts/predict.py new.csv       # batch-score a CSV of new encounters
uvicorn app.main:app --reload           # POST /predict with a single encounter
pytest tests/                           # unit tests for the feature functions

docker build -t readmission-api .
docker run -p 8000:8000 readmission-api
```

## Limitations

- Public, de-identified benchmark dataset — not a live clinical deployment, and not the proprietary EHR data the professional version of this pipeline was built on.
- No unstructured clinical notes in this dataset (the original system's NLP feature engineering doesn't have an equivalent here).
- PR-AUC of 0.21 means there's real room to improve — next steps would be richer medication-sequence features, calibration, and a cost-sensitive threshold chosen with an actual intervention-cost model rather than F1.
