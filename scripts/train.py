#!/usr/bin/env python3
"""Train, tune, and select the readmission-risk model.

Usage: python scripts/train.py

Splits the data by patient (not by encounter) so no patient's history can
leak across the train/test boundary, tunes three candidate model families
under the same class-imbalance-aware setup, selects the one with the best
cross-validated PR-AUC (the right metric here given an ~11.6% positive
rate - ROC-AUC alone looks good for any classifier on this data and hides
how it's actually doing on the minority class), picks an operating
decision threshold from out-of-fold training predictions (never from the
test set), and reports final held-out metrics at that threshold.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_val_predict

from readmission import config
from readmission.features import build_feature_table
from readmission.pipeline import build_preprocessor, candidate_models


def pick_threshold(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[idx]) if len(thresholds) else 0.5


def main():
    t0 = time.time()
    print("Loading + cleaning + engineering features...")
    df = build_feature_table()
    groups = df["patient_nbr"]
    y = df[config.TARGET]
    X = df.drop(columns=["patient_nbr", config.TARGET])
    print(f"  {len(X):,} encounters, {X['prior_utilization'].ne(0).sum():,} with prior utilization, "
          f"positive rate={y.mean():.3%}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    print(f"  train={len(X_train):,} test={len(X_test):,} (split by patient_nbr)")

    preprocessor = build_preprocessor(X_train)
    inner_cv = GroupKFold(n_splits=3)

    cv_results, best_name, best_pipe, best_score = {}, None, None, -np.inf
    for name, (estimator, param_dist) in candidate_models().items():
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        search = RandomizedSearchCV(
            pipe,
            param_dist,
            n_iter=5,
            scoring="average_precision",
            cv=list(inner_cv.split(X_train, y_train, groups_train)),
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        cv_results[name] = {"cv_pr_auc": round(search.best_score_, 4), "best_params": search.best_params_}
        print(f"  [{name}] CV PR-AUC={search.best_score_:.4f}  params={search.best_params_}")
        if search.best_score_ > best_score:
            best_name, best_pipe, best_score = name, search.best_estimator_, search.best_score_

    print(f"\nSelected: {best_name} (CV PR-AUC={best_score:.4f})")
    print("Choosing decision threshold from out-of-fold training predictions...")
    oof_prob = cross_val_predict(
        best_pipe, X_train, y_train,
        cv=list(inner_cv.split(X_train, y_train, groups_train)),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    threshold = pick_threshold(y_train, oof_prob)
    print(f"  threshold={threshold:.4f}")

    best_pipe.fit(X_train, y_train)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": best_name,
        "threshold": threshold,
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "cv_results": cv_results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "train_seconds": round(time.time() - t0, 1),
    }

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "threshold": threshold, "feature_columns": list(X.columns)}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nTest set @ threshold={threshold:.3f}: "
          f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
          f"roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"\nSaved pipeline -> {config.MODEL_PATH}")
    print(f"Saved metrics   -> {config.METRICS_PATH}")
    print(f"Total time: {metrics['train_seconds']}s")


if __name__ == "__main__":
    main()
