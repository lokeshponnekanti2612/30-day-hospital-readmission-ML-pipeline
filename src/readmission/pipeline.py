"""Preprocessing + candidate model pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in config.FORCE_CATEGORICAL:
        if c in X.columns and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # dense output: HistGradientBoostingClassifier doesn't accept sparse input
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def candidate_models():
    """name -> (estimator, randomized-search param distributions).

    All three carry explicit class-imbalance handling (~11.6% positive
    rate) instead of the unweighted baseline the original notebook used,
    which is what drove its near-zero recall on the minority class.

    Gradient boosting here is scikit-learn's HistGradientBoostingClassifier
    rather than XGBoost: same family of algorithm (histogram-based gradient
    boosted trees), but no native OpenMP library dependency to install -
    XGBoost's macOS wheel requires Homebrew's libomp, which isn't present
    on this machine and isn't worth installing a new package manager for.
    """
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [8, 16, None],
                "model__min_samples_leaf": [1, 5, 20],
            },
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced", random_state=config.RANDOM_STATE
            ),
            {
                "model__max_iter": [150, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.1, 0.2],
            },
        ),
    }
