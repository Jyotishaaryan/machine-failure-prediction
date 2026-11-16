"""Train, tune and compare classification models for machine-failure prediction.

Usage:  python -m src.train [--model NAME]

Trains one or all of the four candidate models and writes, per model,
models/<name>.joblib and reports/metrics_<name>.json. Final model selection
and the operating threshold are handled by src/finalize.py.

  1. Loads the data, adds engineered features, drops identifiers/leakage cols.
  2. Makes a stratified 80/20 train/test split (class ratio is ~3.4% positive,
     so a plain random split could leave the test set with too few failures).
  3. Trains four models inside sklearn Pipelines (preprocessing is baked into
     each pipeline, so the saved artifact is self-contained):
       - Logistic Regression  (linear baseline, needs scaling)
       - Random Forest        (bagged trees, handles thresholds/non-linearity)
       - XGBoost              (gradient-boosted trees)
       - LightGBM             (gradient-boosted trees, leaf-wise)
     Tree models are well suited here because the physical failure mechanisms
     are threshold-based (wear limits, power ranges), which is exactly what
     axis-aligned splits learn well. Logistic Regression is kept as the
     interpretable linear baseline.
  4. Tunes each model with stratified 3-fold cross-validation, optimising
     average precision (area under the precision-recall curve). PR-AUC is the
     tuning objective because accuracy is meaningless at 3.4% positives and
     ROC-AUC can look optimistic on imbalanced data.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import TARGET, load_data, quality_report
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_engineered_features

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def model_zoo(n_neg: int, n_pos: int):
    """Return {name: (pipeline, param_grid)}."""
    imbalance_ratio = n_neg / n_pos
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale=True)),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale=False)),
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [None, 8],
                "clf__min_samples_leaf": [1, 4],
            },
        ),
        "xgboost": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale=False)),
                    (
                        "clf",
                        XGBClassifier(
                            scale_pos_weight=imbalance_ratio,
                            eval_metric="logloss",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
        "lightgbm": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale=False)),
                    (
                        "clf",
                        LGBMClassifier(
                            class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [-1, 8],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def evaluate(name: str, y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="train only this model (per-model partial run)")
    args = parser.parse_args()

    df = add_engineered_features(load_data())
    qr = quality_report(df)
    print(f"Data quality: {qr}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    print(f"Train {len(X_train)} rows ({n_pos} failures), test {len(X_test)} rows "
          f"({int(y_test.sum())} failures)")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    zoo = model_zoo(n_neg, n_pos)
    names = [args.model] if args.model else list(zoo)
    for name in names:
        pipe, grid = zoo[name]
        search = GridSearchCV(
            pipe, grid, scoring="average_precision", cv=CV, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        joblib.dump(best, ROOT / "models" / f"{name}.joblib")
        prob = best.predict_proba(X_test)[:, 1]
        row = evaluate(name, y_test, prob)
        row["best_params"] = search.best_params_
        row["cv_pr_auc"] = float(search.best_score_)
        with open(ROOT / "reports" / f"metrics_{name}.json", "w") as f:
            json.dump(row, f, indent=2)
        print(f"{name:20s} PR-AUC={row['pr_auc']:.3f} ROC-AUC={row['roc_auc']:.3f} "
              f"recall={row['recall']:.3f} precision={row['precision']:.3f} F2={row['f2']:.3f}")


if __name__ == "__main__":
    main()
