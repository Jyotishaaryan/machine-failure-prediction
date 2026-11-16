"""Merge per-model results, pick the final model, choose the operating
threshold, and write the final artifact + metrics.json.

Usage:  python -m src.finalize   (after running src.train for all four models)
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_val_predict

from .data import TARGET, load_data, quality_report
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_engineered_features
from .train import CV, RANDOM_STATE, evaluate
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def main() -> None:
    df = add_engineered_features(load_data())
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    for name in MODELS:
        with open(ROOT / "reports" / f"metrics_{name}.json") as f:
            results.append(json.load(f))

    # Model selection uses CROSS-VALIDATION performance on the training set
    # only. The test set stays untouched until the final unbiased evaluation,
    # so choosing the winner does not leak test information into the choice.
    best_row = max(results, key=lambda r: r["cv_pr_auc"])
    best_name = best_row["model"]
    best_model = joblib.load(ROOT / "models" / f"{best_name}.joblib")
    print(f"Final model by CV PR-AUC (train only): {best_name}")

    oof_prob = cross_val_predict(
        best_model, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f2_scores = [fbeta_score(y_train, (oof_prob >= t).astype(int), beta=2) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f2_scores))])
    tuned_row = evaluate(
        f"{best_name} @ F2 threshold {best_threshold:.2f}",
        y_test, best_model.predict_proba(X_test)[:, 1], best_threshold,
    )

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "threshold": best_threshold,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    joblib.dump(artifact, ROOT / "models" / "final_model.joblib")

    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(
            {
                "data_quality": quality_report(df),
                "split": {"test_size": 0.2, "stratified": True, "random_state": RANDOM_STATE},
                "results_at_0.5": results,
                "final_model": best_name,
                "operating_threshold": best_threshold,
                "final_at_operating_threshold": tuned_row,
            },
            f, indent=2,
        )
    print(f"Operating threshold: {best_threshold:.2f}")
    print(json.dumps(tuned_row, indent=2))


if __name__ == "__main__":
    main()
