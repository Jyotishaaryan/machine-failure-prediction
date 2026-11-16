"""Make predictions on new data with the saved final model.

Usage:
    python -m src.predict --input new_machines.csv --output predictions.csv

The input CSV must contain the raw sensor columns:
    Type, Air temperature [K], Process temperature [K],
    Rotational speed [rpm], Torque [Nm], Tool wear [min]
(UDI / Product ID are optional and passed through.)

Output columns: machine_failure_prediction (0/1 at the chosen operating
threshold) and failure_probability.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import add_engineered_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with raw sensor columns")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    artifact = joblib.load(ROOT / "models" / "final_model.joblib")
    model, threshold = artifact["model"], artifact["threshold"]
    features = artifact["features"]

    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    X = add_engineered_features(raw)[features]
    prob = model.predict_proba(X)[:, 1]

    out = raw.copy()
    out["failure_probability"] = prob.round(4)
    out["machine_failure_prediction"] = (prob >= threshold).astype(int)
    out.to_csv(args.output, index=False)
    print(f"Model: {artifact['model_name']} | threshold: {threshold:.2f} | "
          f"{int(out['machine_failure_prediction'].sum())} of {len(out)} rows predicted to fail")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
