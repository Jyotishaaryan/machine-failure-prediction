"""Data loading and data-quality checks for the AI4I 2020 predictive
maintenance dataset (UCI ML Repository, dataset id 601).

Column reference (from the UCI documentation):
    UDI                     row identifier (1..10000)
    Product ID              product serial number (letter = quality variant)
    Type                    product quality variant: L (low), M (medium), H (high)
    Air temperature [K]     ambient air temperature
    Process temperature [K] temperature of the machining process
    Rotational speed [rpm]  rotational speed of the tool
    Torque [Nm]             torque applied by the tool
    Tool wear [min]         accumulated tool wear
    Machine failure         TARGET: 1 if any failure occurred, else 0
    TWF, HDF, PWF, OSF, RNF failure-mode flags (which kind of failure)
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

TARGET = "Machine failure"

# The failure-mode flags (TWF/HDF/PWF/OSF/RNF) are only recorded WHEN a
# failure happens, so they are a perfect leak of the target. Using them as
# features would inflate every metric; they are kept only for error analysis.
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Row identifiers carry no physical information about machine state.
ID_COLS = ["UDI", "Product ID"]


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV. encoding='utf-8-sig' strips the BOM on the UDI header."""
    return pd.read_csv(path or DATA_PATH, encoding="utf-8-sig")


def quality_report(df: pd.DataFrame) -> dict:
    """Basic data-quality facts used in the EDA and the evaluation report."""
    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": df[TARGET].value_counts().to_dict(),
        "positive_rate": float(df[TARGET].mean()),
    }
    return report
