"""Feature engineering for machine-failure prediction.

Two engineered features are added because the physical failure mechanisms in
this dataset are defined in terms of derived quantities:

* temp_diff  = Process temperature - Air temperature.
  Heat-dissipation failures (HDF) occur when the temperature *difference*
  is too small for the given rotational speed, not at any absolute
  temperature, so the difference is the informative signal.

* power_w    = Torque [Nm] x rotational speed converted to rad/s.
  Power failures (PWF) are defined exactly on mechanical power
  (P = torque x angular velocity), so this is the physically correct feature.

The original sensor columns are kept alongside the engineered ones so the
model can use either representation.
"""
import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    # rpm -> rad/s : omega = 2*pi*n/60 ; P [W] = torque * omega
    df["power_w"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    return df


# Feature set fed to the models (after engineering).
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power_w",
]
CATEGORICAL_FEATURES = ["Type"]
