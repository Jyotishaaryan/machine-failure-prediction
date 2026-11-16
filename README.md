# Machine Failure Prediction - AI4I 2020 Predictive Maintenance

Predicts whether an industrial milling machine is likely to fail from its
operating and sensor measurements. Built on the
[UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(10,000 runs, 3.4% failure rate).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
# 1. EDA and experimentation (executed notebook with all outputs)
jupyter notebook notebooks/01_eda.ipynb

# 2. Train + tune the four models (each writes models/<name>.joblib and reports/metrics_<name>.json)
python -m src.train                 # all models
python -m src.train --model lightgbm   # or just one

# 3. Pick the final model (by cross-validation PR-AUC), choose the operating
#    threshold (out-of-fold F2), write models/final_model.joblib + reports/metrics.json
python -m src.finalize

# 4. Regenerate evaluation figures + error analysis
python -m src.evaluate

# 5. Predict on new data
python -m src.predict --input new_machines.csv --output predictions.csv
```

Input CSV for prediction needs the raw columns: `Type`,
`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
`Torque [Nm]`, `Tool wear [min]`.

## Approach in one paragraph

The five failure-mode flags (TWF/HDF/PWF/OSF/RNF) are post-hoc labels - they are
only ever 1 when a failure has already happened - so they are excluded from the
features to avoid target leakage, and used only for error analysis. Two
physically motivated features are engineered: `temp_diff` (process minus air
temperature; heat-dissipation failures happen when this difference is too small
for the given speed) and `power_w` (torque x angular velocity; power failures
are defined on mechanical power). The data is split stratified 80/20, and four
models are compared: Logistic Regression (scaled linear baseline), Random
Forest, XGBoost and LightGBM (all with class-imbalance handling). Tuning uses
3-fold stratified cross-validation optimising PR-AUC, because accuracy is
meaningless at 3.4% positives and PR-AUC reflects ranking quality on the rare
class. The final model is selected by **cross-validation** PR-AUC on the training set
(note: the test set was inspected once during development, so the held-out
numbers are exploratory estimates; the CV scores are the clean comparison - see
the evaluation report's chronology note), and the operating
threshold is chosen on out-of-fold training predictions by maximising F2 -
recall is weighted double because a missed failure (unplanned downtime) costs
far more than a false alarm (an extra inspection).

## Model selection reasoning and benchmarking

See `reports/evaluation_report.md` for the full table. In short:

- **LightGBM won** (CV PR-AUC 0.872, test PR-AUC 0.897) - leaf-wise boosting
  captures the sharp threshold boundaries in the failure physics with the
  fewest compromises between precision and recall.
- **Random Forest** is close on ranking but conservative: highest precision,
  lowest recall - bagging smooths the boundary and misses borderline failures.
- **XGBoost** sits between the two; depth-3/6 trees with 0.05-0.1 learning rate
  behaved almost identically to LightGBM at slightly lower recall.
- **Logistic Regression** lags far behind (test PR-AUC 0.45): the failure
  mechanisms are threshold/region based, not linear, so it can only trade
  precision for recall (recall 0.87 at precision 0.17 - too many false alarms
  to be usable).

## Results (held-out test set, final LightGBM @ threshold 0.22)

| Precision | Recall | F1 | F2 | Confusion matrix |
|---|---|---|---|---|
| 0.731 | 0.838 | 0.781 | 0.814 | TN 1911, FP 21, FN 11, TP 57 |

57 of 68 test failures are caught, at the cost of 21 false alarms (~1% of
healthy runs). Error analysis: all 11 missed failures are tool-wear failures
(TWF), which include a random component in the data-generating process; every
heat-dissipation, power and overstrain failure is caught.

## Repository layout

```
WALKTHROUGH.md               interview prep: pipeline, decisions, likely Q&A
data/ai4i2020.csv            dataset (UCI mirror)
notebooks/01_eda.ipynb       EDA + experimentation (executed)
src/data.py                  loading + data-quality checks
src/features.py              engineered features (temp_diff, power_w)
src/train.py                 per-model training + tuning
src/finalize.py              model selection, threshold, final artifact
src/evaluate.py              figures + error analysis
src/predict.py               prediction CLI for new data
models/                      saved pipelines (joblib)
reports/evaluation_report.md full evaluation report
reports/metrics.json         all metrics
reports/figures/             PR/ROC curves, confusion matrix, importances, SHAP
```

## Limitations

Synthetic dataset with rule-based failure generation, so scores are optimistic
vs. real sensor data; only 68 failures in the test split, so metrics have wide
confidence intervals; no timestamp column, so no temporal validation was
possible. See the evaluation report, section 8.
