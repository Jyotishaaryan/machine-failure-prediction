# Walkthrough - Task 1 (read this before the interview)

## The 60-second version

We predict machine failures from sensor readings (temperature, speed, torque,
tool wear). Only 3.4% of runs fail, so the whole project is about doing
imbalanced classification honestly: remove leakage, engineer two physically
meaningful features, compare four models with cross-validation, evaluate with
precision/recall/PR-AUC instead of accuracy, and pick an operating threshold
that favours catching failures over avoiding false alarms.

## The pipeline in 5 steps

1. **Data** (`src/data.py`): 10,000 rows, no missing values, no duplicates.
   We drop `UDI`/`Product ID` (identifiers) and the five failure-mode flags
   TWF/HDF/PWF/OSF/RNF - those flags are only recorded *after* a failure, so
   using them would be target leakage. They are kept only for error analysis.
2. **Features** (`src/features.py`): add `temp_diff` (process - air temp;
   heat-dissipation failures happen when this difference is too small) and
   `power_w` (torque x angular velocity; power failures are defined on
   mechanical power). One-hot encode `Type`.
3. **Training** (`src/train.py`): stratified 80/20 split; four models in
   sklearn Pipelines - Logistic Regression (linear baseline), Random Forest,
   XGBoost, LightGBM - each tuned with 3-fold stratified CV optimising
   PR-AUC.
4. **Selection** (`src/finalize.py`): the winner is chosen by CV PR-AUC on the
   training set only (LightGBM, 0.872). The operating threshold (0.22) is
   chosen on out-of-fold training predictions by maximising F2. Then, and only
   then, the test set is scored.
5. **Evaluation** (`src/evaluate.py` + `reports/`): PR/ROC curves, confusion
   matrix, feature importance, SHAP, error analysis.

## Numbers to remember

- 3.39% positives (339/10,000). Accuracy of "always healthy" = 96.6% - that
  is why we never use accuracy.
- CV PR-AUC: LightGBM 0.872, Random Forest 0.867, XGBoost 0.844, LR 0.495.
- Test (held-out, exploratory): recall 0.838 (57/68 failures caught),
  precision 0.731, PR-AUC 0.897, 21 false alarms.
- All 11 missed failures are tool-wear failures (TWF), which include a random
  component - that residue is irreducible without more sensors.

## Likely questions and honest answers

- **Why not accuracy?** A model predicting "no failure" always scores 96.6%.
  PR-AUC, recall and precision actually measure the rare class we care about.
- **Why tree models?** The failure mechanisms are threshold rules (wear
  limits, power range, heat-dissipation boundary). Axis-aligned splits learn
  those naturally; a linear boundary cannot - visible in LR's PR-AUC of 0.45.
- **How did you prevent leakage?** Dropped the post-failure mode flags and
  IDs; preprocessing lives inside the CV pipeline; model selection and the
  threshold were computed from training data only. One honesty note: the test
  set was inspected once during development, which the evaluation report
  discloses - the CV scores are the clean comparison.
- **Why F2 and a 0.22 threshold?** A missed failure costs unplanned downtime;
  a false alarm costs one inspection. F2 weights recall double, and 0.22 is
  where F2 peaks on out-of-fold predictions.
- **What would you improve?** More failure examples (68 in test = wide
  confidence intervals), a temporal split if timestamps existed, and
  calibration of predicted probabilities.

## Where to look in the code

| Question | File |
|---|---|
| Leakage handling | `src/data.py` (LEAKAGE_COLS), notebook section 6 |
| Feature engineering | `src/features.py` |
| Model comparison | `reports/metrics.json`, evaluation report section 4 |
| Threshold choice | `src/finalize.py` |
| Error analysis | `reports/error_analysis.json`, evaluation report section 6 |
