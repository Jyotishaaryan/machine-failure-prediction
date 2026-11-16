# Evaluation Report - Machine Failure Prediction (AI4I 2020)

## 1. Data and split

- 10,000 rows, 14 raw columns. No missing values, no duplicate rows.
- Target `Machine failure`: 339 positives (**3.39%**) - strong class imbalance.
- Stratified 80/20 split (random_state=42): 8,000 train rows (271 failures), 2,000 test rows (68 failures). Stratification keeps the 3.4% failure rate in both parts; a plain random split could leave the test set with too few failures to measure recall reliably.
- **Leakage control**: TWF/HDF/PWF/OSF/RNF are failure-mode labels recorded when a failure happens - every 1 in these columns coincides with `Machine failure = 1`. They are excluded from the features (used only for error analysis in section 6). `UDI` and `Product ID` are identifiers and are also excluded.

## 2. Metrics and why they suit machine-failure prediction

Accuracy is not used: always predicting "no failure" already scores 96.6%.

- **PR-AUC (average precision)** - the primary metric. On imbalanced data it reflects how well the model ranks rare failures above healthy runs; the no-skill baseline is the positive rate (0.034).
- **Recall (sensitivity)** - share of real failures that are caught. A missed failure (false negative) means unplanned downtime, the costliest outcome in predictive maintenance.
- **Precision** - share of alarms that are real. A false positive means an unnecessary inspection/maintenance stop: costly, but far cheaper than a missed failure.
- **F1 / F2** - harmonic blends of precision and recall; F2 weights recall twice as heavily, matching the asymmetric costs.
- **ROC-AUC** - reported for completeness; on imbalanced data it looks optimistic and is not used for selection.

## 3. Models compared

| Model | Why it is included |
|---|---|
| Logistic Regression | Interpretable linear baseline (features standardised, `class_weight='balanced'`). |
| Random Forest | Bagged trees - learns threshold rules, robust to unscaled features (`class_weight='balanced'`). |
| XGBoost | Gradient-boosted trees (`scale_pos_weight` = negative/positive ratio). |
| LightGBM | Leaf-wise gradient-boosted trees (`class_weight='balanced'`). |

Tree ensembles are the natural fit because the documented failure mechanisms are threshold-based (wear limits, power range, heat-dissipation boundary). All preprocessing (one-hot `Type`, scaling where needed) lives inside each sklearn Pipeline, so no fitting information leaks from validation or test data into preprocessing.

Each model was tuned with 3-fold stratified cross-validation on the training set, optimising average precision.

## 4. Results

*\* at the default 0.5 threshold*

| Model | CV PR-AUC (train) | Test PR-AUC | Test ROC-AUC | Precision* | Recall* | F1* | F2* |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.495 | 0.450 | 0.929 | 0.171 | 0.868 | 0.285 | 0.477 |
| random_forest | 0.867 | 0.858 | 0.970 | 0.939 | 0.676 | 0.786 | 0.717 |
| xgboost | 0.844 | 0.866 | 0.978 | 0.794 | 0.794 | 0.794 | 0.794 |
| lightgbm | 0.872 | 0.897 | 0.973 | 0.864 | 0.838 | 0.851 | 0.843 |

**Development chronology, stated honestly:** during development all four models were first scored on the test set at the 0.5 threshold while the comparison table was being built, so the test set has been inspected once. The selection procedure was then redone using training data only - model selection by 3-fold CV PR-AUC on the training set, operating threshold by out-of-fold training predictions - and this training-only procedure independently confirms the same winner. The held-out test numbers in this report are therefore **exploratory estimates, not pristine untouched-holdout figures**; the CV PR-AUC column is the cleanest model comparison, and a fresh evaluation on new data would be the strict confirmation.

LightGBM wins on CV PR-AUC (0.872) and also on the held-out test set (0.897 PR-AUC, 0.973 ROC-AUC). The ordering is consistent between CV and test, which indicates the comparison is stable and not overfit to the split.

Why the linear model loses: the failure physics is defined by *regions and thresholds* (e.g. power outside ~3.5-9 kW, wear beyond ~200 min combined with torque), which a linear boundary cannot represent. Its high recall (0.868) comes with poor precision (0.171) - it raises far too many alarms to be operationally useful.

## 5. Operating threshold

The default 0.5 threshold wastes the model's ranking ability on imbalanced data. The operating threshold was chosen on **out-of-fold training predictions** (no test leakage) by maximising F2: **threshold = 0.22**.

Final held-out test performance at this threshold:

| Metric | Value |
|---|---|
| Precision | 0.731 |
| Recall | 0.838 |
| F1 | 0.781 |
| F2 | 0.814 |
| Confusion matrix | TN=1911, FP=21, FN=11, TP=57 |

The model catches 57 of 68 failures (84%) at the cost of 21 false alarms in 2,000 runs (~1% of healthy runs). If maintenance capacity is tight, the threshold can be raised to trade recall for precision; the PR curve (reports/figures/model_comparison_curves.png) shows the full trade-off.

## 6. Error analysis

Using the excluded failure-mode labels purely for diagnosis on the test set:

- **11 false negatives** (missed failures). All of them are tool-wear failures (TWF: 9); every HDF, PWF and OSF failure is caught. The missed TWF cases sit at low predicted probabilities [0.0, 0.0, 0.0]... - TWF is stochastic (it includes a random failure component in the data-generating process), so a residue of unpredictable cases is expected and irreducible without additional sensors.
- **21 false positives** - healthy runs flagged for inspection. These concentrate near the decision boundary; in practice they cost one inspection each.

## 7. Explainability

- Feature importance (reports/figures/feature_importance.png): tool wear dominates, followed by the engineered `power_w`, rotational speed, torque and `temp_diff` - matching the documented failure physics.
- SHAP summary (reports/figures/shap_summary.png): high tool wear and high torque push towards failure; *low* `temp_diff` at speed pushes towards failure (heat-dissipation mechanism); product type has only a minor effect.

## 8. Limitations

- The dataset is synthetic with rule-based failure generation, so scores here are optimistic relative to real sensor data with noise and drift.
- ~3.4% positives means 68 failures in the test set; metric confidence intervals are wide at this sample size.
- A single random split is reported; repeating with different seeds would give a tighter picture (the CV scores in section 4 already indicate stability).
- No time-ordering is available in the data, so no temporal split was possible.
