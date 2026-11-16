"""Generate evaluation figures and the evaluation report from the saved
per-model artifacts and metrics. Run after src.train + src.finalize.

Usage:  python -m src.evaluate
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from .data import LEAKAGE_COLS, TARGET, load_data
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_engineered_features
from .train import RANDOM_STATE

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = add_engineered_features(load_data())
    raw = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    test_idx = X_test.index

    with open(ROOT / "reports" / "metrics.json") as f:
        metrics = json.load(f)
    artifact = joblib.load(ROOT / "models" / "final_model.joblib")
    final_model, threshold = artifact["model"], artifact["threshold"]

    # --- 1. PR + ROC curves for all models --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in MODEL_NAMES:
        m = joblib.load(ROOT / "models" / f"{name}.joblib")
        prob = m.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, prob)
        axes[0].plot(r, p, label=name)
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[1].plot(fpr, tpr, label=name)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (test set)")
    axes[0].axhline(y.mean() * 0 + y_test.mean(), ls="--", c="grey", lw=1,
                    label=f"baseline ({y_test.mean():.3f})")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison_curves.png", dpi=150)
    plt.close(fig)

    # --- 2. Confusion matrix of final model at operating threshold ---------
    final_prob = final_model.predict_proba(X_test)[:, 1]
    final_pred = (final_prob >= threshold).astype(int)
    cm = np.array(metrics["final_at_operating_threshold"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no failure", "failure"],
           yticklabels=["no failure", "failure"],
           xlabel="Predicted", ylabel="Actual",
           title=f"{artifact['model_name']} @ threshold {threshold:.2f}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix_final.png", dpi=150)
    plt.close(fig)

    # --- 3. Feature importance + SHAP for the final model ------------------
    prep = final_model.named_steps["prep"]
    clf = final_model.named_steps["clf"]
    feat_names = NUMERIC_FEATURES + list(
        prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.plot.barh(ax=ax)
    ax.set_title(f"Feature importance - {artifact['model_name']}")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close(fig)

    X_test_t = prep.transform(X_test)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_t)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    shap.summary_plot(sv, X_test_t, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 4. Error analysis: which failures are missed, which alarms are false
    analysis = raw.loc[test_idx, LEAKAGE_COLS].copy()
    analysis["y_true"] = y_test.values
    analysis["y_pred"] = final_pred
    analysis["prob"] = final_prob.round(4)
    fn = analysis[(analysis.y_true == 1) & (analysis.y_pred == 0)]
    fp = analysis[(analysis.y_true == 0) & (analysis.y_pred == 1)]
    fn_modes = {c: int(fn[c].sum()) for c in LEAKAGE_COLS}
    with open(ROOT / "reports" / "error_analysis.json", "w") as f:
        json.dump(
            {
                "n_false_negatives": int(len(fn)),
                "false_negative_failure_modes": fn_modes,
                "n_false_positives": int(len(fp)),
                "false_negative_probabilities": sorted(fn["prob"].round(3).tolist()),
            },
            f, indent=2,
        )
    print("figures + error_analysis.json written")
    print("FN modes:", fn_modes, "| FP count:", len(fp))


if __name__ == "__main__":
    main()
