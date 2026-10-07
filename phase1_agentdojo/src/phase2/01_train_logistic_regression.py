"""
Phase 2, Step 1 - Logistic Regression baseline for `security`.

Source:   PROJECT_PLAN.md Phase 2; AgentDojo_Dataset_Research_Report.md Part 19
Reason:   Simplest, most interpretable baseline - establishes a floor that
          the tree/deep models (Steps 2-4) should beat, and its coefficients
          give a first, cheap read on which of the 176 engineered features
          (Phase 1) drive `security` predictions.

          `class_weight="balanced"` is used because the scoped population is
          imbalanced (~15.9% security=True, report Part 5/19) - without it,
          a linear model would be tempted to just predict the majority
          class. This is the same imbalance-handling decision that will be
          applied consistently across all four Phase 2 models (RF's
          class_weight, XGBoost's scale_pos_weight, the MLP's pos_weight),
          documented once here since it recurs.

Input:    outputs/phase1/04_X_{train,val}.pkl, 04_y_{train,val}.pkl
Output:   models/phase2/logistic_regression.joblib
          evaluation/phase2/logistic_regression_val_metrics.json
          figures/phase2/logistic_regression_confusion_matrix_val.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "phase1"
MODEL_OUT = ROOT / "models" / "phase2" / "logistic_regression.joblib"
METRICS_OUT = ROOT / "evaluation" / "phase2" / "logistic_regression_val_metrics.json"
FIGURE_OUT = ROOT / "figures" / "phase2" / "logistic_regression_confusion_matrix_val.png"

RANDOM_STATE = 42


def main() -> None:
    X_train = pd.read_pickle(DATA_DIR / "04_X_train.pkl")
    y_train = pd.read_pickle(DATA_DIR / "04_y_train.pkl")
    X_val = pd.read_pickle(DATA_DIR / "04_X_val.pkl")
    y_val = pd.read_pickle(DATA_DIR / "04_y_val.pkl")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_proba, model_name="logistic_regression", split="val")
    print_summary(metrics)

    # Top coefficients - cheap first look at which features drive predictions
    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
    top_positive = coefs.tail(10).to_dict()
    top_negative = coefs.head(10).to_dict()
    metrics["top_positive_coefficients"] = {k: round(float(v), 4) for k, v in top_positive.items()}
    metrics["top_negative_coefficients"] = {k: round(float(v), 4) for k, v in top_negative.items()}

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    save_metrics(metrics, METRICS_OUT)

    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: Blocked", "Pred: Breach"],
        yticklabels=["True: Blocked", "True: Breach"],
        ax=ax,
    )
    ax.set_title("Logistic Regression - Confusion Matrix (val)")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=150)
    plt.close(fig)

    print(f"Top positive coefficients: {metrics['top_positive_coefficients']}")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print(f"Saved figure: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
