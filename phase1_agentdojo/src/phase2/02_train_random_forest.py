"""
Phase 2, Step 2 - Random Forest baseline for `security`.

Source:   PROJECT_PLAN.md Phase 2; AgentDojo_Dataset_Research_Report.md Part 19
Reason:   A tree ensemble handles the Phase 1 feature set's mix of one-hot
          categoricals, bag-of-tools binaries, and scaled numerics natively
          (no assumption of linear separability, unlike Step 1's LR), and
          its `feature_importances_` gives a second, independent read on
          which features drive `security` predictions - worth comparing
          against Step 1's coefficient-based ranking for agreement.

          `class_weight="balanced"` again, for the same reason as Step 1
          (report Part 19's imbalance-handling guidance) - RF supports this
          natively per-tree via reweighted bootstrap sampling.

Input:    outputs/phase1/04_X_{train,val}.pkl, 04_y_{train,val}.pkl
Output:   models/phase2/random_forest.joblib
          evaluation/phase2/random_forest_val_metrics.json
          figures/phase2/random_forest_confusion_matrix_val.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "phase1"
MODEL_OUT = ROOT / "models" / "phase2" / "random_forest.joblib"
METRICS_OUT = ROOT / "evaluation" / "phase2" / "random_forest_val_metrics.json"
FIGURE_OUT = ROOT / "figures" / "phase2" / "random_forest_confusion_matrix_val.png"

RANDOM_STATE = 42


def main() -> None:
    X_train = pd.read_pickle(DATA_DIR / "04_X_train.pkl")
    y_train = pd.read_pickle(DATA_DIR / "04_y_train.pkl")
    X_val = pd.read_pickle(DATA_DIR / "04_X_val.pkl")
    y_val = pd.read_pickle(DATA_DIR / "04_y_val.pkl")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_proba, model_name="random_forest", split="val")
    print_summary(metrics)

    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top_features = importances.head(15).to_dict()
    metrics["top_feature_importances"] = {k: round(float(v), 4) for k, v in top_features.items()}

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
    ax.set_title("Random Forest - Confusion Matrix (val)")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=150)
    plt.close(fig)

    print(f"Top feature importances: {metrics['top_feature_importances']}")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print(f"Saved figure: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
