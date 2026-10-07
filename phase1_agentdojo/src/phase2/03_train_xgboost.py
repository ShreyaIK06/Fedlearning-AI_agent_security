"""
Phase 2, Step 3 - XGBoost baseline for `security`.

Source:   PROJECT_PLAN.md Phase 2; AgentDojo_Dataset_Research_Report.md Part 19
Reason:   Gradient-boosted trees are the strongest classical baseline
          typically achievable on tabular data of this shape (mixed
          one-hot/binary/scaled-numeric, moderate dimensionality, ~19K
          training rows) - the natural "best classical" comparison point
          against Step 4's MLP.

          Imbalance handling here is `scale_pos_weight` (XGBoost's native
          mechanism, analogous to Steps 1-2's `class_weight="balanced"`),
          set to the train-set negative:positive ratio rather than a
          hand-picked constant, so it's derived from data, not guessed.

Input:    outputs/phase1/04_X_{train,val}.pkl, 04_y_{train,val}.pkl
Output:   models/phase2/xgboost.joblib
          evaluation/phase2/xgboost_val_metrics.json
          figures/phase2/xgboost_confusion_matrix_val.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "phase1"
MODEL_OUT = ROOT / "models" / "phase2" / "xgboost.joblib"
METRICS_OUT = ROOT / "evaluation" / "phase2" / "xgboost_val_metrics.json"
FIGURE_OUT = ROOT / "figures" / "phase2" / "xgboost_confusion_matrix_val.png"

RANDOM_STATE = 42


def main() -> None:
    X_train = pd.read_pickle(DATA_DIR / "04_X_train.pkl")
    y_train = pd.read_pickle(DATA_DIR / "04_y_train.pkl")
    X_val = pd.read_pickle(DATA_DIR / "04_X_val.pkl")
    y_val = pd.read_pickle(DATA_DIR / "04_y_val.pkl")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_proba, model_name="xgboost", split="val")
    metrics["scale_pos_weight"] = round(scale_pos_weight, 4)
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
    ax.set_title("XGBoost - Confusion Matrix (val)")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=150)
    plt.close(fig)

    print(f"scale_pos_weight (n_neg/n_pos in train): {scale_pos_weight:.4f}")
    print(f"Top feature importances: {metrics['top_feature_importances']}")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print(f"Saved figure: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
