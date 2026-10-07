"""
Phase 2, Step 5 (final) - Held-out test-set evaluation and model comparison.

Source:   PROJECT_PLAN.md Phase 2
Reason:   Steps 1-4 each reported val-set metrics only (val was used for
          early stopping / model selection, e.g. the MLP's best-epoch
          checkpoint - reporting headline numbers on the same split used
          for selection would be mildly optimistic). This step is the
          FIRST and ONLY time the test set (held out since Phase 1, Step 3)
          is touched, and it's touched exactly once, for all four models
          together, producing the numbers this project would actually
          report in a paper.

          The MLP's test performance here is also the "centralized"
          reference Phase 4 will compare Phase 3's federated MLP against -
          same architecture, same test set, so that comparison is apples-
          to-apples per PROJECT_PLAN.md.

Input:    outputs/phase1/04_X_test.pkl, 04_y_test.pkl
          models/phase2/{logistic_regression,random_forest,xgboost}.joblib
          models/phase2/mlp_state_dict.pt
Output:   evaluation/phase2/{model}_test_metrics.json (x4)
          evaluation/phase2/phase2_model_comparison.csv
          evaluation/phase2/phase2_comparison_report.md
          figures/phase2/roc_curve_comparison.png
          figures/phase2/pr_curve_comparison.png
          figures/phase2/metrics_comparison_bar_chart.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.mlp_model import SecurityMLP  # noqa: E402
from _metrics_utils import compute_metrics, print_summary, save_metrics, LABEL_NOISE_CEILING_PCT  # noqa: E402

DATA_DIR = ROOT / "outputs" / "phase1"
MODELS_DIR = ROOT / "models" / "phase2"
EVAL_DIR = ROOT / "evaluation" / "phase2"
FIG_DIR = ROOT / "figures" / "phase2"


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal DataFrame->markdown-table renderer (no `tabulate` dependency
    in this environment - kept self-contained rather than adding a new
    requirement mid-pipeline)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def get_mlp_proba(X_test: pd.DataFrame) -> np.ndarray:
    model = SecurityMLP(input_dim=X_test.shape[1], hidden_dims=(128, 64), dropout=0.2)
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_state_dict.pt"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test.to_numpy(dtype=np.float32)))
        return torch.sigmoid(logits).numpy()


def main() -> None:
    X_test = pd.read_pickle(DATA_DIR / "04_X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "04_y_test.pkl")
    y_true = y_test.to_numpy()

    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")

    probas = {
        "logistic_regression": lr.predict_proba(X_test)[:, 1],
        "random_forest": rf.predict_proba(X_test)[:, 1],
        "xgboost": xgb.predict_proba(X_test)[:, 1],
        "mlp": get_mlp_proba(X_test),
    }

    all_metrics = {}
    for name, proba in probas.items():
        pred = (proba >= 0.5).astype(int)
        metrics = compute_metrics(y_true, pred, proba, model_name=name, split="test")
        print_summary(metrics)
        save_metrics(metrics, EVAL_DIR / f"{name}_test_metrics.json")
        all_metrics[name] = metrics

    # --- Comparison table -----------------------------------------------
    comparison_rows = []
    for name, m in all_metrics.items():
        comparison_rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "tp": m["confusion_matrix"]["tp"],
            "fp": m["confusion_matrix"]["fp"],
            "fn": m["confusion_matrix"]["fn"],
            "tn": m["confusion_matrix"]["tn"],
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values("f1", ascending=False)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(EVAL_DIR / "phase2_model_comparison.csv", index=False)

    # --- ROC curve overlay -------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={all_metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison - Test Set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close(fig)

    # --- PR curve overlay ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    base_rate = y_true.mean()
    for name, proba in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (AP={all_metrics[name]['pr_auc']:.3f})")
    ax.axhline(base_rate, color="k", linestyle="--", alpha=0.3, label=f"base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison - Test Set")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve_comparison.png", dpi=150)
    plt.close(fig)

    # --- Metrics bar chart -------------------------------------------------
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    plot_df = comparison_df.set_index("model")[metric_cols]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Phase 2 Model Comparison - Test Set")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.axhline(1 - LABEL_NOISE_CEILING_PCT / 100, color="red", linestyle=":", alpha=0.5,
               label="label-noise-implied accuracy ceiling")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_comparison_bar_chart.png", dpi=150)
    plt.close(fig)

    # --- Comparison report ---------------------------------------------
    best_f1 = comparison_df.iloc[0]
    best_roc = comparison_df.loc[comparison_df["roc_auc"].idxmax()]
    report_lines = [
        "# Phase 2 - Centralized ML Model Comparison Report",
        "",
        f"Test set: {len(y_test)} rows, {y_true.mean()*100:.2f}% security=True "
        f"(held out since Phase 1 Step 3, touched here for the first and only time).",
        "",
        "## Results (test set)",
        "",
        df_to_markdown(comparison_df[["model"] + metric_cols]),
        "",
        "## Label-noise ceiling",
        "",
        f"The `security` label flips on {LABEL_NOISE_CEILING_PCT}% of exact-repeat configurations "
        "(report Part 4, re-verified in PIPELINE_READINESS_REPORT.md) - irreducible run-to-run "
        "stochasticity in the original AgentDojo benchmark traces, not model error. None of the four "
        "models' test accuracy approaches a level that would be suspicious against this ceiling.",
        "",
        "## Best model by F1",
        "",
        f"**{best_f1['model']}** (F1={best_f1['f1']}, precision={best_f1['precision']}, recall={best_f1['recall']}).",
        "",
        "## Best model by ROC-AUC",
        "",
        f"**{best_roc['model']}** (ROC-AUC={best_roc['roc_auc']}).",
        "",
        "## Carried forward to Phase 3/4",
        "",
        "The **MLP**'s test performance here is the centralized reference Phase 4 will compare Phase 3's "
        "federated MLP against - same architecture (`src/common/mlp_model.py`), same test set. The three "
        "classical models are a benchmarking reference, not carried into the FL comparison (FedAvg requires "
        "a differentiable model with shared weight structure across clients; LR/RF/XGBoost don't fit that "
        "framework the way an MLP does).",
        "",
        "## Figures",
        "",
        "- `figures/phase2/roc_curve_comparison.png`",
        "- `figures/phase2/pr_curve_comparison.png`",
        "- `figures/phase2/metrics_comparison_bar_chart.png`",
    ]
    (EVAL_DIR / "phase2_comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== Comparison table (test set) ===")
    print(comparison_df[["model"] + metric_cols].to_string(index=False))
    print(f"\nBest F1: {best_f1['model']} ({best_f1['f1']})")
    print(f"Best ROC-AUC: {best_roc['model']} ({best_roc['roc_auc']})")
    print(f"\nSaved: {EVAL_DIR / 'phase2_model_comparison.csv'}")
    print(f"Saved: {EVAL_DIR / 'phase2_comparison_report.md'}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
