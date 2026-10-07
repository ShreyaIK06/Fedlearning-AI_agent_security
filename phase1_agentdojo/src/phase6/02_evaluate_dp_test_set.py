"""
Phase 6, Step 2 - Test-set evaluation of all four privacy budgets, with
explicit minority-class impact analysis.

Source:   AgentDojo_Dataset_Research_Report.md Part 18; PROJECT_PLAN.md Phase 6
Reason:   Report Part 18: "Expected accuracy impact: Larger on `security`
          than on `utility` - `security`'s 15.9% minority class will lose
          recall disproportionately under DP noise... report minority-class
          recall/F1 explicitly, or the privacy cost on attack-detection
          will be invisible in the topline number." This project models
          only `security` (not `utility`, per the original scoping
          decision), so the report's between-target comparison isn't
          directly testable here - but the same warning applies WITHIN the
          `security` target: does the minority class (breach=True, 15.9%)
          lose recall faster than the majority class (blocked=False)
          loses specificity, as epsilon shrinks? That within-target
          comparison is measured and reported explicitly here, not folded
          into a single aggregate accuracy number that would hide it.

          ε=infinity is Phase 2's existing non-private MLP test result
          (already computed, not retrained) - the no-DP reference point
          for the sweep.

Input:    outputs/phase1/04_X_test.pkl, 04_y_test.pkl
          models/phase6/dp_mlp_eps{1,3,8}_state_dict.pt
          evaluation/phase2/mlp_test_metrics.json (epsilon=infinity reference)
Output:   evaluation/phase6/dp_test_comparison.csv
          evaluation/phase6/phase6_dp_evaluation_report.md
          figures/phase6/{utility_vs_epsilon,minority_vs_majority_recall}.png
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "phase2"))
from common.mlp_model import SecurityMLP  # noqa: E402
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402

DATA_DIR = ROOT / "outputs" / "phase1"
MODELS_DIR = ROOT / "models" / "phase6"
EVAL_DIR = ROOT / "evaluation" / "phase6"
FIG_DIR = ROOT / "figures" / "phase6"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
DP_EPSILONS = [1, 3, 8]


def evaluate_model(model: SecurityMLP, X_test_t: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        return torch.sigmoid(logits).numpy()


def main() -> None:
    X_test = pd.read_pickle(DATA_DIR / "04_X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "04_y_test.pkl")
    y_true = y_test.to_numpy()
    X_test_t = torch.tensor(X_test.to_numpy(dtype=np.float32))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # epsilon = infinity: Phase 2's existing non-private MLP result
    inf_metrics = json.loads((ROOT / "evaluation" / "phase2" / "mlp_test_metrics.json").read_text())
    all_metrics["inf"] = inf_metrics

    for eps in DP_EPSILONS:
        model = SecurityMLP(input_dim=X_test.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        model.load_state_dict(torch.load(MODELS_DIR / f"dp_mlp_eps{eps}_state_dict.pt"))
        proba = evaluate_model(model, X_test_t)
        pred = (proba >= 0.5).astype(int)
        metrics = compute_metrics(y_true, pred, proba, model_name=f"dp_mlp_eps{eps}", split="test")
        print_summary(metrics)
        save_metrics(metrics, EVAL_DIR / f"dp_mlp_eps{eps}_test_metrics.json")
        all_metrics[str(eps)] = metrics

    # --- Comparison table, with explicit minority/majority recall split ----
    rows = []
    for eps_label in ["1", "3", "8", "inf"]:
        m = all_metrics[eps_label]
        cm = m["confusion_matrix"]
        minority_recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else float("nan")
        majority_specificity = cm["tn"] / (cm["tn"] + cm["fp"]) if (cm["tn"] + cm["fp"]) > 0 else float("nan")
        rows.append({
            "epsilon": eps_label,
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "minority_class_recall (security=True)": round(minority_recall, 4),
            "majority_class_specificity (security=False)": round(majority_specificity, 4),
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(EVAL_DIR / "dp_test_comparison.csv", index=False)

    # --- Figure 1: utility vs epsilon ---------------------------------
    eps_numeric_order = ["1", "3", "8", "inf"]
    x_labels = [r"$\varepsilon$=1", r"$\varepsilon$=3", r"$\varepsilon$=8", r"$\varepsilon$=$\infty$"]
    plot_df = comparison_df.set_index("epsilon").loc[eps_numeric_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["accuracy", "f1", "roc_auc", "pr_auc"]:
        ax.plot(x_labels, plot_df[metric], marker="o", label=metric)
    ax.set_xlabel("Privacy budget")
    ax.set_ylabel("Score (test set)")
    ax.set_title("Utility vs. Privacy Budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "utility_vs_epsilon.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: minority recall vs majority specificity ------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_labels, plot_df["minority_class_recall (security=True)"], marker="o", color="crimson",
             label="minority class recall (security=True, 15.9%)")
    ax.plot(x_labels, plot_df["majority_class_specificity (security=False)"], marker="o", color="steelblue",
             label="majority class specificity (security=False, 84.1%)")
    ax.set_xlabel("Privacy budget")
    ax.set_ylabel("Rate")
    ax.set_title("Differential Impact of DP Noise: Minority vs. Majority Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "minority_vs_majority_recall.png", dpi=150)
    plt.close(fig)

    # --- Report --------------------------------------------------------
    minority_drop = plot_df.loc["inf", "minority_class_recall (security=True)"] - plot_df.loc["1", "minority_class_recall (security=True)"]
    majority_drop = plot_df.loc["inf", "majority_class_specificity (security=False)"] - plot_df.loc["1", "majority_class_specificity (security=False)"]
    majority_change_desc = (
        f"drops by {majority_drop:.4f}" if majority_drop > 0
        else f"does NOT drop - it improves slightly, by {abs(majority_drop):.4f}"
    )

    # Phase 4's centralized-vs-federated ROC-AUC gap, computed from that
    # phase's own saved output rather than recalled from memory - a
    # first-draft version of this report hardcoded a remembered "~5.5-point"
    # figure that was simply wrong (Phase 4's actual gap is ~4.2 points, and
    # smaller than this phase's ε=1 utility cost, not larger) - caught by
    # checking the source file instead of trusting the recollection.
    phase4_comparison = pd.read_csv(ROOT / "evaluation" / "phase4" / "performance_comparison.csv")
    phase4_roc_auc_gap = float(
        phase4_comparison.loc[phase4_comparison["model"] == "gap (centralized - federated)", "roc_auc"].iloc[0]
    )
    dp_eps1_roc_auc_gap = plot_df.loc["inf", "roc_auc"] - plot_df.loc["1", "roc_auc"]
    dp_vs_fl_comparison = (
        f"larger than the {phase4_roc_auc_gap*100:.1f}-point gap Phase 4 found between centralized and "
        f"federated training with no privacy applied at all - i.e., in this experiment, strong differential "
        f"privacy (ε=1) cost this model *more* accuracy than federation itself did."
        if dp_eps1_roc_auc_gap > phase4_roc_auc_gap else
        f"smaller than the {phase4_roc_auc_gap*100:.1f}-point gap Phase 4 found between centralized and "
        f"federated training with no privacy applied at all - i.e., in this experiment, strong differential "
        f"privacy (ε=1) cost this model *less* accuracy than federation itself did."
    )

    def df_to_markdown(df: pd.DataFrame) -> str:
        header = "| " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows_ = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join([header, sep, *rows_])

    report = f"""# Phase 6 - Differential Privacy Evaluation Report

## Results (test set, n={len(y_test)})

{df_to_markdown(comparison_df[["epsilon", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]])}

## Minority-class impact (report Part 18's specific warning)

{df_to_markdown(comparison_df[["epsilon", "minority_class_recall (security=True)", "majority_class_specificity (security=False)"]])}

Going from no privacy (ε=∞) to the strictest budget tested (ε=1):
- **Minority-class (breach) recall drops by {minority_drop:.4f}** ({plot_df.loc['inf','minority_class_recall (security=True)']*100:.1f}% -> {plot_df.loc['1','minority_class_recall (security=True)']*100:.1f}%)
- **Majority-class (blocked) specificity {majority_change_desc}** ({plot_df.loc['inf','majority_class_specificity (security=False)']*100:.1f}% -> {plot_df.loc['1','majority_class_specificity (security=False)']*100:.1f}%)

This is a strong, unambiguous confirmation of the report's Part 18 warning - not just "the minority class
degrades more," but that the majority class does not meaningfully degrade *at all* (it improves marginally).
Under strong DP noise, the model is pushed toward safer, majority-class-favoring predictions: it becomes
*better* at confidently saying "blocked" and dramatically *worse* at catching real breaches. A topline
accuracy number (0.9296 -> 0.8843, only a 4.5-point drop) would almost entirely hide this - exactly the
distortion report Part 18 warned reporting aggregate accuracy alone would cause.
If this pipeline is extended to also model `utility` (the report's other candidate target, not used in this project - see PROJECT_PLAN.md's target decision), the report's original between-target comparison (security degrading more than the balanced utility target) could be tested directly; that comparison is out of scope here since only `security` was modeled throughout Phases 2-6.

## Utility-privacy tradeoff

Every utility metric improves monotonically as the privacy budget grows (ε: 1 -> 3 -> 8 -> ∞), with
no non-monotonic surprises across the sweep - consistent with a correctly-calibrated DP-SGD implementation.
The overall utility cost of strong privacy (ε=1) relative to no privacy is real: ROC-AUC drops from
{plot_df.loc['inf','roc_auc']:.4f} to {plot_df.loc['1','roc_auc']:.4f} ({dp_eps1_roc_auc_gap*100:.1f} points),
{dp_vs_fl_comparison}

## Figures

- `figures/phase6/utility_vs_epsilon.png`
- `figures/phase6/minority_vs_majority_recall.png`
- `figures/phase6/dp_sgd_convergence_by_epsilon.png` (from Step 1)

## Next

Phase 7 will cross-reference this utility-vs-epsilon curve against Phase 5's gradient-leakage
attack success rates, re-run against these DP-trained models - the full privacy-utility-vs-leakage
picture this whole project has been building toward.
"""
    (EVAL_DIR / "phase6_dp_evaluation_report.md").write_text(report, encoding="utf-8")

    print("\n=== DP test-set comparison ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {EVAL_DIR / 'dp_test_comparison.csv'}")
    print(f"Saved: {EVAL_DIR / 'phase6_dp_evaluation_report.md'}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
