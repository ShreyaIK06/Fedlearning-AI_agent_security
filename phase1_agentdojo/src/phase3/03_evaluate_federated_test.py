"""
Phase 3, Step 3 (final) - Evaluate the federated global model on the
held-out test set.

Source:   PROJECT_PLAN.md Phase 3/4
Reason:   Same test set Phase 2 used (touched there for the first and only
          time per-model; this is that same, single evaluation extended to
          the federated model) - the numbers this step produces are what
          Phase 4 will directly compare against Phase 2's centralized MLP
          test metrics. Uses the same `compute_metrics` helper as every
          Phase 2 model for a metric-for-metric identical comparison.

Input:    outputs/phase1/04_X_test.pkl, 04_y_test.pkl
          models/phase3/federated_mlp_state_dict.pt
Output:   evaluation/phase3/federated_mlp_test_metrics.json
          figures/phase3/federated_mlp_confusion_matrix_test.png
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "phase2"))
from common.mlp_model import SecurityMLP  # noqa: E402
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402

DATA_DIR = ROOT / "outputs" / "phase1"
MODEL_IN = ROOT / "models" / "phase3" / "federated_mlp_state_dict.pt"
METRICS_OUT = ROOT / "evaluation" / "phase3" / "federated_mlp_test_metrics.json"
FIGURE_OUT = ROOT / "figures" / "phase3" / "federated_mlp_confusion_matrix_test.png"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2


def main() -> None:
    X_test = pd.read_pickle(DATA_DIR / "04_X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "04_y_test.pkl")

    model = SecurityMLP(input_dim=X_test.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model.load_state_dict(torch.load(MODEL_IN))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test.to_numpy(dtype=np.float32)))
        proba = torch.sigmoid(logits).numpy()
    pred = (proba >= 0.5).astype(int)

    metrics = compute_metrics(y_test.to_numpy(), pred, proba, model_name="federated_mlp", split="test")
    print_summary(metrics)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, METRICS_OUT)

    # Pull in Phase 2's centralized MLP test metrics for a direct printout
    centralized_path = ROOT / "evaluation" / "phase2" / "mlp_test_metrics.json"
    if centralized_path.exists():
        import json
        centralized = json.loads(centralized_path.read_text())
        print("\n=== Centralized vs. Federated MLP (test set) ===")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            print(f"  {k:10s}  centralized={centralized[k]:.4f}   federated={metrics[k]:.4f}   "
                  f"gap={centralized[k]-metrics[k]:+.4f}")

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
    ax.set_title("Federated MLP - Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=150)
    plt.close(fig)

    print(f"\nSaved metrics: {METRICS_OUT}")
    print(f"Saved figure: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
