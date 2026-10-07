"""
Phase 5, Step 2 - DLG (Deep Leakage from Gradients) attack.

Source:   Zhang et al. 2019 "Deep Leakage from Gradients"; PROJECT_PLAN.md Phase 5
Reason:   The original gradient-inversion attack: jointly optimize a dummy
          input AND a dummy (soft) label so that the gradient the dummy
          pair produces matches the real gradient exactly, under L2
          distance. Both x and y start from random noise - DLG does not
          assume the label is known, which is what distinguishes it from
          Step 3's iDLG (closed-form label recovery) and makes it the
          harder, more realistic starting point among the three attacks.

Input:    outputs/phase5/victims.pt
          models/phase3/federated_mlp_state_dict.pt
Output:   outputs/phase5/02_dlg_results.json
          figures/phase5/dlg_convergence_examples.png
          figures/phase5/dlg_reconstruction_quality.png
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.mlp_model import SecurityMLP  # noqa: E402
from common.gradient_leakage import load_feature_blocks, reconstruct, score_reconstruction  # noqa: E402

DATA_DIR = ROOT / "outputs" / "phase5"
MODEL_IN = ROOT / "models" / "phase3" / "federated_mlp_state_dict.pt"
ENCODERS_DIR = ROOT / "models" / "phase1_encoders"
OUT_JSON = DATA_DIR / "02_dlg_results.json"
FIG_DIR = ROOT / "figures" / "phase5"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
N_ITERS = 150
LBFGS_LR = 1.0
RANDOM_STATE = 42


def main() -> None:
    data = torch.load(DATA_DIR / "victims.pt")
    victims_X, victims_y = data["victims_X"], data["victims_y"]
    victim_grads, victim_meta = data["victim_grads"], data["victim_meta"]
    input_dim = data["input_dim"]

    blocks = load_feature_blocks(ENCODERS_DIR)

    model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model.load_state_dict(torch.load(MODEL_IN))
    model.eval()

    per_victim_results = []
    histories = []

    for i, (x_true, y_true, g_true, meta) in enumerate(zip(victims_X, victims_y, victim_grads, victim_meta)):
        x_recon, y_recon, history = reconstruct(
            model, g_true, input_dim, mode="l2", know_label=False,
            n_iters=N_ITERS, lr=LBFGS_LR, seed=RANDOM_STATE + i,
        )
        histories.append(history)

        score = score_reconstruction(x_true.numpy(), x_recon.squeeze(0).numpy(), blocks)
        label_recovered = bool(round(float(y_recon.item())) == int(y_true.item()))

        per_victim_results.append({
            "victim_index": i,
            "suite": meta["suite"],
            "true_label": float(y_true.item()),
            "recovered_label_soft": float(y_recon.item()),
            "label_recovered_correctly": label_recovered,
            "best_gradient_matching_loss": min(history),
            **score,
        })
        print(f"[DLG victim {i:2d} / {meta['suite']:10s}] best_loss={min(history):.6f}  "
              f"overall_mse={score['overall_mse']:.4f}  cat_acc={score['categorical_accuracy']:.2f}  "
              f"tool_P={score['tool_precision']:.2f} tool_R={score['tool_recall']:.2f}  "
              f"label_ok={label_recovered}")

    # --- Aggregate ---------------------------------------------------------
    def agg(key):
        vals = [r[key] for r in per_victim_results]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    aggregate = {
        "overall_mse": agg("overall_mse"),
        "categorical_accuracy": agg("categorical_accuracy"),
        "tool_precision": agg("tool_precision"),
        "tool_recall": agg("tool_recall"),
        "numeric_mae_std_units": agg("numeric_mae_std_units"),
        "label_recovery_rate": round(
            sum(r["label_recovered_correctly"] for r in per_victim_results) / len(per_victim_results), 4
        ),
    }

    null_baseline = json.loads((DATA_DIR / "01_victim_selection_stats.json").read_text())["null_baseline_mse"]

    results = {
        "attack": "DLG",
        "config": {"n_iters": N_ITERS, "lbfgs_lr": LBFGS_LR, "loss": "l2", "label_known": False},
        "per_victim": per_victim_results,
        "aggregate": aggregate,
        "null_baseline_mse": null_baseline,
        "mse_improvement_vs_mean_guess": round(
            1 - aggregate["overall_mse"]["mean"] / null_baseline["training_mean_vector_guess_mean"], 4
        ),
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Figures -------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(0, len(histories), 3):  # one example per suite (every 3rd victim)
        ax.plot(histories[i], label=f"victim {i} ({per_victim_results[i]['suite']})")
    ax.set_xlabel("LBFGS iteration")
    ax.set_ylabel("Gradient-matching L2 loss")
    ax.set_yscale("log")
    ax.set_title("DLG Attack Convergence (one example per suite)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dlg_convergence_examples.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(["null (mean guess)", "DLG"], [null_baseline["training_mean_vector_guess_mean"], aggregate["overall_mse"]["mean"]])
    axes[0].set_title("Reconstruction MSE\n(lower = more leaked)")
    axes[1].bar(["categorical_acc", "tool_precision", "tool_recall"],
                [aggregate["categorical_accuracy"]["mean"], aggregate["tool_precision"]["mean"], aggregate["tool_recall"]["mean"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("DLG - What Leaked")
    axes[2].bar(["label recovery rate"], [aggregate["label_recovery_rate"]])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("DLG - Label Recovery\n(jointly optimized, not closed-form)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dlg_reconstruction_quality.png", dpi=150)
    plt.close(fig)

    print(f"\n=== DLG aggregate (n={len(per_victim_results)}) ===")
    print(json.dumps(aggregate, indent=2))
    print(f"MSE improvement vs. training-mean baseline: {results['mse_improvement_vs_mean_guess']*100:.1f}%")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
