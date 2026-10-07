"""
Phase 5, Step 4 - Inverting Gradients (IG) attack.

Source:   Geiping et al. 2020 "Inverting Gradients - How easy is it to break
          privacy in federated learning?"; PROJECT_PLAN.md Phase 5
Reason:   IG's key change from DLG/iDLG is the matching objective: cosine
          similarity (gradient DIRECTION) instead of L2 distance
          (gradient MAGNITUDE + direction). The IG paper's motivation is
          exactly the instability Step 2 (DLG) hit here empirically - L2
          matching can be dominated by a few large-magnitude gradient
          components and get stuck in poor optima; cosine similarity is
          scale-invariant and, per the paper, more robust for deeper
          networks. Same closed-form label inference as Step 3 (iDLG) -
          this isolates the loss-function choice as the only remaining
          variable across all three attacks (DLG: L2 + optimized label;
          iDLG: L2 + closed-form label; IG: cosine + closed-form label).

          Optimizer: Adam, not LBFGS (unlike Steps 2-3). First attempt
          reused LBFGS for consistency and got stuck at loss=1.000000
          (cos_sim=0, i.e. no matching at all) for essentially every
          victim. Diagnosed directly (not assumed): LBFGS's internal
          quasi-Newton curvature history gets corrupted by this project's
          per-step `clamp_range` safeguard, and a single large early jump
          (x_dummy norm reached ~5865 before clamping) permanently wrecked
          its subsequent steps. Switched to Adam, which is what Geiping et
          al.'s actual Inverting-Gradients implementation uses - not a
          convenience patch, the literature-correct pairing for this loss.
          Full derivation in `src/common/gradient_leakage.py`'s
          `reconstruct` docstring.

Input:    outputs/phase5/victims.pt
          models/phase3/federated_mlp_state_dict.pt
Output:   outputs/phase5/04_ig_results.json
          figures/phase5/ig_convergence_examples.png
          figures/phase5/ig_reconstruction_quality.png
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
from common.gradient_leakage import (  # noqa: E402
    infer_label_from_final_bias_grad,
    load_feature_blocks,
    reconstruct,
    score_reconstruction,
)

DATA_DIR = ROOT / "outputs" / "phase5"
MODEL_IN = ROOT / "models" / "phase3" / "federated_mlp_state_dict.pt"
ENCODERS_DIR = ROOT / "models" / "phase1_encoders"
OUT_JSON = DATA_DIR / "04_ig_results.json"
FIG_DIR = ROOT / "figures" / "phase5"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
N_ITERS = 500  # Adam needs more steps than LBFGS to converge on this loss
ADAM_LR = 0.05
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
    label_inference_correct = []

    for i, (x_true, y_true, g_true, meta) in enumerate(zip(victims_X, victims_y, victim_grads, victim_meta)):
        inferred_label = infer_label_from_final_bias_grad(g_true[-1])
        inference_correct = bool(int(inferred_label.item()) == int(y_true.item()))
        label_inference_correct.append(inference_correct)

        x_recon, _, history = reconstruct(
            model, g_true, input_dim, mode="cosine", know_label=True, true_y=inferred_label,
            n_iters=N_ITERS, lr=ADAM_LR, seed=RANDOM_STATE + i, optimizer_name="adam",
        )
        histories.append(history)

        score = score_reconstruction(x_true.numpy(), x_recon.squeeze(0).numpy(), blocks)

        per_victim_results.append({
            "victim_index": i,
            "suite": meta["suite"],
            "true_label": float(y_true.item()),
            "inferred_label_closed_form": float(inferred_label.item()),
            "label_inference_correct": inference_correct,
            "best_gradient_matching_loss": min(history),
            **score,
        })
        print(f"[IG victim {i:2d} / {meta['suite']:10s}] label_ok={inference_correct}  "
              f"best_loss={min(history):.6f}  overall_mse={score['overall_mse']:.4f}  "
              f"cat_acc={score['categorical_accuracy']:.2f}  tool_P={score['tool_precision']:.2f} tool_R={score['tool_recall']:.2f}")

    def agg(key):
        vals = [r[key] for r in per_victim_results]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    aggregate = {
        "overall_mse": agg("overall_mse"),
        "categorical_accuracy": agg("categorical_accuracy"),
        "tool_precision": agg("tool_precision"),
        "tool_recall": agg("tool_recall"),
        "numeric_mae_std_units": agg("numeric_mae_std_units"),
        "label_inference_accuracy": round(float(np.mean(label_inference_correct)), 4),
    }

    null_baseline = json.loads((DATA_DIR / "01_victim_selection_stats.json").read_text())["null_baseline_mse"]
    dlg_results = json.loads((DATA_DIR / "02_dlg_results.json").read_text())
    idlg_results = json.loads((DATA_DIR / "03_idlg_results.json").read_text())

    results = {
        "attack": "InvertingGradients",
        "config": {"n_iters": N_ITERS, "optimizer": "adam", "adam_lr": ADAM_LR, "loss": "cosine", "label_known": "closed_form"},
        "per_victim": per_victim_results,
        "aggregate": aggregate,
        "null_baseline_mse": null_baseline,
        "mse_improvement_vs_mean_guess": round(
            1 - aggregate["overall_mse"]["mean"] / null_baseline["training_mean_vector_guess_mean"], 4
        ),
        "comparison_vs_dlg_idlg": {
            "dlg_overall_mse_mean": dlg_results["aggregate"]["overall_mse"]["mean"],
            "idlg_overall_mse_mean": idlg_results["aggregate"]["overall_mse"]["mean"],
            "ig_overall_mse_mean": aggregate["overall_mse"]["mean"],
            "dlg_categorical_accuracy": dlg_results["aggregate"]["categorical_accuracy"]["mean"],
            "idlg_categorical_accuracy": idlg_results["aggregate"]["categorical_accuracy"]["mean"],
            "ig_categorical_accuracy": aggregate["categorical_accuracy"]["mean"],
            "dlg_tool_recall": dlg_results["aggregate"]["tool_recall"]["mean"],
            "idlg_tool_recall": idlg_results["aggregate"]["tool_recall"]["mean"],
            "ig_tool_recall": aggregate["tool_recall"]["mean"],
        },
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(0, len(histories), 3):
        ax.plot(histories[i], label=f"victim {i} ({per_victim_results[i]['suite']})")
    ax.set_xlabel("LBFGS iteration")
    ax.set_ylabel("Gradient-matching cosine loss (1 - cos_sim)")
    ax.set_title("Inverting Gradients Attack Convergence (one example per suite)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ig_convergence_examples.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(["null", "DLG", "iDLG", "IG"],
                [null_baseline["training_mean_vector_guess_mean"], dlg_results["aggregate"]["overall_mse"]["mean"],
                 idlg_results["aggregate"]["overall_mse"]["mean"], aggregate["overall_mse"]["mean"]])
    axes[0].set_yscale("log")
    axes[0].set_title("Reconstruction MSE (log scale)\n(lower = more leaked)")
    axes[1].bar(["DLG", "iDLG", "IG"],
                [dlg_results["aggregate"]["categorical_accuracy"]["mean"],
                 idlg_results["aggregate"]["categorical_accuracy"]["mean"],
                 aggregate["categorical_accuracy"]["mean"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Categorical Field Accuracy")
    axes[2].bar(["DLG", "iDLG", "IG"],
                [dlg_results["aggregate"]["tool_recall"]["mean"],
                 idlg_results["aggregate"]["tool_recall"]["mean"],
                 aggregate["tool_recall"]["mean"]])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Tool-Usage Recall")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ig_reconstruction_quality.png", dpi=150)
    plt.close(fig)

    print(f"\n=== IG aggregate (n={len(per_victim_results)}) ===")
    print(json.dumps(aggregate, indent=2))
    print(f"MSE: DLG={dlg_results['aggregate']['overall_mse']['mean']:.2f}  "
          f"iDLG={idlg_results['aggregate']['overall_mse']['mean']:.2f}  IG={aggregate['overall_mse']['mean']:.2f}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
