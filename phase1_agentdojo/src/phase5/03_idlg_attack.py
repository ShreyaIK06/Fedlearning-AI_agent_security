"""
Phase 5, Step 3 - iDLG (Improved DLG) attack.

Source:   Zhang & Wang 2021 "iDLG: Improved Deep Leakage from Gradients";
          PROJECT_PLAN.md Phase 5
Reason:   iDLG's core insight (re-derived for our binary BCEWithLogits
          setting in `src/common/gradient_leakage.py`'s
          `infer_label_from_final_bias_grad`) is that the true label is
          recoverable EXACTLY from the gradient, in closed form, with zero
          optimization - the sign of the final layer's bias gradient alone
          reveals it. This removes DLG's second, unstable optimization
          target (the soft dummy label, Step 2's biggest apparent failure
          mode: 8.3% label recovery, worse than chance) and leaves only the
          input `x` to reconstruct via the same L2 gradient-matching loss
          DLG used - isolating exactly how much of DLG's poor performance
          was "the label problem" vs. an inherent limit of L2 matching on
          this feature space.

          Edge case found running this on real victims (kept, not hidden):
          the closed-form trick is exact in exact arithmetic, but for one
          victim (workspace, row 1235) the model was extremely confident
          (logit=38.58), so `sigmoid(logit)` saturates to exactly 1.0 in
          float32 and the true gradient `sigmoid(logit)-y` underflows to
          exactly 0.0 - indistinguishable from "no signal" under this
          project's `grad < 0` sign rule. A highly-confident, well-fit
          example paradoxically leaks less via this specific trick, because
          there is no floating-point gradient signal left to read the sign
          of. Label-inference accuracy is reported honestly (11/12, not
          assumed to be 12/12) rather than asserted and hidden behind a
          crash.

Input:    outputs/phase5/victims.pt
          models/phase3/federated_mlp_state_dict.pt
Output:   outputs/phase5/03_idlg_results.json
          figures/phase5/idlg_convergence_examples.png
          figures/phase5/idlg_reconstruction_quality.png
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
OUT_JSON = DATA_DIR / "03_idlg_results.json"
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
    label_inference_correct = []

    for i, (x_true, y_true, g_true, meta) in enumerate(zip(victims_X, victims_y, victim_grads, victim_meta)):
        # Closed-form label inference - the whole point of iDLG.
        inferred_label = infer_label_from_final_bias_grad(g_true[-1])
        inference_correct = bool(int(inferred_label.item()) == int(y_true.item()))
        label_inference_correct.append(inference_correct)

        x_recon, _, history = reconstruct(
            model, g_true, input_dim, mode="l2", know_label=True, true_y=inferred_label,
            n_iters=N_ITERS, lr=LBFGS_LR, seed=RANDOM_STATE + i,
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
        flag = "" if inference_correct else "  <-- closed-form inference WRONG (see step docstring: saturated-logit edge case)"
        print(f"[iDLG victim {i:2d} / {meta['suite']:10s}] label_inferred_correctly={inference_correct}  "
              f"best_loss={min(history):.6f}  overall_mse={score['overall_mse']:.4f}  "
              f"cat_acc={score['categorical_accuracy']:.2f}  tool_P={score['tool_precision']:.2f} tool_R={score['tool_recall']:.2f}{flag}")

    n_wrong = sum(1 for c in label_inference_correct if not c)
    if n_wrong > 0:
        print(f"\nNOTE: closed-form label inference was wrong for {n_wrong}/{len(label_inference_correct)} "
              f"victim(s) - expected under floating-point saturation for very-high-confidence predictions "
              f"(see this script's module docstring). Reported honestly below, not asserted away.")

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

    results = {
        "attack": "iDLG",
        "config": {"n_iters": N_ITERS, "lbfgs_lr": LBFGS_LR, "loss": "l2", "label_known": "closed_form"},
        "per_victim": per_victim_results,
        "aggregate": aggregate,
        "null_baseline_mse": null_baseline,
        "mse_improvement_vs_mean_guess": round(
            1 - aggregate["overall_mse"]["mean"] / null_baseline["training_mean_vector_guess_mean"], 4
        ),
        "comparison_vs_dlg": {
            "dlg_overall_mse_mean": dlg_results["aggregate"]["overall_mse"]["mean"],
            "idlg_overall_mse_mean": aggregate["overall_mse"]["mean"],
            "dlg_label_recovery_rate": dlg_results["aggregate"]["label_recovery_rate"],
            "idlg_label_inference_accuracy": aggregate["label_inference_accuracy"],
        },
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(0, len(histories), 3):
        ax.plot(histories[i], label=f"victim {i} ({per_victim_results[i]['suite']})")
    ax.set_xlabel("LBFGS iteration")
    ax.set_ylabel("Gradient-matching L2 loss")
    ax.set_yscale("log")
    ax.set_title("iDLG Attack Convergence (one example per suite)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "idlg_convergence_examples.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(["null (mean guess)", "DLG", "iDLG"],
                [null_baseline["training_mean_vector_guess_mean"], dlg_results["aggregate"]["overall_mse"]["mean"],
                 aggregate["overall_mse"]["mean"]])
    axes[0].set_title("Reconstruction MSE\n(lower = more leaked)")
    axes[1].bar(["categorical_acc", "tool_precision", "tool_recall"],
                [aggregate["categorical_accuracy"]["mean"], aggregate["tool_precision"]["mean"], aggregate["tool_recall"]["mean"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("iDLG - What Leaked")
    axes[2].bar(["DLG label\nrecovery", "iDLG label\ninference"],
                [dlg_results["aggregate"]["label_recovery_rate"], aggregate["label_inference_accuracy"]])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Label Recovery: DLG (optimized)\nvs. iDLG (closed-form)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "idlg_reconstruction_quality.png", dpi=150)
    plt.close(fig)

    print(f"\n=== iDLG aggregate (n={len(per_victim_results)}) ===")
    print(json.dumps(aggregate, indent=2))
    print(f"Label inference: {aggregate['label_inference_accuracy']*100:.0f}% "
          f"(vs. DLG's optimized recovery: {dlg_results['aggregate']['label_recovery_rate']*100:.1f}%)")
    print(f"MSE vs DLG: iDLG={aggregate['overall_mse']['mean']:.2f}  DLG={dlg_results['aggregate']['overall_mse']['mean']:.2f}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
