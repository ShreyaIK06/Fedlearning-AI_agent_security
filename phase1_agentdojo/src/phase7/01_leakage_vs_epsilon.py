"""
Phase 7, Step 1 - Gradient leakage attack success vs. privacy budget.

Source:   PROJECT_PLAN.md Phase 7; AgentDojo_Dataset_Research_Report.md Part 18
Reason:   Phase 6 measured DP-SGD's utility cost across ε ∈ {1, 3, 8, ∞}.
          This step measures the other side of the tradeoff: does that same
          privacy budget actually reduce Phase 5's gradient-leakage attack
          success? This is the whole project's central question, answered
          directly rather than assumed.

          Model-family consistency matters here: Phase 5 attacked the
          FEDERATED model (Phase 3), but Phase 6's DP models are trained by
          Opacus DP-SGD directly on pooled data (a different, centralized
          training procedure). Comparing "federated, no DP" against
          "centralized, DP" would confound two changed variables (FL vs.
          DP) into one number. This step instead builds a clean, single-
          variable sweep: the ε=infinity point is Phase 2's centralized
          MLP (no DP, no FL) and ε=1/3/8 are Phase 6's DP-SGD models
          trained the same way (centralized) - only epsilon changes across
          the sweep, isolating DP's effect on leakage from FL's.

          Attack used: Inverting Gradients (closed-form label + cosine
          loss, Adam optimizer, best-iterate tracking) - Phase 5's most
          balanced attack across the "what did the attacker learn" metrics
          (best categorical accuracy and tool precision of the three).
          Re-running all three attacks x 4 models was judged out of
          scope/proportion for this final synthesis step; IG alone directly
          answers the project's throughline question.

          Same 12 victims as Phase 5 (raw x/y reused from
          `outputs/phase5/victims.pt`), but gradients are recomputed fresh
          against each of the 4 models here - a victim's gradient depends
          on the exact weights it's computed against, so Phase 5's
          federated-model gradients cannot be reused for this sweep.

          MAJOR FINDING mid-experiment, changing this step's scope: the
          first run's IG-only sweep showed the DP models (esp. ε=1)
          reconstructing MUCH better than the non-private model (ε=∞) -
          the opposite of the naive expectation. Diagnosed directly rather
          than reported as-is: this is NOT a privacy failure specific to
          DP-SGD. `src/common/gradient_leakage.py`'s new
          `closed_form_first_layer_attack` proves (verified to ~1e-9
          error) that ANY single-example gradient through this
          architecture's linear first layer exactly determines that
          example's input, via the identity `dL/dW1 = outer(dL/db1, x)` -
          true for every model tested here, DP or not. IG's optimizer
          simply found that exact solution far more often for the
          smaller-weight-norm DP models (whose loss landscape is easier to
          navigate) than for the larger, more complex non-private model,
          where it got stuck in local optima. The closed-form attack is
          now run against all four models too, to show the TRUE ceiling of
          single-gradient vulnerability - which turns out to be almost
          total, for every epsilon, including epsilon=infinity's own
          "well-defended-looking" IG result.

Input:    outputs/phase5/victims.pt (victims_X, victims_y only - gradients recomputed)
          models/phase2/mlp_state_dict.pt (epsilon=infinity)
          models/phase6/dp_mlp_eps{1,3,8}_state_dict.pt
Output:   outputs/phase7/01_leakage_vs_epsilon_results.json
          figures/phase7/leakage_vs_epsilon.png
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
    closed_form_first_layer_attack,
    compute_gradient,
    infer_label_from_final_bias_grad,
    load_feature_blocks,
    reconstruct,
    score_reconstruction,
)

DATA_DIR = ROOT / "outputs" / "phase5"
ENCODERS_DIR = ROOT / "models" / "phase1_encoders"
OUT_DIR = ROOT / "outputs" / "phase7"
FIG_DIR = ROOT / "figures" / "phase7"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
RANDOM_STATE = 42

MODEL_PATHS = {
    "1": ROOT / "models" / "phase6" / "dp_mlp_eps1_state_dict.pt",
    "3": ROOT / "models" / "phase6" / "dp_mlp_eps3_state_dict.pt",
    "8": ROOT / "models" / "phase6" / "dp_mlp_eps8_state_dict.pt",
    "inf": ROOT / "models" / "phase2" / "mlp_state_dict.pt",
}


def attack_model(model, victims_X, victims_y, blocks, input_dim):
    """Both attacks per victim: Inverting Gradients (optimization-based,
    what Phase 5 used) and the closed-form first-layer identity (exact,
    non-iterative) - run side by side so the gap between them is visible
    in the results, not just asserted in prose."""
    per_victim_ig = []
    per_victim_cf = []
    for i, (x_true, y_true) in enumerate(zip(victims_X, victims_y)):
        x_t = x_true.unsqueeze(0)
        y_t = y_true.unsqueeze(0)
        g_true = compute_gradient(model, x_t, y_t)
        inferred_label = infer_label_from_final_bias_grad(g_true[-1])
        label_correct = bool(int(inferred_label.item()) == int(y_true.item()))

        x_recon_ig, _, history = reconstruct(
            model, g_true, input_dim, mode="cosine", know_label=True, true_y=inferred_label,
            n_iters=500, lr=0.05, seed=RANDOM_STATE + i, optimizer_name="adam",
        )
        score_ig = score_reconstruction(x_true.numpy(), x_recon_ig.squeeze(0).numpy(), blocks)
        score_ig["label_inference_correct"] = label_correct
        score_ig["best_loss"] = min(history)
        per_victim_ig.append(score_ig)

        x_recon_cf = closed_form_first_layer_attack(g_true)
        score_cf = score_reconstruction(x_true.numpy(), x_recon_cf.squeeze(0).numpy(), blocks)
        score_cf["label_inference_correct"] = label_correct
        per_victim_cf.append(score_cf)

    return per_victim_ig, per_victim_cf


def main() -> None:
    data = torch.load(DATA_DIR / "victims.pt")
    victims_X, victims_y = data["victims_X"], data["victims_y"]
    input_dim = data["input_dim"]
    blocks = load_feature_blocks(ENCODERS_DIR)

    def summarize(per_victim):
        def agg(key):
            vals = [r[key] for r in per_victim]
            return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        return {
            "overall_mse": agg("overall_mse"),
            "categorical_accuracy": agg("categorical_accuracy"),
            "tool_precision": agg("tool_precision"),
            "tool_recall": agg("tool_recall"),
            "label_inference_accuracy": round(
                sum(r["label_inference_correct"] for r in per_victim) / len(per_victim), 4
            ),
        }

    results = {}
    weight_norms = {}
    for eps_label, path in MODEL_PATHS.items():
        model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        model.load_state_dict(torch.load(path))
        model.eval()
        weight_norms[eps_label] = round(sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5, 4)

        per_victim_ig, per_victim_cf = attack_model(model, victims_X, victims_y, blocks, input_dim)
        agg_ig = summarize(per_victim_ig)
        agg_cf = summarize(per_victim_cf)

        results[eps_label] = {
            "weight_norm": weight_norms[eps_label],
            "inverting_gradients": {"per_victim": per_victim_ig, "aggregate": agg_ig},
            "closed_form_first_layer": {"per_victim": per_victim_cf, "aggregate": agg_cf},
        }
        print(f"[eps={eps_label:4s}] weight_norm={weight_norms[eps_label]:7.3f}  "
              f"IG: mse={agg_ig['overall_mse']['mean']:.4f} cat_acc={agg_ig['categorical_accuracy']['mean']:.3f}  |  "
              f"ClosedForm: mse={agg_cf['overall_mse']['mean']:.6f} cat_acc={agg_cf['categorical_accuracy']['mean']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "01_leakage_vs_epsilon_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    eps_order = ["1", "3", "8", "inf"]
    x_labels = [r"$\varepsilon$=1", r"$\varepsilon$=3", r"$\varepsilon$=8", r"$\varepsilon$=$\infty$"]

    # --- Figure 1: IG (optimization-based) vs epsilon, annotated with weight norm ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    mse_vals = [results[e]["inverting_gradients"]["aggregate"]["overall_mse"]["mean"] for e in eps_order]
    cat_vals = [results[e]["inverting_gradients"]["aggregate"]["categorical_accuracy"]["mean"] for e in eps_order]
    wnorm_vals = [results[e]["weight_norm"] for e in eps_order]

    axes[0].plot(x_labels, mse_vals, marker="o", color="tab:red")
    axes[0].set_title("Inverting Gradients: MSE vs. ε\n(higher = attack failed more)")
    axes[0].set_ylabel("MSE")

    axes[1].plot(x_labels, cat_vals, marker="o", color="tab:orange")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Inverting Gradients: Categorical Accuracy vs. ε")

    axes[2].plot(x_labels, wnorm_vals, marker="o", color="tab:green")
    axes[2].set_title("Model Weight Norm vs. ε\n(explains the pattern at left)")

    fig.suptitle("Optimization-Based Attack (IG) Success Tracks Weight Norm, Not Epsilon Directly")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leakage_vs_epsilon.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: closed-form attack vs epsilon - the real ceiling -----
    cf_mse_vals = [results[e]["closed_form_first_layer"]["aggregate"]["overall_mse"]["mean"] for e in eps_order]
    cf_cat_vals = [results[e]["closed_form_first_layer"]["aggregate"]["categorical_accuracy"]["mean"] for e in eps_order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(x_labels, mse_vals, marker="o", label="Inverting Gradients (optimization)", color="tab:red")
    axes[0].plot(x_labels, cf_mse_vals, marker="s", label="Closed-form (exact identity)", color="tab:blue")
    axes[0].set_yscale("symlog", linthresh=1e-6)
    axes[0].set_title("Reconstruction MSE: Optimization vs. Closed-Form")
    axes[0].set_ylabel("MSE (symlog)")
    axes[0].legend(fontsize=8)

    axes[1].plot(x_labels, cat_vals, marker="o", label="Inverting Gradients (optimization)", color="tab:red")
    axes[1].plot(x_labels, cf_cat_vals, marker="s", label="Closed-form (exact identity)", color="tab:blue")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Categorical Accuracy: Optimization vs. Closed-Form")
    axes[1].legend(fontsize=8)

    fig.suptitle("The True Single-Gradient Vulnerability Ceiling (closed-form) vs.\nWhat Optimization-Based Attacks Actually Find")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "closed_form_vs_optimization.png", dpi=150)
    plt.close(fig)

    print(f"\nWeight norms by epsilon: {weight_norms}")
    print(f"\nSaved: {OUT_DIR / '01_leakage_vs_epsilon_results.json'}")
    print(f"Saved: {FIG_DIR / 'leakage_vs_epsilon.png'}")
    print(f"Saved: {FIG_DIR / 'closed_form_vs_optimization.png'}")


if __name__ == "__main__":
    main()
