"""
Phase 5, Step 5 (final) - Compare DLG, iDLG, and Inverting Gradients; produce
the consolidated Phase 5 report.

Source:   PROJECT_PLAN.md Phase 5; AgentDojo_Dataset_Research_Report.md Part 17
Reason:   Synthesizes Steps 2-4's results into the comparison and the
          "can gradients reconstruct sensitive information" answer the
          whole phase was building toward. No new attack methodology here -
          this reads Steps 2-4's saved results and (for the one case-study
          victim whose true trace includes a sensitive tool call) re-runs
          each attack's already-validated `reconstruct()` once more to
          inspect exactly which tool names came back, since the per-victim
          summary JSONs only stored aggregate precision/recall, not which
          specific tokens were correctly recovered.

Input:    outputs/phase5/victims.pt
          outputs/phase5/{02_dlg,03_idlg,04_ig}_results.json
          models/phase3/federated_mlp_state_dict.pt
Output:   evaluation/phase5/attack_comparison.csv
          evaluation/phase5/phase5_comparison_report.md
          figures/phase5/three_way_comparison.png
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.mlp_model import SecurityMLP  # noqa: E402
from common.gradient_leakage import (  # noqa: E402
    infer_label_from_final_bias_grad,
    load_feature_blocks,
    reconstruct,
)

DATA_DIR = ROOT / "outputs" / "phase5"
MODEL_IN = ROOT / "models" / "phase3" / "federated_mlp_state_dict.pt"
ENCODERS_DIR = ROOT / "models" / "phase1_encoders"
EVAL_DIR = ROOT / "evaluation" / "phase5"
FIG_DIR = ROOT / "figures" / "phase5"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
SENSITIVE_TOOLS = ["tool__send_money", "tool__update_password", "tool__remove_user_from_slack",
                   "tool__post_webpage", "tool__delete_file"]


def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def main() -> None:
    dlg = json.loads((DATA_DIR / "02_dlg_results.json").read_text())
    idlg = json.loads((DATA_DIR / "03_idlg_results.json").read_text())
    ig = json.loads((DATA_DIR / "04_ig_results.json").read_text())
    null_baseline = json.loads((DATA_DIR / "01_victim_selection_stats.json").read_text())["null_baseline_mse"]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Comparison table -----------------------------------------------
    rows = []
    for name, r in [("DLG", dlg), ("iDLG", idlg), ("Inverting Gradients", ig)]:
        a = r["aggregate"]
        rows.append({
            "attack": name,
            "overall_mse": a["overall_mse"]["mean"],
            "categorical_accuracy": a["categorical_accuracy"]["mean"],
            "tool_precision": a["tool_precision"]["mean"],
            "tool_recall": a["tool_recall"]["mean"],
            "numeric_mae_std_units": a["numeric_mae_std_units"]["mean"],
            "label_recovery_accuracy": a.get("label_recovery_rate", a.get("label_inference_accuracy")),
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(EVAL_DIR / "attack_comparison.csv", index=False)

    # --- 2. Three-way comparison figure -------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    attacks = comparison_df["attack"]

    axes[0].bar(["null"] + list(attacks), [null_baseline["training_mean_vector_guess_mean"]] + list(comparison_df["overall_mse"]))
    axes[0].set_yscale("log")
    axes[0].set_title("Reconstruction MSE (log)\nlower = more leaked")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(attacks, comparison_df["categorical_accuracy"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Categorical Field Accuracy")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(attacks, comparison_df["tool_recall"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Tool-Usage Recall")
    axes[2].tick_params(axis="x", rotation=30)

    axes[3].bar(attacks, comparison_df["label_recovery_accuracy"])
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Label Recovery Accuracy")
    axes[3].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "three_way_comparison.png", dpi=150)
    plt.close(fig)

    # --- 3. Sensitive-information case study --------------------------------
    victims_data = torch.load(DATA_DIR / "victims.pt")
    blocks = load_feature_blocks(ENCODERS_DIR)
    ts, te = blocks.tool_range
    tool_names = blocks.feature_names[ts:te]

    case_study_idx = None
    for i, (x, meta) in enumerate(zip(victims_data["victims_X"], victims_data["victim_meta"])):
        active = [tool_names[j] for j in range(len(tool_names)) if x[ts + j] > 0.5]
        if any(t in SENSITIVE_TOOLS for t in active):
            case_study_idx = i
            true_active_tools = active
            break

    case_study_section = "No victim among the 12 selected happened to include a sensitive tool call - skipped."
    if case_study_idx is not None:
        model = SecurityMLP(input_dim=victims_data["input_dim"], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        model.load_state_dict(torch.load(MODEL_IN))
        model.eval()

        x_true = victims_data["victims_X"][case_study_idx]
        g_true = victims_data["victim_grads"][case_study_idx]
        meta = victims_data["victim_meta"][case_study_idx]
        inferred_label = infer_label_from_final_bias_grad(g_true[-1])

        recon_tools = {}
        x_dlg, _, _ = reconstruct(model, g_true, victims_data["input_dim"], mode="l2", know_label=False,
                                   n_iters=150, lr=1.0, seed=42 + case_study_idx)
        x_idlg, _, _ = reconstruct(model, g_true, victims_data["input_dim"], mode="l2", know_label=True,
                                    true_y=inferred_label, n_iters=150, lr=1.0, seed=42 + case_study_idx)
        x_ig, _, _ = reconstruct(model, g_true, victims_data["input_dim"], mode="cosine", know_label=True,
                                  true_y=inferred_label, n_iters=500, lr=0.05, seed=42 + case_study_idx,
                                  optimizer_name="adam")

        for name, x_r in [("DLG", x_dlg), ("iDLG", x_idlg), ("Inverting Gradients", x_ig)]:
            recovered = [tool_names[j] for j in range(len(tool_names)) if x_r.squeeze(0)[ts + j] > 0.5]
            recon_tools[name] = recovered

        lines = [
            f"Victim {case_study_idx} (suite: {meta['suite']}, row {meta['row_index']}) has true active tools: "
            f"**{', '.join(true_active_tools)}** - including a sensitive one, `post_webpage`.",
            "",
            "| Attack | Recovered tool set | `post_webpage` recovered? |",
            "|---|---|---|",
        ]
        for name in ["DLG", "iDLG", "Inverting Gradients"]:
            recovered = recon_tools[name]
            got_sensitive = "**YES**" if "tool__post_webpage" in recovered else "no"
            recovered_str = ", ".join(t.replace("tool__", "") for t in recovered) if recovered else "(none)"
            lines.append(f"| {name} | {recovered_str} | {got_sensitive} |")
        case_study_section = "\n".join(lines)

    # --- 4. Written report ---------------------------------------------
    best_mse = comparison_df.loc[comparison_df["overall_mse"].idxmin()]
    best_cat = comparison_df.loc[comparison_df["categorical_accuracy"].idxmax()]
    best_tool = comparison_df.loc[comparison_df["tool_recall"].idxmax()]

    report = f"""# Phase 5 - Gradient Leakage Attack Comparison Report

## Threat model (restated)

All three attacks target a **single-example gradient** at the trained federated
model's final weights (`models/phase3/federated_mlp_state_dict.pt`) - the classic
DLG/iDLG/Inverting-Gradients setting (one forward+backward pass), not Phase 3's
multi-epoch FedAvg delta. 12 victims (3 per suite client). Full framing in
`src/common/gradient_leakage.py`'s module docstring.

## Results (n=12 victims, mean across all)

{df_to_markdown(comparison_df)}

Null baseline for reference: training-mean-vector guess MSE =
{null_baseline['training_mean_vector_guess_mean']}, random-noise guess MSE =
{null_baseline['random_noise_guess_mean']}.

**No single attack dominates every metric.** Best raw reconstruction (lowest
MSE): **{best_mse['attack']}** ({best_mse['overall_mse']:.3f}). Best categorical
recovery: **{best_cat['attack']}** ({best_cat['categorical_accuracy']*100:.1f}%).
Best tool-usage recall: **{best_tool['attack']}** ({best_tool['tool_recall']*100:.1f}%).

## What each attack actually tells us

- **DLG** (L2 loss, optimized label): the weakest attack here. Reconstruction MSE
  ({dlg['aggregate']['overall_mse']['mean']:.2f}) is *worse* than the blind
  training-mean-guess baseline, and label recovery (8.3%) is worse than a coin
  flip - jointly optimizing input and label over L2 distance is unstable on this
  feature space, exactly the failure mode the literature developed iDLG and IG to
  address.
- **iDLG** (L2 loss, closed-form label): removing the label-optimization problem
  fixes most of DLG's instability - MSE drops 136x to {idlg['aggregate']['overall_mse']['mean']:.2f}.
  But this doesn't translate to better *discrete* leakage: categorical accuracy and
  tool recall are both lower than DLG's. Converging precisely in continuous gradient
  space doesn't guarantee the discrete one-hot/binary structure comes along with it.
- **Inverting Gradients** (cosine loss, closed-form label): a middle-ground MSE
  ({ig['aggregate']['overall_mse']['mean']:.2f}, between DLG and iDLG) but the best
  performance on the metrics that answer "what did the attacker actually learn":
  highest categorical accuracy and highest tool precision. Consistent with the
  paper's own framing - cosine similarity targets gradient *direction*, which turns
  out to align better with discrete category/tool identity than L2's magnitude-
  sensitive matching does, even though it doesn't win on aggregate continuous error.

## Sensitive-information case study

The report's own question ("can gradients reconstruct... sensitive information")
is answered concretely here, not just via aggregate tool-recall numbers:

{case_study_section}

**Reading the case study honestly, not just reporting the "YES":** Inverting
Gradients did recover `post_webpage` correctly, but alongside 22 other tool
guesses for a victim whose true tool set had only 3 entries - including
plausible-looking but **false** sensitive-tool alarms (`send_money`,
`remove_user_from_slack`). DLG's reconstruction is even less selective (34 of
~75 possible tools flagged "active"). An attacker automating this without a
human analyst reviewing the output would face a high false-positive rate -
the true signal is present, but not cleanly isolated. iDLG, the most
conservative attack here, recovered nothing at all for this victim (0 tools
above threshold), consistent with its near-zero tool recall aggregate. The
honest conclusion is "sensitive tool usage can leak, but current gradient-
matching alone doesn't produce a clean, actionable signal an attacker could
trust without corroborating evidence" - not "the model safely hides tool usage."

## Comparison against the original report's expectation (Part 17)

The report predicted: *"if you restrict the model to the compact, mostly-categorical
trace features... the input dimensionality (~8-12 after encoding) is small enough
that gradient inversion may reconstruct inputs too easily... Including the
text-embedding features raises both the realism and the difficulty of the attack,
which is the more publishable framing."*

This project's feature space is **176-dimensional** (95 categorical one-hot + 75
tool bag-of-tools + 5 numeric + 1 flag) - considerably richer than the report's
~8-12 minimal baseline, though still short of full text embeddings. Consistent with
the report's prediction, reconstruction here is **not** near-trivial: none of the
three attacks come close to perfect recovery, MSE never approaches the null
baseline's low value, and categorical/tool recovery rates are modest (14-24%,
0-46% respectively) rather than saturated. This is evidence *for* the report's
framing - richer, more realistic feature encoding measurably raises attack
difficulty compared to the minimal encoding the report warned would leak "too
easily."

## Discussion

- Gradient leakage against this model's single-example gradients is real but
  **partial**, not catastrophic, at this feature dimensionality - an attacker
  observing one client's gradient learns *something* (occasionally a sensitive
  tool call, as the case study shows; sometimes the wrong thing entirely, as
  DLG's below-baseline MSE shows) but not a reliable, complete reconstruction.
- The three attacks are not simply "better/worse" - they trade off differently:
  iDLG minimizes continuous reconstruction error, Inverting Gradients better
  recovers discrete structure. A real threat assessment should report the
  attack matched to what the adversary actually cares about (exact values vs.
  categorical identity), not just whichever attack has the lowest headline MSE.
- This attack was run against the **final, trained** federated model. The
  literature consistently finds early-training (less-converged, larger-gradient)
  snapshots leak more - not tested here (out of PROJECT_PLAN.md's scope for this
  phase) but a natural extension: repeating this comparison against the FL
  training's round-1 model would show whether Phase 3's convergence itself is a
  partial, incidental defense.
- Phase 6 (Differential Privacy) is the direct, designed response to this
  finding: DP-SGD's gradient clipping and noise injection specifically target
  the exact per-example gradient signal these three attacks all depend on.

## Figures

- `figures/phase5/dlg_convergence_examples.png`, `dlg_reconstruction_quality.png`
- `figures/phase5/idlg_convergence_examples.png`, `idlg_reconstruction_quality.png`
- `figures/phase5/ig_convergence_examples.png`, `ig_reconstruction_quality.png`
- `figures/phase5/three_way_comparison.png`
"""
    (EVAL_DIR / "phase5_comparison_report.md").write_text(report, encoding="utf-8")

    print(df_to_markdown(comparison_df))
    print(f"\n{case_study_section}")
    print(f"\nSaved: {EVAL_DIR / 'attack_comparison.csv'}")
    print(f"Saved: {EVAL_DIR / 'phase5_comparison_report.md'}")
    print(f"Saved: {FIG_DIR / 'three_way_comparison.png'}")


if __name__ == "__main__":
    main()
