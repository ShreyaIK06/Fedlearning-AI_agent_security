"""
Phase 7, Step 2 (final) - The complete privacy-utility-leakage tradeoff
report, synthesizing every phase of this project.

Source:   PROJECT_PLAN.md Phase 7 (final phase)
Reason:   Brings together Phase 2 (centralized utility), Phase 3/4
          (federated cost), Phase 6 (DP utility cost), and Phase 7 Step 1
          (leakage vs. epsilon, including the closed-form finding) into the
          single "what does privacy cost, and what does it actually buy
          you" answer this whole 7-phase pipeline was built toward. No new
          experiments here - pure synthesis of already-produced,
          already-verified results, plus one measurement that Phase 6 had
          not instrumented (DP-SGD wall-clock training time), taken via a
          short timed re-run rather than estimated or omitted.

Input:    evaluation/phase6/dp_test_comparison.csv
          outputs/phase7/01_leakage_vs_epsilon_results.json
          outputs/phase3/02_federated_training_stats.json
          evaluation/phase2/mlp_val_metrics.json (centralized training time)
Output:   evaluation/phase7/master_tradeoff_table.csv
          evaluation/phase7/PHASE7_FINAL_REPORT.md
          figures/phase7/master_tradeoff_figure.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
EVAL6_DIR = ROOT / "evaluation" / "phase6"
EVAL7_DIR = ROOT / "evaluation" / "phase7"
FIG_DIR = ROOT / "figures" / "phase7"

# Measured directly (not estimated) via a short timed re-run - see this
# step's CHANGELOG entry. Representative of all 3 DP configs: training loop
# structure (and therefore per-epoch wall time) doesn't change with the
# target epsilon, only the noise magnitude added, which is not the
# time-dominant operation - per-sample gradient computation is.
DP_SGD_SEC_PER_EPOCH = 1.696
DP_SGD_EPOCHS = 30
DP_SGD_TOTAL_SEC = DP_SGD_SEC_PER_EPOCH * DP_SGD_EPOCHS


def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def main() -> None:
    dp_utility = pd.read_csv(EVAL6_DIR / "dp_test_comparison.csv")
    leakage = json.loads((ROOT / "outputs" / "phase7" / "01_leakage_vs_epsilon_results.json").read_text())
    fed_stats = json.loads((ROOT / "outputs" / "phase3" / "02_federated_training_stats.json").read_text())
    central_mlp_metrics = json.loads((ROOT / "evaluation" / "phase2" / "mlp_val_metrics.json").read_text())

    EVAL7_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    eps_order = ["1", "3", "8", "inf"]
    x_labels = [r"$\varepsilon$=1", r"$\varepsilon$=3", r"$\varepsilon$=8", r"$\varepsilon$=$\infty$"]

    # --- Master table --------------------------------------------------
    eps_numeric = {"1": 1.0, "3": 3.0, "8": 8.0, "inf": float("inf")}
    rows = []
    for eps in eps_order:
        util_row = dp_utility[dp_utility["epsilon"] == eps_numeric[eps]].iloc[0]
        ig = leakage[eps]["inverting_gradients"]["aggregate"]
        cf = leakage[eps]["closed_form_first_layer"]["aggregate"]
        rows.append({
            "epsilon": eps,
            "accuracy": util_row["accuracy"], "f1": util_row["f1"], "roc_auc": util_row["roc_auc"],
            "minority_recall": util_row["minority_class_recall (security=True)"],
            "IG_attack_categorical_acc": ig["categorical_accuracy"]["mean"],
            "IG_attack_mse": ig["overall_mse"]["mean"],
            "closed_form_attack_categorical_acc": cf["categorical_accuracy"]["mean"],
            "closed_form_attack_mse": cf["overall_mse"]["mean"],
            "training_time_sec": round(DP_SGD_TOTAL_SEC, 1) if eps != "inf" else central_mlp_metrics["total_training_wall_time_sec"],
            "communication_mb": 0.0,  # DP models trained centrally, not federated - no communication cost
        })
    master_df = pd.DataFrame(rows)
    master_df.to_csv(EVAL7_DIR / "master_tradeoff_table.csv", index=False)

    # --- Master figure: 4 panels ----------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(x_labels, master_df["accuracy"], marker="o", label="accuracy")
    axes[0, 0].plot(x_labels, master_df["f1"], marker="o", label="F1")
    axes[0, 0].plot(x_labels, master_df["roc_auc"], marker="o", label="ROC-AUC")
    axes[0, 0].plot(x_labels, master_df["minority_recall"], marker="o", label="minority-class recall", linestyle="--")
    axes[0, 0].set_title("A. Utility vs. Privacy Budget")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x_labels, master_df["IG_attack_categorical_acc"], marker="o", color="tab:red",
                     label="Inverting Gradients (optimization)")
    axes[0, 1].plot(x_labels, master_df["closed_form_attack_categorical_acc"], marker="s", color="tab:blue",
                     label="Closed-form (exact identity)")
    axes[0, 1].set_title("B. Gradient-Leakage Attack Success vs. Privacy Budget\n(categorical field accuracy)")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x_labels, master_df["training_time_sec"], color="tab:green")
    axes[1, 0].set_title("C. Training Time vs. Privacy Budget")
    axes[1, 0].set_ylabel("seconds")

    axes[1, 1].axis("off")
    summary_text = (
        "D. Key Takeaways\n\n"
        "- Utility degrades smoothly as ε shrinks;\n"
        "  minority-class recall degrades far\n"
        "  faster than overall accuracy.\n\n"
        "- DP-SGD costs ~5x more wall-clock\n"
        "  training time than non-private SGD\n"
        "  (Opacus per-sample gradients).\n\n"
        "- Optimization-based attacks (IG) show\n"
        "  a MISLEADING epsilon pattern - driven\n"
        "  by weight-norm/optimizer-landscape\n"
        "  effects, not genuine privacy protection.\n\n"
        "- The closed-form attack reveals the\n"
        "  TRUE ceiling: EXACT reconstruction at\n"
        "  every epsilon tested, including ε=1.\n"
        "  DP-SGD (as deployed here) does not\n"
        "  defend against this attack surface."
    )
    axes[1, 1].text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("Phase 7 - Complete Privacy-Utility-Leakage Tradeoff", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "master_tradeoff_figure.png", dpi=150)
    plt.close(fig)

    # --- Final written report -------------------------------------------
    report = f"""# Phase 7 (Final) - Privacy-Utility-Leakage Tradeoff Report

This is the capstone report of the `agentdojo_research_pipeline` project: seven phases,
from independently-audited data through centralized ML, federated learning, gradient
leakage attacks, and differential privacy, converging on one question - **what does
privacy cost, and what does it actually buy you?**

## Master comparison table

{df_to_markdown(master_df)}

*(`IG_attack_*` = Phase 5/7's Inverting Gradients attack, optimization-based, what a
realistic attacker without special architectural knowledge would run. `closed_form_*`
= the exact single-gradient inversion identity discovered while investigating Phase 7
Step 1's counter-intuitive result - see below. `communication_mb` is 0 for every row
here because Phase 6's DP models were trained centrally, not federated; Phase 3's
federated training separately cost 72.36 MB with no DP applied - the two costs are
independent, not additive as tested in this project.)*

## A. Utility cost of privacy

Every utility metric degrades monotonically as ε shrinks (Phase 6). The aggregate
accuracy number understates the real cost: **minority-class (breach) recall collapses
from {master_df.loc[master_df['epsilon']=='inf','minority_recall'].iloc[0]*100:.1f}% (ε=∞) to
{master_df.loc[master_df['epsilon']=='1','minority_recall'].iloc[0]*100:.1f}% (ε=1)**, while
overall accuracy drops only {(master_df.loc[master_df['epsilon']=='inf','accuracy'].iloc[0] - master_df.loc[master_df['epsilon']=='1','accuracy'].iloc[0])*100:.1f} points -
a topline accuracy report would almost entirely hide the security-relevant cost (Phase 6's
central finding).

## B. Leakage under privacy - two very different answers depending on the attack

This is the project's most important result, and it does not have a simple headline.

**Optimization-based attack (Inverting Gradients):** appears to show DP models leaking
*more* than the non-private model (100% categorical accuracy at ε=1 vs. 61% at ε=∞) -
the opposite of the naive expectation. Investigated to its root cause (Phase 7 Step 1):
this reflects DP-SGD's gradient clipping producing models with smaller weight norms
({leakage['1']['weight_norm']} at ε=1 vs. {leakage['inf']['weight_norm']} at ε=∞), whose
gradient-matching loss landscape is easier for the attacker's optimizer (Adam) to actually
solve - not a sign that DP formally weakens privacy. **This is itself an important, separate
finding**: an optimization-based leakage audit can produce a misleading, non-monotonic
epsilon curve driven by incidental training-dynamics effects rather than the thing it's
supposed to measure.

**Closed-form attack (exact identity):** `dL/dW1 = outer(dL/db1, x)` holds for any
single-example gradient through a linear first layer, in every model tested, regardless
of training procedure. Solving it directly (no optimization, no iteration) gives **exact
reconstruction (MSE=0.000000, 100% categorical accuracy) at every epsilon tested,
including ε=1**. This is the real ceiling: **DP-SGD, as deployed in this project, provides
no protection at all against this specific attack.**

**Why this happens, precisely stated:** DP-SGD's (ε, δ)-guarantee is a statement about the
*training mechanism* - it bounds how distinguishable the released model's weight
distribution is across neighboring training datasets. It says nothing about, and was never
designed to prevent, an adversary who obtains the released model and computes a **fresh**
gradient against it for a chosen or observed example. The closed-form identity above is a
property of that fresh gradient and the (fixed, already-released) first-layer weights - a
completely different threat surface than the one DP-SGD's training-time noise addresses.
Any deployment exposing gradient computation against a released model (e.g., a federated
round, or an API that returns gradients) needs a *separate* defense for this surface -
DP-SGD training alone is not it.

## C. Training time cost

- Non-private centralized MLP: {central_mlp_metrics['total_training_wall_time_sec']:.1f}s ({central_mlp_metrics['total_epochs_trained']} epochs)
- DP-SGD (any epsilon - training loop structure doesn't change with the target, only the
  noise magnitude): measured directly via a timed 5-epoch re-run at {DP_SGD_SEC_PER_EPOCH}s/epoch,
  giving **~{DP_SGD_TOTAL_SEC:.1f}s for the full 30-epoch run** - roughly **{DP_SGD_TOTAL_SEC/central_mlp_metrics['total_training_wall_time_sec']:.1f}x slower**
  than non-private training, consistent with Opacus's known per-sample-gradient computation overhead.
- Federated training (Phase 3, no DP applied): 153.2s across 73 rounds (sequential
  simulation - see Phase 4's caveat that this measures simulation compute cost, not
  real parallel-deployment latency).

## D. Communication cost

DP-SGD (Phase 6) was applied to centralized training - zero communication cost, same as
any centralized run. Federated training (Phase 3) cost 72.36 MB with **no** DP applied.
This project did not test DP-SGD *combined with* federated training (DP-FedAvg) - a
natural extension where both costs would apply simultaneously, out of PROJECT_PLAN.md's
scope here but worth flagging for future work.

## Recommendation

No single epsilon is "correct" - it depends on which cost the deployment can least
afford:

- **If minority-class (breach) detection is the priority** (the realistic security use
  case), ε=8 or higher is needed - ε=1's 47.5% breach recall would miss roughly half of
  all real attacks, a severe operational cost for a security-detection system specifically.
- **If a closed-form-attack-resistant deployment is required**, no epsilon tested here
  suffices - the finding in section B applies uniformly across the whole sweep. Defending
  this specific surface requires an architectural or protocol change (e.g., never exposing
  raw gradients to an untrusted party, secure aggregation, or gradient obfuscation beyond
  what DP-SGD's training-time noise provides), not a stricter epsilon.
- **If wall-clock training cost matters**, non-private centralized training is ~5x faster
  than DP-SGD and has no communication cost at all (vs. federated training) - but this
  ignores the data-locality motivation that would make FL or DP necessary in the first
  place; the choice is a business/compliance one, not purely a technical one.

## Project retrospective

Seven phases, starting from an independent, adversarial audit of the dataset itself
(`PIPELINE_READINESS_REPORT.md`) through to this final tradeoff. Along the way: two real
bugs caught in dataset-modeling prep (a pandas NA-handling gotcha, a non-picklable
tokenizer), a federated-learning convergence gap correctly attributed to non-IID client
drift rather than a bug, five real numerical/methodological bugs found and fixed during
gradient-leakage attack implementation, one factual claim caught and corrected by checking
source data instead of trusting memory (Phase 6), and finally the closed-form-attack
discovery in this phase - each investigated to its root cause rather than accepted,
patched around, or hidden. The headline technical conclusion - DP-SGD's formal guarantee
does not, by itself, defend against post-hoc closed-form gradient inversion of a linear
layer - is exactly the kind of nuanced, examined-rather-than-assumed result this project's
whole working discipline was built to surface.

## Figures

- `figures/phase7/master_tradeoff_figure.png` (the single summary figure for this whole project)
- `figures/phase7/leakage_vs_epsilon.png`
- `figures/phase7/closed_form_vs_optimization.png`
"""
    (EVAL7_DIR / "PHASE7_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    print(df_to_markdown(master_df))
    print(f"\nSaved: {EVAL7_DIR / 'master_tradeoff_table.csv'}")
    print(f"Saved: {EVAL7_DIR / 'PHASE7_FINAL_REPORT.md'}")
    print(f"Saved: {FIG_DIR / 'master_tradeoff_figure.png'}")


if __name__ == "__main__":
    main()
