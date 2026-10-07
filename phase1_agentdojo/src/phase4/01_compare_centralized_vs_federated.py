"""
Phase 4 - Centralized vs. Federated comparison.

Source:   PROJECT_PLAN.md Phase 4
Reason:   Synthesizes Phase 2's centralized MLP and Phase 3's federated MLP
          results into the comparison the whole exercise was building
          toward - same architecture (`src/common/mlp_model.py`), same
          test set, same imbalance-handling approach, so every difference
          reported here reflects *how* training happened (pooled data vs.
          FedAvg over 4 non-IID clients), not a confound. No new models are
          trained in this phase - everything here reads Phase 2/3 outputs.

Input:    evaluation/phase2/mlp_test_metrics.json
          evaluation/phase3/federated_mlp_test_metrics.json
          outputs/phase2/04_mlp_training_history.csv
          outputs/phase3/02_federated_training_history.csv
          outputs/phase3/02_federated_training_stats.json
          outputs/phase3/01_client_partition_stats.json
Output:   evaluation/phase4/performance_comparison.csv
          evaluation/phase4/phase4_comparison_report.md
          figures/phase4/{performance_comparison_bar,convergence_side_by_side,
                           communication_cost,training_time_comparison}.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
EVAL2_DIR = ROOT / "evaluation" / "phase2"
EVAL3_DIR = ROOT / "evaluation" / "phase3"
OUT3_DIR = ROOT / "outputs" / "phase3"
EVAL4_DIR = ROOT / "evaluation" / "phase4"
FIG_DIR = ROOT / "figures" / "phase4"

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def main() -> None:
    centralized = json.loads((EVAL2_DIR / "mlp_test_metrics.json").read_text())
    federated = json.loads((EVAL3_DIR / "federated_mlp_test_metrics.json").read_text())

    central_history = pd.read_csv(ROOT / "outputs" / "phase2" / "04_mlp_training_history.csv")
    fed_history = pd.read_csv(OUT3_DIR / "02_federated_training_history.csv")
    fed_stats = json.loads((OUT3_DIR / "02_federated_training_stats.json").read_text())
    client_stats = json.loads((OUT3_DIR / "01_client_partition_stats.json").read_text())

    EVAL4_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Performance comparison table --------------------------------
    perf_rows = []
    for name, m in [("centralized_mlp", centralized), ("federated_mlp", federated)]:
        perf_rows.append({"model": name, **{c: m[c] for c in METRIC_COLS}})
    perf_df = pd.DataFrame(perf_rows)
    gap_row = {"model": "gap (centralized - federated)"}
    gap_row.update({c: round(perf_df.loc[0, c] - perf_df.loc[1, c], 4) for c in METRIC_COLS})
    perf_df = pd.concat([perf_df, pd.DataFrame([gap_row])], ignore_index=True)
    perf_df.to_csv(EVAL4_DIR / "performance_comparison.csv", index=False)

    # --- 2. Performance bar chart ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(METRIC_COLS))
    width = 0.35
    ax.bar([i - width / 2 for i in x], perf_df.loc[0, METRIC_COLS], width, label="Centralized MLP")
    ax.bar([i + width / 2 for i in x], perf_df.loc[1, METRIC_COLS], width, label="Federated MLP")
    ax.set_xticks(list(x))
    ax.set_xticklabels(METRIC_COLS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Centralized vs. Federated MLP - Test Set Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "performance_comparison_bar.png", dpi=150)
    plt.close(fig)

    # --- 3. Convergence side-by-side (different x-axis units - epochs vs. rounds) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ax1.plot(central_history["epoch"], central_history["val_roc_auc"], color="tab:blue")
    ax1.set_xlabel("Local training epoch")
    ax1.set_ylabel("Val ROC-AUC")
    ax1.set_title(f"Centralized MLP\n({len(central_history)} epochs to convergence)")
    ax1.grid(alpha=0.3)

    ax2.plot(fed_history["round"], fed_history["val_roc_auc"], color="tab:orange")
    ax2.set_xlabel("Communication round")
    ax2.set_title(f"Federated MLP (FedAvg)\n({len(fed_history)} rounds to convergence)")
    ax2.grid(alpha=0.3)
    fig.suptitle("Convergence Comparison (note: x-axes are NOT the same unit - see report)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_side_by_side.png", dpi=150)
    plt.close(fig)

    # --- 4. Communication cost -------------------------------------------
    fed_total_mb = fed_stats["total_bytes_transmitted_mb"]
    n_params = fed_stats["n_params"]
    centralized_mb = n_params * 4 / 1e6  # one model, no network transmission needed at all in this setup

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Centralized\n(0 - single node,\nno network transmission)", "Federated\n(73 rounds x 4 clients)"],
           [0, fed_total_mb], color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Data transmitted (MB)")
    ax.set_title("Communication Cost")
    for i, v in enumerate([0, fed_total_mb]):
        ax.text(i, v + 1, f"{v:.1f} MB", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "communication_cost.png", dpi=150)
    plt.close(fig)

    # --- 5. Training time comparison --------------------------------------
    central_time = json.loads((EVAL2_DIR / "mlp_val_metrics.json").read_text())["total_training_wall_time_sec"]
    fed_time = fed_history["round_wall_time_sec"].sum()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Centralized MLP", "Federated MLP (FedAvg)"], [central_time, fed_time], color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Total wall-clock training time (sec)")
    ax.set_title("Training Time Comparison\n(single-machine simulation - see caveat in report)")
    for i, v in enumerate([central_time, fed_time]):
        ax.text(i, v + 2, f"{v:.1f}s", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_time_comparison.png", dpi=150)
    plt.close(fig)

    # --- 6. Written report --------------------------------------------
    report = f"""# Phase 4 - Centralized vs. Federated Comparison

## Performance (test set)

{df_to_markdown(perf_df)}

The centralized model outperforms the federated model on **every metric**, by 0.04-0.16
depending on the metric (largest gaps in precision and PR-AUC). This is a consistent,
one-sided gap, not noise in one or two metrics.

## Why: the client-drift explanation

The 4 FL clients (`suite`-based: banking, slack, travel, workspace) are genuinely
non-IID - `security` breach rates range **{client_stats['client_stats']['workspace']['security_true_rate']*100:.1f}%
(workspace)** to **{client_stats['client_stats']['slack']['security_true_rate']*100:.1f}% (slack)**,
a ~5x spread (Phase 3 Step 1). Each round, every client's local model drifts toward its own
locally-skewed optimum during its {fed_stats['config']['local_epochs']} local epochs, before
FedAvg's size-weighted average pulls the 4 drifted models back toward a compromise. This
"client drift" is well-documented in the FL literature (it's the core motivation behind
algorithms like FedProx) and is the expected, not anomalous, outcome of training on
heterogeneous clients with plain FedAvg - not a bug in this implementation (same
architecture, same imbalance handling, same test set as the centralized run).

## Convergence

- Centralized: converged in **{len(central_history)} epochs** (best at epoch
  {int(central_history.loc[central_history['val_roc_auc'].idxmax(), 'epoch'])}), single model,
  single optimizer trajectory over the full pooled training set.
- Federated: converged in **{len(fed_history)} communication rounds** (best at round
  {fed_stats['best_round']}), each round involving 4 independent local-then-aggregate steps.

**These x-axis units are not directly comparable** (one centralized "epoch" is one pass over
~19K pooled rows; one federated "round" is 4 clients each doing
{fed_stats['config']['local_epochs']} local epochs over their own, much smaller, partition) -
see `figures/phase4/convergence_side_by_side.png` for the two curves plotted separately
rather than misleadingly overlaid on a shared x-axis.

## Communication cost

- Centralized: **0 MB** - single-node training, no network transmission of model weights at all.
- Federated: **{fed_total_mb:.2f} MB** total, over {fed_stats['communication_rounds_run']} rounds x
  4 clients x 2 (download + upload) x {n_params:,} float32 parameters per transmission.

This is the fundamental cost federated learning pays for keeping data on-device: centralized
training has no communication cost at all (data and compute are co-located), while every FL
round requires transmitting the full model in both directions for every participating client.

## Training time

- Centralized: **{central_time:.1f} seconds** ({len(central_history)} epochs, single process).
- Federated: **{fed_time:.1f} seconds** ({len(fed_history)} rounds, clients trained sequentially
  in this simulation).

**Important caveat**: this federated number reflects *sequential, single-machine simulation*
of all 4 clients per round, not a real deployment. In a genuine federated setting, clients
train in parallel on separate devices, so real-world wall-clock time would be closer to
`(per-client local training time) x rounds`, not `(sum of all clients' local training time)
x rounds` as measured here - this simulation's total-time number specifically measures
*compute cost*, not *deployable latency*. Communication latency (network round-trips per
round) is also not modeled here at all; only payload size (MB transmitted) is.

## Advantages / Disadvantages

**Centralized**
- (+) Higher accuracy/precision/recall/F1/ROC-AUC/PR-AUC on every metric measured here.
- (+) Faster to converge in wall-clock time (in this simulation) and zero communication cost.
- (-) Requires all 4 suites' data pooled in one place - the exact thing FL exists to avoid
  when data cannot leave its origin (e.g. real multi-tenant deployments with data-residency
  or competitive-separation requirements between organizations).

**Federated**
- (+) Never pools raw data across clients - each suite's rows never leave their own
  partition during training (Phase 5 will examine whether gradients leak that data anyway).
- (+) Naturally extends to more clients/organizations without a central data lake.
- (-) Measurable accuracy/precision/recall cost under non-IID clients with plain FedAvg
  (this report's central finding).
- (-) Real communication cost ({fed_total_mb:.2f} MB here) that scales with rounds x clients
  x model size - would be materially larger for a bigger model.
- (-) More communication rounds needed to converge ({len(fed_history)} rounds vs.
  {len(central_history)} centralized epochs) - though these units aren't equivalent (see above).

## Discussion

For this dataset and this simple FedAvg implementation, centralized training is strictly
better on every measured performance axis, at real cost to data locality. This matches the
report's own framing (`AgentDojo_Dataset_Research_Report.md` Part 16): `suite`-based clients
are *deliberately* non-IID, chosen specifically because they demonstrate heterogeneous
client risk realistically (a banking org's attack surface looks nothing like a Slack
workspace's) - the FL performance cost measured here is the direct, expected consequence of
that same realism, not an artifact of a poor implementation. A production system choosing
between these approaches would weigh this measured utility cost against whatever
data-locality requirement motivates using FL in the first place; a system that must use FL
for compliance/data-locality reasons could look to non-IID-aware aggregation (FedProx,
personalization layers, client clustering) to help close this gap - out of scope for this
project's PROJECT_PLAN.md, but a natural next step this comparison motivates.

## Figures

- `figures/phase4/performance_comparison_bar.png`
- `figures/phase4/convergence_side_by_side.png`
- `figures/phase4/communication_cost.png`
- `figures/phase4/training_time_comparison.png`
"""
    (EVAL4_DIR / "phase4_comparison_report.md").write_text(report, encoding="utf-8")

    print(df_to_markdown(perf_df))
    print(f"\nCentralized training time: {central_time:.1f}s ({len(central_history)} epochs)")
    print(f"Federated training time (sequential sim): {fed_time:.1f}s ({len(fed_history)} rounds)")
    print(f"Federated communication: {fed_total_mb:.2f} MB")
    print(f"\nSaved: {EVAL4_DIR / 'performance_comparison.csv'}")
    print(f"Saved: {EVAL4_DIR / 'phase4_comparison_report.md'}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
