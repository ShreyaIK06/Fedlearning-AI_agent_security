# Phase 4 - Centralized vs. Federated Comparison

## Performance (test set)

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| centralized_mlp | 0.9296 | 0.7145 | 0.8327 | 0.7691 | 0.9616 | 0.8445 |
| federated_mlp | 0.8816 | 0.5581 | 0.7661 | 0.6458 | 0.9201 | 0.6841 |
| gap (centralized - federated) | 0.048 | 0.1564 | 0.0666 | 0.1233 | 0.0415 | 0.1604 |

The centralized model outperforms the federated model on **every metric**, by 0.04-0.16
depending on the metric (largest gaps in precision and PR-AUC). This is a consistent,
one-sided gap, not noise in one or two metrics.

## Why: the client-drift explanation

The 4 FL clients (`suite`-based: banking, slack, travel, workspace) are genuinely
non-IID - `security` breach rates range **7.7%
(workspace)** to **39.7% (slack)**,
a ~5x spread (Phase 3 Step 1). Each round, every client's local model drifts toward its own
locally-skewed optimum during its 3 local epochs, before
FedAvg's size-weighted average pulls the 4 drifted models back toward a compromise. This
"client drift" is well-documented in the FL literature (it's the core motivation behind
algorithms like FedProx) and is the expected, not anomalous, outcome of training on
heterogeneous clients with plain FedAvg - not a bug in this implementation (same
architecture, same imbalance handling, same test set as the centralized run).

## Convergence

- Centralized: converged in **34 epochs** (best at epoch
  26), single model,
  single optimizer trajectory over the full pooled training set.
- Federated: converged in **73 communication rounds** (best at round
  63), each round involving 4 independent local-then-aggregate steps.

**These x-axis units are not directly comparable** (one centralized "epoch" is one pass over
~19K pooled rows; one federated "round" is 4 clients each doing
3 local epochs over their own, much smaller, partition) -
see `figures/phase4/convergence_side_by_side.png` for the two curves plotted separately
rather than misleadingly overlaid on a shared x-axis.

## Communication cost

- Centralized: **0 MB** - single-node training, no network transmission of model weights at all.
- Federated: **72.36 MB** total, over 73 rounds x
  4 clients x 2 (download + upload) x 30,977 float32 parameters per transmission.

This is the fundamental cost federated learning pays for keeping data on-device: centralized
training has no communication cost at all (data and compute are co-located), while every FL
round requires transmitting the full model in both directions for every participating client.

## Training time

- Centralized: **10.8 seconds** (34 epochs, single process).
- Federated: **153.2 seconds** (73 rounds, clients trained sequentially
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
- (-) Real communication cost (72.36 MB here) that scales with rounds x clients
  x model size - would be materially larger for a bigger model.
- (-) More communication rounds needed to converge (73 rounds vs.
  34 centralized epochs) - though these units aren't equivalent (see above).

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
