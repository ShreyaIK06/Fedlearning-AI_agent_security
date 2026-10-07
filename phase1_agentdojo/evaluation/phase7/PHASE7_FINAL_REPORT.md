# Phase 7 (Final) - Privacy-Utility-Leakage Tradeoff Report

This is the capstone report of the `agentdojo_research_pipeline` project: seven phases,
from independently-audited data through centralized ML, federated learning, gradient
leakage attacks, and differential privacy, converging on one question - **what does
privacy cost, and what does it actually buy you?**

## Master comparison table

| epsilon | accuracy | f1 | roc_auc | minority_recall | IG_attack_categorical_acc | IG_attack_mse | closed_form_attack_categorical_acc | closed_form_attack_mse | training_time_sec | communication_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8843 | 0.5359 | 0.8817 | 0.4746 | 1.0 | 0.0155 | 1.0 | 0.0 | 50.9 | 0.0 |
| 3 | 0.8928 | 0.6167 | 0.8945 | 0.6125 | 0.9583 | 0.2108 | 1.0 | 0.0 | 50.9 | 0.0 |
| 8 | 0.8928 | 0.6234 | 0.8967 | 0.6301 | 0.8333 | 0.2951 | 1.0 | 0.0 | 50.9 | 0.0 |
| inf | 0.9296 | 0.7691 | 0.9616 | 0.8327 | 0.6111 | 0.6213 | 1.0 | 0.0 | 10.8116 | 0.0 |

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
from 83.3% (ε=∞) to
47.5% (ε=1)**, while
overall accuracy drops only 4.5 points -
a topline accuracy report would almost entirely hide the security-relevant cost (Phase 6's
central finding).

## B. Leakage under privacy - two very different answers depending on the attack

This is the project's most important result, and it does not have a simple headline.

**Optimization-based attack (Inverting Gradients):** appears to show DP models leaking
*more* than the non-private model (100% categorical accuracy at ε=1 vs. 61% at ε=∞) -
the opposite of the naive expectation. Investigated to its root cause (Phase 7 Step 1):
this reflects DP-SGD's gradient clipping producing models with smaller weight norms
(12.3951 at ε=1 vs. 19.4532 at ε=∞), whose
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

- Non-private centralized MLP: 10.8s (34 epochs)
- DP-SGD (any epsilon - training loop structure doesn't change with the target, only the
  noise magnitude): measured directly via a timed 5-epoch re-run at 1.696s/epoch,
  giving **~50.9s for the full 30-epoch run** - roughly **4.7x slower**
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
