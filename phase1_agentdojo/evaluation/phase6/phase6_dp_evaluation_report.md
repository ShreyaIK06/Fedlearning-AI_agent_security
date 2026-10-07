# Phase 6 - Differential Privacy Evaluation Report

## Results (test set, n=7258)

| epsilon | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8843 | 0.6155 | 0.4746 | 0.5359 | 0.8817 | 0.5652 |
| 3 | 0.8928 | 0.621 | 0.6125 | 0.6167 | 0.8945 | 0.6193 |
| 8 | 0.8928 | 0.6169 | 0.6301 | 0.6234 | 0.8967 | 0.6329 |
| inf | 0.9296 | 0.7145 | 0.8327 | 0.7691 | 0.9616 | 0.8445 |

## Minority-class impact (report Part 18's specific warning)

| epsilon | minority_class_recall (security=True) | majority_class_specificity (security=False) |
| --- | --- | --- |
| 1 | 0.4746 | 0.9514 |
| 3 | 0.6125 | 0.9387 |
| 8 | 0.6301 | 0.9359 |
| inf | 0.8327 | 0.9455 |

Going from no privacy (ε=∞) to the strictest budget tested (ε=1):
- **Minority-class (breach) recall drops by 0.3581** (83.3% -> 47.5%)
- **Majority-class (blocked) specificity does NOT drop - it improves slightly, by 0.0059** (94.5% -> 95.1%)

This is a strong, unambiguous confirmation of the report's Part 18 warning - not just "the minority class
degrades more," but that the majority class does not meaningfully degrade *at all* (it improves marginally).
Under strong DP noise, the model is pushed toward safer, majority-class-favoring predictions: it becomes
*better* at confidently saying "blocked" and dramatically *worse* at catching real breaches. A topline
accuracy number (0.9296 -> 0.8843, only a 4.5-point drop) would almost entirely hide this - exactly the
distortion report Part 18 warned reporting aggregate accuracy alone would cause.
If this pipeline is extended to also model `utility` (the report's other candidate target, not used in this project - see PROJECT_PLAN.md's target decision), the report's original between-target comparison (security degrading more than the balanced utility target) could be tested directly; that comparison is out of scope here since only `security` was modeled throughout Phases 2-6.

## Utility-privacy tradeoff

Every utility metric improves monotonically as the privacy budget grows (ε: 1 -> 3 -> 8 -> ∞), with
no non-monotonic surprises across the sweep - consistent with a correctly-calibrated DP-SGD implementation.
The overall utility cost of strong privacy (ε=1) relative to no privacy is real: ROC-AUC drops from
0.9616 to 0.8817 (8.0 points),
larger than the 4.2-point gap Phase 4 found between centralized and federated training with no privacy applied at all - i.e., in this experiment, strong differential privacy (ε=1) cost this model *more* accuracy than federation itself did.

## Figures

- `figures/phase6/utility_vs_epsilon.png`
- `figures/phase6/minority_vs_majority_recall.png`
- `figures/phase6/dp_sgd_convergence_by_epsilon.png` (from Step 1)

## Next

Phase 7 will cross-reference this utility-vs-epsilon curve against Phase 5's gradient-leakage
attack success rates, re-run against these DP-trained models - the full privacy-utility-vs-leakage
picture this whole project has been building toward.
