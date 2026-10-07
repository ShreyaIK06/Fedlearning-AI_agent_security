# Phase 2 - Centralized ML Model Comparison Report

Test set: 7258 rows, 14.08% security=True (held out since Phase 1 Step 3, touched here for the first and only time).

## Results (test set)

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.9296 | 0.7145 | 0.8327 | 0.7691 | 0.9616 | 0.8445 |
| xgboost | 0.9208 | 0.656 | 0.9198 | 0.7658 | 0.9694 | 0.8505 |
| random_forest | 0.926 | 0.6945 | 0.8474 | 0.7633 | 0.9564 | 0.8351 |
| logistic_regression | 0.8575 | 0.4964 | 0.8033 | 0.6136 | 0.9133 | 0.6468 |

## Label-noise ceiling

The `security` label flips on 26.4% of exact-repeat configurations (report Part 4, re-verified in PIPELINE_READINESS_REPORT.md) - irreducible run-to-run stochasticity in the original AgentDojo benchmark traces, not model error. None of the four models' test accuracy approaches a level that would be suspicious against this ceiling.

## Best model by F1

**mlp** (F1=0.7691, precision=0.7145, recall=0.8327).

## Best model by ROC-AUC

**xgboost** (ROC-AUC=0.9694).

## Carried forward to Phase 3/4

The **MLP**'s test performance here is the centralized reference Phase 4 will compare Phase 3's federated MLP against - same architecture (`src/common/mlp_model.py`), same test set. The three classical models are a benchmarking reference, not carried into the FL comparison (FedAvg requires a differentiable model with shared weight structure across clients; LR/RF/XGBoost don't fit that framework the way an MLP does).

## Figures

- `figures/phase2/roc_curve_comparison.png`
- `figures/phase2/pr_curve_comparison.png`
- `figures/phase2/metrics_comparison_bar_chart.png`