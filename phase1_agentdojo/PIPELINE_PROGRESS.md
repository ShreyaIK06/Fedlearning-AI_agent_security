# Pipeline Progress

Live status. Updated after every step.

| Phase | Step | Status | Date |
|---|---|---|---|
| 0 | Validate cleaned dataset against report (`PIPELINE_READINESS_REPORT.md`) | ✅ Done — verdict: Ready | 2026-08-07 |
| 0 | Scaffold `agentdojo_research_pipeline/` project | ✅ Done | 2026-08-07 |
| 1 | `01_load_data.py` | ✅ Done | 2026-08-07 |
| 1 | `02_scope_security_target.py` | ✅ Done | 2026-08-07 |
| 1 | `03_grouped_split.py` | ✅ Done | 2026-08-07 |
| 1 | `04_feature_encoding.py` | ✅ Done | 2026-08-07 |
| 1 | `05_validate_processed.py` | ✅ Done | 2026-08-07 |
| 2 | Logistic Regression | ✅ Done | 2026-08-07 |
| 2 | Random Forest | ✅ Done | 2026-08-07 |
| 2 | XGBoost | ✅ Done | 2026-08-07 |
| 2 | MLP (PyTorch) | ✅ Done | 2026-08-07 |
| 2 | Test-set evaluation + comparison | ✅ Done | 2026-08-07 |
| 3 | Partition clients by suite | ✅ Done | 2026-08-07 |
| 3 | Manual FedAvg training loop | ✅ Done | 2026-08-07 |
| 3 | Evaluate federated model on test set | ✅ Done | 2026-08-07 |
| 4 | Centralized vs. Federated comparison | ✅ Done | 2026-08-07 |
| 5 | Select victims + true gradients | ✅ Done | 2026-08-07 |
| 5 | DLG attack | ✅ Done | 2026-08-07 |
| 5 | iDLG attack | ✅ Done | 2026-08-07 |
| 5 | Inverting Gradients attack | ✅ Done | 2026-08-07 |
| 5 | Comparison report | ✅ Done | 2026-08-07 |
| 6 | DP-SGD training sweep (ε=1,3,8) | ✅ Done | 2026-08-07 |
| 6 | Test-set evaluation + per-class impact | ✅ Done | 2026-08-07 |
| 7 | Leakage vs. epsilon sweep (+ major closed-form finding) | ✅ Done | 2026-08-07 |
| 7 | Final tradeoff report | ✅ Done | 2026-08-07 |

## Legend
✅ Done · 🔄 In progress · ⏳ Pending · ❌ Blocked

## Current blocker
None. **ALL 7 PHASES COMPLETE.** Project finished — see `evaluation/phase7/PHASE7_FINAL_REPORT.md` for the capstone report.
