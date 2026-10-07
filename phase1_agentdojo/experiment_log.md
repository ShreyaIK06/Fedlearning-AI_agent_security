# Experiment Log

One entry per experiment run. Append-only. Each entry should have enough detail (config + metrics + environment) to be independently reproduced.

## Entry template

```
### [Phase N] <experiment name> — YYYY-MM-DD

**Script:** src/phaseN/script.py
**Config:** (key parameters, or path to a configs/*.yaml file)
**Data:** (which processed dataset version / split)
**Metrics:**
**Artifacts:** (model checkpoint / figure / results file paths)
**Notes:**
```

---

### [Phase 2] Centralized baselines (LR, RF, XGBoost, MLP) — 2026-08-07

**Scripts:** `src/phase2/01_train_logistic_regression.py`, `02_train_random_forest.py`, `03_train_xgboost.py`, `04_train_mlp.py`, `05_evaluate_test_set.py`
**Config:** `class_weight="balanced"` (LR/RF), `scale_pos_weight` derived from train data (XGBoost), `pos_weight` derived from train data (MLP, `BCEWithLogitsLoss`). MLP: hidden dims (128,64), dropout 0.2, Adam lr=1e-3, early stop on val ROC-AUC.
**Data:** `outputs/phase1/04_X_{train,val,test}.pkl` (176 features, `security` target scoped to `measured` rows)
**Metrics (test):** MLP best F1 (0.7691); XGBoost best ROC-AUC/PR-AUC/recall (0.9694/0.8505/0.9198). Full table: `evaluation/phase2/phase2_model_comparison.csv`
**Artifacts:** `models/phase2/*.joblib`, `models/phase2/mlp_state_dict.pt`, `evaluation/phase2/*_test_metrics.json`
**Notes:** XGBoost run required a Phase 1 amendment (feature name containing `<`/`>` rejected by XGBoost's DMatrix) — fixed at the source, Phase 1 Steps 4-5 and Phase 2 Steps 1-2 re-run for consistency.

### [Phase 3] Federated learning (manual FedAvg, 4 suite clients) — 2026-08-07

**Script:** `src/phase3/02_federated_training.py`
**Config:** `SecurityMLP` (same architecture as Phase 2), 4 clients (banking/slack/travel/workspace), 3 local epochs/round, Adam lr=1e-3, size-weighted FedAvg aggregation, early stop on val ROC-AUC (patience 10).
**Data:** Phase 1 train split partitioned by `suite` (non-IID; breach rate 7.7%-39.7% across clients)
**Metrics:** Converged at round 73 (best round 63), val ROC-AUC 0.9235; test ROC-AUC 0.9201 vs. centralized MLP's 0.9616.
**Artifacts:** `models/phase3/federated_mlp_state_dict.pt`, `outputs/phase3/02_federated_training_history.csv`
**Notes:** First run (40-round budget) hadn't converged; re-run with 150-round budget so early stopping could trigger naturally before comparing against the centralized baseline.

### [Phase 5] Gradient leakage attacks (DLG, iDLG, Inverting Gradients) — 2026-08-07

**Scripts:** `src/phase5/02_dlg_attack.py`, `03_idlg_attack.py`, `04_inverting_gradients_attack.py`, `05_compare_attacks_and_report.py`
**Config:** 12 victims (3/suite) attacked against the trained federated model (single-example gradients, `.eval()` mode). DLG/iDLG: LBFGS, L2 loss, 150 iters. IG: Adam (lr=0.05), cosine loss, 500 iters, best-iterate tracking.
**Metrics:** Reconstruction MSE — DLG 86.67, iDLG 0.64, IG 0.84. No single attack dominates every metric (see `evaluation/phase5/attack_comparison.csv`).
**Artifacts:** `outputs/phase5/{02,03,04}_*_results.json`, `evaluation/phase5/phase5_comparison_report.md`
**Notes:** Five real bugs found and fixed in this experiment (numerical divergence, float32 label-inference saturation, LBFGS/clamp state corruption, non-monotonic optimum, stale loss logging) — full diagnostic trail in `CHANGELOG.md`.

### [Phase 6] DP-SGD training sweep (ε=1,3,8) — 2026-08-07

**Script:** `src/phase6/01_train_dp_sgd_sweep.py`
**Config:** Opacus `PrivacyEngine.make_private_with_epsilon`, `max_grad_norm=1.0`, `target_delta=1/19052`, 30 epochs, batch size 256.
**Metrics:** Achieved ε closely matched target in all 3 runs (0.995/2.994/7.993). Test ROC-AUC: 0.8817 (ε=1) → 0.8967 (ε=8) vs. 0.9616 non-private. Minority-class recall collapses 83.3%→47.5% (ε=∞→ε=1) while majority specificity barely moves.
**Artifacts:** `models/phase6/dp_mlp_eps{1,3,8}_state_dict.pt`, `evaluation/phase6/dp_test_comparison.csv`

### [Phase 7] Leakage vs. epsilon + closed-form attack discovery — 2026-08-07

**Script:** `src/phase7/01_leakage_vs_epsilon.py`
**Config:** Same 12 Phase-5 victims, gradients recomputed fresh against Phase 2's centralized MLP (ε=∞) and Phase 6's 3 DP models. Both Inverting Gradients (optimization) and a newly-derived closed-form first-layer identity attack run per victim per model.
**Metrics:** IG showed a misleading pattern (100% cat. accuracy at ε=1 vs. 61% at ε=∞, driven by DP models' smaller weight norms easing the optimizer's job). **Closed-form attack: exact reconstruction (MSE=0.000000, 100% categorical accuracy) at every epsilon, including ε=1.**
**Artifacts:** `outputs/phase7/01_leakage_vs_epsilon_results.json`, `evaluation/phase7/PHASE7_FINAL_REPORT.md`, `figures/phase7/master_tradeoff_figure.png`
**Notes:** The project's central finding — DP-SGD's formal guarantee does not defend against post-hoc closed-form gradient inversion of a linear layer, at any tested epsilon. Full mechanism and derivation in `src/common/gradient_leakage.py`'s `closed_form_first_layer_attack`.
