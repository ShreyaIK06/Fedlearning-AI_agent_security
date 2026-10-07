# Project Plan — AgentDojo Federated Security Research Pipeline

**Status: ✅ Complete — all 7 phases finished 2026-08-07.** See `evaluation/phase7/PHASE7_FINAL_REPORT.md` for the capstone report and `PIPELINE_PROGRESS.md` for the full step-by-step completion record. Note: `PROJECT_PLAN.md` note #3 (dropping/merging the two rare `pipeline` values before FL client partitioning) was not applied in the end — Phase 3 partitioned by `suite`, not `pipeline`/`base_model`, so that specific rule never came into play; left here unmodified as the plan that was made in advance, not edited retroactively to match what happened.

**Research question:** Federated Learning for Multi-Agent AI Security: Gradient Leakage Attack and Differential Privacy.

**Input data:** `data/dataset_cleaned_final.csv` (36,475 rows × 20 columns), produced and independently validated in the sibling `dataset_preparation/` project. See `../PIPELINE_READINESS_REPORT.md` for the full readiness audit (verdict: ✅ Ready, 2026-08-07).

This project is deliberately kept separate from `dataset_preparation/` — that project's job was producing a clean dataset; this project's job is everything downstream of it (feature engineering for modeling, training, attacking, defending, evaluating). `dataset_cleaned_final.csv` is treated as read-only input here, same as the original raw CSV was treated as read-only input there.

---

## Hard rules carried forward from the readiness audit (§4/§5 of `PIPELINE_READINESS_REPORT.md`)

These are non-negotiable constraints on every phase below, not suggestions:

1. **Never row-random split.** Every train/test or train/val/test split must be grouped on `user_prompt` (or `(suite, task)` where finer-grained grouping isn't needed), enforced with `GroupShuffleSplit`/`GroupKFold`, with an assertion that no `user_prompt` value appears on both sides of a split.
2. **`security` modeling must filter to `security_label_status == "measured"`** before any fit/predict call — asserted in code, not just commented.
3. **Two rare `pipeline`/`base_model` values must be dropped or merged before FL client partitioning**: `claude-3-sonnet-20240229-repeat_user_prompt` (40 rows) and `llama-3.3-70b-versatile` (4 rows).
4. **Report the label-noise ceiling** (26.4% `security` / 54.4% `utility` flip rate on exact repeats) alongside every headline accuracy number — don't overclaim.
5. **`final_answer`-derived features, if used, must exclude/ablate outcome-restating keywords** ("successfully", "I cannot", etc.) separately from the main result.

## Environment findings (recorded 2026-08-07, during project setup)

- `flwr` (Flower) is installed but its import chain is broken in this environment: `flwr.server` imports `tensorboard` → `tensorflow` → `ml_dtypes`, which raises `AttributeError: _ARRAY_API not found` under `numpy>=2.0`. Not fixed (would mean downgrading numpy/tensorflow environment-wide, outside this project's scope, and risky to other work in the repo).
- **Decision: Phase 3 (Federated Learning) is implemented as a manual FedAvg loop in PyTorch**, not via the Flower framework. This sidesteps the broken dependency and — more importantly — gives Phase 5 (gradient leakage) direct access to each client's per-round gradients, which a framework's Strategy/Client abstraction would otherwise hide.
- Verified working: `scikit-learn` 1.7.0, `xgboost` 3.2.0, `torch` 2.13.0 (CPU), `opacus` 1.5.4 (DP-SGD), `matplotlib`/`seaborn` for plotting.

---

## Phase-by-Phase Plan

| Phase | Name | Target(s) | Key outputs |
|---|---|---|---|
| 1 | Modeling-Ready Data Preparation | — | Encoded feature matrices, grouped train/test splits, scalers/encoders (fit on train only) |
| 2 | Centralized Machine Learning | `security` (scoped to `measured` rows) | LR, RF, XGBoost, MLP baselines; metrics, confusion matrices, ROC/PR curves |
| 3 | Federated Learning | Same as Phase 2 | FedAvg over `suite`-based clients (primary) and `base_model`-based clients (secondary), convergence curves |
| 4 | Centralized vs. Federated Comparison | — | Comparison tables/plots, communication cost, discussion |
| 5 | Gradient Leakage Attack | — | DLG/iDLG/Inverting-Gradients reconstructions against the Phase 3 FL model, qualitative + quantitative leakage analysis |
| 6 | Differential Privacy | Same as Phase 2/3 | DP-SGD (Opacus) applied to centralized and/or federated training, multiple ε budgets |
| 7 | Privacy-Utility Tradeoff | — | ε-sweep tables/plots: accuracy/F1/leakage-success vs. ε |

Modeling target decision (confirmed before this phase started): **`security`, correctly scoped to `security_label_status == "measured"`** (rule 2 above). `utility` was considered but not selected for this pipeline's primary target — it remains available in the data if a future phase wants a secondary/comparison target, but nothing here should silently default to it.

### Phase 1 — Modeling-Ready Data Preparation
Steps (each its own script under `src/phase1/`, run in order):
1. `01_load_data.py` — load `data/dataset_cleaned_final.csv`, re-assert the invariants the readiness report depends on (0 missing values, `security_label_status` mapping correct, etc.) before anything downstream trusts the file.
2. `02_scope_security_target.py` — filter to `security_label_status == "measured"`, producing the modeling population for Phases 2-7.
3. `03_grouped_split.py` — `GroupShuffleSplit` on `user_prompt` → train/test (and train/val split for model selection), with a leakage assertion.
4. `04_feature_encoding.py` — encode categorical features (`suite`, `base_model`, `defense`, `task`, `attack_type`, `injection_task`), engineer `is_dos_attack` (deferred from dataset-prep, done here as this is exactly modeling-time feature engineering), encode `tool_sequence` (bag-of-tools), select numeric features (dropping the redundant raw `duration_seconds` in favor of `duration_seconds_log`, keeping `num_messages`/`assistant_turns`/`tool_calls` despite their collinearity per the report's tree-model guidance), scale numeric features (fit on train only).
5. `05_validate_processed.py` — shape/leakage/distribution checks on the final processed train/test matrices before Phase 2 touches them.

### Phase 2 — Centralized Machine Learning
Logistic Regression, Random Forest, XGBoost (classical) + a small MLP (PyTorch, reused unmodified in Phase 3 for a fair comparison). Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, full classification report — always reported alongside the 26.4% label-noise ceiling.

### Phase 3 — Federated Learning
Manual FedAvg (see Environment Findings). Primary partitioning: `suite` (4 clients — banking/slack/travel/workspace). Secondary: `base_model` (after pruning the 2 rare values per rule 3). Same MLP architecture as Phase 2's, so centralized-vs-federated (Phase 4) is an apples-to-apples comparison.

### Phase 4 — Comparison
Directly built from Phase 2 + Phase 3 outputs — no new training.

### Phase 5 — Gradient Leakage
Attack the Phase 3 FL model's shared gradients directly (this is why Phase 3 is manual PyTorch, not a black-box framework). Start on the compact categorical/numeric feature set; the report (Part 17) expects near-trivial reconstruction there given the low dimensionality — that's a documented, expected finding, not a bug. Optionally enrich with `user_prompt` embeddings for a harder, more realistic attack surface, as a clearly separate ablation.

### Phase 6 — Differential Privacy
Opacus DP-SGD on the Phase 2 MLP and/or the Phase 3 FL clients. Sweep noise multiplier / clipping norm to hit target ε ∈ {1, 3, 8, ∞} per the original report's Part 18 starting recommendation.

### Phase 7 — Privacy-Utility Tradeoff
ε-sweep results from Phase 6 cross-referenced against Phase 5's leakage-success metric — the core "what does privacy cost you, and what does it buy you" result the whole pipeline builds toward.

---

## Working process for this project

Same discipline as `dataset_preparation/`: one logical step at a time. After each step — explain why, implement, validate, summarize, update `CHANGELOG.md`/`PIPELINE_PROGRESS.md`/`experiment_log.md` — then stop for review before the next step. No multi-phase batching.
