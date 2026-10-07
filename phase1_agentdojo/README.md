# AgentDojo Federated Security Research Pipeline

Federated Learning for Multi-Agent AI Security: Gradient Leakage Attack and Differential Privacy — built on the AgentDojo prompt-injection benchmark dataset.

**Status: All 7 phases complete.** Start with **[`evaluation/phase7/PHASE7_FINAL_REPORT.md`](evaluation/phase7/PHASE7_FINAL_REPORT.md)** — the capstone report synthesizing every phase into the project's central question: what does privacy cost, and what does it actually buy you? Headline finding: DP-SGD's formal guarantee does not, by itself, defend against post-hoc closed-form gradient inversion of a linear layer — exact reconstruction was achieved at every tested epsilon, including the strictest (ε=1). See `PIPELINE_PROGRESS.md` for the full completion record.

## Relationship to the rest of the repository

```
d:/MP2026-27/
├── agentdojo_dataset.csv                        # Raw AgentDojo benchmark export (untouched)
├── AgentDojo_Dataset_Research_Report.md          # Dataset suitability report (Part 1-20)
├── dataset_preparation/                          # Stage 1: raw CSV -> cleaned CSV (separate project)
│   └── outputs/dataset_cleaned_final.csv
├── PIPELINE_READINESS_REPORT.md                  # Stage 1->2 gate: independent audit, verdict Ready
└── agentdojo_research_pipeline/                  # Stage 2: THIS project - modeling onward
    └── data/dataset_cleaned_final.csv            # Read-only copy of the cleaned dataset
```

This project does not modify `dataset_cleaned_final.csv`. Every phase below reads it (or a prior phase's processed output) and writes new files under this project's own `outputs/`/`models/`/`figures/`.

## Documentation map

| File | Purpose |
|---|---|
| `PROJECT_PLAN.md` | Full 7-phase plan, hard rules carried from the readiness audit, environment findings |
| `PIPELINE_PROGRESS.md` | Live status table — which phase/step is done, in progress, pending |
| `CHANGELOG.md` | Chronological record of every change made in this project |
| `experiment_log.md` | One entry per experiment run — config, metrics, notes |
| `docs/` | Any additional per-phase writeups |

## Project structure

```
agentdojo_research_pipeline/
├── data/                 Read-only input (copy of the cleaned dataset)
├── src/phase1.../phase7/  One script per logical step, per phase
├── outputs/               Processed datasets, metrics tables, JSON results
├── figures/               Publication-quality plots
├── models/                Trained model checkpoints
├── evaluation/            Evaluation reports/artifacts
├── configs/                Run configuration files
├── notebooks/              Exploratory notebooks (not part of the reproducible pipeline)
└── docs/                    Extra documentation
```

## Modeling target

`security`, scoped to `security_label_status == "measured"` rows (`injection == True`) — the leakage-safe scoping established in `dataset_preparation/`. See `PROJECT_PLAN.md` for the full rationale and the hard rules that follow from it.

## A CSV-loading gotcha (found in Phase 1, Step 1 — read this before loading the CSV yourself)

`pd.read_csv()`'s default NA-sniffing treats an empty field as `NaN` regardless of why it's empty. `dataset_cleaned_final.csv` has 140 rows with a deliberately empty-string `final_answer` (encoding "no verbal answer", not "missing" — see `dataset_preparation`'s Phase 3). A naive `pd.read_csv(...)` reload silently turns those back into `NaN`, undoing that choice. Every script from Step 1 onward either loads `keep_default_na=False, na_values=[]` or reads the prior step's `.pkl` checkpoint (which has no such ambiguity). If you load the raw CSV yourself outside this pipeline, do the same.

## Running the pipeline

Each phase's scripts are numbered and meant to run in order, e.g.:

```bash
python src/phase1/01_load_data.py
python src/phase1/02_scope_security_target.py
python src/phase1/03_grouped_split.py
python src/phase1/04_feature_encoding.py
python src/phase1/05_validate_processed.py
```

See `PIPELINE_PROGRESS.md` for what's implemented so far.

## Working process

One logical step at a time, same discipline as `dataset_preparation/`: explain why → implement → validate → summarize → update logs → stop for review. See `PROJECT_PLAN.md`'s closing section.
