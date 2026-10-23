# Federated Learning for Multi-Agent AI Security: A Gradient Leakage and Mitigation Study

Federated learning (FL) lets organisations jointly train a model that detects prompt-injection attacks on AI agents, without sharing their agent logs. This project tests whether those logs actually stay private. We train breach-detection models with FedAvg on agent interaction logs, attack the shared model updates with gradient-inversion attacks, and evaluate common defences.

**In short:** a single shared gradient can reveal a client's training record exactly. When the record is a text embedding, an attacker who can guess the surrounding text can then recover secrets in it, such as account numbers and email addresses; without that knowledge, recovery mostly fails. A malicious server that rigs the model can still rebuild many records from large batches. Where differential privacy is applied decides whether it helps: DP-SGD used only during training does not protect the updates clients send afterwards, while DP applied inside each client (DP-FedAvg) stopped every attack we tested, including the malicious server, at an accuracy cost that depends strongly on how much data each client holds.

## Project structure

The project has two phases that share the same model, FedAvg implementation and attack code.

- **Phase 1 — AgentDojo** (`phase1_agentdojo/`): a 7-stage pipeline on the [AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark results: feature encoding, centralized baselines, federated training over 4 suite-based clients, gradient-inversion attacks (DLG, iDLG, Inverting Gradients, closed-form) and DP-SGD with Opacus.
- **Phase 2 — Synthetic agent logs** (`phase2_synthetic/`): a new dataset of 576 LLM-agent runs (LangChain + Groq) across 4 suites with indirect prompt injections and full message trajectories. It replicates Phase 1's experiments, then tests content leakage from sentence embeddings of the trajectories and further defences (batching, secure aggregation, attacks on FedAvg updates, and DP-FedAvg on both datasets).

```
fed-agent-leakage/
├── phase1_agentdojo/
│   └── src/
│       ├── common/          shared code: SecurityMLP, FedAvg, gradient-leakage attacks
│       └── phase1 … phase7/ Phase 1 pipeline stages
├── phase2_synthetic/        dataset generation (runner.py, suites/) and Phase 2 experiments (phase2_*.py)
├── data/                    raw/ (Phase 2 agent runs, one JSON each) and processed/
├── models/  evaluation/  outputs/  figures/
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install torch opacus scikit-learn xgboost pandas numpy scipy matplotlib joblib sentence-transformers
pip install langchain langchain-groq   # only to regenerate the Phase 2 dataset
```

Regenerating the dataset needs a Groq API key in `phase2_synthetic/.env` (`GROQ_API_KEY=...`, not committed). Training and attacks run locally without it.

## Running

**Phase 1:** see `phase1_agentdojo/PIPELINE_PROGRESS.md`. Requires the AgentDojo results CSV (`dataset_cleaned_final.csv`) in `phase1_agentdojo/data/` (not included). The processed splits used in our results are in `phase1_agentdojo/outputs/phase1/`. Note: re-running `src/phase1/03_grouped_split.py` under pandas 3.x produces a *different* split; to reproduce the original one, run stages `01` and `02`, then `python phase2_synthetic/rebuild_phase1_train.py`, then stages `04` and `05`.

**Phase 2 dataset** (optional; the runs are already in `data/raw/`). From `phase2_synthetic/`:

```bash
python runner.py banking        # or slack, workspace, travel
python validate_all.py
python summary.py banking
```

**Phase 2 experiments.** From `phase2_synthetic/`, in order:

```bash
python phase2_split.py && python phase2_features.py
python phase2_train_central.py && python phase2_train_fed.py && python phase2_seeds.py
python phase2_attacks.py
python phase2_dp.py --lr 1e-2 --batch 64 --epochs 60 --baseline-tuned --delta 1e-5
python phase2_embed.py && python phase2_p2_train.py && python phase2_p2_attacks.py && python phase2_p2_dp.py
python phase2_aggregation.py
python phase2_fedavg_attack.py
python phase2_dpfedavg.py                    # DP-FedAvg (DP inside each client), Phase 2
python phase1_dpfedavg.py                    # DP-FedAvg on Phase 1's AgentDojo data
python phase2_robbing_fed.py                 # malicious server ("Robbing the Fed"); add --agentdojo for Phase 1 data
python phase2_realistic_attacker.py          # secret extraction when the attacker does not know the victim's text
```

Results are written to `evaluation/` and `outputs/` (Phase 1 DP-FedAvg: `phase1_agentdojo/evaluation/dpfedavg/`).

## Limitations

- Phase 2 is small (563 usable runs), generated with a single LLM on synthetic tasks.
- Most attacks assume an honest-but-curious server (one experiment tests a malicious one). Secret-extraction numbers depend heavily on how much of the victim's text the attacker already knows; the headline figures are upper bounds.
- The models detect breaches in completed runs; they do not predict them in advance.