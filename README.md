# Federated Learning for Multi-Agent AI Security: A Gradient Leakage and Mitigation Study

Federated learning (FL) lets organisations jointly train a model that detects prompt-injection attacks on AI agents, without sharing their agent logs. This project tests whether those logs actually stay private. We train breach-detection models with FedAvg on agent interaction logs, attack the shared model updates with gradient-inversion attacks, and evaluate common defences.

**In short:** a single shared gradient can reveal a client's training record exactly, including sensitive content such as account numbers and email addresses. DP-SGD did not prevent this in our experiments; multi-step local training (standard FedAvg) was the setting that held up best.

**Team:** Shreya, Niharika, Jagath · **Guide:** Ms. Madhushree, Department of Cybersecurity, NMAMIT

## Project structure

The project has two phases that share the same model, FedAvg implementation and attack code.

- **Phase 1 — AgentDojo** (`phase1_agentdojo/`): a 7-stage pipeline on the [AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark results: feature encoding, centralized baselines, federated training over 4 suite-based clients, gradient-inversion attacks (DLG, iDLG, Inverting Gradients, closed-form) and DP-SGD with Opacus.
- **Phase 2 — Synthetic agent logs** (`phase2_synthetic/`): a new dataset of 576 LLM-agent runs (LangChain + Groq) across 4 suites with indirect prompt injections and full message trajectories. It replicates Phase 1's experiments, then tests content leakage from sentence embeddings of the trajectories and further defences (batching, secure aggregation, attacks on FedAvg updates).

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

**Phase 1:** see `phase1_agentdojo/PIPELINE_PROGRESS.md`. Requires the AgentDojo results CSV in `phase1_agentdojo/data/` (not included).

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
python phase2_dp.py --lr 1e-2 --batch 64 --epochs 60 --baseline-tuned
python phase2_embed.py && python phase2_p2_train.py && python phase2_p2_attacks.py && python phase2_p2_dp.py
python phase2_aggregation.py
python phase2_fedavg_attack.py
```

Results are written to `evaluation/` and `outputs/`.

## Limitations

- Phase 2 is small (563 usable runs), generated with a single LLM on synthetic tasks.
- Attacks assume an honest-but-curious server and, in places, a well-informed attacker; results are upper bounds.
- The models detect breaches in completed runs; they do not predict them in advance.