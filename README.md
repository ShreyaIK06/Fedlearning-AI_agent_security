# fed-agent-leakage

Gradient leakage attacks and DP-SGD defenses in federated learning on AI agent interaction logs (AgentDojo + LangChain synthetic data).

## Phase 2: Synthetic agent logs

Generate a suite's records:

    cd phase2_synthetic
    python runner.py banking    # or slack, workspace, travel

Validate and summarize:

    python validate_all.py
    python summary.py <suite>

Requires a `.env` file with `GROQ_API_KEY=...` (not committed).

### Dataset
576 records across 4 suites (banking, slack, workspace, travel), labelled for
security breach and task completion. See `data/raw/`.