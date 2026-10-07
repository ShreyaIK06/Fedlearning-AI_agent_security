"""
Phase 3, Step 2 - Manual FedAvg training loop over the 4 suite-based clients.

Source:   PROJECT_PLAN.md Phase 3; AgentDojo_Dataset_Research_Report.md Part 16
Reason:   Trains the SAME `SecurityMLP` architecture as Phase 2's centralized
          MLP (imported from `src/common/mlp_model.py`, not redefined), so
          Phase 4's centralized-vs-federated comparison isolates the effect
          of *how* training happens (one model on pooled data vs. FedAvg
          over 4 non-IID clients), not a difference in what's being trained.

          Each communication round: broadcast the current global weights to
          all 4 clients, each trains locally for a few epochs on its own
          data only (never sees another client's rows - not even
          indirectly, since `train_local_model` in
          `src/common/fedavg_utils.py` takes only that one client's X/y),
          then the server averages the 4 updated weight sets, weighted by
          client size (standard FedAvg). The global model is evaluated each
          round on the same global val set Phase 2 used, for a directly
          comparable convergence curve.

          Global model starts from a fresh random init (same architecture,
          different seed context than Phase 2's centralized run) rather than
          the trained centralized weights - both Phase 2 and Phase 3 train
          from scratch, which is the fair comparison PROJECT_PLAN.md Phase 4
          calls for.

Input:    outputs/phase3/client_{suite}_X.pkl, client_{suite}_y.pkl
          outputs/phase1/04_X_val.pkl, 04_y_val.pkl
Output:   models/phase3/federated_mlp_state_dict.pt
          outputs/phase3/02_federated_training_history.csv
          outputs/phase3/02_federated_training_stats.json
          figures/phase3/federated_convergence.png
          figures/phase3/federated_communication_cost.png
          figures/phase3/federated_client_timing.png
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.mlp_model import SecurityMLP  # noqa: E402
from common.fedavg_utils import average_state_dicts, train_local_model  # noqa: E402

PHASE1_DIR = ROOT / "outputs" / "phase1"
PHASE3_DIR = ROOT / "outputs" / "phase3"
MODEL_OUT = ROOT / "models" / "phase3" / "federated_mlp_state_dict.pt"
HISTORY_CSV = PHASE3_DIR / "02_federated_training_history.csv"
STATS_JSON = PHASE3_DIR / "02_federated_training_stats.json"
FIG_DIR = ROOT / "figures" / "phase3"

SUITES = ["banking", "slack", "travel", "workspace"]
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
NUM_ROUNDS = 150
LOCAL_EPOCHS = 3
LOCAL_LR = 1e-3
LOCAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10
RANDOM_STATE = 42
BYTES_PER_PARAM = 4  # float32


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    clients = {}
    for suite in SUITES:
        X = pd.read_pickle(PHASE3_DIR / f"client_{suite}_X.pkl").to_numpy(dtype=np.float32)
        y = pd.read_pickle(PHASE3_DIR / f"client_{suite}_y.pkl").to_numpy(dtype=np.float32)
        clients[suite] = (X, y)

    input_dim = next(iter(clients.values()))[0].shape[1]

    X_val = pd.read_pickle(PHASE1_DIR / "04_X_val.pkl")
    y_val = pd.read_pickle(PHASE1_DIR / "04_y_val.pkl")
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32))
    y_val_np = y_val.to_numpy()

    global_model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    global_state = global_model.state_dict()
    n_params = sum(p.numel() for p in global_model.parameters())

    history = []
    best_val_roc_auc = -1.0
    best_state = None
    best_round = 0
    rounds_without_improvement = 0
    cumulative_bytes = 0

    for rnd in range(1, NUM_ROUNDS + 1):
        round_start = time.perf_counter()
        client_states, client_sizes, client_times, client_losses = [], [], {}, {}

        for suite in SUITES:
            X, y = clients[suite]
            state, elapsed, local_loss = train_local_model(
                global_state=global_state,
                X=X, y=y,
                input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                local_epochs=LOCAL_EPOCHS, lr=LOCAL_LR, batch_size=LOCAL_BATCH_SIZE,
                seed=RANDOM_STATE + rnd,
            )
            client_states.append(state)
            client_sizes.append(len(X))
            client_times[suite] = round(elapsed, 4)
            client_losses[suite] = round(local_loss, 4)

        global_state = average_state_dicts(client_states, [float(s) for s in client_sizes])
        global_model.load_state_dict(global_state)

        # communication: each client downloads the global model and uploads
        # its local update once per round
        cumulative_bytes += n_params * BYTES_PER_PARAM * 2 * len(SUITES)

        global_model.eval()
        with torch.no_grad():
            val_logits = global_model(X_val_t)
            val_proba = torch.sigmoid(val_logits).numpy()
        val_roc_auc = roc_auc_score(y_val_np, val_proba)
        round_time = time.perf_counter() - round_start

        history.append({
            "round": rnd,
            "val_roc_auc": val_roc_auc,
            "round_wall_time_sec": round(round_time, 4),
            "cumulative_bytes_transmitted": cumulative_bytes,
            **{f"client_{s}_local_loss": client_losses[s] for s in SUITES},
            **{f"client_{s}_local_time_sec": client_times[s] for s in SUITES},
        })

        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_state = {k: v.clone() for k, v in global_state.items()}
            best_round = rnd
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rnd % 5 == 0 or rnd == 1:
            print(f"round {rnd:3d}  val_roc_auc={val_roc_auc:.4f}  round_time={round_time:.2f}s")

        if rounds_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at round {rnd} (no val_roc_auc improvement for {EARLY_STOP_PATIENCE} rounds)")
            break

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)

    history_df = pd.DataFrame(history)
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(HISTORY_CSV, index=False)

    stats = {
        "step": "phase3_02_federated_training",
        "n_clients": len(SUITES),
        "n_params": int(n_params),
        "communication_rounds_run": len(history),
        "best_round": best_round,
        "best_val_roc_auc": round(best_val_roc_auc, 4),
        "total_bytes_transmitted": int(cumulative_bytes),
        "total_bytes_transmitted_mb": round(cumulative_bytes / 1e6, 2),
        "config": {
            "hidden_dims": list(HIDDEN_DIMS), "dropout": DROPOUT,
            "num_rounds_max": NUM_ROUNDS, "local_epochs": LOCAL_EPOCHS,
            "local_lr": LOCAL_LR, "local_batch_size": LOCAL_BATCH_SIZE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history_df["round"], history_df["val_roc_auc"], marker="o", markersize=3)
    ax.axvline(best_round, color="gray", linestyle="--", alpha=0.6, label=f"best round ({best_round})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global model val ROC-AUC")
    ax.set_title("Federated Learning Convergence (FedAvg, 4 suite clients)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "federated_convergence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history_df["round"], history_df["cumulative_bytes_transmitted"] / 1e6)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative data transmitted (MB)")
    ax.set_title("Federated Learning Communication Cost")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "federated_communication_cost.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for suite in SUITES:
        ax.plot(history_df["round"], history_df[f"client_{suite}_local_time_sec"], label=suite)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Local training time (sec)")
    ax.set_title("Per-Client Local Training Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "federated_client_timing.png", dpi=150)
    plt.close(fig)

    print(f"\nBest round: {best_round}  val_roc_auc={best_val_roc_auc:.4f}")
    print(f"Total communication: {cumulative_bytes/1e6:.2f} MB over {len(history)} rounds")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved history: {HISTORY_CSV}")
    print(f"Saved stats: {STATS_JSON}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
