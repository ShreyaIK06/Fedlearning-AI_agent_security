"""
phase2_train_fed.py  -  Phase 2, FedAvg over the 4 suite clients
--------------------------------------------------------------------
Mirrors Phase 1's src/phase3/01_partition_clients.py + 02_federated_train.py:
  * clients = TRAIN split partitioned by suite; val/test stay global
  * each round: broadcast global weights -> each client trains locally
    (Phase 1's train_local_model, unchanged) -> size-weighted FedAvg
    (Phase 1's average_state_dicts, unchanged)
  * up to 150 rounds, 3 local epochs, lr 1e-3, local batch 64,
    early stop after 10 rounds w/o global val ROC-AUC gain, seed 42 (+round)
  * best round's weights are kept

Then evaluates on TEST and, if phase2_train_central.py has been run,
prints the centralized-vs-federated "cost of federation".

Run from phase2_synthetic/:   python phase2_train_fed.py
"""

import json
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from phase2_common import (EVAL_DIR, FIG_DIR, MODEL_DIR, OUT_DIR, RANDOM_STATE, SUITES,
                           SecurityMLP, average_state_dicts, label_noise_ceiling,
                           load_features, metrics_for, per_suite_recall, print_summary,
                           save_metrics, train_local_model)

matplotlib.use("Agg")

# ---- identical to Phase 1's 02_federated_train.py
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
NUM_ROUNDS = 150
LOCAL_EPOCHS = 3
LOCAL_LR = 1e-3
LOCAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10
BYTES_PER_PARAM = 4


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    d = load_features()

    # ---- partition TRAIN by suite (Phase 1 step 01)
    clients, stats = {}, {}
    for s in SUITES:
        m = d["suite_train"] == s
        assert m.sum() > 0, f"suite '{s}' has zero training rows - cannot form a client"
        clients[s] = (d["X_train"][m], d["y_train"][m])
        stats[s] = {"n_rows": int(m.sum()), "pct_of_train": round(100 * m.mean(), 2),
                    "breach_rate": round(float(d["y_train"][m].mean()), 4)}
    assert sum(v["n_rows"] for v in stats.values()) == len(d["X_train"]), "every train row must be in exactly one client"
    print("Clients (train split by suite):")
    for s, v in stats.items():
        print(f"  {s:10s} n={v['n_rows']:4d} ({v['pct_of_train']}%)  breach_rate={v['breach_rate']}")

    input_dim = d["X_train"].shape[1]
    X_va = torch.tensor(d["X_val"]); y_va = d["y_val"]
    X_te = torch.tensor(d["X_test"]); y_te = d["y_test"]

    gmodel = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    gstate = gmodel.state_dict()
    n_params = sum(p.numel() for p in gmodel.parameters())

    history, best_auc, best_state, best_round, wait, cum_bytes = [], -1.0, None, 0, 0, 0
    print()
    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.perf_counter()
        states, sizes, times, losses = [], [], {}, {}
        for s in SUITES:
            X, y = clients[s]
            st, el, ll = train_local_model(global_state=gstate, X=X, y=y, input_dim=input_dim,
                                           hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                                           local_epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
                                           batch_size=LOCAL_BATCH_SIZE, seed=RANDOM_STATE + rnd)
            states.append(st); sizes.append(float(len(X)))
            times[s] = round(el, 4); losses[s] = round(ll, 4)

        gstate = average_state_dicts(states, sizes)
        gmodel.load_state_dict(gstate)
        cum_bytes += n_params * BYTES_PER_PARAM * 2 * len(SUITES)

        gmodel.eval()
        with torch.no_grad():
            val_auc = roc_auc_score(y_va, torch.sigmoid(gmodel(X_va)).numpy())
        history.append({"round": rnd, "val_roc_auc": val_auc,
                        "round_wall_time_sec": round(time.perf_counter() - t0, 4),
                        "cumulative_bytes_transmitted": cum_bytes,
                        **{f"client_{s}_local_loss": losses[s] for s in SUITES},
                        **{f"client_{s}_local_time_sec": times[s] for s in SUITES}})

        if val_auc > best_auc:
            best_auc, best_round, wait = val_auc, rnd, 0
            best_state = {k: v.clone() for k, v in gstate.items()}
        else:
            wait += 1
        if rnd % 5 == 0 or rnd == 1:
            print(f"round {rnd:3d}  val_roc_auc={val_auc:.4f}")
        if wait >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at round {rnd} (no val_roc_auc improvement for {EARLY_STOP_PATIENCE} rounds)")
            break

    # ---- evaluate best global model
    gmodel.load_state_dict(best_state); gmodel.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(gmodel(X_va)).numpy()
        p_te = torch.sigmoid(gmodel(X_te)).numpy()

    noise = label_noise_ceiling()
    cfg = {"hidden_dims": list(HIDDEN_DIMS), "dropout": DROPOUT, "num_rounds_max": NUM_ROUNDS,
           "local_epochs": LOCAL_EPOCHS, "local_lr": LOCAL_LR, "local_batch_size": LOCAL_BATCH_SIZE,
           "early_stop_patience": EARLY_STOP_PATIENCE, "best_round": best_round,
           "rounds_run": len(history)}
    print()
    fed = {}
    for split, y, p in [("val", y_va, p_va), ("test", y_te, p_te)]:
        m = metrics_for(y, p, "federated_mlp", split, noise); m["architecture"] = cfg
        print_summary(m)
        save_metrics(m, EVAL_DIR / f"federated_mlp_{split}_metrics.json")
        fed[split] = m

    ps = per_suite_recall(y_te, p_te, d["suite_test"])
    print("\nPer-suite TEST (indicative only - small n):")
    print(ps.to_string(index=False))

    # ---- save
    MODEL_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / "federated_mlp_state_dict.pt")
    hdf = pd.DataFrame(history); hdf.to_csv(OUT_DIR / "federated_training_history.csv", index=False)
    ps.to_csv(OUT_DIR / "federated_per_suite_test.csv", index=False)
    (OUT_DIR / "federated_training_stats.json").write_text(json.dumps({
        "n_clients": len(SUITES), "client_stats": stats, "n_params": int(n_params),
        "best_round": best_round, "best_val_roc_auc": round(best_auc, 4),
        "total_bytes_transmitted_mb": round(cum_bytes / 1e6, 2), "config": cfg}, indent=2), encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hdf["round"], hdf["val_roc_auc"], marker="o", markersize=3)
    ax.axvline(best_round, color="gray", ls="--", alpha=.6, label=f"best round ({best_round})")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Global model val ROC-AUC")
    ax.set_title("Phase 2 (synthetic) - FedAvg convergence, 4 suite clients"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "federated_convergence.png", dpi=150); plt.close(fig)

    # ---- cost of federation (Phase 1: ~4.2 accuracy points)
    cpath = EVAL_DIR / "central_mlp_test_metrics.json"
    if cpath.exists():
        c = json.loads(cpath.read_text(encoding="utf-8")); f = fed["test"]
        print("\nCentralized vs federated (TEST)   [Phase 1 federation cost: ~4.2 accuracy pts]")
        for k in ["accuracy", "recall", "f1", "roc_auc"]:
            print(f"  {k:9s} central={c[k]:.4f}  federated={f[k]:.4f}  cost={100*(c[k]-f[k]):+.1f} pts")
    else:
        print("\n(run phase2_train_central.py to get the centralized-vs-federated comparison)")

    print(f"\nBest round: {best_round} of {len(history)}  |  comms: {cum_bytes/1e6:.2f} MB")
    print(f"Saved: {MODEL_DIR / 'federated_mlp_state_dict.pt'}\n       {EVAL_DIR}\n       {FIG_DIR / 'federated_convergence.png'}")


if __name__ == "__main__":
    main()