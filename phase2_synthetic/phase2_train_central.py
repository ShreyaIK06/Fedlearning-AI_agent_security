"""
phase2_train_central.py  -  Phase 2, centralized MLP baseline
--------------------------------------------------------------------
Mirrors Phase 1's src/phase2/04_train_mlp.py exactly - same SecurityMLP
(imported unchanged), same settings:
  hidden (128, 64), dropout 0.2, Adam lr 1e-3, batch 256,
  up to 60 epochs, BCEWithLogitsLoss(pos_weight = n_neg / n_pos),
  threshold 0.5, seed 42. Only input_dim differs (67 vs 176).

TWO documented deviations from Phase 1 (small-data protocol, see phase2_seeds.py):
  * EARLY_STOP_PATIENCE 8 -> 25: Phase 1's epochs were ~700 updates, ours are
    2, so 8 epochs stopped training prematurely (2 of 5 seeds collapsed).
  * WARMUP = 20: no checkpoint is selected (and patience doesn't count)
    before epoch 20 - with ~90 val records ROC-AUC can peak by chance
    before the 0.5 decision threshold has settled.
  Over 5 seeds this cut the spread from +-16.0 to +-8.5 accuracy points.

Also evaluates the best model on TEST (Phase 1 did this in a separate
05_evaluate_test.py; same metrics function).

Run from phase2_synthetic/:   python phase2_train_central.py
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from phase2_common import (EVAL_DIR, FIG_DIR, MODEL_DIR, OUT_DIR, RANDOM_STATE,
                           SecurityMLP, label_noise_ceiling, load_features,
                           metrics_for, per_suite_recall, print_summary, save_metrics)

matplotlib.use("Agg")

# ---- identical to Phase 1's 04_train_mlp.py
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 25   # Phase 1: 8
WARMUP = 20                # Phase 1: none


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    d = load_features()
    X_tr, y_tr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    X_va, y_va = torch.tensor(d["X_val"]), torch.tensor(d["y_val"])
    X_te, y_te = torch.tensor(d["X_test"]), d["y_test"]

    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model = SecurityMLP(input_dim=X_tr.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}  features={X_tr.shape[1]}  "
          f"pos_weight={pos_weight.item():.3f}  steps/epoch={len(loader)}")

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_roc_auc": [], "epoch_wall_time_sec": []}
    best_auc, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train(); tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        train_loss = tot / len(X_tr)

        model.eval()
        with torch.no_grad():
            vl = model(X_va)
            val_loss = crit(vl, y_va).item()
            val_auc = roc_auc_score(y_va.numpy(), torch.sigmoid(vl).numpy())

        for k, v in zip(hist, (epoch, train_loss, val_loss, val_auc, time.perf_counter() - t0)):
            hist[k].append(v)
        if epoch >= WARMUP:
            if val_auc > best_auc:
                best_auc, best_epoch, wait = val_auc, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_roc_auc={val_auc:.4f}")
        if wait >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val_roc_auc improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(X_va)).numpy()
        p_te = torch.sigmoid(model(X_te)).numpy()

    noise = label_noise_ceiling()
    cfg = {"hidden_dims": list(HIDDEN_DIMS), "dropout": DROPOUT, "lr": LR, "batch_size": BATCH_SIZE,
           "early_stop_patience": EARLY_STOP_PATIENCE, "warmup": WARMUP, "best_epoch": best_epoch, "total_epochs_trained": len(hist["epoch"]),
           "total_training_wall_time_sec": round(sum(hist["epoch_wall_time_sec"]), 4)}

    print()
    for split, y, p in [("val", d["y_val"], p_va), ("test", y_te, p_te)]:
        m = metrics_for(y, p, "central_mlp", split, noise); m["architecture"] = cfg
        print_summary(m)
        save_metrics(m, EVAL_DIR / f"central_mlp_{split}_metrics.json")

    ps = per_suite_recall(y_te, p_te, d["suite_test"])
    print("\nPer-suite TEST (indicative only - small n):")
    print(ps.to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / "central_mlp_state_dict.pt")
    pd.DataFrame(hist).to_csv(OUT_DIR / "central_training_history.csv", index=False)
    ps.to_csv(OUT_DIR / "central_per_suite_test.csv", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    a1.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    a1.axvline(best_epoch, color="gray", ls="--", alpha=.6, label="best epoch")
    a1.set_xlabel("epoch"); a1.set_ylabel("BCE loss"); a1.legend(); a1.set_title("Training/validation loss")
    a2.plot(hist["epoch"], hist["val_roc_auc"], color="green")
    a1.axvspan(0, WARMUP, color="gray", alpha=.08); a2.axvspan(0, WARMUP, color="gray", alpha=.08)
    a2.axvline(best_epoch, color="gray", ls="--", alpha=.6)
    a2.set_xlabel("epoch"); a2.set_ylabel("val ROC-AUC"); a2.set_title("Validation ROC-AUC")
    fig.suptitle("Phase 2 (synthetic) - centralized MLP"); fig.tight_layout()
    fig.savefig(FIG_DIR / "central_training_curve.png", dpi=150); plt.close(fig)

    print(f"\nLabel-noise ceiling (Phase 2): {noise['pct_scenarios_flipping']}% of "
          f"{noise['n_scenarios_compared']} scenarios flip across repeats "
          f"({noise['pct_records_minority']}% of records disagree with their scenario's majority)  "
          f"[Phase 1: 26.4%]")
    print(f"Best epoch: {best_epoch} of {len(hist['epoch'])}")
    print(f"Saved: {MODEL_DIR / 'central_mlp_state_dict.pt'}\n       {EVAL_DIR}\n       {FIG_DIR / 'central_training_curve.png'}")


if __name__ == "__main__":
    main()