"""
phase2_diag_fed.py  -  why did FedAvg fail on one seed?
--------------------------------------------------------------------
Re-runs the federated model for ONE seed (default 0) exactly as
phase2_seeds.py does, but prints what happens inside:
  * each client's size and breach rate on this split
  * global val ROC-AUC every round + each client's local loss
  * the val/test breach balance
so we can tell apart:
  (a) a lucky early val peak that early stopping locked in,
  (b) FedAvg genuinely not converging (clients pulling apart),
  (c) a hard split.

Run from phase2_synthetic/:   python phase2_diag_fed.py --seed 0 --patience 25
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from phase2_common import SUITES, SecurityMLP, average_state_dicts, train_local_model
from phase2_features import build_table
from phase2_seeds import make_data, score
import phase2_train_fed as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=F.NUM_ROUNDS)
    a = ap.parse_args()

    d = make_data(build_table(), a.seed)
    print(f"\nseed {a.seed}  val breach={d['y_val'].mean():.2f}  test breach={d['y_test'].mean():.2f}")
    print("Clients:")
    for s in SUITES:
        m = d["suite_train"] == s
        print(f"  {s:10s} n={m.sum():3d}  breach_rate={d['y_train'][m].mean():.3f}  "
              f"| val n={(d['suite_val']==s).sum():2d}  test n={(d['suite_test']==s).sum():2d}")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    clients = {s: (d["X_train"][d["suite_train"] == s], d["y_train"][d["suite_train"] == s]) for s in SUITES}
    dim = d["X_train"].shape[1]
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    X_te = torch.tensor(d["X_test"])
    g = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
    gstate = g.state_dict()

    best, best_rnd, wait = -1.0, 0, 0
    print(f"\n{'rnd':>3}  {'val_auc':>7}  {'test_acc':>8}  {'pred_breach%':>12}  local losses " + " ".join(f"{s[:5]:>6}" for s in SUITES))
    for rnd in range(1, a.rounds + 1):
        states, sizes, losses = [], [], []
        for s in SUITES:
            X, y = clients[s]
            st, _, ll = train_local_model(global_state=gstate, X=X, y=y, input_dim=dim,
                                          hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT,
                                          local_epochs=F.LOCAL_EPOCHS, lr=F.LOCAL_LR,
                                          batch_size=F.LOCAL_BATCH_SIZE, seed=a.seed + rnd)
            states.append(st); sizes.append(float(len(X))); losses.append(ll)
        gstate = average_state_dicts(states, sizes)
        g.load_state_dict(gstate); g.eval()
        with torch.no_grad():
            pv = torch.sigmoid(g(X_va)).numpy(); pt = torch.sigmoid(g(X_te)).numpy()
        auc = roc_auc_score(y_va, pv)
        mark = ""
        if auc > best:
            best, best_rnd, wait, mark = auc, rnd, 0, "  <- best"
        else:
            wait += 1
        if rnd <= 10 or rnd % 5 == 0 or mark:
            print(f"{rnd:3d}  {auc:7.4f}  {score(d['y_test'], pt)['accuracy']:8.4f}  "
                  f"{100*(pt >= .5).mean():11.1f}%  {'':12s}" + " ".join(f"{l:6.3f}" for l in losses) + mark)
        if wait >= a.patience:
            print(f"early stop at round {rnd}"); break
    print(f"\nbest round {best_rnd} (val_auc={best:.4f})")


if __name__ == "__main__":
    main()