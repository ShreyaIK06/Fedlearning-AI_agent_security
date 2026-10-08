"""
phase2_p2_train.py  -  Phase 2, Path 2 step 2: train on trajectory EMBEDDINGS
-------------------------------------------------------------------------------
Same models, same protocol, same 5 seeds as Path 1 - only the input changes from the 67
Path-1 features to the 384-dim sentence embeddings made by phase2_embed.py.

  * SecurityMLP (Phase 1, unchanged; input_dim 384)
  * centralized: batch 256, Adam 1e-3, max 60 epochs        (phase2_train_central.py)
  * federated:   FedAvg, 4 suite clients, 3 local epochs, local batch 64, max 150 rounds
                 (phase2_train_fed.py, Phase 1's fedavg_utils unchanged)
  * both: patience 25, warm-up 20, best val ROC-AUC, threshold 0.5
  * seeds 42, 0, 1, 2, 3 - each re-splits exactly as Path 1 did (same scenarios in each split)
  * StandardScaler fit on each seed's TRAIN embeddings (Phase 1 style: encoders fit train-only)

Seed 42's models (and its scaler) are saved: they are what phase2_p2_attacks.py attacks.
If Path 1's 5-seed summary exists, a Path 1 vs Path 2 table is printed.

Run from phase2_synthetic/:   python phase2_p2_train.py              (both views)
                              python phase2_p2_train.py --views actions
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import phase2_train_central as C
import phase2_train_fed as F
from phase2_common import (EVAL_DIR, OUT_DIR, ROOT, SUITES, SecurityMLP, average_state_dicts,
                           train_local_model)
from phase2_embed import EMB_NPZ
from phase2_features import build_table
from phase2_split import one_fold

SEEDS = [42, 0, 1, 2, 3]
P2_MODEL_DIR = ROOT / "models" / "phase2_path2"
P2_OUT = OUT_DIR / "path2"
P1_SUMMARY = OUT_DIR / "seeds_summary_patience25_warmup20.csv"
METRICS = ["accuracy", "recall", "f1", "roc_auc"]


def load_embeddings(view):
    d = np.load(EMB_NPZ, allow_pickle=False)
    ids = np.concatenate([d[f"run_id_{s}"] for s in ("train", "val", "test")])
    X = np.concatenate([d[f"X_{view}_{s}"] for s in ("train", "val", "test")])
    return dict(zip(ids, X)), str(d["model_name"])


def make_data(df, seed, emb):
    """Path 1's split (StratifiedGroupKFold x2 by scenario, stratified on breach) -> embedding rows."""
    trainval, test = one_fold(df, seed)
    train, val = one_fold(trainval.reset_index(drop=True), seed)
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(a.scenario_id) & set(b.scenario_id), "scenario leaked across splits"
    raw = {s: np.stack([emb[r] for r in part["run_id"]]) for s, part in [("train", train), ("val", val), ("test", test)]}
    scaler = StandardScaler().fit(raw["train"])
    d = {"scaler": scaler}
    for s, part in [("train", train), ("val", val), ("test", test)]:
        d[f"X_{s}"] = scaler.transform(raw[s]).astype(np.float32)
        d[f"y_{s}"] = part["breach"].to_numpy(np.float32)
        d[f"suite_{s}"] = part["suite"].to_numpy(str)
        d[f"run_id_{s}"] = part["run_id"].to_numpy(str)
    return d


def score(y, p):
    y = y.astype(int); pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p)}


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def train_central(d, seed):
    """Same loop as phase2_train_central.py (patience 25, warm-up 20)."""
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_tr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    n_pos = int(y_tr.sum()); pw = torch.tensor((len(y_tr) - n_pos) / n_pos, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=C.BATCH_SIZE, shuffle=True)
    model = SecurityMLP(input_dim=X_tr.shape[1], hidden_dims=C.HIDDEN_DIMS, dropout=C.DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR); crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, state, wait, ep = -1.0, None, 0, 0
    for ep in range(1, C.MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(model(X_va)).numpy())
        if ep < C.WARMUP:
            continue
        if auc > best:
            best, wait = auc, 0; state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= C.EARLY_STOP_PATIENCE:
            break
    model.load_state_dict(state); model.eval()
    return model, ep


def train_fed(d, seed):
    """Same loop as phase2_train_fed.py (Phase 1's FedAvg utils, patience 25, warm-up 20)."""
    torch.manual_seed(seed); np.random.seed(seed)
    clients = {s: (d["X_train"][d["suite_train"] == s], d["y_train"][d["suite_train"] == s]) for s in SUITES}
    dim = d["X_train"].shape[1]
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    g = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT); gstate = g.state_dict()
    best, state, wait, rnd = -1.0, None, 0, 0
    for rnd in range(1, F.NUM_ROUNDS + 1):
        sts, sz = [], []
        for s in SUITES:
            X, y = clients[s]
            st, _, _ = train_local_model(global_state=gstate, X=X, y=y, input_dim=dim, hidden_dims=F.HIDDEN_DIMS,
                                         dropout=F.DROPOUT, local_epochs=F.LOCAL_EPOCHS, lr=F.LOCAL_LR,
                                         batch_size=F.LOCAL_BATCH_SIZE, seed=seed + rnd)
            sts.append(st); sz.append(float(len(X)))
        gstate = average_state_dicts(sts, sz); g.load_state_dict(gstate); g.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(g(X_va)).numpy())
        if rnd < F.WARMUP:
            continue
        if auc > best:
            best, wait = auc, 0; state = {k: v.clone() for k, v in gstate.items()}
        else:
            wait += 1
        if wait >= F.EARLY_STOP_PATIENCE:
            break
    g.load_state_dict(state); g.eval()
    return g, rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["actions", "full"], choices=["actions", "full"])
    a = ap.parse_args()

    df = build_table()
    rows = []
    for view in a.views:
        emb, model_name = load_embeddings(view)
        print(f"\n=== view '{view}'  ({model_name}, {len(next(iter(emb.values())))}-dim) ===")
        for seed in SEEDS:
            d = make_data(df, seed, emb)
            cm, cep = train_central(d, seed)
            fm, frd = train_fed(d, seed)
            for name, m, n in [("central", cm, cep), ("federated", fm, frd)]:
                sc = score(d["y_test"], predict(m, d["X_test"]))
                rows.append({"view": view, "seed": seed, "model": name, **{k: round(v, 4) for k, v in sc.items()}})
                print(f"  seed {seed:2d} {name:9s} acc={sc['accuracy']:.4f} recall={sc['recall']:.4f} "
                      f"f1={sc['f1']:.4f} auc={sc['roc_auc']:.4f}  ({n} {'epochs' if name == 'central' else 'rounds'})")
            if seed == 42:
                P2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(cm.state_dict(), P2_MODEL_DIR / f"central_{view}_state_dict.pt")
                torch.save(fm.state_dict(), P2_MODEL_DIR / f"federated_{view}_state_dict.pt")
                joblib.dump(d["scaler"], P2_MODEL_DIR / f"scaler_{view}.joblib")

    runs = pd.DataFrame(rows)
    P2_OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(P2_OUT / "path2_seeds_per_run.csv", index=False)
    summ = runs.groupby(["view", "model"])[METRICS].agg(["mean", "std"]).round(4)
    summ.to_csv(EVAL_DIR / "path2_training_summary.csv")

    print("\n" + "=" * 90 + "\nPath 2 TEST metrics over 5 seeds (mean +- std)\n" + "=" * 90)
    for (view, model), s in summ.iterrows():
        print(f"  {view:8s} {model:9s} " + "  ".join(f"{k}={s[(k,'mean')]:.3f}+-{s[(k,'std')]:.3f}" for k in METRICS))

    print("\nCost of federation, paired per seed (central - federated, points)")
    for view in a.views:
        p = runs[runs.view == view].pivot(index="seed", columns="model", values=METRICS).reindex(SEEDS)
        for k in ["accuracy", "roc_auc"]:
            diff = 100 * (p[(k, "central")] - p[(k, "federated")])
            print(f"  {view:8s} {k:9s} mean={diff.mean():+5.1f}  std={diff.std():4.1f}")

    if P1_SUMMARY.exists():
        p1 = pd.read_csv(P1_SUMMARY, header=[0, 1], index_col=0)
        print("\nPath 1 (67 tool/category features) vs Path 2 (384-dim embeddings), test accuracy, 5 seeds:")
        for model, p1name in [("central", "central_b256"), ("federated", "federated")]:
            line = f"  {model:9s} Path1={p1.loc[p1name, ('accuracy','mean')]:.3f}+-{p1.loc[p1name, ('accuracy','std')]:.3f}"
            for view in a.views:
                s = summ.loc[(view, model)]
                line += f"   Path2[{view}]={s[('accuracy','mean')]:.3f}+-{s[('accuracy','std')]:.3f}"
            print(line)
    print(f"\nSaved: {P2_MODEL_DIR}/ (seed-42 models + scalers)\n       {P2_OUT / 'path2_seeds_per_run.csv'}"
          f"\n       {EVAL_DIR / 'path2_training_summary.csv'}")


if __name__ == "__main__":
    main()