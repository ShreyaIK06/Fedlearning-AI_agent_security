"""
phase2_seeds.py  -  Phase 2 robustness check over 5 seeds
--------------------------------------------------------------------
Question: is "federated ~= centralized" (no measurable cost of federation)
real, or luck of one split / one initialisation?

For each seed, the WHOLE Path 1 pipeline is re-run in memory:
  1. new grouped split   (StratifiedGroupKFold x2, random_state = seed)
  2. encoders re-fit on that seed's TRAIN only
  3. centralized MLP     - Phase 1 settings (batch 256)            [main]
  4. centralized MLP     - batch 64 (sensitivity: similar number of
                           weight updates to federated)            [secondary]
  5. federated FedAvg    - Phase 1 settings                        [main]
All evaluated on that seed's TEST split, threshold 0.5.

Seed 42 is included and MUST reproduce phase2_train_central.py /
phase2_train_fed.py exactly (0.9204 / 0.9558 test accuracy) - a built-in
correctness check. The main seed-42 files on disk are NOT overwritten.

Output: outputs/phase2_synthetic/seeds_per_run.csv
        outputs/phase2_synthetic/seeds_summary.csv

Run from phase2_synthetic/:   python phase2_seeds.py
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from phase2_common import (OUT_DIR, SUITES, SecurityMLP, average_state_dicts,
                           label_noise_ceiling, train_local_model)
from phase2_features import build_table, fit_encoders, transform
from phase2_split import one_fold
import phase2_train_central as C
import phase2_train_fed as F

SEEDS = [42, 0, 1, 2, 3]
PATIENCE = None   # set from --patience; None = Phase 1 values (8 epochs central, 10 rounds federated)
WARMUP = 0        # set from --warmup; no checkpoint is selected (and patience doesn't count) before this epoch/round
METRICS = ["accuracy", "recall", "f1", "roc_auc"]


# ------------------------------------------------------------ data per seed
def make_data(df, seed):
    trainval, test = one_fold(df, seed)
    train, val = one_fold(trainval.reset_index(drop=True), seed)
    assert not set(train.scenario_id) & set(test.scenario_id)
    assert not set(train.scenario_id) & set(val.scenario_id)
    assert not set(val.scenario_id) & set(test.scenario_id)
    enc = fit_encoders(train)
    out = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out[f"X_{name}"] = transform(part, *enc)
        out[f"y_{name}"] = part["breach"].to_numpy(np.float32)
        out[f"suite_{name}"] = part["suite"].to_numpy(str)
    return out


def score(y, p):
    pred = (p >= 0.5).astype(int); y = y.astype(int)
    return {"accuracy": accuracy_score(y, pred), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p)}


# --------------------------------------------- centralized (= C's loop)
def run_central(d, seed, batch_size):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_tr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    n_pos = int(y_tr.sum()); pos_weight = torch.tensor((len(y_tr) - n_pos) / n_pos, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = SecurityMLP(input_dim=X_tr.shape[1], hidden_dims=C.HIDDEN_DIMS, dropout=C.DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state, wait, updates, epoch = -1.0, None, 0, 0, 0
    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step(); updates += 1
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(model(X_va)).numpy())
        if epoch < WARMUP:
            continue
        if auc > best:
            best, wait = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= (PATIENCE or C.EARLY_STOP_PATIENCE):
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(d["X_test"]))).numpy()
    return score(d["y_test"], p), {"epochs_or_rounds": epoch, "weight_updates": updates}


# --------------------------------------------- federated (= F's loop)
def run_fed(d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    clients = {s: (d["X_train"][d["suite_train"] == s], d["y_train"][d["suite_train"] == s]) for s in SUITES}
    assert all(len(c[0]) > 0 for c in clients.values()), "a suite has no train rows for this seed"
    input_dim = d["X_train"].shape[1]
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    g = SecurityMLP(input_dim=input_dim, hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
    gstate = g.state_dict()
    best, best_state, wait, updates, rnd = -1.0, None, 0, 0, 0
    for rnd in range(1, F.NUM_ROUNDS + 1):
        states, sizes = [], []
        for s in SUITES:
            X, y = clients[s]
            st, _, _ = train_local_model(global_state=gstate, X=X, y=y, input_dim=input_dim,
                                         hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT,
                                         local_epochs=F.LOCAL_EPOCHS, lr=F.LOCAL_LR,
                                         batch_size=F.LOCAL_BATCH_SIZE, seed=seed + rnd)
            states.append(st); sizes.append(float(len(X)))
            updates += F.LOCAL_EPOCHS * int(np.ceil(len(X) / min(F.LOCAL_BATCH_SIZE, len(X))))
        gstate = average_state_dicts(states, sizes)
        g.load_state_dict(gstate); g.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(g(X_va)).numpy())
        if rnd < WARMUP:
            continue
        if auc > best:
            best, wait = auc, 0
            best_state = {k: v.clone() for k, v in gstate.items()}
        else:
            wait += 1
        if wait >= (PATIENCE or F.EARLY_STOP_PATIENCE):
            break
    g.load_state_dict(best_state); g.eval()
    with torch.no_grad():
        p = torch.sigmoid(g(torch.tensor(d["X_test"]))).numpy()
    return score(d["y_test"], p), {"epochs_or_rounds": rnd, "weight_updates": updates}


# ================================================================== main
def main():
    global PATIENCE, WARMUP
    ap = argparse.ArgumentParser()
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience for ALL models (default: Phase 1's 8 epochs / 10 rounds)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="minimum epochs/rounds before a best checkpoint can be chosen (default 0 = Phase 1)")
    args = ap.parse_args()
    PATIENCE, WARMUP = args.patience, args.warmup
    tag = (f"_patience{PATIENCE}" if PATIENCE else "") + (f"_warmup{WARMUP}" if WARMUP else "")
    print(f"Warm-up before checkpoint selection: {WARMUP} epochs/rounds")
    print(f"Early-stopping patience: {PATIENCE or 'Phase 1 defaults (8 epochs central / 10 rounds federated)'}")
    df = build_table()
    rows = []
    for seed in SEEDS:
        d = make_data(df, seed)
        print(f"\nseed {seed}: train={len(d['y_train'])} val={len(d['y_val'])} test={len(d['y_test'])} "
              f"features={d['X_train'].shape[1]} test_breach={d['y_test'].mean():.2f}")
        for model, fn in [("central_b256", lambda: run_central(d, seed, C.BATCH_SIZE)),
                          ("central_b64", lambda: run_central(d, seed, 64)),
                          ("federated", lambda: run_fed(d, seed))]:
            m, info = fn()
            rows.append({"seed": seed, "model": model, **{k: round(v, 4) for k, v in m.items()}, **info})
            print(f"  {model:13s} acc={m['accuracy']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} "
                  f"auc={m['roc_auc']:.4f}  ({info['epochs_or_rounds']} ep/rounds, {info['weight_updates']} updates)")

    runs = pd.DataFrame(rows)

    # ---- sanity check: seed 42 must match the main scripts
    r42 = runs[runs.seed == 42].set_index("model")
    if not PATIENCE and not WARMUP:
      print(f"\nSeed-42 check (should be 0.9204 / 0.9558): central_b256={r42.loc['central_b256','accuracy']:.4f}  "
          f"federated={r42.loc['federated','accuracy']:.4f}")

    # ---- mean +- std per model
    summ = runs.groupby("model")[METRICS + ["weight_updates"]].agg(["mean", "std"]).round(4)
    print("\nTEST metrics over 5 seeds (mean +- std):")
    for model in ["central_b256", "central_b64", "federated"]:
        s = summ.loc[model]
        print(f"  {model:13s} " + "  ".join(f"{k}={s[(k,'mean')]:.3f}+-{s[(k,'std')]:.3f}" for k in METRICS)
              + f"  updates~{s[('weight_updates','mean')]:.0f}")

    # ---- paired cost of federation (same split for both models, per seed)
    piv = runs.pivot(index="seed", columns="model", values=METRICS).reindex(SEEDS)   # keep 42,0,1,2,3 order
    print("\nCost of federation, paired per seed (central - federated, in points; + = federation costs)")
    print(f"   [Phase 1: ~4.2 accuracy pts]   per-seed order = {SEEDS}")
    for base in ["central_b256", "central_b64"]:
        for k in METRICS:
            diff = 100 * (piv[(k, base)] - piv[(k, "federated")])
            print(f"  vs {base:12s} {k:9s} mean={diff.mean():+5.1f}  std={diff.std():4.1f}  "
                  f"per-seed=[{', '.join(f'{x:+.1f}' for x in diff)}]")

    noise = label_noise_ceiling()
    print(f"\nLabel-noise ceiling: {noise['pct_scenarios_flipping']}% of scenarios flip "
          f"(max achievable accuracy is roughly {100 - noise['pct_records_minority']:.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT_DIR / f"seeds_per_run{tag}.csv", index=False)
    summ.to_csv(OUT_DIR / f"seeds_summary{tag}.csv")
    print(f"\nSaved: {OUT_DIR / f'seeds_per_run{tag}.csv'}\n       {OUT_DIR / f'seeds_summary{tag}.csv'}")


if __name__ == "__main__":
    main()