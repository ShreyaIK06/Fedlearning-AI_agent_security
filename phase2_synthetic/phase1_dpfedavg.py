"""
phase1_dpfedavg.py  -  DP-FedAvg (DP inside each client) on PHASE 1's AgentDojo data
--------------------------------------------------------------------------------------
Same mechanism, attacks and settings as phase2_dpfedavg.py (imported, not copied), applied to the
AgentDojo pipeline's processed data, so Phase 1's DP result can be corrected the same way and the
two datasets compared. The key difference is CLIENT SIZE: AgentDojo suite clients hold thousands of
records, ours ~90, so the noise needed for the same epsilon should be far smaller.

Input (made by Phase 1's stage-1 scripts, run them first if missing):
  phase1_agentdojo/outputs/phase1/04_X_{train,val,test}.pkl   176 features (DataFrames)
  phase1_agentdojo/outputs/phase1/04_y_{train,val,test}.pkl   security label
  phase1_agentdojo/outputs/phase1/03_train.pkl                 'suite' column = the 4 clients
Phase 1 has one fixed split, so the 5 seeds vary initialisation, sampling and noise (not the split).

Leakage clients: 1 record (12 victims, 3 per suite, fixed RNG), 4, 8 and 64 records (a full local
batch; whole AgentDojo clients are too large to score on a laptop and leak nothing anyway).
No noise sweep secrets here (Phase 1 features carry no text).

Run from phase2_synthetic/:   python phase1_dpfedavg.py
  options: --rounds 40 --batch 64 --lr 1e-2 --seeds 42 0 1 2 3 --tag _x
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch

import phase2_dpfedavg as D
from phase2_common import ROOT, SUITES

P1_OUT = ROOT / "phase1_agentdojo" / "outputs" / "phase1"
EVAL = ROOT / "phase1_agentdojo" / "evaluation" / "dpfedavg"
MODELS = ROOT / "phase1_agentdojo" / "models" / "dpfedavg"


def load_phase1():
    need = [P1_OUT / f"04_{a}_{s}.pkl" for a in ("X", "y") for s in ("train", "val", "test")] + [P1_OUT / "03_train.pkl"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Missing Phase 1 files in {P1_OUT}: {missing}\nRun Phase 1's src/phase1/01-05 scripts first.")
    d = {}
    for s in ("train", "val", "test"):
        X = pd.read_pickle(P1_OUT / f"04_X_{s}.pkl"); y = pd.read_pickle(P1_OUT / f"04_y_{s}.pkl")
        d[f"X_{s}"] = np.asarray(X, dtype=np.float32); d[f"y_{s}"] = np.asarray(y, dtype=np.float32).reshape(-1)
        if s == "train":
            src = pd.read_pickle(P1_OUT / "03_train.pkl")
            assert len(src) == len(X) and (src.index == X.index).all(), "03_train and 04_X_train rows differ"
            d["suite_train"] = src["suite"].astype(str).to_numpy()
    d["run_id_train"] = np.array([f"row{i}" for i in range(len(d["X_train"]))])
    assert set(np.unique(d["suite_train"])) == set(SUITES), f"unexpected suites {np.unique(d['suite_train'])}"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D.CFG.update(rounds=a.rounds, local_epochs=a.local_epochs, batch=a.batch, lr=a.lr)
    d = load_phase1()
    sizes = {s: int((d["suite_train"] == s).sum()) for s in SUITES}
    print(f"Phase 1 (AgentDojo): train {len(d['X_train'])}  val {len(d['X_val'])}  test {len(d['X_test'])}  "
          f"features {d['X_train'].shape[1]}  breach rate train {d['y_train'].mean():.3f}")
    print(f"DP-FedAvg: rounds={D.CFG['rounds']} local_epochs={D.CFG['local_epochs']} batch={D.CFG['batch']} "
          f"Adam lr={D.CFG['lr']} C={D.CLIP} delta={D.DELTA}")
    for s, n in sizes.items():
        q, spr = D.steps_per_round(n)
        print(f"  client {s:10s} n={n:6d}  sample rate {q:.4f}  steps/round {spr}  sigma eps8/3/1 = "
              + " / ".join(f"{D.sigma_for(e, n):.2f}" for e in D.EPS))

    # ---------------- A. utility
    util, gmodels = [], {}
    for seed in a.seeds:
        line = f"  seed {seed:2d}"
        for eps in [float("inf")] + D.EPS:
            g, sig, eps_real = D.train_dpfedavg(d, seed, eps)
            m = D.metrics(d["y_test"], D.predict(g, d["X_test"]))
            lbl = "inf" if eps == float("inf") else str(int(eps))
            util.append({"seed": seed, "epsilon": lbl, **{k: round(v, 4) for k, v in m.items()},
                         "sigma_mean": round(float(np.mean(list(sig.values()))), 3),
                         "eps_achieved_max": round(max(eps_real.values()), 2) if eps != float("inf") else None})
            line += f" | eps={lbl} acc={m['accuracy']:.3f} rec={m['breach_recall']:.3f}"
            if seed == a.seeds[0]:
                gmodels[lbl] = g
        print(line, flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for lbl, g in gmodels.items():
        torch.save(g.state_dict(), MODELS / f"phase1_dpfedavg_eps{lbl}_state_dict.pt")

    # ---------------- B. leakage
    rng = np.random.default_rng(42)
    clients = {s: np.where(d["suite_train"] == s)[0] for s in SUITES}
    victims = [i for s in SUITES for i in rng.choice(clients[s], 3, replace=False)]
    groups = {"1": [np.array([i]) for i in victims],
              **{str(k): [rng.choice(clients[s], k, replace=False) for s in SUITES for _ in range(2)] for k in (4, 8, 64)}}
    Xtr, ytr, ids = d["X_train"], d["y_train"], d["run_id_train"]
    rows = []
    for lbl, gm in gmodels.items():
        gstate = copy.deepcopy(gm.state_dict()); eps = float(lbl)
        for opt_name in ("system", "sgd"):
            for k, sets in groups.items():
                for j, idx in enumerate(sets):
                    sig = D.sigma_for(eps, len(idx))
                    delta, _ = D.local_dp(gm, gstate, Xtr[idx], ytr[idx], sig, None if eps == float("inf") else D.CLIP,
                                          opt_name, False, 7000 + j, D.F.DROPOUT)
                    rows.append({"epsilon": lbl, "client_opt": opt_name, "k": k, "set": j, "sigma": round(sig, 3),
                                 **D.attack_update(delta, Xtr[idx], ids[idx], None)})
        print(f"  leakage done for eps={lbl}", flush=True)
    L = pd.DataFrame(rows)

    # ---------------- C. noise sweep (single-record SGD client, non-private global model)
    W = D.sweep("Phase 1", gmodels["inf"], Xtr, ytr, ids, victims, None, None, None)

    U = pd.DataFrame(util)
    print("\n" + "=" * 100 + "\nA. Utility (test, seeds) and paired cost vs the non-private run of the same loop\n" + "=" * 100)
    piv = U.pivot(index="seed", columns="epsilon", values=["accuracy", "roc_auc", "breach_recall", "specificity"])
    cost = []
    for e in ["inf", "8", "3", "1"]:
        s = U[U.epsilon == e]
        row = {"epsilon": e, "accuracy": f"{s.accuracy.mean():.3f}+-{s.accuracy.std():.3f}",
               "roc_auc": f"{s.roc_auc.mean():.3f}+-{s.roc_auc.std():.3f}",
               "breach_recall": f"{s.breach_recall.mean():.3f}+-{s.breach_recall.std():.3f}"}
        if e != "inf":
            for k in ["accuracy", "roc_auc", "breach_recall", "specificity"]:
                dl = 100 * (piv[(k, "inf")] - piv[(k, e)])
                row[f"{k}_cost"] = f"{dl.mean():+.1f}+-{dl.std():.1f}"
            row["sigma_mean"] = round(s.sigma_mean.mean(), 3)
        cost.append(row)
    C = pd.DataFrame(cost); print(C.to_string(index=False))

    print("\n" + "=" * 100 + "\nB. Leakage from DP-FedAvg client updates (first-seed global models)\n" + "=" * 100)
    L["exact_rate"] = L.cf_exact / L.records
    LB = L.groupby(["epsilon", "client_opt", "k"], sort=False).agg(
        updates=("records", "size"), exact_rate=("exact_rate", "mean"), best_cos=("best_cos", "mean")).reset_index()
    print(LB.round(3).to_string(index=False))

    print("\n" + "=" * 100 + "\nC. Noise sweep: single-record SGD client\n" + "=" * 100)
    W["exact_rate"] = W.cf_exact / W.records
    WB = W.groupby("sigma").agg(exact_rate=("exact_rate", "mean"), best_cos=("best_cos", "mean"),
                                eps_if_used_all_training=("eps_if_used_all_training", "first")).reset_index()
    print(WB.round(4).to_string(index=False))

    EVAL.mkdir(parents=True, exist_ok=True)
    U.to_csv(EVAL / f"phase1_dpfedavg_utility_runs{a.tag}.csv", index=False)
    C.to_csv(EVAL / f"phase1_dpfedavg_utility_summary{a.tag}.csv", index=False)
    L.to_csv(EVAL / f"phase1_dpfedavg_leakage_per_update{a.tag}.csv", index=False)
    LB.to_csv(EVAL / f"phase1_dpfedavg_leakage_summary{a.tag}.csv", index=False)
    WB.to_csv(EVAL / f"phase1_dpfedavg_noise_sweep_summary{a.tag}.csv", index=False)
    print(f"\nSaved: {EVAL}/\n       {MODELS}/")


if __name__ == "__main__":
    main()