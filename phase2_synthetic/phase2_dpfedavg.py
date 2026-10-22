"""
phase2_dpfedavg.py  -  Phase 2: DP applied WHERE THE DATA LEAVES THE CLIENT (DP-FedAvg)
-----------------------------------------------------------------------------------------
Why this script exists
  phase2_dp.py trained a model with DP-SGD and then attacked a fresh, CLEAN gradient computed on
  the released model. In a federated system that uses DP properly, each client runs DP-SGD
  locally, so every update the server ever sees is already clipped and noised. This script tests
  that setting, which is the one a reviewer will expect.

Mechanism (record-level DP-SGD inside every client, the standard DP-FedAvg variant)
  every round, each suite client starts from the global model and runs local DP-SGD:
    Poisson-sample a batch (expected size B), per-example gradients, clip each to norm C,
    sum, add Gaussian noise N(0, sigma^2 C^2), divide by the expected batch size, optimizer step.
  The client sends (global - local); the server averages (Phase 1's size-weighted FedAvg).
  Privacy is accounted per client over all its steps (Opacus RDP accountant), delta = 1e-5:
  each client's records are protected at the stated epsilon against anyone who sees its updates,
  including the server. sigma is chosen per client for its own size (sample rate B / n_client).

Part A  Utility: 5 seeds, both paths (Path 1 features, Path 2 'full' embeddings), eps 1/3/8 vs a
        non-private run of the SAME loop (no clipping, no noise). Paired cost per seed.
Part B  Leakage: attack the noisy client updates sent to the seed-42 DP-FedAvg global model.
        Clients holding 1 record (the 12 victims of every earlier attack), 4, 8 (2 sets per suite)
        or the whole suite. Local optimizer: 'system' = Adam (as trained) and 'sgd' = plain SGD
        (the attacker's best case: without noise, single-record SGD clients leaked fully).
        Attacks: all-neuron closed-form (phase2_aggregation.py); Path 2 also names runs (no oracle)
        and, for single-record clients, runs the dictionary secret extraction (phase2_p2_dp.py).
Part C  Noise sweep: single-record SGD client, sigma from 0 to 1, to find how little noise is
        needed to stop the leak, with the epsilon that noise would give over the whole training.

Leakage tests use plain BCE (no pos_weight) so a single breach record still sends a non-zero
update; clipping removes the loss scale anyway. Utility training uses Phase 1's per-client
pos_weight, as in our system.

Run from phase2_synthetic/:   python phase2_dpfedavg.py                 (~20-40 min)
                              python phase2_dpfedavg.py --no-secrets    (skip the text encoder)
"""

import argparse
import copy
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from torch.func import functional_call, grad, vmap

import phase2_p2_train as P2
import phase2_seeds as S
import phase2_train_fed as F
from phase2_aggregation import first_layer, identify, score, twins
from phase2_common import (EVAL_DIR, OUT_DIR, ROOT, SUITES, SecurityMLP, average_state_dicts,
                           load_features)
from phase2_features import build_table

warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")

SEEDS = [42, 0, 1, 2, 3]
EPS = [8.0, 3.0, 1.0]
DELTA = 1e-5
CLIP = 1.0
VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"
OUT = OUT_DIR / "dpfedavg"
MODELS = ROOT / "models" / "phase2_dpfedavg"
SWEEP_SIGMAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

CFG = {"rounds": 40, "local_epochs": 1, "batch": 64, "lr": 1e-2, "sgd_lr": 0.05, "warmup": 10}


# ============================================================ DP accounting
def steps_per_round(n):
    q = min(1.0, CFG["batch"] / n)
    return q, CFG["local_epochs"] * max(1, math.ceil(1 / q))


def sigma_for(eps, n):
    if eps == float("inf"):
        return 0.0
    q, spr = steps_per_round(n)
    return float(get_noise_multiplier(target_epsilon=eps, target_delta=DELTA, sample_rate=q,
                                      steps=CFG["rounds"] * spr, accountant="rdp"))


def eps_for(sigma, n):
    if sigma <= 0:
        return float("inf")
    q, spr = steps_per_round(n)
    acc = RDPAccountant(); acc.history = [(sigma, q, CFG["rounds"] * spr)]
    try:
        return float(acc.get_epsilon(DELTA))
    except Exception:
        return float("inf")


# ============================================================ one client's local DP-SGD
def local_dp(model, gstate, X, y, sigma, clip, opt_name, pos_weight, seed, dropout):
    """Local training from the global weights; returns (update = global - local) as a list per parameter.
    sigma = 0 and clip = None gives ordinary (non-private) local training through the same code."""
    torch.manual_seed(seed)
    m = SecurityMLP(input_dim=X.shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=dropout)
    m.load_state_dict(gstate); m.train()
    names = [k for k, _ in m.named_parameters()]
    opt = (torch.optim.Adam(m.parameters(), lr=CFG["lr"]) if opt_name == "system"
           else torch.optim.SGD(m.parameters(), lr=CFG["sgd_lr"]))
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(Xt)
    q, spr = steps_per_round(n)
    pw = None
    if pos_weight:
        n_pos = int(y.sum()); pw = torch.tensor((n - n_pos) / max(n_pos, 1), dtype=torch.float32)

    def loss_one(p, x, t):
        out = functional_call(m, p, (x.unsqueeze(0),)).view(-1)
        return Fn.binary_cross_entropy_with_logits(out, t.view(-1), pos_weight=pw)

    per_ex = vmap(grad(loss_one), in_dims=(None, 0, 0), randomness="different")
    for _ in range(spr):
        mask = torch.rand(n) < q if q < 1 else torch.ones(n, dtype=torch.bool)
        params = {k: v.detach() for k, v in m.named_parameters()}
        if mask.any():
            g = per_ex(params, Xt[mask], yt[mask])
            if clip is not None:
                norms = torch.sqrt(sum(g[k].reshape(len(g[k]), -1).pow(2).sum(1) for k in names))
                f = (clip / (norms + 1e-12)).clamp(max=1.0)
                g = {k: g[k] * f.view(-1, *([1] * (g[k].dim() - 1))) for k in names}
            summed = {k: g[k].sum(0) for k in names}
        else:
            summed = {k: torch.zeros_like(params[k]) for k in names}
        denom = q * n if clip is not None else max(int(mask.sum()), 1)
        opt.zero_grad()
        for k, p in m.named_parameters():
            noise = torch.randn_like(p) * sigma * (clip or 0.0)
            p.grad = (summed[k] + noise) / denom
        opt.step()
    g0 = SecurityMLP(input_dim=X.shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=0.0); g0.load_state_dict(gstate)
    return [a.detach() - b.detach() for a, b in zip(g0.parameters(), m.parameters())], m.state_dict()


# ============================================================ Part A: DP-FedAvg training
def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def metrics(y, p):
    y = y.astype(int); pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    return {"accuracy": accuracy_score(y, pred), "breach_recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"), "roc_auc": roc_auc_score(y, p)}


def train_dpfedavg(d, seed, eps):
    torch.manual_seed(seed); np.random.seed(seed)
    dim = d["X_train"].shape[1]
    clients = {s: (d["X_train"][d["suite_train"] == s], d["y_train"][d["suite_train"] == s]) for s in SUITES}
    sig = {s: sigma_for(eps, len(c[0])) for s, c in clients.items()}
    g = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
    gstate = copy.deepcopy(g.state_dict())
    best, best_state = -1.0, None
    for rnd in range(1, CFG["rounds"] + 1):
        states, sizes = [], []
        for k, s in enumerate(SUITES):
            X, y = clients[s]
            _, st = local_dp(g, gstate, X, y, sig[s], None if eps == float("inf") else CLIP, "system",
                             True, seed * 1000 + rnd * 10 + k, F.DROPOUT)
            states.append(st); sizes.append(float(len(X)))
        gstate = average_state_dicts(states, sizes); g.load_state_dict(gstate)
        auc = roc_auc_score(d["y_val"], predict(g, d["X_val"]))
        if rnd >= CFG["warmup"] and auc > best:
            best, best_state = auc, copy.deepcopy(gstate)
    g.load_state_dict(best_state); g.eval()
    eps_real = {s: eps_for(sig[s], len(clients[s][0])) for s in SUITES}
    return g, sig, eps_real


# ============================================================ Part B / C helpers
def secrets_eval(x_row, scaler, cand):
    e = scaler.inverse_transform(x_row[None])[0]
    right = total = 0
    for _, true, cands, ce in cand:
        total += 1
        right += int(cands[int(np.argmax(ce @ e))] == true)
    return right, total


def attack_update(delta, X, ids, ident, scaler=None, cand=None):
    W, b = first_layer(delta)
    r, R = score(W, b, X)
    out = {"records": len(X), "cf_exact": r["exact"], "cf_near": r["near"], "best_cos": r["best_cos_mean"],
           "p1_neuron_exact": r["p1_neuron_exact"]}
    if ident is not None and len(R):
        out.update(identify(R, ident["scaler"], ident["E_all"], ident["ids_all"], list(ids)))
    if cand is not None:
        j = int(b.abs().argmax())
        out["secrets_right"], out["secrets"] = secrets_eval((W[j] / b[j]).float().numpy(), scaler, cand)
    return out


def leakage(name, gmodels, sigmas_by_eps, Xtr, ytr, suites, ids, victim_idx, ident, scaler, cands, rng):
    clients = {s: np.where(suites == s)[0] for s in SUITES}
    groups = {"1": [np.array([i]) for i in victim_idx],
              "4": [rng.choice(clients[s], 4, replace=False) for s in SUITES for _ in range(2)],
              "8": [rng.choice(clients[s], 8, replace=False) for s in SUITES for _ in range(2)],
              "client": [clients[s] for s in SUITES]}
    rows = []
    for eps_lbl, gm in gmodels.items():
        gstate = copy.deepcopy(gm.state_dict())
        eps = float(eps_lbl)
        for opt_name in ("system", "sgd"):
            for k, sets in groups.items():
                for j, idx in enumerate(sets):
                    sig = sigma_for(eps, len(idx))
                    delta, _ = local_dp(gm, gstate, Xtr[idx], ytr[idx], sig, None if eps == float("inf") else CLIP,
                                        opt_name, False, 7000 + j, F.DROPOUT)
                    r = attack_update(delta, Xtr[idx], ids[idx], ident, scaler,
                                      cands[j] if (cands is not None and k == "1") else None)
                    rows.append({"path": name, "epsilon": eps_lbl, "client_opt": opt_name, "k": k, "set": j,
                                 "sigma": round(sig, 3), **r})
        print(f"  leakage done for eps={eps_lbl}", flush=True)
    return pd.DataFrame(rows)


def sweep(name, gm, Xtr, ytr, ids, victim_idx, ident, scaler, cands):
    gstate = copy.deepcopy(gm.state_dict())
    rows = []
    for sig in SWEEP_SIGMAS:
        for j, i in enumerate(victim_idx):
            idx = np.array([i])
            delta, _ = local_dp(gm, gstate, Xtr[idx], ytr[idx], sig, CLIP, "sgd", False, 9000 + j, 0.0)
            r = attack_update(delta, Xtr[idx], ids[idx], ident, scaler, cands[j] if cands is not None else None)
            rows.append({"path": name, "sigma": sig, "eps_if_used_all_training": eps_for(sig, 1), "set": j, **r})
    return pd.DataFrame(rows)


# ================================================================== per path
def run_path(name, data_for_seed, d42, victim_ids, ident=None, scaler=None, texts=None, enc=None):
    print(f"\n{'=' * 100}\n{name}\n{'=' * 100}", flush=True)
    util, gmodels = [], {}
    for seed in SEEDS:
        d = d42 if seed == 42 else data_for_seed(seed)
        line = f"  seed {seed:2d}"
        for eps in [float("inf")] + EPS:
            g, sig, eps_real = train_dpfedavg(d, seed, eps)
            m = metrics(d["y_test"], predict(g, d["X_test"]))
            lbl = "inf" if eps == float("inf") else str(int(eps))
            util.append({"path": name, "seed": seed, "epsilon": lbl, **{k: round(v, 4) for k, v in m.items()},
                         "sigma_mean": round(float(np.mean(list(sig.values()))), 2),
                         "eps_achieved_max": round(max(eps_real.values()), 2) if eps != float("inf") else None})
            line += f" | eps={lbl} acc={m['accuracy']:.3f}"
            if seed == 42:
                gmodels[lbl] = g
        print(line, flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    for lbl, g in gmodels.items():
        torch.save(g.state_dict(), MODELS / f"{name.split()[0].lower()}{name.split()[1]}_eps{lbl}_state_dict.pt")

    pos = {r: i for i, r in enumerate(d42["run_id_train"])}
    victim_idx = [pos[r] for r in victim_ids]
    cands = None
    if enc is not None:
        from phase2_p2_dp import secret_candidates
        print("  embedding secret candidates for the 12 victims ...", flush=True)
        cands = secret_candidates(enc, texts, "full", victim_ids, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    leak = leakage(name, gmodels, None, d42["X_train"], d42["y_train"], d42["suite_train"], d42["run_id_train"],
                   victim_idx, ident, scaler, cands, rng)
    sw = sweep(name, gmodels["inf"], d42["X_train"], d42["y_train"], d42["run_id_train"], victim_idx, ident,
               scaler, cands)
    return pd.DataFrame(util), leak, sw


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["1", "2"], choices=["1", "2"])
    ap.add_argument("--rounds", type=int, default=CFG["rounds"])
    ap.add_argument("--local-epochs", type=int, default=CFG["local_epochs"])
    ap.add_argument("--batch", type=int, default=CFG["batch"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--no-secrets", action="store_true")
    ap.add_argument("--stub", action="store_true", help="test encoder (NOT for results)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    CFG.update(rounds=a.rounds, local_epochs=a.local_epochs, batch=a.batch, lr=a.lr)
    print(f"DP-FedAvg: record-level DP-SGD inside each client | rounds={CFG['rounds']} local_epochs={CFG['local_epochs']} "
          f"expected batch={CFG['batch']} Adam lr={CFG['lr']} | C={CLIP} delta={DELTA} | eps {EPS} + non-private")
    victim_ids = [m["run_id"] for m in torch.load(VICTIMS_PT)["victim_meta"]]
    df = build_table()
    U, L, W = [], [], []

    if "1" in a.paths:
        d42 = load_features()
        d42 = {k: (v.astype(np.float32) if k.startswith(("X_", "y_")) else v) for k, v in d42.items()}
        for s in SUITES:
            n = int((d42["suite_train"] == s).sum())
            print(f"  Path 1 client {s:10s} n={n:3d}  sigma eps8/3/1 = "
                  + " / ".join(f"{sigma_for(e, n):.2f}" for e in EPS))
        u, l, w = run_path("Path 1 (features)", lambda sd: S.make_data(df, sd), d42, victim_ids)
        U.append(u); L.append(l); W.append(w)

    if "2" in a.paths:
        emb, _ = P2.load_embeddings("full")
        d42 = P2.make_data(df, 42, emb)
        ids_all = np.array(list(emb.keys())); E_all = np.stack([emb[i] for i in ids_all])
        print(f"\nRuns with an identical twin: {twins(E_all)} of {len(E_all)}")
        enc = texts = None
        if not a.no_secrets:
            from phase2_embed import MODEL_NAME, TEXTS_CSV, STEncoder, StubEncoder
            enc = StubEncoder() if a.stub else STEncoder(MODEL_NAME)
            texts = pd.read_csv(TEXTS_CSV, keep_default_na=False).set_index("run_id")
        u, l, w = run_path("Path 2 (embeddings)", lambda sd: P2.make_data(df, sd, emb), d42, victim_ids,
                           ident={"scaler": d42["scaler"], "E_all": E_all, "ids_all": ids_all},
                           scaler=d42["scaler"], texts=texts, enc=enc)
        U.append(u); L.append(l); W.append(w)

    U, L, W = pd.concat(U), pd.concat(L), pd.concat(W)
    OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = a.tag
    U.to_csv(OUT / f"dpfedavg_utility_runs{tag}.csv", index=False)
    L.to_csv(OUT / f"dpfedavg_leakage_per_update{tag}.csv", index=False)
    W.to_csv(OUT / f"dpfedavg_noise_sweep_per_update{tag}.csv", index=False)

    print("\n" + "=" * 100 + "\nA. Utility (test, 5 seeds) and paired cost vs the non-private run of the same loop\n" + "=" * 100)
    cost = []
    for path, u in U.groupby("path", sort=False):
        piv = u.pivot(index="seed", columns="epsilon", values=["accuracy", "roc_auc", "breach_recall", "specificity"])
        for e in ["inf", "8", "3", "1"]:
            s = u[u.epsilon == e]
            row = {"path": path, "epsilon": e, "accuracy": f"{s.accuracy.mean():.3f}+-{s.accuracy.std():.3f}",
                   "roc_auc": f"{s.roc_auc.mean():.3f}+-{s.roc_auc.std():.3f}"}
            if e != "inf":
                for k in ["accuracy", "roc_auc", "breach_recall", "specificity"]:
                    dlt = 100 * (piv[(k, "inf")] - piv[(k, e)])
                    row[f"{k}_cost"] = f"{dlt.mean():+.1f}+-{dlt.std():.1f}"
                row["sigma_mean"] = round(s.sigma_mean.mean(), 2)
            cost.append(row)
    C = pd.DataFrame(cost); print(C.to_string(index=False))

    print("\n" + "=" * 100 + "\nB. Leakage from DP-FedAvg client updates (seed-42 global models)\n" + "=" * 100)
    L["exact_rate"] = L.cf_exact / L.records
    agg = {"updates": ("records", "size"), "exact_rate": ("exact_rate", "mean"), "best_cos": ("best_cos", "mean")}
    if "id_recall" in L:
        agg.update(id_recall=("id_recall", "mean"), id_precision=("id_precision", "mean"))
    LB = L.groupby(["path", "epsilon", "client_opt", "k"], sort=False).agg(**agg).reset_index()
    if "secrets" in L:
        sec = L[L.k == "1"].groupby(["path", "epsilon", "client_opt"], sort=False)[["secrets_right", "secrets"]].sum()
        LB = LB.merge(sec.reset_index(), on=["path", "epsilon", "client_opt"], how="left")
        LB.loc[LB.k != "1", ["secrets_right", "secrets"]] = np.nan
    print(LB.round(3).to_string(index=False))

    print("\n" + "=" * 100 + "\nC. Noise sweep: single-record SGD client on the non-private global model\n" + "=" * 100)
    W["exact_rate"] = W.cf_exact / W.records
    aggw = {"exact_rate": ("exact_rate", "mean"), "best_cos": ("best_cos", "mean"),
            "eps_if_used_all_training": ("eps_if_used_all_training", "first")}
    if "id_recall" in W:
        aggw.update(id_recall=("id_recall", "mean"))
    WB = W.groupby(["path", "sigma"], sort=False).agg(**aggw).reset_index()
    if "secrets" in W:
        WB = WB.merge(W.groupby(["path", "sigma"])[["secrets_right", "secrets"]].sum().reset_index(), on=["path", "sigma"])
    print(WB.round(4).to_string(index=False))

    C.to_csv(EVAL_DIR / f"dpfedavg_utility_summary{tag}.csv", index=False)
    LB.to_csv(EVAL_DIR / f"dpfedavg_leakage_summary{tag}.csv", index=False)
    WB.to_csv(EVAL_DIR / f"dpfedavg_noise_sweep_summary{tag}.csv", index=False)
    print(f"\nSaved: {EVAL_DIR}/dpfedavg_*_summary{tag}.csv\n       {OUT}/\n       {MODELS}/ (seed-42 global models)")


if __name__ == "__main__":
    main()