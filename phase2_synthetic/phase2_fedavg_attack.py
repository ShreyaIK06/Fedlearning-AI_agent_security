"""
phase2_fedavg_attack.py  -  Phase 2: can a STRONGER attack break multi-step FedAvg updates?
--------------------------------------------------------------------------------------------
phase2_aggregation.py found that the closed-form attack recovers NOTHING from our FedAvg client
updates (Adam, 3 local epochs). But closed-form assumes the update is a plain gradient. Here we run
the standard attack on FedAvg updates instead (Geiping et al. 2020, "Inverting Gradients", FedAvg
setting): the attacker

  1. starts from random guesses X' for the client's k records (labels assumed KNOWN - strong attacker),
  2. SIMULATES the client's local training on X' from the same global model, with the same
     optimizer, learning rate, epochs and batch size (differentiably),
  3. compares the simulated weight change with the real one (cosine distance, as IG),
  4. moves X' to make them match (Adam lr 0.05, Phase 1's IG setting).

Settings (both paths, seed-42 federated models):
  client data k     1 (12 non-breach records, 3 per suite, each as its own client),
                    4 and 8 (2 random subsets per suite), and the whole client (~90, one per suite).
                    Every set holds >= 1 non-breach record: under Phase 1's train_local_model,
                    pos_weight = n_neg / max(n_pos, 1), so a client holding ONLY breach records gets
                    pos_weight 0, learns nothing and sends a zero update (nothing to attack).
  client optimizer  system = Phase 1's train_local_model UNCHANGED (what our FL system does)
                    sgd    = plain SGD lr 0.01, same epochs/batch (the textbook FedAvg client)
  client dropout    on (0.2, as trained) - the attacker cannot know the dropout masks, so it
                    simulates without dropout; and off (upper bound: simulation can be exact)

The attacker's simulator for "system" is chosen automatically: before attacking, the script checks
which simulator (Adam or SGD, with or without per-client pos_weight) reproduces train_local_model
on known data, and stops if none does (then Phase 1's fedavg_utils.py must be checked).

Scores per attack (k records, guesses matched to true records by the Hungarian method):
  IG exact / near   records rebuilt exactly (MSE < 1e-4) / at cosine >= 0.99
  CF exact          all-neuron closed-form on the same update (phase2_aggregation.py), for comparison
  fit               how well the attacker's simulated update matched the real one (cosine)
  oracle fit        the same with the TRUE records plugged in = the best fit possible;
                    < 1 means the simulation itself cannot be exact (dropout masks, batch order)
  Path 2 only: id   runs named from the reconstructions alone (nearest of 563, tie-aware, cos >= 0.99)

Run from phase2_synthetic/:   python phase2_fedavg_attack.py               (both paths, ~1-1.5 h)
  stronger attacker on our system's (Adam) clients:
                              python phase2_fedavg_attack.py --client system --sizes 1 4 8 --iters 2000 --restarts 3 --tag _strong
                              python phase2_fedavg_attack.py --quick       (k = 1 and 4, 200 iters)
"""

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call

import phase2_train_fed as F
from phase2_aggregation import TIE_EPS, first_layer, identify, score, twins
from phase2_common import EVAL_DIR, MODEL_DIR, OUT_DIR, SUITES, SecurityMLP, load_features, train_local_model

ATT_ITERS, ATT_LR = 500, 0.05          # Phase 1's Inverting Gradients settings
SGD_LR = 0.01
EXACT_MSE, NEAR_COS = 1e-4, 0.99
VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"
OUT = OUT_DIR / "fedavg_attack"


# ============================================================ local training: real and simulated
def client_pos_weight(y):
    """Exactly as Phase 1's train_local_model: computed ONCE over the client's whole local data.
    NB: a client holding only breach records gets pos_weight 0 -> zero loss -> it sends a zero update."""
    n_pos = int(np.asarray(y).sum()); n_neg = len(y) - n_pos
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def bce(logits, y, pw):
    return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, y) if pw is not None else nn.BCEWithLogitsLoss()(logits, y)


def simulate(model, X, y, sim, create_graph):
    """Differentiable copy of a client's local training from the global weights (dropout off).
    Returns the update (global - local) per parameter."""
    names = [n for n, _ in model.named_parameters()]
    p0 = {n: p.detach() for n, p in model.named_parameters()}
    p = {n: v.clone().requires_grad_(True) for n, v in p0.items()}
    m = {n: torch.zeros_like(v) for n, v in p0.items()}
    v2 = {n: torch.zeros_like(v) for n, v in p0.items()}
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    model.eval()
    pw = client_pos_weight(y.detach().numpy()) if sim["pw"] == "ratio" else None
    bs = min(sim["batch"], len(X))
    for _ in range(sim["epochs"]):
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs], y[i:i + bs]
            loss = bce(functional_call(model, p, (xb,)).view(-1), yb, pw)
            g = torch.autograd.grad(loss, [p[n] for n in names], create_graph=create_graph,
                                    allow_unused=True)
            t += 1
            new = {}
            for n, gn in zip(names, g):
                gn = torch.zeros_like(p[n]) if gn is None else gn
                if sim["opt"] == "adam":
                    m[n] = b1 * m[n] + (1 - b1) * gn
                    v2[n] = b2 * v2[n] + (1 - b2) * gn * gn
                    mh, vh = m[n] / (1 - b1 ** t), v2[n] / (1 - b2 ** t)
                    new[n] = p[n] - sim["lr"] * mh / ((vh + 1e-24).sqrt() + eps)
                else:
                    new[n] = p[n] - sim["lr"] * gn
            p = new
            if not create_graph:
                p = {n: v.detach().requires_grad_(True) for n, v in p.items()}
    return [p0[n] - p[n] for n in names]


def real_update(model, gstate, X, y, client_opt, dropout, seed):
    """What the server receives from a client: global - local weights after real local training."""
    dim = X.shape[1]
    if client_opt == "system":
        st, _, _ = train_local_model(global_state=gstate, X=X, y=y, input_dim=dim, hidden_dims=F.HIDDEN_DIMS,
                                     dropout=dropout, local_epochs=F.LOCAL_EPOCHS, lr=F.LOCAL_LR,
                                     batch_size=F.LOCAL_BATCH_SIZE, seed=seed)
        local = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=dropout)
        local.load_state_dict(st)
    else:
        torch.manual_seed(seed)
        local = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=dropout)
        local.load_state_dict(gstate); local.train()
        opt = torch.optim.SGD(local.parameters(), lr=SGD_LR)
        Xt, yt = torch.tensor(X), torch.tensor(y)
        for _ in range(F.LOCAL_EPOCHS):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), F.LOCAL_BATCH_SIZE):
                idx = perm[i:i + F.LOCAL_BATCH_SIZE]
                opt.zero_grad(); nn.BCEWithLogitsLoss()(local(Xt[idx]).view(-1), yt[idx]).backward(); opt.step()
    gm = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=0.0); gm.load_state_dict(gstate)
    return [a.detach() - b.detach() for a, b in zip(gm.parameters(), local.parameters())]


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(flat(a).detach().double(), flat(b).detach().double(), dim=0))


def pick_system_simulator(model, gstate, X, y):
    """Which simulator reproduces Phase 1's train_local_model (dropout off, full-batch data)?"""
    real = real_update(model, gstate, X, y, "system", 0.0, seed=0)
    best = None
    for opt in ("adam", "sgd"):
        for pw in (None, "ratio"):
            sim = {"opt": opt, "lr": F.LOCAL_LR, "epochs": F.LOCAL_EPOCHS, "batch": F.LOCAL_BATCH_SIZE, "pw": pw}
            d = simulate(model, torch.tensor(X), torch.tensor(y), sim, create_graph=False)
            err = float((flat(d).detach() - flat(real)).abs().max() / (flat(real).abs().max() + 1e-12))
            print(f"    simulator {opt:4s} pos_weight={str(pw):5s}: max relative error {err:.2e}")
            if best is None or err < best[0] - 1e-9:
                best = (err, sim)
    if best[0] > 1e-3:
        raise SystemExit("No simulator reproduces train_local_model - send Phase 1's src/common/fedavg_utils.py "
                         "so the attacker's simulation can be matched to it.")
    return best[1]


# ============================================================ the attack
def attack(model, delta, y, k, dim, sim, iters, seed, restarts=1):
    """Inverting Gradients on a FedAvg update. Learning-rate drops at 3/8, 5/8, 7/8 of the run
    (Geiping et al.'s schedule); with restarts > 1 the best-fitting of several random starts is kept."""
    yt = torch.tensor(y)
    target = flat(delta).detach()
    best, best_X = -2.0, None
    for r in range(restarts):
        torch.manual_seed(seed + 1000 * r)
        Xg = torch.randn(k, dim, requires_grad=True)
        opt = torch.optim.Adam([Xg], lr=ATT_LR)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [iters * 3 // 8, iters * 5 // 8, iters * 7 // 8], 0.1)
        for _ in range(iters):
            opt.zero_grad()
            d = flat(simulate(model, Xg, yt, sim, create_graph=True))
            c = torch.nn.functional.cosine_similarity(d, target, dim=0)
            (1 - c).backward()
            opt.step(); sched.step()
            cv = float(c.detach())
            if cv > best:
                best, best_X = cv, Xg.detach().clone()
    return best_X.numpy(), best


def match(Xr, X):
    mse = ((Xr[:, None] - X[None]) ** 2).mean(-1)
    r, c = linear_sum_assignment(mse)
    cos = (Xr[r] * X[c]).sum(1) / (np.linalg.norm(Xr[r], axis=1) * np.linalg.norm(X[c], axis=1) + 1e-12)
    return int((mse[r, c] < EXACT_MSE).sum()), int((cos >= NEAR_COS).sum()), float(cos.mean())


# ================================================================== one path
def run_path(name, model, Xtr, ytr, suites, ids, sizes_k, iters, client_opts, restarts, ident=None):
    gstate = copy.deepcopy(model.state_dict())
    dim = Xtr.shape[1]
    rng = np.random.default_rng(42)
    clients = {s: np.where(suites == s)[0] for s in SUITES}
    pos = {r: i for i, r in enumerate(ids)}
    print(f"\n{'=' * 100}\n{name}   input dim {dim}\n{'=' * 100}")

    # known data for the simulator check: UNEQUAL class counts (6 non-breach + 2 breach), so that
    # "with pos_weight" and "without" give different updates and can be told apart
    neg, posi = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
    mixed = np.concatenate([rng.choice(neg, 6, replace=False), rng.choice(posi, 2, replace=False)])
    print("  Matching the attacker's simulator to Phase 1's train_local_model:")
    sys_sim = pick_system_simulator(model, gstate, Xtr[mixed], ytr[mixed])
    sims = {"system": sys_sim,
            "sgd": {"opt": "sgd", "lr": SGD_LR, "epochs": F.LOCAL_EPOCHS, "batch": F.LOCAL_BATCH_SIZE, "pw": None}}
    print(f"  -> system client = {sys_sim['opt']}, pos_weight={sys_sim['pw']}, lr {sys_sim['lr']}, "
          f"{sys_sim['epochs']} epochs, batch {sys_sim['batch']}")

    # Client data sets. Every set holds at least one NON-breach record: a client with only breach
    # records sends a zero update under Phase 1's pos_weight rule (nothing to attack, nothing leaks).
    def draw(s, k):
        while True:
            idx = rng.choice(clients[s], size=k, replace=False)
            if (ytr[idx] == 0).any():
                return idx
    groups = {}
    if 1 in sizes_k:
        groups[1] = [np.array([i]) for s in SUITES
                     for i in rng.choice(clients[s][ytr[clients[s]] == 0], size=3, replace=False)]
    for k in [k for k in sizes_k if k not in (1, "client")]:
        groups[k] = [draw(s, k) for s in SUITES for _ in range(2)]
    if "client" in sizes_k:
        groups["client"] = [clients[s] for s in SUITES]

    rows, t0 = [], time.time()
    n_total = sum(len(v) for v in groups.values()) * 2 * len(client_opts)
    for client_opt in client_opts:
        for dropout in (F.DROPOUT, 0.0):
            for k, sets in groups.items():
                for j, idx in enumerate(sets):
                    X, y = Xtr[idx], ytr[idx]
                    delta = real_update(model, gstate, X, y, client_opt, dropout, seed=1000 + j)
                    assert float(flat(delta).abs().max()) > 0, "zero update - should not happen with a non-breach record"
                    sim = sims[client_opt]
                    oracle = cosine(simulate(model, torch.tensor(X), torch.tensor(y), sim, create_graph=False), delta)
                    Xr, fit = attack(model, delta, y, len(idx), dim, sim, iters, seed=42 + j, restarts=restarts)
                    ex, near, mcos = match(Xr, X)
                    W, b = first_layer(delta)
                    cf, _ = score(W, b, X)
                    r = {"client_opt": client_opt, "dropout": dropout, "k": k, "set": j, "records": len(idx),
                         "ig_exact": ex, "ig_near": near, "ig_mean_cos": mcos, "fit": fit, "oracle_fit": oracle,
                         "cf_exact": cf["exact"]}
                    if ident is not None:
                        r.update(identify(Xr, ident["scaler"], ident["E_all"], ident["ids_all"], list(ids[idx])))
                    rows.append(r)
                    done = len(rows)
                    print(f"  [{done:3d}/{n_total}] client={client_opt:6s} dropout={dropout:.1f} k={str(k):6s} "
                          f"IG exact {ex}/{len(idx)} near {near} cos {mcos:.3f} | fit {fit:.4f} (oracle {oracle:.4f}) | "
                          f"CF exact {cf['exact']}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df["ig_exact_rate"] = df.ig_exact / df.records
    df["ig_near_rate"] = df.ig_near / df.records
    df["cf_exact_rate"] = df.cf_exact / df.records
    agg = {"attacks": ("records", "size"), "records": ("records", "mean"),
           "ig_exact_rate": ("ig_exact_rate", "mean"), "ig_near_rate": ("ig_near_rate", "mean"),
           "ig_mean_cos": ("ig_mean_cos", "mean"), "fit": ("fit", "mean"), "oracle_fit": ("oracle_fit", "mean"),
           "cf_exact_rate": ("cf_exact_rate", "mean")}
    if ident is not None:
        agg.update({"id_recall": ("id_recall", "mean"), "id_precision": ("id_precision", "mean")})
    summ = df.assign(k=df.k.astype(str)).groupby(["client_opt", "dropout", "k"], sort=False).agg(**agg).reset_index()
    summ.insert(0, "path", name)
    print("\n" + summ.round(3).to_string(index=False))
    return df.assign(path=name), summ


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["1", "2"], choices=["1", "2"])
    ap.add_argument("--iters", type=int, default=ATT_ITERS)
    ap.add_argument("--restarts", type=int, default=1, help="random starts per attack (best fit kept)")
    ap.add_argument("--client", default="both", choices=["system", "sgd", "both"])
    ap.add_argument("--sizes", nargs="+", default=["1", "4", "8", "client"])
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--quick", action="store_true", help="k = 1 and 4 only, 200 iterations")
    a = ap.parse_args()
    sizes_k = [1, 4] if a.quick else [s if s == "client" else int(s) for s in a.sizes]
    iters = 200 if a.quick else a.iters
    client_opts = ["system", "sgd"] if a.client == "both" else [a.client]
    torch.manual_seed(42)
    runs, summs = [], []

    if "1" in a.paths:
        d = load_features()
        m = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        m.load_state_dict(torch.load(MODEL_DIR / "federated_mlp_state_dict.pt")); m.eval()
        r, s = run_path("Path 1 (features)", m, d["X_train"].astype(np.float32), d["y_train"].astype(np.float32),
                        d["suite_train"], d["run_id_train"], sizes_k, iters, client_opts, a.restarts)
        runs.append(r); summs.append(s)

    if "2" in a.paths:
        import phase2_p2_train as P2
        from phase2_features import build_table
        emb, _ = P2.load_embeddings("full")
        d = P2.make_data(build_table(), 42, emb)
        ids_all = np.array(list(emb.keys())); E_all = np.stack([emb[i] for i in ids_all])
        print(f"Runs with an identical twin: {twins(E_all)} of {len(E_all)}")
        m = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        m.load_state_dict(torch.load(P2.P2_MODEL_DIR / "federated_full_state_dict.pt")); m.eval()
        r, s = run_path("Path 2 (embeddings, full)", m, d["X_train"], d["y_train"], d["suite_train"],
                        d["run_id_train"], sizes_k, iters, client_opts, a.restarts,
                        ident={"scaler": d["scaler"], "E_all": E_all, "ids_all": ids_all})
        runs.append(r); summs.append(s)

    OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = ("_quick" if a.quick else "") + a.tag
    pd.concat(runs, ignore_index=True).to_csv(OUT / f"fedavg_attack_per_update{tag}.csv", index=False)
    S = pd.concat(summs, ignore_index=True)
    S.round(4).to_csv(EVAL_DIR / f"fedavg_attack_summary{tag}.csv", index=False)
    print("\n" + "=" * 100 + "\nSummary\n" + "=" * 100)
    print(S.round(3).to_string(index=False))
    print("\nig_exact/near_rate = records rebuilt by Inverting Gradients on the FedAvg update (labels known)")
    print("fit / oracle_fit = attacker's simulated update vs the real one (oracle = true records plugged in)")
    print(f"\nSaved: {EVAL_DIR / f'fedavg_attack_summary{tag}.csv'}\n       {OUT / f'fedavg_attack_per_update{tag}.csv'}")


if __name__ == "__main__":
    main()