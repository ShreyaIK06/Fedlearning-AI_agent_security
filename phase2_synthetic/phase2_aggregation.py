"""
phase2_aggregation.py  -  Phase 2 defence experiment: does AGGREGATION stop the closed-form attack?
---------------------------------------------------------------------------------------------------
DP-SGD did not stop the closed-form attack (x = dW1[j] / db1[j]): one record's gradient was rebuilt
exactly at every epsilon. That attack assumes the server sees the gradient of ONE record. Here the
server instead sees an update that MIXES many records, as in real federated training:

  S1  FedSGD, batch B        one client sends the gradient of a mini-batch of B of its records
                             (B = 1, 2, 4, 8, 16, 32, 64; B = 1 is the earlier single-record attack)
  S2  FedSGD + secure agg.   the server only sees the size-weighted SUM over the 4 clients'
                             batch gradients (4 x B records mixed)
  S3  FedAvg local update    a client trains locally from the global model and sends its update
                             (global - local weights):
                               adam : Phase 1's train_local_model unchanged (Adam lr 1e-3,
                                      3 local epochs, batch 64) = exactly what our FL system does
                               sgd  : plain SGD (lr 0.01, 1 local epoch, batch 64) - the textbook
                                      FedAvg client, for comparison
  S4  FedAvg + secure agg.   the server only sees the size-weighted average of the 4 clients'
                             S3 updates (= what average_state_dicts produces)

The attack, on every update: for EVERY first-layer neuron j that the update touched, compute
x_j = dW1[j] / db1[j]. With one record each x_j is that record. With a batch, each x_j is a weighted
mix of the records that switched neuron j on - so it is still EXACT whenever only one record in the
batch activated neuron j ("neuron isolation", the reason batching alone is a weak defence).
Phase 1's version uses only the single neuron with the largest |db1|; both are reported.

Metrics per update (records = the records mixed into it):
  exact recovered   records rebuilt exactly by some neuron (MSE < 1e-4, Phase 1's definition)
  near recovered    records with some neuron reconstruction at cosine >= 0.99
  Phase 1 neuron    does the single argmax-|db1| reconstruction equal one of the records?
  Path 2 only - identification WITHOUT knowing the batch (no oracle): each neuron's reconstruction
  is matched to its nearest run among all 563 embeddings; confident matches (cosine >= 0.99) are the
  attacker's guesses of "who was in the batch". Reported as recall (batch records named) and
  precision (guesses that were really in the batch).
"exact/near recovered" use the true records to score (an UPPER bound, like the earlier attacks);
the Path 2 identification does not.

Models: the seed-42 FEDERATED models already attacked earlier (Path 1 features, Path 2 'full' view).
Batches are drawn from each client's TRAIN rows with a fixed RNG; model in .eval() for FedSGD
(as Phase 1's compute_gradient), local training uses train mode (dropout on), as in real FedAvg.

Run from phase2_synthetic/:   python phase2_aggregation.py                (both paths)
                              python phase2_aggregation.py --paths 1      (Path 1 only, fast)
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import phase2_train_fed as F
from phase2_common import EVAL_DIR, MODEL_DIR, OUT_DIR, SUITES, SecurityMLP, load_features, train_local_model
from common.gradient_leakage import closed_form_first_layer_attack, compute_gradient  # Phase 1, unchanged

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SECAGG_BATCH = [1, 4, 16]
N_BATCHES = 20                 # mini-batches per client per batch size
EXACT_MSE = 1e-4               # Phase 1's "exactly reconstructed"
NEAR_COS = 0.99
ID_COS = 0.99                  # Path 2: confidence needed before the attacker names a run
TIE_EPS = 1e-6                 # Path 2: runs this close to the best match count as the same content
SGD_LR, SGD_EPOCHS = 0.01, 1
OUT = OUT_DIR / "aggregation"


# ============================================================ update + attack helpers
def first_layer(params):
    W, b = params[0], params[1]
    assert W.dim() == 2 and b.dim() == 1 and W.shape[0] == b.shape[0], "first two parameters must be W1, b1"
    return W.detach().double(), b.detach().double()


def batch_gradient(model, X, y):
    """Gradient of the mean BCE loss over a batch, model in eval mode (Phase 1's compute_gradient, batched)."""
    model.eval()
    params = [p for p in model.parameters()]
    loss = nn.BCEWithLogitsLoss()(model(X).view(-1), y.view(-1))
    return [g.detach() for g in torch.autograd.grad(loss, params)]


def all_neuron_recon(W, b, rel_eps=1e-6):
    """x_j = W[j] / b[j] for every neuron the update touched."""
    keep = b.abs() > rel_eps * b.abs().max().clamp_min(1e-30)
    return (W[keep] / b[keep, None]).float().numpy(), int(keep.sum())


def score(W, b, X):
    """Exact / near recovery of the records X (n, d) from one update's first layer."""
    R, n_neurons = all_neuron_recon(W, b)
    out = {"records": len(X), "neurons_used": n_neurons}
    if n_neurons == 0:
        return {**out, "exact": 0, "near": 0, "p1_neuron_exact": False, "best_cos_mean": 0.0}, R
    mse = ((R[:, None, :] - X[None, :, :]) ** 2).mean(-1)                         # neurons x records
    cosm = (R @ X.T) / (np.linalg.norm(R, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None] + 1e-12)
    j = int(b.abs().argmax())
    x1 = (W[j] / b[j]).float().numpy()
    out.update({"exact": int((mse.min(0) < EXACT_MSE).sum()),
                "near": int((cosm.max(0) >= NEAR_COS).sum()),
                "p1_neuron_exact": bool((((x1[None] - X) ** 2).mean(-1) < EXACT_MSE).any()),
                "best_cos_mean": float(cosm.max(0).mean())})
    return out, R


def identify(R, scaler, E_all, ids_all, batch_ids):
    """Path 2: name runs from the reconstructions alone (no knowledge of the batch)."""
    E = scaler.inverse_transform(R)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    An = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)
    S = En @ An.T
    conf = S.max(1)
    batch = set(batch_ids)
    # A reconstruction "names" every run tied for its best match: repeat runs with identical text have
    # identical embeddings, so the attacker cannot (and need not) tell them apart - same content leaked.
    named, n_guess, n_good = set(), 0, 0
    seen = set()
    for i in np.where(conf >= ID_COS)[0]:
        tied = frozenset(ids_all[S[i] >= conf[i] - TIE_EPS])
        if tied in seen:
            continue
        seen.add(tied); n_guess += 1
        n_good += int(bool(tied & batch))
        named |= tied & batch
    return {"id_guesses": n_guess, "id_correct": len(named), "id_recall": len(named) / len(batch_ids),
            "id_precision": (n_good / n_guess) if n_guess else float("nan")}


def twins(E_all):
    """How many runs share an identical embedding with another run (identical trajectory text)."""
    An = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)
    S = An @ An.T
    np.fill_diagonal(S, -1)
    return int((S.max(1) >= 1 - TIE_EPS).sum())


def local_update(model, gstate, X, y, mode, seed):
    """Client update = global - local weights, as the server receives it."""
    dim = X.shape[1]
    if mode == "adam":
        st, _, _ = train_local_model(global_state=gstate, X=X, y=y, input_dim=dim, hidden_dims=F.HIDDEN_DIMS,
                                     dropout=F.DROPOUT, local_epochs=F.LOCAL_EPOCHS, lr=F.LOCAL_LR,
                                     batch_size=F.LOCAL_BATCH_SIZE, seed=seed)
        local = SecurityMLP(input_dim=dim, hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        local.load_state_dict(st)
    else:
        torch.manual_seed(seed)
        local = copy.deepcopy(model); local.train()
        opt = torch.optim.SGD(local.parameters(), lr=SGD_LR)
        Xt, yt = torch.tensor(X), torch.tensor(y)
        for _ in range(SGD_EPOCHS):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), F.LOCAL_BATCH_SIZE):
                idx = perm[i:i + F.LOCAL_BATCH_SIZE]
                opt.zero_grad(); nn.BCEWithLogitsLoss()(local(Xt[idx]).view(-1), yt[idx]).backward(); opt.step()
    Wg, bg = first_layer(list(model.parameters()))
    Wl, bl = first_layer(list(local.parameters()))
    return Wg - Wl, bg - bl


# ================================================================== one path
def run_path(name, model, Xtr, ytr, suites, ids, ident=None):
    rng = np.random.default_rng(42)
    clients = {s: np.where(suites == s)[0] for s in SUITES}
    sizes = {s: len(ix) for s, ix in clients.items()}
    print(f"\n{'=' * 100}\n{name}   input dim {Xtr.shape[1]}, client train sizes {sizes}\n{'=' * 100}")

    # sanity: batch of 1 = Phase 1's compute_gradient + closed-form attack
    i0 = clients[SUITES[0]][0]
    x0, y0 = torch.tensor(Xtr[i0:i0 + 1]), torch.tensor(ytr[i0:i0 + 1])
    W, b = first_layer(batch_gradient(model, x0, y0))
    ref = closed_form_first_layer_attack(compute_gradient(model, x0, y0)).squeeze(0).numpy()
    j = int(b.abs().argmax())
    assert np.allclose((W[j] / b[j]).float().numpy(), ref, atol=1e-4), "batched gradient disagrees with Phase 1's"

    rows = []

    def add(setting, B, client, W, b, idx):
        r, R = score(W, b, Xtr[idx])
        r.update({"setting": setting, "batch_per_client": B, "client": client})
        if ident is not None:
            r.update(identify(R, ident["scaler"], ident["E_all"], ident["ids_all"], list(ids[idx])))
        rows.append(r)

    # S1 FedSGD, one client, batch B
    for B in BATCH_SIZES:
        for s in SUITES:
            for _ in range(N_BATCHES):
                idx = rng.choice(clients[s], size=min(B, sizes[s]), replace=False)
                W, b = first_layer(batch_gradient(model, torch.tensor(Xtr[idx]), torch.tensor(ytr[idx])))
                add("S1 FedSGD", B, s, W, b, idx)
    # S2 FedSGD + secure aggregation (size-weighted sum over the 4 clients)
    for B in SECAGG_BATCH:
        for _ in range(N_BATCHES):
            Ws, bs, all_idx = 0, 0, []
            for s in SUITES:
                idx = rng.choice(clients[s], size=min(B, sizes[s]), replace=False)
                W, b = first_layer(batch_gradient(model, torch.tensor(Xtr[idx]), torch.tensor(ytr[idx])))
                w = sizes[s] / sum(sizes.values())
                Ws, bs = Ws + w * W, bs + w * b
                all_idx.extend(idx)
            add("S2 FedSGD + SecAgg", B, "all 4", Ws, bs, np.array(all_idx))
    # S3 / S4 FedAvg local updates (whole client = the records mixed)
    gstate = copy.deepcopy(model.state_dict())
    for mode in ("adam", "sgd"):
        Ws, bs = 0, 0
        for k, s in enumerate(SUITES):
            idx = clients[s]
            W, b = local_update(model, gstate, Xtr[idx], ytr[idx], mode, seed=42 + k)
            add(f"S3 FedAvg update ({mode})", "whole client", s, W, b, idx)
            w = sizes[s] / sum(sizes.values())
            Ws, bs = Ws + w * W, bs + w * b
        add(f"S4 FedAvg + SecAgg ({mode})", "whole client", "all 4", Ws, bs, np.arange(len(Xtr))[np.isin(suites, SUITES)])

    df = pd.DataFrame(rows)
    df["exact_rate"] = df.exact / df.records
    df["near_rate"] = df.near / df.records
    g = df.groupby(["setting", "batch_per_client"], sort=False)
    summ = g.agg(updates=("records", "size"), records_per_update=("records", "mean"),
                 exact_rate=("exact_rate", "mean"), near_rate=("near_rate", "mean"),
                 p1_neuron_exact=("p1_neuron_exact", "mean"), neurons_used=("neurons_used", "mean"),
                 best_cos=("best_cos_mean", "mean"))
    if ident is not None:
        summ = summ.join(g.agg(id_recall=("id_recall", "mean"), id_precision=("id_precision", "mean")))
    summ = summ.reset_index()
    summ.insert(0, "path", name)
    print(summ.round(3).to_string(index=False))
    return df.assign(path=name), summ


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["1", "2"], choices=["1", "2"])
    a = ap.parse_args()
    torch.manual_seed(42)
    all_runs, all_summ = [], []

    if "1" in a.paths:
        d = load_features()
        m = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        m.load_state_dict(torch.load(MODEL_DIR / "federated_mlp_state_dict.pt")); m.eval()
        r, s = run_path("Path 1 (features)", m, d["X_train"].astype(np.float32), d["y_train"].astype(np.float32),
                        d["suite_train"], d["run_id_train"])
        all_runs.append(r); all_summ.append(s)

    if "2" in a.paths:
        import phase2_p2_train as P2
        from phase2_features import build_table
        emb, _ = P2.load_embeddings("full")
        d = P2.make_data(build_table(), 42, emb)
        ids_all = np.array(list(emb.keys())); E_all = np.stack([emb[i] for i in ids_all])
        print(f"\nRuns whose embedding is identical to another run's (identical trajectory text): "
              f"{twins(E_all)} of {len(E_all)}")
        m = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        m.load_state_dict(torch.load(P2.P2_MODEL_DIR / "federated_full_state_dict.pt")); m.eval()
        r, s = run_path("Path 2 (embeddings, full)", m, d["X_train"], d["y_train"], d["suite_train"],
                        d["run_id_train"], ident={"scaler": d["scaler"], "E_all": E_all, "ids_all": ids_all})
        all_runs.append(r); all_summ.append(s)

    OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_runs, ignore_index=True).to_csv(OUT / "aggregation_per_update.csv", index=False)
    S = pd.concat(all_summ, ignore_index=True)
    S.round(4).to_csv(EVAL_DIR / "aggregation_defence_summary.csv", index=False)
    print("\nexact_rate = share of the mixed records rebuilt exactly by some neuron (oracle-scored upper bound)")
    print("p1_neuron_exact = share of updates where Phase 1's single-neuron attack still returns a real record")
    print(f"\nSaved: {EVAL_DIR / 'aggregation_defence_summary.csv'}\n       {OUT / 'aggregation_per_update.csv'}")


if __name__ == "__main__":
    main()