"""
phase2_robbing_fed.py  -  a MALICIOUS server: "Robbing the Fed" (Fowl et al., ICLR 2022) on agent logs
-------------------------------------------------------------------------------------------------------
Every earlier attack assumed an honest-but-curious server that sends clients the real model. Here the
server cheats: it keeps the model's shape (SecurityMLP 128-64-1, unchanged) but overwrites its weights
so that each client record ends up isolated in its own "bin", then reads records out of a normal
client update - even one that mixes many records.

The rigged model ("imprint" layer):
  layer 1: all 128 units use the SAME weight vector m (a fixed random direction), with increasing
           thresholds t_1 < ... < t_128 taken from quantiles of m.x on the server's own data (the
           validation set stands in for public data). Unit k switches on for records with m.x > t_k.
  layer 2/3: constant weights, so every layer-1 unit receives the same error signal per record.
  The update for unit k is then (weighted) the SUM of the records with m.x > t_k. Subtracting
  neighbouring units leaves only the records between two thresholds - usually one record - and
  dW / db of that difference is the record itself (same algebra as the closed-form attack).

Settings (the same batches for every setting; each batch = 1 of the 12 earlier victims + B-1 other
records from the same suite client):
  gradient            FedSGD: one gradient over the batch (model in eval mode)
  fedavg_sgd          FedAvg client, plain SGD lr 0.01, 3 local epochs, batch 64 (dropout off / on)
  fedavg_adam         FedAvg client = our system (Phase 1's train_local_model: Adam, 3 epochs, dropout)
  dpfedavg_eps8       DP-SGD inside the client (clip C=1, noise for eps=8, delta=1e-5; SGD)
  + honest baseline   the same gradient/update on the HONEST model, all-neuron closed-form attack
Metrics: records rebuilt exactly (MSE < 1e-4) / near (cos >= 0.99), mean best cosine; Path 2 also
names runs without knowing the batch (tie-aware) and extracts the victim's secrets (dictionary attack),
counted only when the victim's record itself was rebuilt (cosine >= 0.99).

Run from phase2_synthetic/:   python phase2_robbing_fed.py                (Phase 2, both paths)
                              python phase2_robbing_fed.py --agentdojo    (also Phase 1 / AgentDojo)
                              python phase2_robbing_fed.py --no-secrets   (skip the text encoder)
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch

import phase2_dpfedavg as D
import phase2_fedavg_attack as FA
import phase2_p2_train as P2
import phase2_train_fed as F
from phase2_aggregation import batch_gradient, first_layer, identify, score, twins
from phase2_common import EVAL_DIR, MODEL_DIR, OUT_DIR, ROOT, SUITES, SecurityMLP, load_features
from phase2_features import build_table

EXACT_MSE, NEAR_COS = 1e-4, 0.99
OUT = OUT_DIR / "robbing_fed"
VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"


# ============================================================ the rigged model
def malicious_state(honest_model, X_aux, seed=0):
    """Same architecture, imprint weights. Returns (state_dict, m, thresholds)."""
    m_model = copy.deepcopy(honest_model)
    W1, b1, W2, b2, W3, b3 = list(m_model.parameters())
    K, d = W1.shape
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(d, generator=g); m = m / m.norm()
    s = torch.tensor(X_aux) @ m
    qs = torch.quantile(s, torch.linspace(0, 1, K + 1)[1:-1])          # K-1 interior quantiles
    t = torch.cat([s.min().reshape(1) - 10 * s.std(), qs])             # unit 0 catches every record
    with torch.no_grad():
        W1.copy_(m.repeat(K, 1)); b1.copy_(-t)
        W2.fill_(1.0 / K); b2.fill_(1.0)                                  # every hidden-2 unit active
        W3.fill_(0.1 / W3.shape[1]); b3.zero_()                           # small logit: no saturation
    return copy.deepcopy(m_model.state_dict()), m, t


def rtf_recon(W, b, rel_eps=1e-9):
    """Neighbouring-unit differences -> one reconstruction per bin (units are in threshold order)."""
    W, b = W.double(), b.double()
    dW = torch.cat([W[:-1] - W[1:], W[-1:]]); db = torch.cat([b[:-1] - b[1:], b[-1:]])
    keep = db.abs() > rel_eps * db.abs().max().clamp_min(1e-30)
    return (dW[keep] / db[keep, None]).float().numpy()


def score_R(R, X):
    if len(R) == 0:
        return {"exact": 0, "near": 0, "best_cos": 0.0}
    mse = ((R[:, None, :] - X[None, :, :]) ** 2).mean(-1)
    cos = (R @ X.T) / (np.linalg.norm(R, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None] + 1e-12)
    return {"exact": int((mse.min(0) < EXACT_MSE).sum()), "near": int((cos.max(0) >= NEAR_COS).sum()),
            "best_cos": float(cos.max(0).mean())}


# ============================================================ one client update
def client_update(setting, model, gstate, X, y, seed):
    """Returns the first-layer (W, b) of what the server sees (gradient, or global - local)."""
    if setting == "gradient":
        m = SecurityMLP(input_dim=X.shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        m.load_state_dict(gstate)
        return first_layer(batch_gradient(m, torch.tensor(X), torch.tensor(y)))
    if setting.startswith("fedavg_sgd"):
        drop = 0.0 if setting.endswith("nodrop") else F.DROPOUT
        return first_layer(FA.real_update(model, gstate, X, y, "sgd", drop, seed))
    if setting == "fedavg_adam":
        return first_layer(FA.real_update(model, gstate, X, y, "system", F.DROPOUT, seed))
    if setting == "dpfedavg_eps8":
        delta, _ = D.local_dp(model, gstate, X, y, D.sigma_for(8.0, len(X)), D.CLIP, "sgd", False, seed, 0.0)
        return first_layer(delta)
    raise ValueError(setting)


SETTINGS = ["gradient", "fedavg_sgd_nodrop", "fedavg_sgd", "fedavg_adam", "dpfedavg_eps8"]


# ================================================================== one path
def run_path(name, honest, Xtr, ytr, suites, ids, X_aux, victim_idx, sizes, ident=None, scaler=None, cands=None):
    print(f"\n{'=' * 100}\n{name}   input dim {Xtr.shape[1]}\n{'=' * 100}", flush=True)
    hstate = copy.deepcopy(honest.state_dict())
    mstate, m, t = malicious_state(honest, X_aux)
    mal = copy.deepcopy(honest); mal.load_state_dict(mstate); mal.eval()
    s_tr = Xtr @ m.numpy()
    print(f"  imprint: {len(t)} bins from the server's data; train records per bin ~{len(Xtr) / len(t):.1f}")

    rng = np.random.default_rng(42)
    clients = {s: np.where(suites == s)[0] for s in SUITES}
    rows = []
    for B in sizes:
        for j, v in enumerate(victim_idx):
            pool = clients[suites[v]][clients[suites[v]] != v]
            idx = np.array([v] + list(rng.choice(pool, min(B - 1, len(pool)), replace=False)))
            X, y = Xtr[idx], ytr[idx]
            occupied = len(np.unique(np.searchsorted(t.numpy(), s_tr[idx], side="left")))
            for setting in SETTINGS:
                if setting.startswith("fedavg") and B == 1:
                    continue
                for server, gs, mdl in (("malicious", mstate, mal), ("honest", hstate, honest)):
                    if server == "honest" and setting != "gradient":
                        continue                          # honest FedAvg/DP results exist from earlier scripts
                    W, b = client_update(setting, mdl, gs, X, y, seed=500 + j)
                    if server == "malicious":
                        R = rtf_recon(W, b); r = score_R(R, X)
                    else:
                        r0, R = score(W, b, X); r = {"exact": r0["exact"], "near": r0["near"], "best_cos": r0["best_cos_mean"]}
                    row = {"path": name, "server": server, "setting": setting, "B": len(idx), "victim": j,
                           "bins_hit": occupied, **r}
                    if ident is not None and len(R):
                        row.update(identify(R, ident["scaler"], ident["E_all"], ident["ids_all"], list(ids[idx])))
                    if cands is not None:
                        # a secret counts only if the victim's record was actually rebuilt (cos >= 0.99);
                        # picking the "closest" of many noisy reconstructions would otherwise leak the answer
                        xv = X[0]
                        right, total = 0, len(cands[j])
                        if len(R):
                            c = (R @ xv) / (np.linalg.norm(R, axis=1) * np.linalg.norm(xv) + 1e-12)
                            if c.max() >= NEAR_COS:
                                right, total = D.secrets_eval(R[int(np.argmax(c))], scaler, cands[j])
                        row["secrets_right"], row["secrets"] = right, total
                    rows.append(row)
        print(f"  B={B:3d} done", flush=True)
    df = pd.DataFrame(rows)
    df["exact_rate"] = df.exact / df.B
    df["near_rate"] = df.near / df.B
    agg = {"batches": ("B", "size"), "exact_rate": ("exact_rate", "mean"), "near_rate": ("near_rate", "mean"),
           "best_cos": ("best_cos", "mean")}
    if "id_recall" in df:
        agg.update(id_recall=("id_recall", "mean"), id_precision=("id_precision", "mean"))
    S = df.groupby(["path", "server", "setting", "B"], sort=False).agg(**agg).reset_index()
    if "secrets" in df:
        sec = df.groupby(["path", "server", "setting", "B"], sort=False)[["secrets_right", "secrets"]].sum().reset_index()
        S = S.merge(sec, on=["path", "server", "setting", "B"])
    print(S.round(3).to_string(index=False))
    return df, S


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["1", "2"], choices=["1", "2"])
    ap.add_argument("--agentdojo", action="store_true", help="also run on Phase 1 (AgentDojo) data")
    ap.add_argument("--no-secrets", action="store_true")
    ap.add_argument("--stub", action="store_true", help="test encoder (NOT for results)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(42)
    victim_ids = [mm["run_id"] for mm in torch.load(VICTIMS_PT)["victim_meta"]]
    runs, summ = [], []

    if "1" in a.paths:
        d = load_features()
        h = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        h.load_state_dict(torch.load(MODEL_DIR / "federated_mlp_state_dict.pt")); h.eval()
        pos = {r: i for i, r in enumerate(d["run_id_train"])}
        r, s = run_path("Phase 2 / Path 1 (features)", h, d["X_train"].astype(np.float32), d["y_train"].astype(np.float32),
                        d["suite_train"], d["run_id_train"], d["X_val"].astype(np.float32),
                        [pos[v] for v in victim_ids], [1, 4, 16, 64])
        runs.append(r); summ.append(s)

    if "2" in a.paths:
        df = build_table()
        emb, _ = P2.load_embeddings("full")
        d = P2.make_data(df, 42, emb)
        ids_all = np.array(list(emb.keys())); E_all = np.stack([emb[i] for i in ids_all])
        print(f"Runs with an identical twin: {twins(E_all)} of {len(E_all)}")
        h = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        h.load_state_dict(torch.load(P2.P2_MODEL_DIR / "federated_full_state_dict.pt")); h.eval()
        cands = None
        if not a.no_secrets:
            from phase2_embed import MODEL_NAME, TEXTS_CSV, STEncoder, StubEncoder
            from phase2_p2_dp import secret_candidates
            enc = StubEncoder() if a.stub else STEncoder(MODEL_NAME)
            texts = pd.read_csv(TEXTS_CSV, keep_default_na=False).set_index("run_id")
            print("  embedding secret candidates for the 12 victims ...", flush=True)
            cands = secret_candidates(enc, texts, "full", victim_ids, np.random.default_rng(42))
        pos = {r: i for i, r in enumerate(d["run_id_train"])}
        r, s = run_path("Phase 2 / Path 2 (embeddings)", h, d["X_train"], d["y_train"], d["suite_train"], d["run_id_train"],
                        d["X_val"], [pos[v] for v in victim_ids], [1, 4, 16, 64],
                        ident={"scaler": d["scaler"], "E_all": E_all, "ids_all": ids_all}, scaler=d["scaler"], cands=cands)
        runs.append(r); summ.append(s)

    if a.agentdojo:
        from phase1_dpfedavg import load_phase1
        d = load_phase1()
        h = SecurityMLP(input_dim=d["X_train"].shape[1], hidden_dims=F.HIDDEN_DIMS, dropout=F.DROPOUT)
        p = ROOT / "phase1_agentdojo" / "models" / "dpfedavg" / "phase1_dpfedavg_epsinf_state_dict.pt"
        if p.exists():
            h.load_state_dict(torch.load(p))
        h.eval()
        rng = np.random.default_rng(42)
        vic = [int(i) for s in SUITES for i in rng.choice(np.where(d["suite_train"] == s)[0], 3, replace=False)]
        r, s = run_path("Phase 1 (AgentDojo)", h, d["X_train"], d["y_train"], d["suite_train"], d["run_id_train"],
                        d["X_val"], vic, [1, 4, 16, 64, 256])
        runs.append(r); summ.append(s)

    OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(runs, ignore_index=True).to_csv(OUT / f"robbing_fed_per_batch{a.tag}.csv", index=False)
    S = pd.concat(summ, ignore_index=True)
    S.round(4).to_csv(EVAL_DIR / f"robbing_fed_summary{a.tag}.csv", index=False)
    print("\n" + "=" * 100 + "\nSummary\n" + "=" * 100)
    print(S.round(3).to_string(index=False))
    print("\nexact/near_rate = share of the batch's records rebuilt exactly / at cosine >= 0.99")
    print("server=honest: the real model + all-neuron closed-form attack (for comparison)")
    print(f"\nSaved: {EVAL_DIR / f'robbing_fed_summary{a.tag}.csv'}\n       {OUT}/")


if __name__ == "__main__":
    main()