"""
phase2_p2_dp.py  -  Phase 2, Path 2 step 4: DP-SGD on the EMBEDDING models
---------------------------------------------------------------------------
Two questions, same as Path 1's DP stage (phase2_dp.py), now on 384-dim trajectory embeddings:
  A. What does DP-SGD cost in accuracy?   (5 seeds, paired per seed, Path 1's split per seed)
  B. Does DP-SGD stop the leak?  On the seed-42 models: is the closed-form reconstruction still
     exact, can the attacker still identify the run, and can they still extract the secrets
     (account numbers, emails, passport) by the same dictionary attack as phase2_p2_attacks.py?

Settings = the FAIR protocol from Path 1 (phase2_dp.py --lr 1e-2 --batch 64 --epochs 60 --baseline-tuned):
  DP models: centralized SecurityMLP, Opacus, eps 1/3/8, C=1.0, delta=1/N, lr 1e-2, batch 64, 60 epochs
  eps=inf:   non-private model with the SAME lr / batch / max epochs (patience 25, warm-up 20)
  Checkpoints: best val ROC-AUC after warm-up 20.
  --phase1-settings  uses Phase 1's DP settings instead (lr 1e-3, batch 256, 30 epochs; parity check)

Only the DP training loop and the attack code are reused (phase2_dp.py, Phase 1's gradient_leakage.py,
phase2_p2_attacks.py's secret helpers) - nothing is reimplemented differently.
Speed: candidate texts for the secret test depend only on the victim, not the model, so they are
embedded ONCE per view and reused for all 4 models.

Run from phase2_synthetic/:   python phase2_p2_dp.py
Test without the model:       python phase2_p2_dp.py --stub --seeds 42     (NOT for results)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import torch

import phase2_dp as D
import phase2_p2_train as P2
from phase2_common import EVAL_DIR, OUT_DIR, ROOT
from phase2_embed import MODEL_NAME, TEXTS_CSV, STEncoder, StubEncoder
from phase2_features import build_table
from phase2_p2_attacks import EXACT_TOL, N_CANDIDATES, PATTERNS, canon, lookalikes, swap
from common.gradient_leakage import closed_form_first_layer_attack, compute_gradient  # Phase 1, unchanged

warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook is firing")

SEEDS = [42, 0, 1, 2, 3]
EPS = [1.0, 3.0, 8.0]
VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"
P2DP_OUT = OUT_DIR / "path2_dp"
P2DP_MODELS = ROOT / "models" / "phase2_path2_dp"


def fast_encode(enc, texts, batch_size=64):
    """All chunks of all texts through the model in batches (same pooling as phase2_embed)."""
    if not hasattr(enc, "m"):
        return enc.encode(texts)
    parts, owner = [], []
    for i, t in enumerate(texts):
        for c in enc.chunks(t):
            parts.append(c); owner.append(i)
    E = enc.m.encode(parts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    owner = np.array(owner)
    out = np.stack([E[owner == i].mean(axis=0) for i in range(len(texts))])
    return (out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def secret_candidates(enc, texts, view, vids, rng):
    """Per victim: list of (kind, true value, candidate list, candidate embeddings). Built once per view."""
    seen = {k: set() for k in PATTERNS}
    for col in ("actions", "full"):
        for t in texts[col]:
            for k, p in PATTERNS.items():
                seen[k].update(canon(k, m) for m in p.findall(t))
    out = []
    for rid in vids:
        vtext = texts.loc[rid, view]
        items = []
        for k, true in sorted({(k, canon(k, m)) for k, p in PATTERNS.items() for m in p.findall(vtext)}):
            pool = sorted(seen[k] - {true})
            cands = [true] + list(rng.permutation(pool))[:N_CANDIDATES // 2]
            cands += sorted(lookalikes(k, true, rng, N_CANDIDATES - len(cands)) - set(cands))
            cands = cands[:N_CANDIDATES]
            items.append((k, true, cands, fast_encode(enc, [swap(vtext, k, true, c) for c in cands])))
        out.append(items)
    return out


def leak_eval(model, scaler, emb, ids_all, E_all, vids, Vy, cand):
    model.eval()
    res = {"cf_exact": 0, "zero_grad": 0, "identified": 0, "secrets": 0, "secrets_right": 0, "max_logit": 0.0}
    for i, rid in enumerate(vids):
        x = torch.tensor(scaler.transform(emb[rid][None]).astype(np.float32))
        g = compute_gradient(model, x, Vy[i].reshape(1))
        with torch.no_grad():
            res["max_logit"] = max(res["max_logit"], abs(float(model(x).item())))
        if all(float(t.abs().max()) == 0 for t in g):
            res["zero_grad"] += 1
        x_cf = closed_form_first_layer_attack(g).numpy()
        res["cf_exact"] += int(np.max(np.abs(x_cf - x.numpy())) < EXACT_TOL)
        e = scaler.inverse_transform(x_cf)[0]
        sims = E_all @ e / (np.linalg.norm(E_all, axis=1) * np.linalg.norm(e) + 1e-12)
        res["identified"] += int((sims > sims[ids_all == rid][0]).sum() == 0)
        for k, true, cands, ce in cand[i]:
            res["secrets"] += 1
            res["secrets_right"] += int(cands[int(np.argmax(ce @ e))] == true)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["actions", "full"], choices=["actions", "full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--phase1-settings", action="store_true")
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    if 42 not in a.seeds:
        raise SystemExit("seed 42 must be included (part B attacks the seed-42 models)")

    if a.phase1_settings:
        D.LR, D.BATCH_SIZE, D.EPOCHS, D.BASELINE_TUNED = 1e-3, 256, 30, False
        tag = "_phase1settings"
    else:
        D.LR, D.BATCH_SIZE, D.EPOCHS, D.BASELINE_TUNED = 1e-2, 64, 60, True
        tag = ""
    print(f"DP settings: lr={D.LR:g} batch={D.BATCH_SIZE} epochs={D.EPOCHS} C={D.MAX_GRAD_NORM} delta=1/N "
          f"warm-up={D.WARMUP}; non-private baseline {'tuned the same way' if D.BASELINE_TUNED else 'Phase 1 settings'}")

    df = build_table()
    texts = pd.read_csv(TEXTS_CSV, keep_default_na=False).set_index("run_id")
    V = torch.load(VICTIMS_PT)
    vids = [m["run_id"] for m in V["victim_meta"]]
    enc = StubEncoder() if a.stub else STEncoder(MODEL_NAME)

    util_rows, leak_rows = [], []
    for view in a.views:
        emb, _ = P2.load_embeddings(view)
        ids_all = np.array(list(emb.keys())); E_all = np.stack([emb[i] for i in ids_all])
        print(f"\n{'=' * 100}\nview '{view}'\n{'=' * 100}", flush=True)
        keep42 = {}
        for seed in a.seeds:
            d = P2.make_data(df, seed, emb)
            delta = 1.0 / len(d["y_train"])
            cm, _ = D.train_central(d, seed)
            m = D.full_metrics(d["y_test"], D.predict(cm, d["X_test"]))
            util_rows.append({"view": view, "seed": seed, "epsilon": "inf", **{k: round(v, 4) for k, v in m.items()}})
            line = f"  seed {seed:2d}  eps=inf acc={m['accuracy']:.3f}"
            if seed == 42:
                keep42["inf"] = cm; scaler42 = d["scaler"]
            for e in EPS:
                model, state, info = D.train_dp(d, seed, e, delta)
                m = D.full_metrics(d["y_test"], D.predict(model, d["X_test"]))
                util_rows.append({"view": view, "seed": seed, "epsilon": D.eps_label(e),
                                  **{k: round(v, 4) for k, v in m.items()}})
                line += f" | eps={D.eps_label(e)} acc={m['accuracy']:.3f}"
                if seed == 42:
                    keep42[D.eps_label(e)] = model
            print(line, flush=True)

        P2DP_MODELS.mkdir(parents=True, exist_ok=True)
        for e, mdl in keep42.items():
            torch.save(mdl.state_dict(), P2DP_MODELS / f"{view}_eps{e}{tag}_state_dict.pt")

        print(f"  embedding candidate texts for the secret test (once for all models) ...", flush=True)
        cand = secret_candidates(enc, texts, view, vids, np.random.default_rng(42))
        for e in ["1", "3", "8", "inf"]:
            r = leak_eval(keep42[e], scaler42, emb, ids_all, E_all, vids, V["victims_y"], cand)
            r.update({"view": view, "epsilon": e, "weight_norm": round(D.weight_norm(keep42[e]), 2)})
            leak_rows.append(r)
            print(f"  leakage eps={e:3s}: closed-form exact {r['cf_exact']}/12 (zero-gradient {r['zero_grad']}), "
                  f"run identified {r['identified']}/12, secrets extracted {r['secrets_right']}/{r['secrets']}, "
                  f"max |logit| {r['max_logit']:.1f}", flush=True)

    U = pd.DataFrame(util_rows)
    cols = ["accuracy", "breach_recall", "specificity", "roc_auc"]
    print("\n" + "=" * 100 + "\nA. Utility (test, 5 seeds) and paired DP cost (non-private minus DP, points)\n" + "=" * 100)
    cost_rows = []
    for view in a.views:
        u = U[U.view == view]
        piv = u.pivot(index="seed", columns="epsilon", values=cols).reindex(a.seeds)
        for e in ["inf", "8", "3", "1"]:
            s = u[u.epsilon == e][cols]
            row = f"  {view:8s} eps={e:3s} " + "  ".join(f"{k}={s[k].mean():.3f}+-{s[k].std():.3f}" for k in cols)
            if e != "inf":
                acc = 100 * (piv[("accuracy", "inf")] - piv[("accuracy", e)])
                auc = 100 * (piv[("roc_auc", "inf")] - piv[("roc_auc", e)])
                row += f"   | cost: acc {acc.mean():+.1f}+-{acc.std():.1f}, AUC {auc.mean():+.1f}+-{auc.std():.1f}"
                cost_rows.append({"view": view, "epsilon": e, "acc_cost_mean": round(acc.mean(), 1), "acc_cost_std": round(acc.std(), 1),
                                  "auc_cost_mean": round(auc.mean(), 1), "auc_cost_std": round(auc.std(), 1)})
            print(row)
    L = pd.DataFrame(leak_rows)[["view", "epsilon", "cf_exact", "zero_grad", "identified", "secrets_right", "secrets",
                                 "max_logit", "weight_norm"]]
    print("\n" + "=" * 100 + "\nB. Leakage vs epsilon (seed-42 models, 12 victims)\n" + "=" * 100)
    print(L.round(2).to_string(index=False))
    print("  (Path 2 federated non-DP result for reference: closed-form 12/12, secrets 18/18 actions, 24/24 full)")

    P2DP_OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    U.to_csv(P2DP_OUT / f"p2_dp_runs{tag}.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(EVAL_DIR / f"path2_dp_cost{tag}.csv", index=False)
    L.to_csv(EVAL_DIR / f"path2_dp_leakage{tag}.csv", index=False)
    print(f"\nSaved: {P2DP_MODELS}/\n       {EVAL_DIR / f'path2_dp_cost{tag}.csv'}\n       {EVAL_DIR / f'path2_dp_leakage{tag}.csv'}")


if __name__ == "__main__":
    main()