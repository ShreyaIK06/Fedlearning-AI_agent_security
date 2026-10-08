"""
phase2_p2_attacks.py  -  Phase 2, Path 2 step 3: does ONE gradient leak the CONTENT of an agent run?
-----------------------------------------------------------------------------------------------------
Attacks the seed-42 FEDERATED Path-2 model (as phase2_attacks.py did for Path 1), on the SAME 12
victims, with Phase 1's code unchanged (compute_gradient, closed_form_first_layer_attack,
reconstruct for Inverting Gradients). The model input is the scaled 384-dim sentence embedding.

Step A  Reconstruct the embedding
        closed-form (exact) and Inverting Gradients (Phase 1 settings: cosine, Adam 0.05, 500 its).
        The reconstruction is un-scaled with the seed-42 StandardScaler (the preprocessing is part
        of the shared FL pipeline, so the attacker knows it - same assumption as Path 1's encoders).

Step B  What can the attacker READ from a reconstructed embedding?  Two tests, both stated with
        their assumptions so they aren't over-read:
  B1  Identification: rank the victim's true run among all 563 runs' embeddings by cosine
      similarity to the reconstruction. Assumption: the attacker holds a pool of candidate
      runs (an UPPER bound on what identification needs). Top-1 = picked out the exact run.
  B2  Secret extraction (dictionary attack): for each sensitive value in the victim's text -
      account numbers (ACC-ddd-NAME), email addresses, passport numbers - the attacker
      embeds the victim's text with that value swapped for each of ~150 candidates
      (every value of that kind seen in the dataset + generated look-alikes such as
      ACC-321-ANANYA or neha.s@techcorp.in) and picks the candidate whose embedding is
      closest to the reconstruction. Assumption: the attacker knows the rest of the text
      (template) and the kind of secret - a standard dictionary-attack setting.
      Baseline: the same guess made from the TRAINING-MEAN embedding (no gradient) - what
      the attacker gets without the leak. Chance = 1 / number of candidates.

Run from phase2_synthetic/:   python phase2_p2_attacks.py                 (both views)
Test without the model:       python phase2_p2_attacks.py --stub         (NOT for results)
"""

import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch

from phase2_common import EVAL_DIR, OUT_DIR, ROOT, SecurityMLP
from phase2_embed import MODEL_NAME, TEXTS_CSV, STEncoder, StubEncoder
from phase2_p2_train import P2_MODEL_DIR, load_embeddings
from common.gradient_leakage import (closed_form_first_layer_attack, compute_gradient,  # Phase 1, unchanged
                                     infer_label_from_final_bias_grad, reconstruct)

VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"
P2_OUT = OUT_DIR / "path2"
N_CANDIDATES = 150
EXACT_TOL = 1e-4
HYPHENS = ["-", "‑", "‐", "–"]          # the agent sometimes writes non-breaking hyphens

PATTERNS = {
    "account": re.compile(r"\bACC[-‑‐–]\d{3}[-‑‐–][A-Z]+\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "passport": re.compile(r"\b[A-Z]\d{7}\b"),
}


def canon(kind, v):
    return re.sub("[‑‐–]", "-", v) if kind == "account" else v.lower() if kind == "email" else v


def variants(kind, v):
    """All spellings of a value that should be swapped together."""
    if kind == "account":
        return {v.replace("-", h) for h in HYPHENS}
    if kind == "email":
        return {v, v.lower()}
    return {v}


def _edit(s, rng):
    """One look-alike edit: homoglyph swap, doubled/dropped character, or neighbouring letter."""
    swaps = {"o": "0", "0": "o", "l": "1", "1": "l", "i": "l", "e": "3", "a": "4", "s": "5", "m": "rn", "rn": "m"}
    i = int(rng.integers(len(s)))
    c = s[i]
    r = rng.random()
    if r < .4 and c in swaps:
        return s[:i] + swaps[c] + s[i + 1:]
    if r < .6:
        return s[:i] + c + s[i:]                       # doubled character
    if r < .8 and len(s) > 3:
        return s[:i] + s[i + 1:]                       # dropped character
    return s[:i] + chr(97 + int(rng.integers(26))) + s[i + 1:]


def lookalikes(kind, v, rng, n):
    """Up to n plausible look-alikes of v (fewer if the space is small; never loops forever)."""
    out, tries = set(), 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        if kind == "account":
            _, digits, name = v.split("-")
            d = list(digits); d[int(rng.integers(3))] = str(int(rng.integers(10)))
            out.add(f"ACC-{''.join(d)}-{name}" if rng.random() < .7 else f"ACC-{int(rng.integers(100, 1000))}-{name}")
        elif kind == "email":
            local, dom = v.split("@", 1)
            base, _, tld = dom.partition(".")
            r = rng.random()
            if r < .35:
                out.add(f"{local}@{_edit(base, rng)}.{tld}")
            elif r < .7:
                out.add(f"{_edit(local, rng)}@{dom}")
            elif r < .85:
                out.add(f"{local}@{base}.{rng.choice(['com', 'in', 'co.in', 'net', 'org', 'io'])}")
            else:
                out.add(f"{local}{int(rng.integers(100))}@{dom}")
        else:
            out.add(f"{chr(65 + int(rng.integers(26)))}{int(rng.integers(10**6, 10**7))}")
    out.discard(v)
    return out


def swap(text, kind, true, cand):
    for t in variants(kind, true):
        text = text.replace(t, cand)
    return text


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["actions", "full"], choices=["actions", "full"])
    ap.add_argument("--model", default="federated", choices=["federated", "central"])
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(42)

    enc = StubEncoder() if a.stub else STEncoder(MODEL_NAME)
    texts = pd.read_csv(TEXTS_CSV, keep_default_na=False).set_index("run_id")
    V = torch.load(VICTIMS_PT)
    vids = [m["run_id"] for m in V["victim_meta"]]
    vsuites = [m["suite"] for m in V["victim_meta"]]

    # dictionary of every sensitive value seen anywhere in the dataset, by kind
    seen = {k: set() for k in PATTERNS}
    for col in ("actions", "full"):
        for t in texts[col]:
            for k, p in PATTERNS.items():
                seen[k].update(canon(k, m) for m in p.findall(t))
    print(f"Encoder: {'STUB (test only)' if a.stub else MODEL_NAME}")
    print("Sensitive values in the dataset: " + ", ".join(f"{k}={len(v)}" for k, v in seen.items()))

    summary, examples = [], []
    for view in a.views:
        emb, _ = load_embeddings(view)
        ids_all = np.array(list(emb.keys())); E_all = np.stack([emb[i] for i in ids_all])
        scaler = joblib.load(P2_MODEL_DIR / f"scaler_{view}.joblib")
        model = SecurityMLP(input_dim=E_all.shape[1], hidden_dims=(128, 64), dropout=0.2)
        model.load_state_dict(torch.load(P2_MODEL_DIR / f"{a.model}_{view}_state_dict.pt")); model.eval()
        mean_emb = scaler.mean_.astype(np.float32)                   # attacker's no-gradient guess

        print(f"\n{'=' * 100}\nview '{view}'  -  {a.model} seed-42 model, 12 victims\n{'=' * 100}")
        rows = []
        for i, (rid, suite) in enumerate(zip(vids, vsuites)):
            e_true = emb[rid]
            x = torch.tensor(scaler.transform(e_true[None]).astype(np.float32))
            y = V["victims_y"][i].reshape(1)
            g = compute_gradient(model, x, y)
            zero = all(float(t.abs().max()) == 0 for t in g)
            x_cf = closed_form_first_layer_attack(g).numpy()
            y_inf = infer_label_from_final_bias_grad(g[-1])
            x_ig, _, _ = reconstruct(model, g, E_all.shape[1], mode="cosine", know_label=True, true_y=y_inf,
                                     n_iters=500, lr=0.05, seed=42 + i, optimizer_name="adam")
            e_cf = scaler.inverse_transform(x_cf)[0]; e_ig = scaler.inverse_transform(x_ig.numpy())[0]

            r = {"victim": i, "suite": suite, "breach": int(y.item()), "zero_grad": zero,
                 "cf_exact": bool(np.max(np.abs(x_cf - x.numpy())) < EXACT_TOL),
                 "cf_cos": cos(e_cf, e_true), "ig_cos": cos(e_ig, e_true)}
            # B1 identification
            for tag, e in [("cf", e_cf), ("ig", e_ig), ("mean", mean_emb)]:
                sims = E_all @ e / (np.linalg.norm(E_all, axis=1) * np.linalg.norm(e) + 1e-12)
                r[f"{tag}_rank"] = int((sims > sims[ids_all == rid][0]).sum()) + 1
            # B2 secret extraction
            vtext = texts.loc[rid, view]
            secrets = {(k, canon(k, m)) for k, p in PATTERNS.items() for m in p.findall(vtext)}
            r["n_secrets"] = len(secrets)
            hits = {"cf": 0, "ig": 0, "mean": 0}; chance = []
            for k, true in sorted(secrets):
                pool = sorted(seen[k] - {true})
                cands = [true] + list(rng.permutation(pool))[:N_CANDIDATES // 2]
                cands += sorted(lookalikes(k, true, rng, N_CANDIDATES - len(cands)) - set(cands))
                cands = cands[:N_CANDIDATES]
                ce = enc.encode([swap(vtext, k, true, c) for c in cands])
                for tag, e in [("cf", e_cf), ("ig", e_ig), ("mean", mean_emb)]:
                    guess = cands[int(np.argmax(ce @ e / (np.linalg.norm(e) + 1e-12)))]
                    hits[tag] += int(guess == true)
                    if tag == "cf":
                        examples.append({"view": view, "victim": i, "suite": suite, "kind": k, "true": true,
                                         "guess_from_gradient": guess, "correct": guess == true,
                                         "n_candidates": len(cands)})
                chance.append(1 / len(cands))
            r.update({f"{t}_secrets_right": h for t, h in hits.items()})
            r["chance"] = float(np.mean(chance)) if chance else float("nan")
            rows.append(r)
            print(f"  victim {i:2d} {suite:10s} breach={r['breach']}  CF exact={r['cf_exact']} cos={r['cf_cos']:.4f} | "
                  f"IG cos={r['ig_cos']:.3f} | rank of true run: CF {r['cf_rank']}, IG {r['ig_rank']}, no-grad {r['mean_rank']} | "
                  f"secrets right (CF/IG/no-grad): {hits['cf']}/{hits['ig']}/{hits['mean']} of {len(secrets)}")

        df = pd.DataFrame(rows)
        ns = int(df["n_secrets"].sum())
        s = {"view": view, "model": a.model,
             "CF_exact": f"{int(df.cf_exact.sum())}/12", "CF_mean_cos": round(df.cf_cos.mean(), 4),
             "IG_mean_cos": round(df.ig_cos.mean(), 3),
             "identify_top1_CF": f"{int((df.cf_rank == 1).sum())}/12",
             "identify_top1_IG": f"{int((df.ig_rank == 1).sum())}/12",
             "identify_top1_nograd": f"{int((df.mean_rank == 1).sum())}/12",
             "secrets_total": ns,
             "secrets_CF": f"{int(df.cf_secrets_right.sum())}/{ns}",
             "secrets_IG": f"{int(df.ig_secrets_right.sum())}/{ns}",
             "secrets_nograd": f"{int(df.mean_secrets_right.sum())}/{ns}",
             "chance_per_secret": round(float(np.nanmean(df.chance)), 4) if ns else None}
        summary.append(s)
        P2_OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(P2_OUT / f"p2_attack_{a.model}_{view}.csv", index=False)

    summ = pd.DataFrame(summary)
    ex = pd.DataFrame(examples)
    print("\n" + "=" * 100 + "\nSummary\n" + "=" * 100)
    print(summ.to_string(index=False))
    if len(ex):
        print("\nSecrets: true value vs the attacker's guess from the closed-form reconstruction")
        print(ex.drop(columns=["n_candidates"]).to_string(index=False))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summ.to_csv(EVAL_DIR / f"path2_attack_summary_{a.model}.csv", index=False)
    ex.to_csv(EVAL_DIR / f"path2_secret_extraction_{a.model}.csv", index=False)
    print(f"\nSaved: {EVAL_DIR / f'path2_attack_summary_{a.model}.csv'}\n       {EVAL_DIR / f'path2_secret_extraction_{a.model}.csv'}"
          f"\n       {P2_OUT}/p2_attack_{a.model}_<view>.csv")


if __name__ == "__main__":
    main()