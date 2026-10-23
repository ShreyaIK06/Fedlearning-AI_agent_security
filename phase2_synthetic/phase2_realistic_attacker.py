"""
phase2_realistic_attacker.py  -  does secret extraction still work if the attacker does NOT know the victim's text?
--------------------------------------------------------------------------------------------------------------------
The headline result (42/42 secrets) used a dictionary attack in which the attacker knows the victim's
own text and only guesses the secret. That is an upper bound. Here the same attack is repeated with
attackers who know progressively less. The reconstruction is the one the honest single-gradient
closed-form attack produces, which was exact for all 12 victims (12/12), so it equals the victim's
stored embedding.

Templates the attacker fills each candidate secret into:
  own_text       the victim's own text (the earlier, upper-bound setting)
  sibling_run    another repeat of the SAME scenario (same suite, task and injection style), choosing
                 a repeat whose text differs from the victim's when one exists
  other_task     a run of a DIFFERENT task in the same suite that contains a secret of the same kind
  bare_value     the candidate value on its own, no surrounding text
For sibling_run / other_task, the template's own value of that kind is the slot that is swapped.
Candidates are exactly as before: the true value, other values of that kind seen in the dataset, and
generated look-alikes (~150; chance ~0.67%). Both text views ('actions', 'full').

Run from phase2_synthetic/:   python phase2_realistic_attacker.py        (~20-60 min, mostly text encoding)
"""

import argparse

import numpy as np
import pandas as pd
import torch

import phase2_p2_train as P2
from phase2_common import EVAL_DIR, OUT_DIR
from phase2_embed import MODEL_NAME, TEXTS_CSV, STEncoder, StubEncoder
from phase2_features import build_table
from phase2_p2_attacks import N_CANDIDATES, PATTERNS, canon, lookalikes, swap
from phase2_p2_dp import fast_encode

VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"
TEMPLATES = ["own_text", "sibling_run", "other_task", "bare_value"]


def values_in(text, kind):
    return sorted({canon(kind, m) for m in PATTERNS[kind].findall(text)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["actions", "full"], choices=["actions", "full"])
    ap.add_argument("--stub", action="store_true", help="test encoder (NOT for results)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    enc = StubEncoder() if a.stub else STEncoder(MODEL_NAME)
    texts = pd.read_csv(TEXTS_CSV, keep_default_na=False).set_index("run_id")
    table = build_table().set_index("run_id")
    vids = [m["run_id"] for m in torch.load(VICTIMS_PT)["victim_meta"]]
    seen = {k: set() for k in PATTERNS}
    for col in ("actions", "full"):
        for t in texts[col]:
            for k in PATTERNS:
                seen[k].update(values_in(t, k))

    rows = []
    for view in a.views:
        emb, _ = P2.load_embeddings(view)
        rng = np.random.default_rng(42)
        print(f"\nview '{view}'", flush=True)
        for vi, rid in enumerate(vids):
            e = emb[rid]                                   # = exact closed-form reconstruction
            vtext = texts.loc[rid, view]
            scen, suite = table.loc[rid, "scenario_id"], table.loc[rid, "suite"]
            sibs = [r for r in table.index[(table.scenario_id == scen) & (table.index != rid)] if r in texts.index]
            sibs = sorted(sibs, key=lambda r: texts.loc[r, view] == vtext)          # different text first
            others = [r for r in table.index[(table.suite == suite) & (table.scenario_id != scen)] if r in texts.index]
            for k in PATTERNS:
                for true in values_in(vtext, k):
                    pool = sorted(seen[k] - {true})
                    cands = [true] + list(rng.permutation(pool))[:N_CANDIDATES // 2]
                    cands += sorted(lookalikes(k, true, rng, N_CANDIDATES - len(cands)) - set(cands))
                    cands = cands[:N_CANDIDATES]
                    tmpl = {"own_text": (vtext, true)}
                    for r in sibs:
                        vals = values_in(texts.loc[r, view], k)
                        if vals:
                            tmpl["sibling_run"] = (texts.loc[r, view], true if true in vals else vals[0]); break
                    for r in rng.permutation(others):
                        vals = values_in(texts.loc[r, view], k)
                        if vals:
                            tmpl["other_task"] = (texts.loc[r, view], vals[0]); break
                    for name in TEMPLATES:
                        if name == "bare_value":
                            ce = fast_encode(enc, list(cands))
                        elif name in tmpl:
                            text, slot = tmpl[name]
                            ce = fast_encode(enc, [swap(text, k, slot, c) for c in cands])
                        else:
                            rows.append({"view": view, "victim": vi, "kind": k, "template": name, "available": False})
                            continue
                        guess = cands[int(np.argmax(ce @ (e / (np.linalg.norm(e) + 1e-12))))]
                        rank = int((ce @ e > (ce @ e)[0]).sum()) + 1                     # rank of the true value
                        rows.append({"view": view, "victim": vi, "kind": k, "template": name, "available": True,
                                     "correct": guess == true, "true_rank": rank, "n_candidates": len(cands)})
            print(f"  victim {vi:2d} done", flush=True)

    df = pd.DataFrame(rows)
    av = df[df.available]
    S = av.groupby(["view", "template"], sort=False).agg(
        secrets=("correct", "size"), extracted=("correct", "sum"), top5=("true_rank", lambda r: int((r <= 5).sum())),
        median_rank=("true_rank", "median"), candidates=("n_candidates", "mean")).reset_index()
    S["chance"] = (1 / S.candidates).round(4)
    miss = df[~df.available].groupby(["view", "template"]).size().rename("template_unavailable").reset_index()
    S = S.merge(miss, on=["view", "template"], how="left").fillna({"template_unavailable": 0})
    print("\n" + "=" * 100 + "\nSecret extraction by what the attacker knows (12 victims, exact reconstructions)\n" + "=" * 100)
    print(S.to_string(index=False))
    print("\nextracted = true value ranked first among ~150 candidates; top5 = true value in the top 5")
    (OUT_DIR / "realistic_attacker").mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "realistic_attacker" / f"realistic_attacker_per_secret{a.tag}.csv", index=False)
    S.to_csv(EVAL_DIR / f"realistic_attacker_summary{a.tag}.csv", index=False)
    print(f"\nSaved: {EVAL_DIR / f'realistic_attacker_summary{a.tag}.csv'}")


if __name__ == "__main__":
    main()