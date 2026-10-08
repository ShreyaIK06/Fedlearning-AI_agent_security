"""
phase2_embed.py  -  Phase 2, Path 2 step 1: sentence-transformer embeddings of each trajectory
------------------------------------------------------------------------------------------------
Path 1 encoded WHICH tools ran (counts, categories). Path 2 encodes WHAT the agent did and read,
as text, so a gradient attack that reconstructs the input reconstructs CONTENT (account numbers,
recipients, passport data) rather than tool names. Phase 1 never did this.

Two text views per record (both embedded; training picks one):
  actions : user task + every tool call WITH its arguments, in order + the agent's final reply
            -> what the agent DID and SENT
  full    : the whole trajectory: task, each agent message, each tool call with arguments,
            and each tool OUTPUT -> also what the agent READ (incl. injected text, private data)

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, unit-normalised).
Long texts: MiniLM reads at most max_seq_length (256) tokens and silently cuts the rest -
often exactly where a breach happens (late in the run). So a text longer than that is split
into token windows, each window embedded, and the windows averaged then re-normalised
("chunk-mean pooling"). The script reports how many texts needed it.

Same records, same order, same split as Path 1: rows are built with phase2_features.build_table,
merged with data/processed/phase2_splits.csv, and the run_id order is ASSERTED equal to
phase2_features.npz, so Path 1 and Path 2 share train/val/test and the 12 attack victims.

Output: data/processed/phase2_embeddings.npz  (X_<view>_<split>, y_<split>, suite_<split>, run_id_<split>)
        data/processed/phase2_texts.csv       (the exact text embedded, for inspection)

Run from phase2_synthetic/:   python phase2_embed.py
Needs:  pip install sentence-transformers   (first run downloads ~90 MB model)
Test:   python phase2_embed.py --stub        (no download; a hashing stand-in encoder - NOT for results)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_features import EXCLUDED_OUTCOMES, FEATURES_NPZ, PROC_DIR, RAW_DIR, SPLITS_CSV

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_NPZ = PROC_DIR / "phase2_embeddings.npz"
TEXTS_CSV = PROC_DIR / "phase2_texts.csv"
VIEWS = ["actions", "full"]
MAX_TOOL_OUTPUT_CHARS = 1500     # cap a single tool output in the 'full' view (a huge inbox dump)


# ============================================================ text building
def fmt_args(args):
    if not args:
        return ""
    if isinstance(args, dict):
        return ", ".join(f"{k}={v}" for k, v in args.items())
    return str(args)


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, list):    # some LangChain messages hold a list of parts
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def build_texts(r):
    msgs = r.get("messages") or []
    task = r.get("user_task") or next((text_of(m.get("content")) for m in msgs if m.get("type") == "human"), "")
    actions, full = [f"TASK: {task}"], [f"TASK: {task}"]
    final = ""
    n = 0
    for m in msgs:
        t = m.get("type")
        if t == "ai":
            body = text_of(m.get("content")).strip()
            if body:
                full.append(f"AGENT: {body}")
                final = body
            for call in m.get("tool_calls") or []:
                n += 1
                line = f"ACTION {n}: {call.get('name')}({fmt_args(call.get('args'))})"
                actions.append(line); full.append(line)
        elif t == "tool":
            out = text_of(m.get("content")).strip()
            if len(out) > MAX_TOOL_OUTPUT_CHARS:
                out = out[:MAX_TOOL_OUTPUT_CHARS] + " ..."
            full.append(f"RESULT {m.get('tool_name') or ''}: {out}")
    if final:
        actions.append(f"FINAL: {final}")
    return {"actions": "\n".join(actions), "full": "\n".join(full)}


def load_rows():
    rows = []
    for f in sorted(RAW_DIR.glob("*/*.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("outcome") in EXCLUDED_OUTCOMES:
            continue
        rows.append({"run_id": r["run_id"], "suite": r["suite"], "breach": r["breach"], **build_texts(r)})
    return pd.DataFrame(rows)


# ============================================================ encoders
class STEncoder:
    """sentence-transformers model with chunk-mean pooling for texts longer than the model reads."""

    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)
        self.tok = self.m.tokenizer
        self.max_tokens = self.m.max_seq_length - 2          # room for [CLS] / [SEP]
        self.dim = self.m.get_sentence_embedding_dimension()

    def n_tokens(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def chunks(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tokens:
            return [text]
        return [self.tok.decode(ids[i:i + self.max_tokens]) for i in range(0, len(ids), self.max_tokens)]

    def encode(self, texts):
        out = []
        for t in texts:
            parts = self.chunks(t)
            e = self.m.encode(parts, normalize_embeddings=True, show_progress_bar=False).mean(axis=0)
            out.append(e / (np.linalg.norm(e) + 1e-12))
        return np.stack(out).astype(np.float32)


class StubEncoder:
    """TEST ONLY: deterministic hashed bag-of-words, same shape/normalisation as MiniLM. Not for results."""
    dim, max_tokens = 384, 254

    def n_tokens(self, text):
        return len(text.split())

    def chunks(self, text):
        w = text.split()
        return [" ".join(w[i:i + self.max_tokens]) for i in range(0, max(len(w), 1), self.max_tokens)]

    def encode(self, texts):
        import hashlib
        out = []
        for t in texts:
            v = np.zeros(self.dim, np.float32)
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
            out.append(v / (np.linalg.norm(v) + 1e-12))
        return np.stack(out)


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="test with a stand-in encoder (no download)")
    a = ap.parse_args()

    df = load_rows()
    splits = pd.read_csv(SPLITS_CSV)[["run_id", "split"]]
    df = df.merge(splits, on="run_id", how="left", validate="one_to_one")
    assert df["split"].notna().all(), "records without a split - run phase2_split.py first"

    enc = StubEncoder() if a.stub else STEncoder(MODEL_NAME)
    print(f"Encoder: {'STUB (test only - not for results)' if a.stub else MODEL_NAME}  dim={enc.dim}  "
          f"reads {enc.max_tokens} tokens per chunk")
    print(f"Records: {len(df)}")

    for v in VIEWS:
        lens = df[v].map(enc.n_tokens)
        n_long = int((lens > enc.max_tokens).sum())
        df[f"{v}_tokens"] = lens
        print(f"  view '{v}': tokens median {int(lens.median())}, max {int(lens.max())}; "
              f"{n_long} of {len(df)} longer than {enc.max_tokens} -> chunk-mean pooled "
              f"(up to {int(np.ceil(lens.max() / enc.max_tokens))} chunks)")

    emb = {v: enc.encode(df[v].tolist()) for v in VIEWS}

    # ---- same order as Path 1's feature file, so splits and victims line up
    ref = np.load(FEATURES_NPZ, allow_pickle=False)
    out = {}
    for s in ("train", "val", "test"):
        m = (df["split"] == s).to_numpy()
        ids = df.loc[m, "run_id"].to_numpy(str)
        assert np.array_equal(ids, ref[f"run_id_{s}"]), f"{s} order differs from phase2_features.npz"
        for v in VIEWS:
            out[f"X_{v}_{s}"] = emb[v][m]
        out[f"y_{s}"] = df.loc[m, "breach"].to_numpy(np.float32)
        out[f"suite_{s}"] = df.loc[m, "suite"].to_numpy(str)
        out[f"run_id_{s}"] = ids
    out["model_name"] = np.array("STUB" if a.stub else MODEL_NAME)
    np.savez(EMB_NPZ, **out)
    df[["run_id", "suite", "split", "breach", "actions_tokens", "full_tokens", "actions", "full"]].to_csv(
        TEXTS_CSV, index=False, encoding="utf-8")

    # ---- report
    print("\nShapes:", {k: v.shape for k, v in out.items() if k.startswith("X_")})
    for v in VIEWS:
        X = np.concatenate([out[f"X_{v}_{s}"] for s in ("train", "val", "test")])
        sims = X @ X.T
        np.fill_diagonal(sims, np.nan)
        print(f"  view '{v}': value range [{X.min():.3f}, {X.max():.3f}]  "
              f"mean cosine to other records {np.nanmean(sims):.3f}  (lower = records more distinguishable)")
    ex = df[df["breach"] == 1].iloc[0]
    print(f"\nExample breach record ({ex['suite']}, run {ex['run_id']}) - 'actions' text:\n{'-' * 60}\n{ex['actions'][:1200]}")
    print(f"\nSaved: {EMB_NPZ}\n       {TEXTS_CSV}  (open it to check exactly what was embedded)")


if __name__ == "__main__":
    main()