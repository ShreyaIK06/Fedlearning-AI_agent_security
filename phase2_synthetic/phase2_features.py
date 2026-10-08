"""
phase2_features.py  -  Phase 2 (synthetic LangChain logs), Path 1
--------------------------------------------------------------------
Turns the JSON run records into a feature matrix in the SAME STYLE as
Phase 1's src/phase1/04_feature_encoding.py, so the two datasets can be
compared apples-to-apples.

Feature blocks (Phase 1 equivalent in brackets):
  1. One-hot  : suite, task_id, injection_style
                 [suite, task, attack_family; Phase 1 also had base_model,
                  defense, injection_task - no variety in our data, so dropped]
  2. Bag-of-tools on the ordered tool-call sequence  [identical recipe]
  3. StandardScaler on num_messages, assistant_turns, tool_calls, tool_errors
                 [Phase 1 also had duration_seconds_log - our records store
                  only one timestamp, so duration cannot be computed]
  (Phase 1's is_dos_attack flag is dropped - no DoS attacks in our data.)

The one genuinely new piece of logic: Phase 1 had a ready-made
`tool_sequence` column; here we BUILD it from each record's `messages`
by reading the tool calls the agent made, in order.

Encoders are fit on the TRAIN split only, then applied to val/test, and
saved with joblib - exactly as Phase 1 does.

Run order:
  1. python phase2_split.py       (writes data/processed/phase2_splits.csv)
  2. python phase2_features.py    (this script)

Place this file next to runner.py (it uses the same folder convention).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent     # same convention as runner.py
RAW_DIR = ROOT / "data" / "raw"                    # data/raw/<suite>/*.json  (raw_120b is NOT read)
PROC_DIR = ROOT / "data" / "processed"
ENC_DIR = ROOT / "models" / "phase2_encoders"

SPLITS_CSV = PROC_DIR / "phase2_splits.csv"        # written by phase2_split.py
TABLE_CSV = PROC_DIR / "phase2_table.csv"          # one row per usable record (human-readable)
FEATURES_NPZ = PROC_DIR / "phase2_features.npz"    # model-ready arrays

EXCLUDED_OUTCOMES = ("empty_response", "generation_error")
EXPECTED_USABLE = 563

CATEGORICAL_COLS = ["suite", "task_id", "injection_style"]
NUMERIC_COLS = ["num_messages", "assistant_turns", "tool_calls"]


# ======================================================= 1. record -> row
def tool_sequence(messages):
    """Ordered tool names the agent called, joined with ' -> '.

    In our saved trajectory, an AI message that uses tools has a non-empty
    `tool_calls` list, e.g. [{"name": "read_inbox", "args": {}, ...}].
    One AI turn can call several tools at once; we keep them in the order
    the model listed them, so the sequence reads like a timeline.
    """
    names = []
    for m in messages:
        if m.get("type") == "ai":
            for call in m.get("tool_calls") or []:
                name = call.get("name")
                if name:
                    names.append(name)
    return " -> ".join(names)


def is_tool_error(m):
    """A tool message counts as an error if LangChain returned an error string.

    Our serializer did not save LangChain's `status` field, so we detect
    errors from the content: LangChain's ToolNode error messages start
    with "Error" (e.g. "Error: ..." / "Error invoking tool ...").
    """
    content = m.get("content")
    if not isinstance(content, str):
        return False
    return content.strip().lower().startswith("error")


def record_to_row(r):
    msgs = r.get("messages") or []
    style = r.get("injection_style") or "benign"      # None -> "benign" (one clear category)
    return {
        "run_id": r["run_id"],
        "suite": r["suite"],
        "task_id": f"{r['suite']}/{r['task_id']}",       # prefixed: slack & workspace both have 'notes_*'
        "task_type": r.get("task_type"),
        "injection_style": style,
        # GROUP key for the split: the 3 repeats of a scenario share it
        "scenario_id": f"{r['suite']}|{r['task_id']}|{style}",
        "repeat": r.get("repeat"),
        # numeric counts (Phase 1 style)
        "num_messages": len(msgs),
        "assistant_turns": sum(1 for m in msgs if m.get("type") == "ai"),
        "tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs if m.get("type") == "ai"),
        "tool_errors": sum(1 for m in msgs if m.get("type") == "tool" and is_tool_error(m)),
        "tool_sequence": tool_sequence(msgs),
        # labels
        "breach": r["breach"],                         # FL training label
        "data_leaked": r.get("data_leaked"),           # analysis only, never a feature
        "outcome": r["outcome"],
    }


def build_table(raw_dir=RAW_DIR):
    """Read every JSON under data/raw/<suite>/ and return one row per usable record.

    phase2_split.py imports this too, so both scripts see the same records.
    """
    files = sorted(raw_dir.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON records found under {raw_dir}")

    rows, skipped = [], 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("outcome") in EXCLUDED_OUTCOMES:
            skipped += 1
            continue
        rows.append(record_to_row(r))

    df = pd.DataFrame(rows)

    # --- integrity checks
    assert df["run_id"].is_unique, "duplicate run_id found"
    assert set(df["breach"].unique()) <= {0, 1}, "breach must be 0/1 for usable records"
    print(f"Loaded {len(files)} files -> {len(df)} usable, {skipped} excluded")
    if len(df) != EXPECTED_USABLE:
        print(f"  WARNING: expected {EXPECTED_USABLE} usable records, got {len(df)}")
    return df


# ===================================================== 2. encode features
def clean_tool_sequence(s):
    """Phase 1 recipe, copied verbatim: ' -> ' becomes a space, '<' '>' removed."""
    return s.str.replace(" -> ", " ", regex=False) \
            .str.replace("<", "", regex=False) \
            .str.replace(">", "", regex=False)


def fit_encoders(train):
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot.fit(train[CATEGORICAL_COLS])

    tool_vec = CountVectorizer(token_pattern=r"[^\s]+", binary=True, lowercase=False)
    tool_vec.fit(clean_tool_sequence(train["tool_sequence"]))

    scaler = StandardScaler()
    scaler.fit(train[NUMERIC_COLS].astype(float))
    return onehot, tool_vec, scaler


def transform(df, onehot, tool_vec, scaler):
    X_cat = onehot.transform(df[CATEGORICAL_COLS])
    X_tool = tool_vec.transform(clean_tool_sequence(df["tool_sequence"])).toarray()
    X_num = scaler.transform(df[NUMERIC_COLS].astype(float))
    return np.hstack([X_cat, X_tool, X_num]).astype(np.float32)   # float32 for PyTorch


def feature_names(onehot, tool_vec):
    return (list(onehot.get_feature_names_out(CATEGORICAL_COLS))
            + [f"tool_{t}" for t in tool_vec.get_feature_names_out()]
            + NUMERIC_COLS)


# ================================================================== main
def main():
    df = build_table()

    # ---- attach the split made by phase2_split.py
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"{SPLITS_CSV} not found - run phase2_split.py first")
    splits = pd.read_csv(SPLITS_CSV)[["run_id", "split"]]
    df = df.merge(splits, on="run_id", how="left", validate="one_to_one")
    assert df["split"].notna().all(), "some records have no split - re-run phase2_split.py"
    assert set(df["split"]) == {"train", "val", "test"}

    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))

    # ---- leakage guard: no scenario may appear in two splits
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(a["scenario_id"]) & set(b["scenario_id"]), "scenario leaked across splits"

    # ---- fit on TRAIN only, apply to all three
    onehot, tool_vec, scaler = fit_encoders(train)
    X_train, X_val, X_test = (transform(d, onehot, tool_vec, scaler) for d in (train, val, test))
    names = feature_names(onehot, tool_vec)
    assert X_train.shape[1] == len(names)

    # ---- save encoders (mirrors models/phase1_encoders/)
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(onehot, ENC_DIR / "onehot.joblib")
    joblib.dump(tool_vec, ENC_DIR / "tool_vectorizer.joblib")
    joblib.dump(scaler, ENC_DIR / "scaler.joblib")

    # ---- save arrays; suite kept per row so FedAvg can build the 4 clients
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURES_NPZ,
        X_train=X_train, y_train=train["breach"].to_numpy(np.float32),
        X_val=X_val,     y_val=val["breach"].to_numpy(np.float32),
        X_test=X_test,   y_test=test["breach"].to_numpy(np.float32),
        suite_train=train["suite"].to_numpy(str),
        suite_val=val["suite"].to_numpy(str),
        suite_test=test["suite"].to_numpy(str),
        run_id_train=train["run_id"].to_numpy(str),
        run_id_val=val["run_id"].to_numpy(str),
        run_id_test=test["run_id"].to_numpy(str),
        feature_names=np.array(names),
    )
    df.drop(columns=["split"]).to_csv(TABLE_CSV, index=False)

    # ---- report
    n_cat = len(onehot.get_feature_names_out())
    n_tool = len(tool_vec.get_feature_names_out())
    print("\nFeature matrix")
    print(f"  one-hot     : {n_cat}")
    print(f"  bag-of-tools: {n_tool}  {list(tool_vec.get_feature_names_out())}")
    print(f"  numeric     : {len(NUMERIC_COLS)}")
    print(f"  TOTAL       : {len(names)}   (Phase 1 was 176 - smaller is expected)")
    print("\nShapes / breach rate")
    for name, X, d in [("train", X_train, train), ("val", X_val, val), ("test", X_test, test)]:
        print(f"  {name:<5} {str(X.shape):<12} breach={d['breach'].mean():.2f}  "
              f"scenarios={d['scenario_id'].nunique()}")
    print("\nNumeric columns (all usable records)")
    print(df[NUMERIC_COLS].describe().loc[["mean", "min", "max"]].round(2).to_string())
    print(f"\nRecords with no tool calls: {(df['tool_sequence'] == '').sum()}")
    print("Most common tool sequences:")
    print(df["tool_sequence"].replace("", "(none)").value_counts().head(5).to_string())
    print(f"\nSaved: {FEATURES_NPZ}\n       {TABLE_CSV}\n       {ENC_DIR}/")


if __name__ == "__main__":
    main()