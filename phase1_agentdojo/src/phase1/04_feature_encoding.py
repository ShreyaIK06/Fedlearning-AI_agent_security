"""
Phase 1, Step 4 - Feature engineering and encoding for the modeling matrix.

Source:   AgentDojo_Dataset_Research_Report.md Part 6, Part 8, Part 9, Part 12, Part 20.9
Reason:   Turns the (already-cleaned, already-scoped, already-split) tabular
          rows into numeric feature matrices a classical/deep model can
          train on. All fitting (one-hot categories, tool vocabulary, numeric
          scaling) happens on TRAIN ONLY and is then applied to val/test -
          fitting on the full dataset before splitting would leak
          distributional information from val/test into training.

          Feature engineering decisions and why:

          - `is_dos_attack` + collapsed `attack_family` (Part 20.9): the 5
            *_dos attack_type values are rare (97 rows each in the scoped
            population - verified below) and semantically one family. Raw
            one-hot on `attack_type` would give a model 5 near-duplicate,
            data-starved dummy columns; collapsing them into one
            `dos_family` category (used for `attack_family`'s one-hot) plus
            a separate `is_dos_attack` boolean satisfies the report's
            literal recommendation. These two are correlated by
            construction (is_dos_attack == 1 iff attack_family=="dos_family")
            - documented here rather than hidden, and harmless for the
            regularized/tree/MLP models this project uses.

          - `tool_sequence` -> bag-of-tools (Part 8: "not standard one-hot -
            treat as a token sequence... given the vocabulary is only ~90
            distinct tool names"). Tokenized on " -> ", binary
            (presence/absence) CountVectorizer, vocabulary fit on train.
            `no_tool_call` (the Phase 2 sentinel from dataset_preparation)
            and the raw `<empty-function-name>` artifact both survive as
            ordinary vocabulary tokens - not stripped, since both are
            genuine information about the trace.

            Implementation note: tokenization is done by replacing " -> "
            with a plain space and using CountVectorizer's regex
            `token_pattern` (whitespace-delimited tokens) rather than a
            custom Python `tokenizer=` callable. A custom callable defined
            at script level pickles as a reference to `__main__` and fails
            to reload (`AttributeError: Can't get attribute ... on
            <module '__main__'>`) from any other script - caught by
            actually trying to reload the saved encoder before moving on,
            not assumed to work.

            Amendment (found in Phase 2, Step 3 - XGBoost): the raw
            `tool_sequence` data contains one literal artifact token,
            `<empty-function-name>` (an AgentDojo malformed/empty tool-call
            placeholder), which survives into the feature name
            `tool__<empty-function-name>`. XGBoost's DMatrix rejects feature
            names containing `[`, `]`, or `<` and fails at `.fit()` time -
            sklearn's LR/RF don't care about column-name characters, so this
            didn't surface until the first XGBoost run. Fixed by stripping
            angle brackets from tool tokens before vectorizing, so every
            downstream model (Steps 2-4 here, and every later phase that
            reuses this feature schema) sees one consistent, universally-
            safe naming convention rather than an XGBoost-only special case.

          - Categorical one-hot: `suite`, `base_model`, `defense`, `task`,
            `injection_task`, `attack_family` (Part 8's stated encoding).
            `handle_unknown="ignore"` because the grouped split (Step 3) can
            put a task/base_model value disproportionately into one split.

          - Numeric: `num_messages`, `assistant_turns`, `tool_calls`,
            `tool_errors`, `duration_seconds_log` (the de-skewed column from
            dataset_preparation, not raw `duration_seconds` - using both
            would be redundant). StandardScaler fit on train only.

          - Explicitly EXCLUDED from this baseline feature set:
              * `pipeline` (raw) - redundant with `base_model`+`defense`.
              * `injection`, `security_label_status` - constant within the
                scoped population (`True` / `"measured"` for every row),
                zero information.
              * `utility` - a second ground-truth label, not a feature.
                Using one target to predict another would conflate two
                correlated-but-distinct outcomes of the same trace rather
                than modeling `security` from behavioral/categorical
                evidence, which is the report's Part 6 framing (`utility`
                and `security` are both listed under "Labels", not
                "Features"). Excluded as a deliberate design choice, not an
                oversight.
              * `user_prompt`, `final_answer` (raw text) - deferred to a
                later, explicitly-labeled ablation (report Part 9/17); this
                baseline stays tabular, consistent with PROJECT_PLAN.md's
                Phase 2 scope.

Input:    outputs/phase1/03_train.pkl, 03_val.pkl, 03_test.pkl
Output:   outputs/phase1/04_X_train.pkl, 04_X_val.pkl, 04_X_test.pkl
          outputs/phase1/04_y_train.pkl, 04_y_val.pkl, 04_y_test.pkl
          models/phase1_encoders/{onehot,tool_vectorizer,scaler}.joblib
          outputs/phase1/04_feature_encoding_stats.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "outputs" / "phase1"
OUT_DIR = ROOT / "outputs" / "phase1"
ENCODERS_DIR = ROOT / "models" / "phase1_encoders"
OUT_JSON = OUT_DIR / "04_feature_encoding_stats.json"

DOS_ATTACK_TYPES = {"captcha_dos", "dos", "felony_dos", "offensive_email_dos", "swearwords_dos"}
CATEGORICAL_COLS = ["suite", "base_model", "defense", "task", "injection_task", "attack_family"]
NUMERIC_COLS = ["num_messages", "assistant_turns", "tool_calls", "tool_errors", "duration_seconds_log"]
TARGET_COL = "security"


def engineer_attack_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_dos_attack"] = df["attack_type"].isin(DOS_ATTACK_TYPES)
    df["attack_family"] = df["attack_type"].where(~df["is_dos_attack"], "dos_family")
    return df


def to_whitespace_tokens(series: pd.Series) -> pd.Series:
    """" -> "-delimited tool sequences -> whitespace-delimited, so a plain
    CountVectorizer regex token_pattern can split them (no custom callable,
    which would fail to unpickle outside the defining script - see module
    docstring). Also strips `<`/`>` from the `<empty-function-name>`
    artifact token so it can't produce an XGBoost-incompatible feature name
    (see module docstring "Amendment")."""
    return (
        series.str.replace(" -> ", " ", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )


def main() -> None:
    train = pd.read_pickle(IN_DIR / "03_train.pkl")
    val = pd.read_pickle(IN_DIR / "03_val.pkl")
    test = pd.read_pickle(IN_DIR / "03_test.pkl")

    train = engineer_attack_features(train)
    val = engineer_attack_features(val)
    test = engineer_attack_features(test)

    dos_rows_by_type = train.loc[train["is_dos_attack"], "attack_type"].value_counts().to_dict()

    # --- Fit encoders on TRAIN ONLY -----------------------------------------
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train[CATEGORICAL_COLS])
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))

    vec = CountVectorizer(token_pattern=r"[^\s]+", binary=True, lowercase=False)
    vec.fit(to_whitespace_tokens(train["tool_sequence"]))
    tool_feature_names = [f"tool__{t}" for t in vec.get_feature_names_out()]

    scaler = StandardScaler()
    scaler.fit(train[NUMERIC_COLS])

    feature_names = cat_feature_names + tool_feature_names + NUMERIC_COLS + ["is_dos_attack"]

    def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        cat = ohe.transform(df[CATEGORICAL_COLS])
        # count categories not seen during fit (all-zero rows after ignore)
        unknown_mask = (cat.sum(axis=1) < len(CATEGORICAL_COLS))
        n_unknown = int(unknown_mask.sum())

        tool = vec.transform(to_whitespace_tokens(df["tool_sequence"])).toarray()
        num = scaler.transform(df[NUMERIC_COLS])
        is_dos = df[["is_dos_attack"]].astype(int).to_numpy()

        X = np.hstack([cat, tool, num, is_dos]).astype(np.float32)
        return pd.DataFrame(X, columns=feature_names, index=df.index), n_unknown

    X_train, unk_train = transform(train)
    X_val, unk_val = transform(val)
    X_test, unk_test = transform(test)

    # OOV tool tokens: the vectorizer's vocabulary is fixed from train, so any
    # tool name appearing only in val/test is silently dropped from that
    # row's bag-of-tools encoding (CountVectorizer.transform ignores unknown
    # tokens rather than erroring). Small in volume here but real information
    # loss - measured and reported, not swept under the rug.
    tool_vocab = set(vec.get_feature_names_out())

    def oov_tool_tokens(df: pd.DataFrame) -> list[str]:
        toks: set[str] = set()
        for s in to_whitespace_tokens(df["tool_sequence"]):
            toks.update(s.split(" "))
        return sorted(toks - tool_vocab)

    oov_val = oov_tool_tokens(val)
    oov_test = oov_tool_tokens(test)

    y_train = train[TARGET_COL].astype(int)
    y_val = val[TARGET_COL].astype(int)
    y_test = test[TARGET_COL].astype(int)

    # --- Save ----------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_pickle(OUT_DIR / "04_X_train.pkl")
    X_val.to_pickle(OUT_DIR / "04_X_val.pkl")
    X_test.to_pickle(OUT_DIR / "04_X_test.pkl")
    y_train.to_pickle(OUT_DIR / "04_y_train.pkl")
    y_val.to_pickle(OUT_DIR / "04_y_val.pkl")
    y_test.to_pickle(OUT_DIR / "04_y_test.pkl")

    joblib.dump(ohe, ENCODERS_DIR / "onehot.joblib")
    joblib.dump(vec, ENCODERS_DIR / "tool_vectorizer.joblib")
    joblib.dump(scaler, ENCODERS_DIR / "scaler.joblib")
    (ENCODERS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    # Portability check: reload each encoder in a subprocess (a fresh Python
    # process, not just a fresh import in this same process) and confirm it
    # still transforms correctly. This is the check that would have caught
    # the custom-tokenizer __main__-pickling bug before it reached Phase 2.
    import subprocess
    import sys

    check_script = (
        "import joblib, pandas as pd; "
        f"ohe = joblib.load(r'{ENCODERS_DIR / 'onehot.joblib'}'); "
        f"vec = joblib.load(r'{ENCODERS_DIR / 'tool_vectorizer.joblib'}'); "
        f"scaler = joblib.load(r'{ENCODERS_DIR / 'scaler.joblib'}'); "
        "print('RELOAD_OK')"
    )
    result = subprocess.run([sys.executable, "-c", check_script], capture_output=True, text=True)
    reload_ok = "RELOAD_OK" in result.stdout
    if not reload_ok:
        raise RuntimeError(
            f"Encoder reload check failed in a fresh subprocess - encoders would not be "
            f"reusable by Phase 2+. stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    stats = {
        "step": "phase1_04_feature_encoding",
        "dos_attack_rows_by_type_train": dos_rows_by_type,
        "n_categorical_onehot_features": len(cat_feature_names),
        "n_tool_bag_of_tools_features": len(tool_feature_names),
        "n_numeric_features": len(NUMERIC_COLS),
        "n_is_dos_attack_features": 1,
        "total_feature_count": len(feature_names),
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "unknown_category_rows": {"train": unk_train, "val": unk_val, "test": unk_test},
        "y_positive_rate": {
            "train": round(float(y_train.mean()), 4),
            "val": round(float(y_val.mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "excluded_columns": [
            "pipeline", "injection", "security_label_status", "utility",
            "user_prompt", "final_answer", "attack_type", "tool_sequence", "duration_seconds",
        ],
        "encoder_reload_check_passed": reload_ok,
        "oov_tool_tokens_val": oov_val,
        "oov_tool_tokens_test": oov_test,
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Feature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"  categorical one-hot: {len(cat_feature_names)} cols")
    print(f"  tool bag-of-tools:   {len(tool_feature_names)} cols")
    print(f"  numeric:             {len(NUMERIC_COLS)} cols")
    print(f"  is_dos_attack:       1 col")
    print(f"Unknown-category rows (handle_unknown='ignore'): train={unk_train}, val={unk_val}, test={unk_test}")
    print(f"OOV tool tokens (dropped from bag-of-tools): val={oov_val}, test={oov_test}")
    print(f"DOS attack_type rows in train: {dos_rows_by_type}")
    print(f"Saved encoders to: {ENCODERS_DIR}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
