"""
Phase 1, Step 3 - Grouped train/validation/test split on `user_prompt`.

Source:   AgentDojo_Dataset_Research_Report.md Part 13.4
          PROJECT_PLAN.md hard rule #1
Reason:   Only 167 distinct `user_prompt` strings exist across the whole
          dataset (121 within this project's `security`-scoped population -
          see the pre-flight check this script asserts against), each
          reused across many `pipeline`/`attack_type` combinations. A
          row-random split would put near-identical rows (same prompt,
          different model) on both sides of the split, letting a model
          memorize per-prompt base rates instead of learning generalizable
          signal - producing an optimistic, non-generalizing result. The
          report's fix (Part 13.4): group the split on `user_prompt` so an
          entire prompt's rows land on exactly one side.

          Per-`user_prompt`-group `security` rates vary widely (0% to 63%,
          std ~0.15 - checked directly on this data before writing this
          script), so a plain GroupShuffleSplit risks an unlucky draw that
          skews the class balance between splits. `StratifiedGroupKFold`
          (sklearn >=1.1) is used instead - it keeps groups intact like
          GroupShuffleSplit, but explicitly tries to balance the stratify
          target's distribution across folds.

Input:    outputs/phase1/02_scoped.pkl
Output:   outputs/phase1/03_train.pkl
          outputs/phase1/03_val.pkl
          outputs/phase1/03_test.pkl
          outputs/phase1/03_grouped_split_stats.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
IN_PKL = ROOT / "outputs" / "phase1" / "02_scoped.pkl"
OUT_DIR = ROOT / "outputs" / "phase1"

TRAIN_PKL = OUT_DIR / "03_train.pkl"
VAL_PKL = OUT_DIR / "03_val.pkl"
TEST_PKL = OUT_DIR / "03_test.pkl"
OUT_JSON = OUT_DIR / "03_grouped_split_stats.json"

RANDOM_STATE = 42


def describe_split(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "unique_user_prompts": int(df["user_prompt"].nunique()),
        "security_true_pct": round(float(df["security"].mean()) * 100, 2),
    }


def main() -> None:
    df = pd.read_pickle(IN_PKL)

    groups = df["user_prompt"]
    y = df["security"]

    # Split 1: carve off the TEST fold (~20%), stratified on `security`,
    # grouped on `user_prompt`.
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(sgkf_outer.split(df, y, groups))

    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Split 2: carve VAL out of the remaining train_val (~20% of train_val,
    # ~16% overall), same technique, independent random_state.
    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    tv_groups = train_val_df["user_prompt"]
    tv_y = train_val_df["security"]
    train_idx, val_idx = next(sgkf_inner.split(train_val_df, tv_y, tv_groups))

    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # --- Leakage assertions: no user_prompt may appear in more than one split ---
    train_prompts = set(train_df["user_prompt"])
    val_prompts = set(val_df["user_prompt"])
    test_prompts = set(test_df["user_prompt"])

    assert train_prompts.isdisjoint(val_prompts), "user_prompt leakage between train and val"
    assert train_prompts.isdisjoint(test_prompts), "user_prompt leakage between train and test"
    assert val_prompts.isdisjoint(test_prompts), "user_prompt leakage between val and test"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Row count doesn't reconcile after split"
    assert len(train_prompts | val_prompts | test_prompts) == df["user_prompt"].nunique(), (
        "Some user_prompt values were lost or duplicated across splits"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(TRAIN_PKL)
    val_df.to_pickle(VAL_PKL)
    test_df.to_pickle(TEST_PKL)

    stats = {
        "step": "phase1_03_grouped_split",
        "method": "StratifiedGroupKFold(n_splits=5) x2, grouped on user_prompt, stratified on security",
        "random_state": RANDOM_STATE,
        "total_rows": int(len(df)),
        "total_unique_user_prompts": int(df["user_prompt"].nunique()),
        "train": describe_split(train_df),
        "val": describe_split(val_df),
        "test": describe_split(test_df),
        "leakage_checks_passed": [
            "train_val_prompt_disjoint",
            "train_test_prompt_disjoint",
            "val_test_prompt_disjoint",
            "row_count_reconciles",
            "no_user_prompt_lost_or_duplicated",
        ],
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Train: {stats['train']}")
    print(f"Val:   {stats['val']}")
    print(f"Test:  {stats['test']}")
    print("All leakage checks passed (zero user_prompt overlap across splits).")
    print(f"Saved: {TRAIN_PKL}, {VAL_PKL}, {TEST_PKL}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
