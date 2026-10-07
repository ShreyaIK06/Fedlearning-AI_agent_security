"""
Phase 1, Step 5 - Final validation of the processed modeling matrices.

Source:   Same discipline as dataset_preparation/'s own Phase 7 - an
          independent, holistic check of the finished artifact before
          anything downstream (Phase 2's model training) starts trusting
          it, rather than only trusting each individual step's own
          in-script checks.
Reason:   Steps 1-4 each validated their own transformation in isolation.
          This step looks at the four *processed* artifacts (X_train,
          X_val, X_test, and the split source dataframes) together, the way
          Phase 2 will actually consume them, and checks things that only
          become visible at that level: row-index alignment between X and
          its source dataframe, NaN/Inf contamination, feature-matrix
          column consistency across splits, zero-variance columns, and a
          final re-confirmation that the train/val/test partition (now at
          the feature-matrix level, not just the raw dataframe level from
          Step 3) still has zero row overlap and zero user_prompt leakage.

Input:    outputs/phase1/03_{train,val,test}.pkl
          outputs/phase1/04_X_{train,val,test}.pkl
          outputs/phase1/04_y_{train,val,test}.pkl
Output:   outputs/phase1/05_validate_processed_report.json
          (no new dataset artifacts - this step is pure validation)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "outputs" / "phase1"
OUT_JSON = IN_DIR / "05_validate_processed_report.json"

SPLITS = ["train", "val", "test"]


def main() -> None:
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    def check(name: str, condition: bool) -> None:
        (checks_passed if condition else checks_failed).append(name)

    source = {s: pd.read_pickle(IN_DIR / f"03_{s}.pkl") for s in SPLITS}
    X = {s: pd.read_pickle(IN_DIR / f"04_X_{s}.pkl") for s in SPLITS}
    y = {s: pd.read_pickle(IN_DIR / f"04_y_{s}.pkl") for s in SPLITS}

    # 1. Row-count / index alignment between X, y, and their source dataframe
    for s in SPLITS:
        check(f"{s}_X_y_row_count_match", len(X[s]) == len(y[s]))
        check(f"{s}_X_source_row_count_match", len(X[s]) == len(source[s]))
        check(f"{s}_X_index_matches_source_index", X[s].index.equals(source[s].index))
        check(f"{s}_y_index_matches_source_index", y[s].index.equals(source[s].index))

    # 2. Column consistency across splits
    cols_train, cols_val, cols_test = list(X["train"].columns), list(X["val"].columns), list(X["test"].columns)
    check("columns_identical_train_val", cols_train == cols_val)
    check("columns_identical_train_test", cols_train == cols_test)

    # 3. No NaN/Inf anywhere in the feature matrices
    for s in SPLITS:
        arr = X[s].to_numpy()
        check(f"{s}_no_nan", not np.isnan(arr).any())
        check(f"{s}_no_inf", not np.isinf(arr).any())

    # 4. y is strictly binary {0, 1}
    for s in SPLITS:
        check(f"{s}_y_binary_only", set(y[s].unique()) <= {0, 1})

    # 5. Categorical one-hot block: each row should have exactly 6 active
    #    dummies (one per categorical field: suite, base_model, defense,
    #    task, injection_task, attack_family), since Step 4 already
    #    confirmed zero handle_unknown="ignore" rows for any split.
    cat_cols = [c for c in cols_train if not c.startswith("tool__") and c not in
                ["num_messages", "assistant_turns", "tool_calls", "tool_errors",
                 "duration_seconds_log", "is_dos_attack"]]
    for s in SPLITS:
        row_sums = X[s][cat_cols].sum(axis=1)
        check(f"{s}_categorical_rows_sum_to_6", bool((row_sums == 6).all()))

    # 6. Zero-variance (constant) columns in X_train - not necessarily an
    #    error, but worth surfacing before Phase 2/5 treat every column as
    #    equally informative.
    zero_var_cols = X["train"].columns[X["train"].nunique() <= 1].tolist()

    # 7. Numeric feature scaling sanity: train columns should be
    #    approximately standardized (mean ~0, std ~1); val/test need not be
    #    exactly so (that's expected - they weren't used to fit the scaler)
    #    but should not be wildly divergent.
    numeric_cols = ["num_messages", "assistant_turns", "tool_calls", "tool_errors", "duration_seconds_log"]
    train_numeric_mean = X["train"][numeric_cols].mean().to_dict()
    train_numeric_std = X["train"][numeric_cols].std().to_dict()
    check(
        "train_numeric_features_standardized",
        all(abs(m) < 0.01 for m in train_numeric_mean.values())
        and all(abs(s - 1.0) < 0.01 for s in train_numeric_std.values()),
    )

    # 8. Zero row overlap between splits (index-level) + zero user_prompt leakage
    idx_train, idx_val, idx_test = set(X["train"].index), set(X["val"].index), set(X["test"].index)
    check("zero_index_overlap_train_val", idx_train.isdisjoint(idx_val))
    check("zero_index_overlap_train_test", idx_train.isdisjoint(idx_test))
    check("zero_index_overlap_val_test", idx_val.isdisjoint(idx_test))

    prompts = {s: set(source[s]["user_prompt"]) for s in SPLITS}
    check("zero_user_prompt_leakage_train_val", prompts["train"].isdisjoint(prompts["val"]))
    check("zero_user_prompt_leakage_train_test", prompts["train"].isdisjoint(prompts["test"]))
    check("zero_user_prompt_leakage_val_test", prompts["val"].isdisjoint(prompts["test"]))

    # 9. Total row count reconciles against the scoped population (Step 2)
    scoped = pd.read_pickle(IN_DIR / "02_scoped.pkl")
    total_processed = len(X["train"]) + len(X["val"]) + len(X["test"])
    check("total_rows_reconcile_with_scoped_population", total_processed == len(scoped))

    # 10. dtype check - float32 as designed (memory/perf, relevant for
    #     later PyTorch phases)
    for s in SPLITS:
        check(f"{s}_X_dtype_float32", (X[s].dtypes == np.float32).all())

    class_balance = {s: round(float(y[s].mean()), 4) for s in SPLITS}

    report = {
        "step": "phase1_05_validate_processed",
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "all_checks_passed": len(checks_failed) == 0,
        "shapes": {s: list(X[s].shape) for s in SPLITS},
        "class_balance_security_true_rate": class_balance,
        "zero_variance_columns_in_train": zero_var_cols,
        "train_numeric_mean": train_numeric_mean,
        "train_numeric_std": train_numeric_std,
        "total_rows_processed": total_processed,
        "total_rows_scoped_population": int(len(scoped)),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Checks passed: {len(checks_passed)}/{len(checks_passed) + len(checks_failed)}")
    if checks_failed:
        print("FAILED CHECKS:")
        for c in checks_failed:
            print(f"  - {c}")
    else:
        print("All checks passed.")
    print(f"Zero-variance columns in train: {zero_var_cols}")
    print(f"Class balance (security=True rate): {class_balance}")
    print(f"Saved: {OUT_JSON}")

    if checks_failed:
        raise AssertionError(f"{len(checks_failed)} validation check(s) failed: {checks_failed}")


if __name__ == "__main__":
    main()
