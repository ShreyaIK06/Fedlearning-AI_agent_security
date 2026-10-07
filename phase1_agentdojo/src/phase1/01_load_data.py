"""
Phase 1, Step 1 - Load the cleaned dataset and re-assert the invariants the
rest of this project depends on.

Source:   PIPELINE_READINESS_REPORT.md (verdict: Ready, 2026-08-07)
Reason:   `dataset_cleaned_final.csv` was independently validated before this
          project started. That validation is not re-litigated here - but
          this project must not *silently trust* a file it didn't produce.
          This script re-checks the small set of structural invariants every
          later Phase 1-7 script implicitly relies on (zero missing values,
          correct dtypes, a total and correct `security_label_status`
          mapping, the exact row/column shape) and fails loudly if any of
          them don't hold, rather than letting a violated assumption surface
          as a confusing bug three phases later.

          This is intentionally a much smaller check than the full Phase 0
          audit (PIPELINE_READINESS_REPORT.md) - it only re-verifies the
          handful of facts this project's code will assume are true from
          here on, not the entire cleaning history.

Input:    data/dataset_cleaned_final.csv
Output:   outputs/phase1/01_loaded.pkl        (dtype-preserving checkpoint)
          outputs/phase1/01_load_data_stats.json
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "dataset_cleaned_final.csv"
OUT_DIR = ROOT / "outputs" / "phase1"

OUT_PKL = OUT_DIR / "01_loaded.pkl"
OUT_JSON = OUT_DIR / "01_load_data_stats.json"

EXPECTED_ROWS = 36475
EXPECTED_COLUMNS = [
    "suite", "pipeline", "task", "attack_type", "injection", "injection_task",
    "num_messages", "assistant_turns", "tool_calls", "tool_sequence", "tool_errors",
    "user_prompt", "final_answer", "utility", "security", "duration_seconds",
    "base_model", "defense", "duration_seconds_log", "security_label_status",
]


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. This project expects a copy of "
            f"dataset_preparation/outputs/dataset_cleaned_final.csv under data/."
        )

    # keep_default_na=False is required: pandas' default NA-sniffing treats an
    # empty CSV field as NaN regardless of *why* it's empty, which silently
    # undoes dataset_preparation Phase 3's deliberate choice to encode
    # `final_answer`'s 140 "no verbal answer" rows as an empty string rather
    # than a null (report Part 3: "a genuinely informative state, don't
    # drop"). Without this flag, reloading the CSV reintroduces 140 "missing"
    # values that dataset_preparation explicitly eliminated - caught by the
    # invariant check below on first run.
    df = pd.read_csv(
        DATA_CSV,
        dtype={"utility": bool, "security": bool, "injection": bool},
        keep_default_na=False,
        na_values=[],
    )

    # --- Structural invariants -------------------------------------------------
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Column set/order changed.\nExpected: {EXPECTED_COLUMNS}\nGot: {list(df.columns)}"
    )
    total_missing = int(df.isna().sum().sum())
    assert total_missing == 0, f"Expected 0 missing values, found {total_missing}"

    for col in ("injection", "utility", "security"):
        assert df[col].dtype == bool, f"{col} expected dtype bool, got {df[col].dtype}"

    # --- Semantic invariants (relied on by every later Phase 1-7 script) -------
    expected_status = df["injection"].map({True: "measured", False: "not_applicable_placeholder"})
    assert (expected_status == df["security_label_status"]).all(), (
        "security_label_status no longer matches injection - the security-scoping "
        "rule this whole project depends on would silently break downstream."
    )

    recon = df.apply(
        lambda r: r["base_model"] if r["defense"] == "none" else f"{r['base_model']}-{r['defense']}",
        axis=1,
    )
    assert (recon == df["pipeline"]).all(), "base_model + defense no longer reconstructs pipeline"

    assert df.duplicated().sum() == 0, "Unexpected exact duplicate rows in the cleaned dataset"

    # --- Save --------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PKL)

    stats = {
        "step": "phase1_01_load_data",
        "source": str(DATA_CSV),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": total_missing,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "security_label_status_counts": df["security_label_status"].value_counts().to_dict(),
        "suite_counts": df["suite"].value_counts().to_dict(),
        "invariant_checks_passed": [
            "row_count",
            "column_set_and_order",
            "zero_missing_values",
            "bool_dtypes_injection_utility_security",
            "security_label_status_matches_injection",
            "base_model_defense_reconstructs_pipeline",
            "zero_exact_duplicates",
        ],
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {DATA_CSV}")
    print("All invariant checks passed:")
    for check in stats["invariant_checks_passed"]:
        print(f"  - {check}")
    print(f"Saved: {OUT_PKL}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
