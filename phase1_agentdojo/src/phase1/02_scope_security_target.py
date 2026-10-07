"""
Phase 1, Step 2 - Scope the dataset to the rows where `security` is a real
measurement, establishing the modeling population for Phases 2-7.

Source:   AgentDojo_Dataset_Research_Report.md Part 5, Part 13.1
          PROJECT_PLAN.md hard rule #2
Reason:   `security` is only a genuine attack-outcome label when
          `injection == True`. When `injection == False`, the AgentDojo
          codebase hard-codes `security = True` as a vacuous placeholder -
          dataset_preparation's Phase 6 made this explicit via the
          `security_label_status` column rather than dropping those rows
          (since `utility` is valid on the full dataset). This project's
          chosen target is `security` (see PROJECT_PLAN.md), so this is the
          step where that scoping is actually *applied*, not just available.

          Modeling the unscoped column would let a classifier trivially
          learn "no injection -> secure", inflating accuracy with no real
          signal (report Part 13.1's "structural leakage" risk) - this is
          exactly the failure mode PROJECT_PLAN.md's hard rule #2 exists to
          prevent.

Input:    outputs/phase1/01_loaded.pkl
Output:   outputs/phase1/02_scoped.pkl
          outputs/phase1/02_scope_security_target_stats.json
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_PKL = ROOT / "outputs" / "phase1" / "01_loaded.pkl"
OUT_DIR = ROOT / "outputs" / "phase1"

OUT_PKL = OUT_DIR / "02_scoped.pkl"
OUT_JSON = OUT_DIR / "02_scope_security_target_stats.json"

MEASURED = "measured"


def main() -> None:
    df = pd.read_pickle(IN_PKL)
    before_rows = int(len(df))
    before_status_counts = df["security_label_status"].value_counts().to_dict()

    scoped = df.loc[df["security_label_status"] == MEASURED].copy()

    # Hard checks on the result, not just the filter condition itself.
    assert (scoped["injection"] == True).all(), "Scoped rows must all have injection==True"  # noqa: E712
    assert (scoped["security_label_status"] == MEASURED).all()
    assert scoped["security"].isna().sum() == 0
    dropped = before_rows - len(scoped)
    assert dropped == before_status_counts.get("not_applicable_placeholder", 0), (
        "Row count dropped by scoping doesn't match the placeholder count - "
        "the filter condition and the column's own accounting disagree."
    )

    security_dist = scoped["security"].value_counts(normalize=True).to_dict()
    security_dist_pct = {str(k): round(v * 100, 2) for k, v in security_dist.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scoped.to_pickle(OUT_PKL)

    stats = {
        "step": "phase1_02_scope_security_target",
        "rows_before": before_rows,
        "rows_after": int(len(scoped)),
        "rows_dropped": int(dropped),
        "security_label_status_counts_before": before_status_counts,
        "security_distribution_pct_after": security_dist_pct,
        "report_expected_security_true_pct": 15.86,
        "suite_counts_after": scoped["suite"].value_counts().to_dict(),
        "checks_passed": [
            "all_scoped_rows_injection_true",
            "all_scoped_rows_status_measured",
            "zero_null_security_in_scope",
            "row_drop_matches_placeholder_count",
        ],
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Rows: {before_rows} -> {len(scoped)} (dropped {dropped} not_applicable_placeholder rows)")
    print(f"security distribution (scoped): {security_dist_pct}")
    print(f"Saved: {OUT_PKL}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
