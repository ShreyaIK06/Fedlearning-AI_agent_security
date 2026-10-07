"""
Phase 3, Step 1 - Partition the training data into per-`suite` FL clients.

Source:   AgentDojo_Dataset_Research_Report.md Part 16; PROJECT_PLAN.md Phase 3
Reason:   `suite` is the report's recommended primary FL-client axis: "literally
          maps to 'a banking org, a Slack workspace, a travel agency, an
          office suite'... naturally non-IID (different tools, different
          attack susceptibility)". Unlike the `base_model`/`pipeline` axis,
          `suite` needs no rare-value pruning (PROJECT_PLAN.md hard rule #3
          is specific to `base_model`/`pipeline` clients, not this one) -
          all 4 suites are well-populated.

          Partitioning happens on the Phase 1 TRAIN split only
          (`04_X_train.pkl`/`04_y_train.pkl`) - val and test stay global and
          unpartitioned, since they're used to monitor convergence and
          produce the final apples-to-apples comparison against Phase 2's
          centralized MLP (PROJECT_PLAN.md Phase 4), not to simulate
          per-client behavior.

          `suite` itself is one-hot encoded within X_train (`suite_banking`,
          `suite_slack`, `suite_travel`, `suite_workspace`), not a separate
          column - so client membership is recovered from Phase 1 Step 3's
          source dataframe (`03_train.pkl`), which Step 5's validation
          already confirmed shares an identical row index with `04_X_train.pkl`.

Input:    outputs/phase1/03_train.pkl (source dataframe, for the `suite` column)
          outputs/phase1/04_X_train.pkl, 04_y_train.pkl
Output:   outputs/phase3/client_{suite}_X.pkl, client_{suite}_y.pkl  (x4)
          outputs/phase3/01_client_partition_stats.json
          figures/phase3/client_class_balance.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = ROOT / "outputs" / "phase1"
OUT_DIR = ROOT / "outputs" / "phase3"
FIG_DIR = ROOT / "figures" / "phase3"
OUT_JSON = OUT_DIR / "01_client_partition_stats.json"

SUITES = ["banking", "slack", "travel", "workspace"]


def main() -> None:
    source_train = pd.read_pickle(PHASE1_DIR / "03_train.pkl")
    X_train = pd.read_pickle(PHASE1_DIR / "04_X_train.pkl")
    y_train = pd.read_pickle(PHASE1_DIR / "04_y_train.pkl")

    assert X_train.index.equals(source_train.index), (
        "X_train and its source dataframe have diverged - cannot safely partition by suite"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    client_stats = {}
    for suite in SUITES:
        mask = source_train["suite"] == suite
        client_X = X_train.loc[mask]
        client_y = y_train.loc[mask]

        assert len(client_X) == len(client_y)
        assert len(client_X) > 0, f"Suite '{suite}' has zero training rows - cannot form a client"

        client_X.to_pickle(OUT_DIR / f"client_{suite}_X.pkl")
        client_y.to_pickle(OUT_DIR / f"client_{suite}_y.pkl")

        client_stats[suite] = {
            "n_rows": int(len(client_X)),
            "pct_of_train": round(len(client_X) / len(X_train) * 100, 2),
            "security_true_rate": round(float(client_y.mean()), 4),
        }

    total_partitioned = sum(s["n_rows"] for s in client_stats.values())
    assert total_partitioned == len(X_train), (
        f"Partitioned rows ({total_partitioned}) don't sum to total train rows ({len(X_train)}) - "
        f"every training row must belong to exactly one suite"
    )

    stats = {
        "step": "phase3_01_partition_clients_by_suite",
        "n_clients": len(SUITES),
        "total_train_rows": int(len(X_train)),
        "client_stats": client_stats,
        "non_iid_evidence": (
            "security_true_rate varies by suite: "
            f"min={min(s['security_true_rate'] for s in client_stats.values())}, "
            f"max={max(s['security_true_rate'] for s in client_stats.values())} - "
            "genuine label-distribution heterogeneity across clients, matching report Part 16's "
            "expectation of non-IID clients along the suite axis."
        ),
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sizes = [client_stats[s]["n_rows"] for s in SUITES]
    rates = [client_stats[s]["security_true_rate"] * 100 for s in SUITES]
    ax1.bar(SUITES, sizes, color="steelblue")
    ax1.set_ylabel("Training rows")
    ax1.set_title("Client size by suite")
    ax2.bar(SUITES, rates, color="indianred")
    ax2.set_ylabel("security=True rate (%)")
    ax2.set_title("Client label distribution (non-IID)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "client_class_balance.png", dpi=150)
    plt.close(fig)

    print(f"Partitioned {len(X_train)} training rows into {len(SUITES)} clients:")
    for suite, s in client_stats.items():
        print(f"  {suite:10s}  n={s['n_rows']:5d}  ({s['pct_of_train']}%)  security_true_rate={s['security_true_rate']}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {FIG_DIR / 'client_class_balance.png'}")


if __name__ == "__main__":
    main()
