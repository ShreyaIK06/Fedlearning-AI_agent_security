"""
phase2_common.py  -  shared helpers for Phase 2 training scripts
--------------------------------------------------------------------
Not a pipeline step itself. Holds:
  * paths (Phase 2 outputs + where Phase 1's reusable code lives)
  * the import hook that makes Phase 1's SecurityMLP, fedavg_utils and
    _metrics_utils importable UNCHANGED
  * loading the feature arrays made by phase2_features.py
  * Phase 2's OWN label-noise ceiling (replaces Phase 1's 26.4%)
  * a metrics wrapper around Phase 1's compute_metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent          # fed-agent-leakage/
P1_SRC = ROOT / "phase1_agentdojo" / "src"
PROC_DIR = ROOT / "data" / "processed"
FEATURES_NPZ = PROC_DIR / "phase2_features.npz"
TABLE_CSV = PROC_DIR / "phase2_table.csv"

MODEL_DIR = ROOT / "models" / "phase2_mlp"
EVAL_DIR = ROOT / "evaluation" / "phase2_synthetic"
OUT_DIR = ROOT / "outputs" / "phase2_synthetic"
FIG_DIR = ROOT / "figures" / "phase2_synthetic"

SUITES = ["banking", "slack", "travel", "workspace"]   # same order as Phase 1
RANDOM_STATE = 42

# ------------------------------------------------- reuse Phase 1 code as-is
if not (P1_SRC / "common" / "mlp_model.py").exists():
    raise FileNotFoundError(f"Phase 1 code not found at {P1_SRC} - is phase1_agentdojo/ next to phase2_synthetic/?")
sys.path.insert(0, str(P1_SRC))                 # -> `from common.mlp_model import ...` (fedavg_utils needs this)
sys.path.insert(0, str(P1_SRC / "phase2"))      # -> `from _metrics_utils import ...`

from common.mlp_model import SecurityMLP                          # noqa: E402,F401
from common.fedavg_utils import average_state_dicts, train_local_model  # noqa: E402,F401
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402,F401


# ------------------------------------------------------------- data
def load_features():
    """Return dict of arrays saved by phase2_features.py."""
    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(f"{FEATURES_NPZ} not found - run phase2_features.py first")
    d = np.load(FEATURES_NPZ, allow_pickle=False)
    return {k: d[k] for k in d.files}


# ------------------------------------------------ Phase 2 label noise
def label_noise_ceiling():
    """Phase 2's equivalent of Phase 1's 26.4% label-noise ceiling.

    Every scenario was run 3 times with identical inputs. If the repeats
    disagree on `breach`, the label is partly luck (LLM run-to-run
    randomness), and no model can get all of those runs right.
      pct_scenarios_flipping : % of scenarios whose repeats don't all agree
      pct_records_minority   : % of records that disagree with their
                               scenario's majority label = records even a
                               perfect per-scenario model would get wrong
    """
    t = pd.read_csv(TABLE_CSV)
    g = t.groupby("scenario_id")["breach"]
    multi = g.count() >= 2                                  # need >=2 repeats to compare
    flips = (g.nunique() > 1)[multi]
    majority = g.transform(lambda s: int(s.mean() >= 0.5))
    minority = (t["breach"] != majority)[t["scenario_id"].map(multi)]
    return {
        "pct_scenarios_flipping": round(100 * flips.mean(), 1),
        "n_scenarios_compared": int(multi.sum()),
        "pct_records_minority": round(100 * minority.mean(), 1),
        "definition": ("pct_scenarios_flipping = % of scenarios (suite+task+injection_style, "
                       "3 identical repeats) whose breach labels are not all equal"),
    }


def metrics_for(y_true, proba, model_name, split, noise):
    """Phase 1's compute_metrics (threshold 0.5), with Phase 2's own noise ceiling swapped in."""
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= 0.5).astype(int)
    m = compute_metrics(y_true, pred, proba, model_name=model_name, split=split)
    m["label_noise_ceiling_pct"] = noise["pct_scenarios_flipping"]
    m["label_noise_note"] = noise["definition"] + " (Phase 2 synthetic; Phase 1's value was 26.4)"
    m["label_noise_detail"] = noise
    return m


def per_suite_recall(y_true, proba, suites):
    """Indicative only - per-suite test sets are small (~20-35 records)."""
    pred = (proba >= 0.5).astype(int)
    rows = []
    for s in SUITES:
        m = suites == s
        yt, yp = y_true[m].astype(int), pred[m]
        n_pos = int(yt.sum())
        rows.append({"suite": s, "n": int(m.sum()), "n_breach": n_pos,
                     "accuracy": round(float((yt == yp).mean()), 3) if m.any() else None,
                     "breach_recall": round(float(yp[yt == 1].mean()), 3) if n_pos else None})
    return pd.DataFrame(rows)