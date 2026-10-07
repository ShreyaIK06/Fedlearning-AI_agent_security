"""
Shared evaluation helper for Phase 2's four model scripts.

Not a numbered pipeline step itself (hence the leading underscore) - kept as
a small shared module so accuracy/precision/recall/F1/ROC-AUC/PR-AUC/
confusion-matrix computation isn't copy-pasted four times across
01_train_logistic_regression.py, 02_train_random_forest.py,
03_train_xgboost.py, and 04_train_mlp.py.

Every metrics dict includes `label_noise_ceiling_pct` (the report's
26.4% security-label flip rate on exact-key repeats, re-verified in
PIPELINE_READINESS_REPORT.md) alongside the numbers themselves, per
PROJECT_PLAN.md hard rule #4 - so no downstream consumer of these JSON files
can report a headline accuracy without the ceiling sitting right next to it.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABEL_NOISE_CEILING_PCT = 26.4  # security flip rate on exact-key repeats, Part 4 / PIPELINE_READINESS_REPORT.md


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str, split: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model": model_name,
        "split": split,
        "n_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "label_noise_ceiling_pct": LABEL_NOISE_CEILING_PCT,
        "label_noise_note": (
            "security labels flip on 26.4% of exact-repeat configurations (irreducible "
            "run-to-run stochasticity, not model error) - treat this as a ceiling on "
            "achievable accuracy, not a target to exceed without scrutiny."
        ),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def print_summary(metrics: dict) -> None:
    print(f"[{metrics['model']} / {metrics['split']}] n={metrics['n_samples']}")
    print(
        f"  accuracy={metrics['accuracy']}  precision={metrics['precision']}  "
        f"recall={metrics['recall']}  f1={metrics['f1']}  "
        f"roc_auc={metrics['roc_auc']}  pr_auc={metrics['pr_auc']}"
    )
    cm = metrics["confusion_matrix"]
    print(f"  confusion_matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"  (label-noise ceiling: {metrics['label_noise_ceiling_pct']}% flip rate on exact repeats)")
