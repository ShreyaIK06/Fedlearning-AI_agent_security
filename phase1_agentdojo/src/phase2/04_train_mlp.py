"""
Phase 2, Step 4 - MLP (PyTorch) baseline for `security`.

Source:   PROJECT_PLAN.md Phase 2 & Phase 3; AgentDojo_Dataset_Research_Report.md Part 19
Reason:   The deep-learning baseline requested alongside the three classical
          models (Steps 1-3) - and, more importantly, THE model Phase 3
          (Federated Learning) trains identically per-client and Phase 5
          (gradient leakage) attacks. Its architecture therefore lives in
          `src/common/mlp_model.py`, imported here rather than redefined,
          so "centralized vs. federated" (Phase 4) is a true apples-to-
          apples comparison and not an artifact of two slightly different
          model definitions.

          Imbalance handling: `BCEWithLogitsLoss(pos_weight=...)`, the
          PyTorch analogue of Steps 1-2's `class_weight="balanced"` and
          Step 3's `scale_pos_weight` - same design decision, same
          data-derived ratio, applied consistently across all four models
          (report Part 19).

Input:    outputs/phase1/04_X_{train,val}.pkl, 04_y_{train,val}.pkl
Output:   models/phase2/mlp_state_dict.pt
          evaluation/phase2/mlp_val_metrics.json
          outputs/phase2/04_mlp_training_history.csv (added retroactively for
            Phase 4's centralized-vs-federated convergence comparison, which
            needs the raw per-epoch numbers, not just the rendered figure)
          figures/phase2/mlp_confusion_matrix_val.png
          figures/phase2/mlp_training_curve.png
"""

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.mlp_model import SecurityMLP  # noqa: E402
from _metrics_utils import compute_metrics, print_summary, save_metrics  # noqa: E402

DATA_DIR = ROOT / "outputs" / "phase1"
MODEL_OUT = ROOT / "models" / "phase2" / "mlp_state_dict.pt"
METRICS_OUT = ROOT / "evaluation" / "phase2" / "mlp_val_metrics.json"
HISTORY_CSV_OUT = ROOT / "outputs" / "phase2" / "04_mlp_training_history.csv"
CM_FIGURE_OUT = ROOT / "figures" / "phase2" / "mlp_confusion_matrix_val.png"
CURVE_FIGURE_OUT = ROOT / "figures" / "phase2" / "mlp_training_curve.png"

RANDOM_STATE = 42
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 8


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_tensor(df_or_series) -> torch.Tensor:
    return torch.tensor(df_or_series.to_numpy(dtype=np.float32))


def main() -> None:
    set_seed(RANDOM_STATE)

    X_train = pd.read_pickle(DATA_DIR / "04_X_train.pkl")
    y_train = pd.read_pickle(DATA_DIR / "04_y_train.pkl")
    X_val = pd.read_pickle(DATA_DIR / "04_X_val.pkl")
    y_val = pd.read_pickle(DATA_DIR / "04_y_val.pkl")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)
    X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    model = SecurityMLP(input_dim=X_train.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_roc_auc": [], "epoch_wall_time_sec": []}
    best_val_roc_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_proba = torch.sigmoid(val_logits).numpy()
        from sklearn.metrics import roc_auc_score

        val_roc_auc = roc_auc_score(y_val_t.numpy(), val_proba)
        epoch_wall_time = time.perf_counter() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_roc_auc"].append(val_roc_auc)
        history["epoch_wall_time_sec"].append(epoch_wall_time)

        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_roc_auc={val_roc_auc:.4f}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val_roc_auc improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(X_val_t)).numpy()
    val_pred = (val_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_val_t.numpy().astype(int), val_pred, val_proba, model_name="mlp", split="val")
    metrics["architecture"] = {"hidden_dims": list(HIDDEN_DIMS), "dropout": DROPOUT, "lr": LR, "batch_size": BATCH_SIZE}
    metrics["best_epoch"] = int(history["epoch"][np.argmax(history["val_roc_auc"])])
    metrics["total_epochs_trained"] = len(history["epoch"])
    metrics["total_training_wall_time_sec"] = round(float(sum(history["epoch_wall_time_sec"])), 4)
    print_summary(metrics)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    save_metrics(metrics, METRICS_OUT)

    HISTORY_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(HISTORY_CSV_OUT, index=False)

    CM_FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: Blocked", "Pred: Breach"],
        yticklabels=["True: Blocked", "True: Breach"],
        ax=ax,
    )
    ax.set_title("MLP - Confusion Matrix (val)")
    fig.tight_layout()
    fig.savefig(CM_FIGURE_OUT, dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax1.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax1.axvline(metrics["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="best epoch")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.legend()
    ax1.set_title("Training/validation loss")

    ax2.plot(history["epoch"], history["val_roc_auc"], color="green")
    ax2.axvline(metrics["best_epoch"], color="gray", linestyle="--", alpha=0.6)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val ROC-AUC")
    ax2.set_title("Validation ROC-AUC")
    fig.tight_layout()
    fig.savefig(CURVE_FIGURE_OUT, dpi=150)
    plt.close(fig)

    print(f"Best epoch: {metrics['best_epoch']} (of {metrics['total_epochs_trained']} trained)")
    print(f"Saved model state_dict: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print(f"Saved figures: {CM_FIGURE_OUT}, {CURVE_FIGURE_OUT}")


if __name__ == "__main__":
    main()
