"""
Phase 6, Step 1 - DP-SGD training sweep across privacy budgets.

Source:   AgentDojo_Dataset_Research_Report.md Part 18; PROJECT_PLAN.md Phase 6
Reason:   Trains the SAME `SecurityMLP` architecture as Phase 2/3 (imported,
          not redefined) under Opacus DP-SGD at each of the report's
          recommended target epsilon values {1, 3, 8} - the designed
          response to Phase 5's finding that single-example gradients leak
          real (if noisy) information. epsilon=infinity is not retrained
          here; it's Phase 2's existing non-private MLP, already trained
          and evaluated, reused as the no-DP reference point.

          Before training anything, `ModuleValidator.validate()` confirmed
          `SecurityMLP` has zero Opacus-incompatibility errors - validates
          the architecture decision made back in Phase 2 (no BatchNorm,
          specifically because it breaks per-sample gradients) rather than
          discovering a problem here.

          Parameters, and why:
          - `max_grad_norm` (C) = 1.0: midpoint of the report's suggested
            0.5-1.5 starting range (Part 18), not separately tuned per
            epsilon - a scope limitation stated explicitly, not hidden.
          - `target_delta` = 1/N where N = 19,052 (THIS project's actual
            Phase 1 train-split size) ~= 5.25e-5 - derived from this
            project's real N, not copied from the report's figure (which
            was computed against the raw dataset's larger N).
          - `epochs` = 30, `batch_size` = 256: matched to Phase 2's
            centralized MLP (which converged in 26-34 epochs) so DP and
            non-DP training run for a comparable duration - a fairer utility
            comparison than picking an arbitrary different epoch count.
          - Opacus's Poisson-sampling `DPDataLoader` and per-sample gradient
            clipping/noising are handled internally by
            `PrivacyEngine.make_private_with_epsilon`, which also computes
            the noise multiplier needed to hit each target epsilon given
            these fixed epochs/batch-size/delta - not manually tuned.

Input:    outputs/phase1/04_X_{train,val}.pkl, 04_y_{train,val}.pkl
Output:   models/phase6/dp_mlp_eps{1,3,8}_state_dict.pt
          outputs/phase6/01_dp_sgd_training_history.csv
          outputs/phase6/01_dp_sgd_sweep_stats.json
          figures/phase6/dp_sgd_convergence_by_epsilon.png
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook is firing")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.mlp_model import SecurityMLP  # noqa: E402

DATA_DIR = ROOT / "outputs" / "phase1"
MODELS_DIR = ROOT / "models" / "phase6"
OUT_DIR = ROOT / "outputs" / "phase6"
FIG_DIR = ROOT / "figures" / "phase6"

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
TARGET_EPSILONS = [1.0, 3.0, 8.0]
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
MAX_GRAD_NORM = 1.0
RANDOM_STATE = 42


def to_tensor(df_or_series) -> torch.Tensor:
    return torch.tensor(df_or_series.to_numpy(dtype=np.float32))


def main() -> None:
    X_train = pd.read_pickle(DATA_DIR / "04_X_train.pkl")
    y_train = pd.read_pickle(DATA_DIR / "04_y_train.pkl")
    X_val = pd.read_pickle(DATA_DIR / "04_X_val.pkl")
    y_val = pd.read_pickle(DATA_DIR / "04_y_val.pkl")

    input_dim = X_train.shape[1]
    n_train = len(X_train)
    target_delta = 1.0 / n_train

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)
    X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val)

    # Sanity check (should be empty - confirms Phase 2's no-BatchNorm design
    # decision was actually necessary and sufficient for Opacus compatibility).
    probe_model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    validation_errors = ModuleValidator.validate(probe_model, strict=False)
    assert not validation_errors, f"SecurityMLP is not Opacus-compatible: {validation_errors}"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_history = []
    sweep_stats = {}

    for target_eps in TARGET_EPSILONS:
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

        privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
            module=model, optimizer=optimizer, data_loader=train_loader,
            target_epsilon=target_eps, target_delta=target_delta,
            epochs=EPOCHS, max_grad_norm=MAX_GRAD_NORM,
        )
        noise_multiplier = dp_optimizer.noise_multiplier

        best_val_roc_auc = -1.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            dp_model.train()
            epoch_loss = 0.0
            n_seen = 0
            for xb, yb in dp_loader:
                if len(xb) == 0:  # Poisson sampling can produce empty batches
                    continue
                dp_optimizer.zero_grad()
                logits = dp_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                dp_optimizer.step()
                epoch_loss += loss.item() * len(xb)
                n_seen += len(xb)

            dp_model.eval()
            with torch.no_grad():
                val_logits = dp_model(X_val_t)
                val_proba = torch.sigmoid(val_logits).numpy()
            val_roc_auc = roc_auc_score(y_val_t.numpy(), val_proba)
            eps_so_far = privacy_engine.get_epsilon(target_delta)

            all_history.append({
                "target_epsilon": target_eps, "epoch": epoch,
                "train_loss": epoch_loss / max(n_seen, 1),
                "val_roc_auc": val_roc_auc, "epsilon_spent": eps_so_far,
            })

            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_state = {k: v.clone() for k, v in dp_model._module.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(f"[eps={target_eps}] epoch {epoch:2d}  val_roc_auc={val_roc_auc:.4f}  eps_spent={eps_so_far:.3f}")

        final_eps = privacy_engine.get_epsilon(target_delta)
        torch.save(best_state, MODELS_DIR / f"dp_mlp_eps{int(target_eps)}_state_dict.pt")

        sweep_stats[f"eps_{target_eps}"] = {
            "target_epsilon": target_eps,
            "achieved_epsilon": round(final_eps, 4),
            "target_delta": target_delta,
            "noise_multiplier": round(noise_multiplier, 4),
            "max_grad_norm": MAX_GRAD_NORM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "best_val_roc_auc": round(best_val_roc_auc, 4),
        }
        print(f"[eps={target_eps}] DONE - achieved_epsilon={final_eps:.3f}  noise_multiplier={noise_multiplier:.3f}  "
              f"best_val_roc_auc={best_val_roc_auc:.4f}")

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(OUT_DIR / "01_dp_sgd_training_history.csv", index=False)

    stats = {
        "step": "phase6_01_train_dp_sgd_sweep",
        "n_train": n_train,
        "target_delta": target_delta,
        "sweep": sweep_stats,
        "opacus_module_validation_errors": validation_errors,
    }
    (OUT_DIR / "01_dp_sgd_sweep_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    for target_eps in TARGET_EPSILONS:
        sub = history_df[history_df["target_epsilon"] == target_eps]
        ax.plot(sub["epoch"], sub["val_roc_auc"], label=f"target ε={target_eps}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val ROC-AUC")
    ax.set_title("DP-SGD Convergence by Privacy Budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dp_sgd_convergence_by_epsilon.png", dpi=150)
    plt.close(fig)

    print("\n=== Sweep summary ===")
    print(json.dumps(sweep_stats, indent=2))
    print(f"Saved: {OUT_DIR / '01_dp_sgd_training_history.csv'}")
    print(f"Saved: {OUT_DIR / '01_dp_sgd_sweep_stats.json'}")
    print(f"Saved: {FIG_DIR / 'dp_sgd_convergence_by_epsilon.png'}")


if __name__ == "__main__":
    main()
