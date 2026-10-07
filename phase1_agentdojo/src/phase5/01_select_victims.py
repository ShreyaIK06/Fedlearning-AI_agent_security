"""
Phase 5, Step 1 - Select victim examples and compute their true gradients.

Source:   PROJECT_PLAN.md Phase 5; AgentDojo_Dataset_Research_Report.md Part 17
Reason:   All three attack scripts (DLG, iDLG, Inverting Gradients) need the
          exact same victims and true gradients to be comparable - computed
          once here, reused by all three, rather than each attack script
          re-selecting its own (which could accidentally compare different
          victims and produce a misleading "attack A beat attack B" result
          that was really just "victim set A was easier than victim set B").

          12 victims, 3 from each of the 4 suite clients (Phase 3's
          partition) - deliberately spans the non-IID clients rather than
          sampling globally, so Phase 5's results can be read per-suite if
          a pattern emerges (e.g. "banking's send_money-heavy traces are
          easier/harder to invert than workspace's").

          Attacked model: the TRAINED federated global model
          (`models/phase3/federated_mlp_state_dict.pt`), in `.eval()` mode
          (disables Dropout - see `src/common/gradient_leakage.py` docstring
          for why that matters). A null baseline (random-noise guess vs.
          each victim) is also computed here, to give the three attacks'
          eventual reconstruction quality a reference point: "how much
          better than a blind guess did the attack do."

Input:    models/phase3/federated_mlp_state_dict.pt
          outputs/phase3/client_{suite}_X.pkl, client_{suite}_y.pkl
          models/phase1_encoders/{onehot,tool_vectorizer}.joblib, feature_names.json
Output:   outputs/phase5/victims.pt   (X, y, true gradients, per-victim metadata)
          outputs/phase5/01_victim_selection_stats.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.mlp_model import SecurityMLP  # noqa: E402
from common.gradient_leakage import compute_gradient, load_feature_blocks  # noqa: E402

PHASE3_DIR = ROOT / "outputs" / "phase3"
MODEL_IN = ROOT / "models" / "phase3" / "federated_mlp_state_dict.pt"
ENCODERS_DIR = ROOT / "models" / "phase1_encoders"
OUT_DIR = ROOT / "outputs" / "phase5"
OUT_JSON = OUT_DIR / "01_victim_selection_stats.json"

SUITES = ["banking", "slack", "travel", "workspace"]
VICTIMS_PER_SUITE = 3
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
RANDOM_STATE = 42


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    blocks = load_feature_blocks(ENCODERS_DIR)
    input_dim = len(blocks.feature_names)

    model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model.load_state_dict(torch.load(MODEL_IN))
    model.eval()

    victims_X, victims_y, victim_meta, victim_grads = [], [], [], []

    for suite in SUITES:
        X = pd.read_pickle(PHASE3_DIR / f"client_{suite}_X.pkl")
        y = pd.read_pickle(PHASE3_DIR / f"client_{suite}_y.pkl")
        chosen_idx = rng.choice(len(X), size=VICTIMS_PER_SUITE, replace=False)

        for i in chosen_idx:
            x_np = X.iloc[i].to_numpy(dtype=np.float32)
            y_np = np.array([y.iloc[i]], dtype=np.float32)

            x_t = torch.tensor(x_np).unsqueeze(0)
            y_t = torch.tensor(y_np)

            g_true = compute_gradient(model, x_t, y_t)

            victims_X.append(x_np)
            victims_y.append(y_np[0])
            victim_grads.append(g_true)
            victim_meta.append({"suite": suite, "row_index": int(i), "true_label": float(y_np[0])})

    victims_X_arr = np.stack(victims_X)
    victims_y_arr = np.array(victims_y, dtype=np.float32)

    # Null baseline: how far a random-noise guess sits from each real victim
    # (same distribution the attacks' own dummy init uses), and from the
    # training-set mean vector (a slightly-informed blind guess) - both give
    # the eventual attacks' MSE numbers a reference point.
    train_all_X = pd.concat(
        [pd.read_pickle(PHASE3_DIR / f"client_{s}_X.pkl") for s in SUITES], ignore_index=True
    ).to_numpy(dtype=np.float32)
    mean_vector = train_all_X.mean(axis=0)

    torch.manual_seed(RANDOM_STATE)
    random_guess_mse = []
    mean_guess_mse = []
    for x in victims_X_arr:
        noise_guess = torch.randn(input_dim).numpy()
        random_guess_mse.append(float(np.mean((x - noise_guess) ** 2)))
        mean_guess_mse.append(float(np.mean((x - mean_vector) ** 2)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "victims_X": torch.tensor(victims_X_arr),
        "victims_y": torch.tensor(victims_y_arr),
        "victim_grads": victim_grads,
        "victim_meta": victim_meta,
        "input_dim": input_dim,
        "mean_vector": torch.tensor(mean_vector),
    }, OUT_DIR / "victims.pt")

    stats = {
        "step": "phase5_01_select_victims",
        "n_victims": len(victim_meta),
        "victims_per_suite": VICTIMS_PER_SUITE,
        "input_dim": input_dim,
        "victim_meta": victim_meta,
        "null_baseline_mse": {
            "random_noise_guess_mean": round(float(np.mean(random_guess_mse)), 4),
            "training_mean_vector_guess_mean": round(float(np.mean(mean_guess_mse)), 4),
        },
        "attacked_model": str(MODEL_IN),
        "note": (
            "Attacks target a single-example gradient at the trained federated model's "
            "final weights (the classic DLG/iDLG/IG threat model - one forward+backward pass, "
            "not a multi-epoch FedAvg delta). See src/common/gradient_leakage.py module docstring."
        ),
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Selected {len(victim_meta)} victims across {len(SUITES)} suites ({VICTIMS_PER_SUITE} each)")
    print(f"Null baseline MSE - random noise guess: {stats['null_baseline_mse']['random_noise_guess_mean']}")
    print(f"Null baseline MSE - training mean guess: {stats['null_baseline_mse']['training_mean_vector_guess_mean']}")
    print(f"Saved: {OUT_DIR / 'victims.pt'}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
