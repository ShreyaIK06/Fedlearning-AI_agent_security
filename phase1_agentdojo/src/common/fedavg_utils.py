"""
Manual FedAvg building blocks, shared between Phase 3 (federated training)
and any later phase that needs to simulate per-client local training (Phase
5's gradient leakage attack targets exactly this local-update step - a
single client's gradient/weight-delta for one local round is the attack
surface, so the same local-training function is reused there rather than
redefined).

This exists instead of using the Flower framework because Flower's import
chain is broken in this environment (see PROJECT_PLAN.md "Environment
Findings") - and because gradient leakage needs direct access to each
client's local update, which a framework's Strategy/Client abstraction
would otherwise hide behind an API boundary.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.mlp_model import SecurityMLP


def average_state_dicts(state_dicts: list[dict], weights: list[float]) -> dict:
    """Weighted average of a list of model state_dicts (standard FedAvg
    aggregation: each client's update is weighted by its local dataset
    size / the round's total dataset size)."""
    assert len(state_dicts) == len(weights)
    total = sum(weights)
    norm_weights = [w / total for w in weights]

    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        avg_state[key] = sum(sd[key] * w for sd, w in zip(state_dicts, norm_weights))
    return avg_state


def train_local_model(
    global_state: dict,
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    local_epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[dict, float, float]:
    """One client's local training round: start from the broadcast global
    weights, train `local_epochs` epochs on this client's own data only,
    return the updated state_dict, elapsed wall time, and final local loss.
    """
    torch.manual_seed(seed)

    model = SecurityMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(global_state)
    model.train()

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=min(batch_size, len(X)), shuffle=True)

    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(local_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        final_loss = epoch_loss / len(X)
    elapsed = time.perf_counter() - start

    return model.state_dict(), elapsed, final_loss
