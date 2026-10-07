"""
Shared MLP architecture, used unmodified across Phase 2 (centralized),
Phase 3 (federated - each client trains a local copy of this exact class),
Phase 5 (gradient leakage - attacks this exact model's gradients), and
Phase 6 (differential privacy - Opacus wraps this exact model).

Living in `src/common/` (not inside `src/phase2/`) specifically so every
later phase imports the *same* class rather than each phase re-defining its
own copy that could quietly drift out of sync - the user's own instructions
for Phase 3 are explicit that centralized-vs-federated must differ only in
*how* training happens, not in what's being trained.

Design constraint: no BatchNorm. BatchNorm's running statistics mix
information across the batch in a way that breaks per-sample gradient
computation - Opacus (Phase 6) cannot privatize it without substituting
GroupNorm/LayerNorm, and per-sample gradients are also central to Phase 5's
gradient leakage attack. Dropout is fine (Opacus supports it; it's a no-op
under `model.eval()` for both the leakage attack and the FL weight-sharing
step, where it doesn't matter which client's dropout mask was used since
inference/attack happens on a set of weights, not against a filtered
activation trace).
"""

import torch
import torch.nn as nn


class SecurityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logits, shape (batch,)
