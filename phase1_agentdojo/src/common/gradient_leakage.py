"""
Shared building blocks for Phase 5's gradient leakage attacks (DLG, iDLG,
Inverting Gradients), reused by all three attack scripts so the actual
attack scripts stay focused on "which loss / which label strategy", not
plumbing.

Threat model (stated explicitly, since it matters for interpreting results):
the classic DLG/iDLG/Inverting-Gradients papers attack a SINGLE gradient
computed from ONE example at the model's CURRENT weights (the "FedSGD"
setting: one forward+backward pass, batch size 1). Phase 3's FedAvg clients
instead send a WEIGHT DELTA after several local epochs over mini-batches -
attacking that directly is a harder, less-standard problem (the delta
entangles many examples' gradients across several optimizer steps). To stay
faithful to the literature these three named attacks come from, Phase 5
attacks the idealized single-example gradient instead: take the trained
federated global model (`models/phase3/federated_mlp_state_dict.pt`) as the
snapshot an honest-but-curious server holds, pick one real client example,
compute the gradient that example alone would produce, and try to invert
it. This is a real, well-motivated leakage question (does the model's
architecture leak individual examples given their exact gradient?) but it
is a different, narrower question than "can I invert a FedAvg round's
aggregated multi-step delta" - noted here so results aren't over-read.

Dropout note: the model is kept in `.eval()` mode for the ENTIRE attack
(both computing the true gradient and every reconstruction step) - Dropout
is stochastic per forward pass, and comparing gradients across different
random dropout masks would make gradient matching ill-posed. `.eval()`
makes Dropout a deterministic no-op, which is the standard setting these
attacks are studied in.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """Gradient of BCEWithLogits loss w.r.t. every model parameter, for a
    single (x, y) example. Model must already be in .eval() mode."""
    model.zero_grad()
    logit = model(x)
    loss = F.binary_cross_entropy_with_logits(logit, y)
    grads = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in grads]


def infer_label_from_final_bias_grad(grad_final_bias: torch.Tensor) -> torch.Tensor:
    """iDLG-style closed-form label inference, derived for our binary
    BCEWithLogits setting (iDLG's original derivation is for multi-class
    cross-entropy; this is the same idea - the true label is recoverable
    exactly, without optimizing for it - re-derived for the binary case):

    dL/dlogit = sigmoid(logit) - y. The final layer's bias gradient equals
    dL/dlogit exactly (bias contributes additively to the logit). Since
    sigmoid(logit) in (0, 1):
      y=1  =>  dL/dlogit = sigmoid(logit) - 1 < 0
      y=0  =>  dL/dlogit = sigmoid(logit) - 0 > 0
    so sign(grad_final_bias) alone reveals y with 100% accuracy, for any
    model weights and any input - no optimization needed.
    """
    return (grad_final_bias < 0).float().reshape(1)


def gradient_matching_loss(g_dummy: list[torch.Tensor], g_true: list[torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "l2":
        return sum(((gd - gt) ** 2).sum() for gd, gt in zip(g_dummy, g_true))
    if mode == "cosine":
        flat_dummy = torch.cat([g.flatten() for g in g_dummy])
        flat_true = torch.cat([g.flatten() for g in g_true])
        cos_sim = F.cosine_similarity(flat_dummy.unsqueeze(0), flat_true.unsqueeze(0)).squeeze()
        return 1.0 - cos_sim
    raise ValueError(f"Unknown gradient_matching_loss mode: {mode}")


def reconstruct(
    model: nn.Module,
    g_true: list[torch.Tensor],
    input_dim: int,
    mode: str,
    know_label: bool,
    true_y: torch.Tensor | None = None,
    n_iters: int = 150,
    lr: float = 1.0,
    seed: int = 0,
    clamp_range: tuple[float, float] | None = (-15.0, 25.0),
    optimizer_name: str = "lbfgs",
) -> tuple[torch.Tensor, torch.Tensor | None, list[float]]:
    """Reconstructs a dummy input (and, if `know_label` is False, a dummy
    label) by gradient descent on the gradient-matching loss. Returns
    (x_reconstructed, y_reconstructed_or_None, loss_history).

    `clamp_range` guards against a real failure mode observed running this
    on Phase 5's data: unconstrained optimization occasionally drives
    x_dummy to numerically enormous values (observed up to ~1e17 magnitude,
    and separately to norm ~5865 mid-optimization) that still happen to
    match the true gradient closely (many wildly different inputs can
    produce similar gradients once a ReLU saturates or a weight is
    near-zero). The clamp range (-15, 25) is set from this project's actual
    engineered feature range (checked directly: real values span roughly
    -1.9 to +20.9, 99.9% within +/-2.1) - loose enough to not constrain a
    genuine reconstruction, tight enough to stop pure float blow-up.

    `optimizer_name`: "lbfgs" (default - matches the original DLG paper,
    used for the L2-loss attacks, DLG/iDLG) or "adam" (matches Geiping et
    al.'s actual Inverting-Gradients implementation, used for the
    cosine-loss attack). This is not a stylistic choice: LBFGS maintains an
    internal quasi-Newton curvature history built from consecutive
    (parameter-change, gradient-change) pairs, and externally clamping
    x_dummy between steps silently corrupts that history - confirmed by
    direct diagnosis (see Phase 5 CHANGELOG entry) that LBFGS+clamp on the
    cosine loss gets permanently stuck (loss frozen at exactly 1.0) after
    one large, clamp-triggering jump. Adam's per-parameter moving-average
    state tolerates external modification between steps far better, and is
    what the Inverting Gradients paper itself uses - this is the literature-
    correct pairing, not an ad hoc patch to make numbers look better.

    Returns the BEST iterate found (lowest gradient-matching loss), not
    necessarily the last one - also confirmed by direct diagnosis that
    Adam's cosine-loss trajectory can be non-monotonic (observed: loss dips
    to 0.87 around iteration 205, then drifts back up to 1.0 by iteration
    500 with no learning-rate decay to lock in the improvement). Tracking
    the best iterate is standard practice in DLG-family reference
    implementations for exactly this reason, and applies uniformly to all
    three attacks (harmless for DLG/iDLG's already-mostly-monotonic runs).
    """
    torch.manual_seed(seed)
    model.eval()

    x_dummy = torch.randn(1, input_dim, requires_grad=True)

    if know_label:
        assert true_y is not None
        y_fixed = true_y.clone().detach()
        params = [x_dummy]
    else:
        y_logit_dummy = torch.randn(1, requires_grad=True)
        params = [x_dummy, y_logit_dummy]

    def compute_loss():
        model.zero_grad()
        y_for_loss = y_fixed if know_label else torch.sigmoid(y_logit_dummy)
        logit = model(x_dummy)
        loss = F.binary_cross_entropy_with_logits(logit, y_for_loss)
        g_dummy = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return gradient_matching_loss(g_dummy, g_true, mode)

    history: list[float] = []
    best_loss = float("inf")
    best_x = x_dummy.detach().clone()
    best_y_logit = y_logit_dummy.detach().clone() if not know_label else None

    def record_if_best(loss_value: float) -> None:
        nonlocal best_loss, best_x, best_y_logit
        if loss_value < best_loss:
            best_loss = loss_value
            best_x = x_dummy.detach().clone()
            if not know_label:
                best_y_logit = y_logit_dummy.detach().clone()

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=lr)

        def closure():
            optimizer.zero_grad()
            gm_loss = compute_loss()
            gm_loss.backward()
            return gm_loss

        for _ in range(n_iters):
            loss_val = optimizer.step(closure)
            if clamp_range is not None:
                with torch.no_grad():
                    x_dummy.clamp_(*clamp_range)
            loss_float = float(loss_val.detach())
            history.append(loss_float)
            record_if_best(loss_float)

    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in range(n_iters):
            optimizer.zero_grad()
            gm_loss = compute_loss()
            gm_loss.backward()
            optimizer.step()
            if clamp_range is not None:
                with torch.no_grad():
                    x_dummy.clamp_(*clamp_range)
            loss_float = float(gm_loss.detach())
            history.append(loss_float)
            record_if_best(loss_float)

    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    y_out = None if know_label else torch.sigmoid(best_y_logit)
    return best_x, y_out, history


@dataclass
class FeatureBlocks:
    """Column-index boundaries for each semantic block of the 176-dim
    feature vector, derived from the fitted Phase 1 encoders (not
    hardcoded) - used to score *what kind* of information a reconstruction
    recovered (which suite, which tools, ...), not just raw MSE."""

    categorical_fields: dict[str, tuple[int, int]]  # field -> (start, end) column range
    tool_range: tuple[int, int]
    numeric_range: tuple[int, int]
    is_dos_attack_index: int
    feature_names: list[str]


def load_feature_blocks(encoders_dir) -> FeatureBlocks:
    import json

    ohe = joblib.load(encoders_dir / "onehot.joblib")
    vec = joblib.load(encoders_dir / "tool_vectorizer.joblib")
    feature_names = json.loads((encoders_dir / "feature_names.json").read_text())

    cat_input_cols = ["suite", "base_model", "defense", "task", "injection_task", "attack_family"]
    categorical_fields = {}
    idx = 0
    for field, cats in zip(cat_input_cols, ohe.categories_):
        categorical_fields[field] = (idx, idx + len(cats))
        idx += len(cats)

    n_cat = idx
    n_tool = len(vec.get_feature_names_out())
    tool_range = (n_cat, n_cat + n_tool)
    numeric_range = (n_cat + n_tool, n_cat + n_tool + 5)
    is_dos_attack_index = n_cat + n_tool + 5

    return FeatureBlocks(
        categorical_fields=categorical_fields,
        tool_range=tool_range,
        numeric_range=numeric_range,
        is_dos_attack_index=is_dos_attack_index,
        feature_names=feature_names,
    )


def closed_form_first_layer_attack(g_true: list[torch.Tensor]) -> torch.Tensor:
    """Exact, non-iterative reconstruction of a single example's input from
    its gradient through a linear first layer - discovered while
    investigating a counter-intuitive Phase 7 result (DP-SGD models
    appeared MORE vulnerable to the optimization-based DLG/iDLG/IG attacks
    than the non-private model). Root cause, verified directly: for a
    linear layer `y = W1 @ x + b1`, a single example's gradient satisfies
    the EXACT identity `dL/dW1 = outer(dL/db1, x)` - true for any model,
    any weights, any training procedure, always, because it falls straight
    out of the chain rule for a linear layer with no approximation. Given
    `dL/db1` (the first layer's bias gradient, `g_true[1]`) and `dL/dW1`
    (`g_true[0]`), x is recoverable exactly via `x = dW1[j] / db1[j]` for
    any row j where `db1[j] != 0` - no optimization needed at all.

    This is NOT a property of DP-SGD, or of this project's specific
    architecture beyond "has a Linear first layer" (true of every MLP in
    this project). It reveals that the iterative gradient-matching attacks
    (DLG/iDLG/IG) can dramatically understate a model's TRUE single-
    gradient vulnerability, since they only find what their optimizer
    happens to converge to, not the exact solution that provably exists.
    Verified numerically: max absolute error ~1e-9 (float32 precision
    noise) against both a DP-SGD model and the non-private model.
    """
    dW1, db1 = g_true[0], g_true[1]
    j = int(db1.abs().argmax().item())
    if db1[j].abs().item() < 1e-12:
        # Degenerate case: this example's bias gradient is (numerically)
        # zero for every hidden unit - the same saturated-logit style
        # degeneracy documented elsewhere in this module. No exact solution
        # is extractable; return zeros rather than dividing by ~0.
        return torch.zeros_like(dW1[j]).unsqueeze(0)
    return (dW1[j] / db1[j]).unsqueeze(0)


def score_reconstruction(x_true: np.ndarray, x_recon: np.ndarray, blocks: FeatureBlocks) -> dict:
    """Quantitative reconstruction-quality scoring, split by semantic
    feature block rather than one aggregate number - directly answers
    "what did the attacker actually learn" (which suite, which tools, ...)."""
    result: dict = {}

    result["overall_mse"] = float(np.mean((x_true - x_recon) ** 2))

    cat_correct = 0
    cat_total = 0
    for field, (s, e) in blocks.categorical_fields.items():
        true_cat = int(np.argmax(x_true[s:e]))
        recon_cat = int(np.argmax(x_recon[s:e]))
        correct = true_cat == recon_cat
        result[f"categorical_{field}_correct"] = bool(correct)
        cat_correct += int(correct)
        cat_total += 1
    result["categorical_accuracy"] = cat_correct / cat_total

    ts, te = blocks.tool_range
    true_tools = (x_true[ts:te] > 0.5).astype(int)
    recon_tools = (x_recon[ts:te] > 0.5).astype(int)
    tp = int(((true_tools == 1) & (recon_tools == 1)).sum())
    fp = int(((true_tools == 0) & (recon_tools == 1)).sum())
    fn = int(((true_tools == 1) & (recon_tools == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if true_tools.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    result["tool_precision"] = precision
    result["tool_recall"] = recall
    result["tool_true_count"] = int(true_tools.sum())

    ns, ne = blocks.numeric_range
    result["numeric_mae_std_units"] = float(np.mean(np.abs(x_true[ns:ne] - x_recon[ns:ne])))

    true_dos = x_true[blocks.is_dos_attack_index] > 0.5
    recon_dos = x_recon[blocks.is_dos_attack_index] > 0.5
    result["is_dos_attack_correct"] = bool(true_dos == recon_dos)

    return result
