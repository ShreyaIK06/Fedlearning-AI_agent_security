"""
phase2_dp.py  -  Phase 2 DP-SGD stage (mirrors Phase 1's src/phase6/01-02 + src/phase7/01-02)
--------------------------------------------------------------------------------------------
Part A  DP-SGD utility sweep (Phase 1 phase6/01 + 02)
        CENTRALIZED SecurityMLP (unchanged) trained with Opacus DP-SGD at target
        eps in {1, 3, 8}; eps=inf is the non-private centralized MLP.
        Phase 1 settings: 30 epochs, batch 256, Adam lr 1e-3, max_grad_norm C=1.0,
        delta = 1/N_train, noise multiplier chosen by Opacus to hit each eps,
        checkpoint = best val ROC-AUC.
        Phase 2 additions:
          * run over the SAME 5 seeds as phase2_seeds.py (each re-splits + re-fits
            encoders), because a single run on 361 rows can't separate DP's cost
            from split noise (+-8.5 accuracy pts non-private); DP cost is reported
            PAIRED per seed (eps=inf minus eps=k on the same split)
          * checkpoint selection uses the Phase 2 warm-up (no epoch < 20 is chosen),
            same as every other Phase 2 model
          * breach recall AND non-breach specificity reported separately (Phase 1's
            minority/majority split; our classes are 50/50, so this tests whether
            Phase 1's recall collapse was an imbalance effect or a DP effect)

Part B  Leakage vs eps (Phase 1 phase7/01), seed-42 models only
        Same 12 victims as phase2_attacks.py (raw x/y from victims.pt), gradients
        RECOMPUTED against each model. Only eps changes: eps=inf is the CENTRALIZED
        non-private model (models/phase2_mlp/central_mlp_state_dict.pt), not the
        federated one, so FL and DP aren't confounded. Attacks: Inverting Gradients
        (Phase 1 settings) + closed-form. Weight norm per model.
        Phase 2 addition: exact-reconstruction counts + median MSE.

Part C  Master tradeoff table + figure (Phase 1 phase7/02), plus a Phase 1 vs
        Phase 2 side-by-side if Phase 1's outputs are present locally.

Options:  --delta 1e-5   stricter delta (sensitivity run; default is Phase 1's 1/N)
          --lr 1e-2 --batch 64 --epochs 60
                         small-data DP settings (sensitivity run; defaults are Phase 1's
                         1e-3 / 256 / 30). Same target eps: Opacus raises the noise to
                         pay for the extra steps, so the privacy guarantee is unchanged.
          --seeds 42     quick single-seed run
Any non-default option writes to separately tagged files; the main results are never overwritten.

Run from phase2_synthetic/:   python phase2_dp.py
"""

import argparse
import json
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

import phase2_seeds as S
import phase2_train_central as C
from phase2_attacks import (EXACT_MSE, load_feature_blocks, score_reconstruction)
from phase2_common import (EVAL_DIR, FIG_DIR, MODEL_DIR, OUT_DIR, ROOT, SecurityMLP, load_features)
from phase2_features import build_table
from common.gradient_leakage import (closed_form_first_layer_attack, compute_gradient,  # Phase 1, unchanged
                                     infer_label_from_final_bias_grad, reconstruct)

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
warnings.filterwarnings("ignore", message="Full backward hook is firing")

# ---- identical to Phase 1's src/phase6/01_train_dp_sgd.py
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
TARGET_EPSILONS = [1.0, 3.0, 8.0]
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
MAX_GRAD_NORM = 1.0
# ---- identical to Phase 1's src/phase7/01 (Inverting Gradients)
IG_ITERS, IG_LR = 500, 0.05
# ---- Phase 2 protocol
WARMUP = C.WARMUP          # 20
SEEDS = [42, 0, 1, 2, 3]

DP_MODEL_DIR = ROOT / "models" / "phase2_dp"
DP_OUT = OUT_DIR / "dp"
VICTIMS_PT = OUT_DIR / "attacks" / "victims.pt"
P1 = ROOT / "phase1_agentdojo"


def eps_label(e):
    return "inf" if e == float("inf") else str(int(e))


def full_metrics(y, p):
    y = y.astype(int); pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "breach_recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p)}


def weight_norm(model):
    return float(sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5)


# ============================================================ Part A: DP training
def train_dp(d, seed, target_eps, delta):
    """Phase 1's DP-SGD loop, + Phase 2 warm-up for checkpoint selection."""
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_tr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    n_pos = int(y_tr.sum()); pos_weight = torch.tensor((len(y_tr) - n_pos) / n_pos, dtype=torch.float32)

    model = SecurityMLP(input_dim=X_tr.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    pe = PrivacyEngine()
    dp_model, dp_opt, dp_loader = pe.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=loader, target_epsilon=target_eps,
        target_delta=delta, epochs=EPOCHS, max_grad_norm=MAX_GRAD_NORM)

    best, best_state, best_epoch, steps = -1.0, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        dp_model.train()
        for xb, yb in dp_loader:
            if len(xb) == 0:                       # Poisson sampling can give empty batches
                continue
            dp_opt.zero_grad(); crit(dp_model(xb), yb).backward(); dp_opt.step(); steps += 1
        dp_model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(dp_model(X_va)).numpy())
        if epoch >= WARMUP and auc > best:
            best, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in dp_model._module.state_dict().items()}
    secs = time.perf_counter() - t0

    final = SecurityMLP(input_dim=X_tr.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    final.load_state_dict(best_state); final.eval()
    info = {"achieved_epsilon": pe.get_epsilon(delta), "noise_multiplier": float(dp_opt.noise_multiplier),
            "sample_rate": float(dp_loader.sample_rate) if hasattr(dp_loader, "sample_rate") else None,
            "dp_steps": steps, "best_epoch": best_epoch, "train_sec": secs}
    return final, best_state, info


BASELINE_TUNED = False   # set from --baseline-tuned


def train_central(d, seed):
    """Non-private centralized MLP (eps=inf) - the same loop as phase2_seeds.run_central /
    phase2_train_central.py (batch 256, patience 25, warm-up 20), but returns the model."""
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_tr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    X_va, y_va = torch.tensor(d["X_val"]), d["y_val"]
    n_pos = int(y_tr.sum()); pos_weight = torch.tensor((len(y_tr) - n_pos) / n_pos, dtype=torch.float32)
    # --baseline-tuned: the non-private model gets the SAME lr / batch / max epochs as the DP
    # models (patience 25 + warm-up 20 unchanged), so DP cost isn't inflated by unequal tuning
    bs, lr, max_ep = (BATCH_SIZE, LR, EPOCHS) if BASELINE_TUNED else (C.BATCH_SIZE, C.LR, C.MAX_EPOCHS)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    model = SecurityMLP(input_dim=X_tr.shape[1], hidden_dims=C.HIDDEN_DIMS, dropout=C.DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state, wait, updates = -1.0, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, max_ep + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step(); updates += 1
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(model(X_va)).numpy())
        if epoch < C.WARMUP:
            continue
        if auc > best:
            best, wait = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= C.EARLY_STOP_PATIENCE:
            break
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state); model.eval()
    return model, {"weight_updates": updates, "train_sec": secs}


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


# ============================================================ Part B: leakage
def attack(model, VX, Vy, blocks, input_dim):
    ig_rows, cf_rows = [], []
    for i, (x, y) in enumerate(zip(VX, Vy)):
        g = compute_gradient(model, x.unsqueeze(0), y.reshape(1))
        with torch.no_grad():
            logit = float(model(x.unsqueeze(0)).item())
        zero_grad = all(float(t.abs().max()) == 0.0 for t in g)   # float32 saturation: sigmoid(logit)-y == 0
        y_inf = infer_label_from_final_bias_grad(g[-1]); ok = int(y_inf.item()) == int(y.item())
        xr, _, hist = reconstruct(model, g, input_dim, mode="cosine", know_label=True, true_y=y_inf,
                                  n_iters=IG_ITERS, lr=IG_LR, seed=42 + i, optimizer_name="adam")
        s = score_reconstruction(x.numpy(), xr.squeeze(0).numpy(), blocks); s["label_ok"] = ok; s["best_loss"] = min(hist)
        s["logit"], s["true_label"], s["zero_gradient"] = logit, float(y.item()), zero_grad
        ig_rows.append(s)
        xc = closed_form_first_layer_attack(g).squeeze(0).numpy()
        s = score_reconstruction(x.numpy(), xc, blocks); s["label_ok"] = ok
        s["logit"], s["true_label"], s["zero_gradient"] = logit, float(y.item()), zero_grad
        s["max_abs_error"] = float(np.max(np.abs(x.numpy() - xc))); cf_rows.append(s)
    return ig_rows, cf_rows


def summ(rows):
    mse = [r["overall_mse"] for r in rows]
    return {"exact": int(sum(m < EXACT_MSE for m in mse)), "n": len(rows),
            "median_mse": float(np.median(mse)), "mean_mse": float(np.mean(mse)),
            "cat_acc": float(np.mean([r["categorical_accuracy"] for r in rows])),
            "tool_precision": float(np.mean([r["tool_precision"] for r in rows])),
            "tool_recall": float(np.mean([r["tool_recall"] for r in rows])),
            "label_acc": float(np.mean([r["label_ok"] for r in rows])),
            "zero_gradient": int(sum(r["zero_gradient"] for r in rows)),
            "exact_of_nonzero": int(sum(r["overall_mse"] < EXACT_MSE for r in rows if not r["zero_gradient"])),
            "max_abs_logit": float(max(abs(r["logit"]) for r in rows))}


# ================================================================== main
def main():
    global LR, BATCH_SIZE, EPOCHS, BASELINE_TUNED
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=None, help="default: Phase 1's 1/N_train")
    ap.add_argument("--lr", type=float, default=LR, help="DP Adam learning rate (Phase 1: 1e-3)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="DP batch size (Phase 1: 256)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="DP epochs (Phase 1: 30)")
    ap.add_argument("--baseline-tuned", action="store_true",
                    help="train the non-private eps=inf model with the same --lr/--batch/--epochs as DP")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    tag = f"_delta{a.delta:g}" if a.delta else ""
    if (a.lr, a.batch, a.epochs) != (LR, BATCH_SIZE, EPOCHS):
        tag += f"_lr{a.lr:g}_b{a.batch}_e{a.epochs}"
    BASELINE_TUNED = a.baseline_tuned
    if BASELINE_TUNED:
        tag += "_tunedbase"
    LR, BATCH_SIZE, EPOCHS = a.lr, a.batch, a.epochs
    if EPOCHS <= WARMUP:
        raise SystemExit(f"--epochs must be > warm-up ({WARMUP})")
    print(f"DP settings: lr={LR:g}  batch={BATCH_SIZE}  epochs={EPOCHS}  C={MAX_GRAD_NORM}  "
          f"delta={'1/N' if not a.delta else a.delta}  warm-up={WARMUP}  "
          f"{'[Phase 1 settings]' if not tag else '[SENSITIVITY RUN, files tagged ' + tag + ']'}")
    if BASELINE_TUNED:
        print("Non-private eps=inf baseline: SAME lr/batch/max-epochs as DP (patience 25, warm-up 20)")
    if 42 not in a.seeds:
        raise SystemExit("seed 42 must be included (Part B attacks the seed-42 models)")

    probe = SecurityMLP(input_dim=10, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    errs = ModuleValidator.validate(probe, strict=False)
    assert not errs, f"SecurityMLP not Opacus-compatible: {errs}"
    print("Opacus ModuleValidator: SecurityMLP compatible (no BatchNorm) - OK")

    # ---------------------------------------------------------------- Part A
    df = build_table()
    rows, keep42 = [], {}
    for seed in a.seeds:
        d = S.make_data(df, seed)
        if seed == 42:   # must be the exact arrays the main scripts used
            ref = load_features()
            assert np.array_equal(ref["X_train"], d["X_train"]) and np.array_equal(ref["X_test"], d["X_test"])
            d42 = d
        n = len(d["y_train"]); delta = a.delta or 1.0 / n
        print(f"\nseed {seed}: train={n}  delta={delta:.2e}  test breach={d['y_test'].mean():.2f}")

        cmodel, cinfo = train_central(d, seed)
        if seed == 42 and BASELINE_TUNED:
            keep42["inf"] = cmodel
            DP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(cmodel.state_dict(), DP_MODEL_DIR / f"nonprivate_mlp{tag}_state_dict.pt")
        cm = full_metrics(d["y_test"], predict(cmodel, d["X_test"]))
        rows.append({"seed": seed, "epsilon": "inf", **{k: round(v, 4) for k, v in cm.items()},
                     "achieved_epsilon": float("inf"), "noise_multiplier": 0.0, "dp_steps": cinfo["weight_updates"],
                     "weight_norm": round(weight_norm(cmodel), 3), "train_sec": round(cinfo["train_sec"], 2)})
        print(f"  eps=inf  acc={cm['accuracy']:.4f} recall={cm['breach_recall']:.4f} "
              f"spec={cm['specificity']:.4f} auc={cm['roc_auc']:.4f}  (non-private, {cinfo['weight_updates']} steps)")

        for e in TARGET_EPSILONS:
            model, state, info = train_dp(d, seed, e, delta)
            met = full_metrics(d["y_test"], predict(model, d["X_test"]))
            rows.append({"seed": seed, "epsilon": eps_label(e), **{k: round(v, 4) for k, v in met.items()},
                         "achieved_epsilon": round(info["achieved_epsilon"], 3),
                         "noise_multiplier": round(info["noise_multiplier"], 3), "dp_steps": info["dp_steps"],
                         "weight_norm": round(weight_norm(model), 3), "train_sec": round(info["train_sec"], 2)})
            print(f"  eps={eps_label(e):3s}  acc={met['accuracy']:.4f} recall={met['breach_recall']:.4f} "
                  f"spec={met['specificity']:.4f} auc={met['roc_auc']:.4f}  (achieved eps={info['achieved_epsilon']:.2f}, "
                  f"noise x{info['noise_multiplier']:.2f}, {info['dp_steps']} noisy steps, best epoch {info['best_epoch']})")
            if seed == 42:
                DP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(state, DP_MODEL_DIR / f"dp_mlp_eps{eps_label(e)}{tag}_state_dict.pt")
                keep42[eps_label(e)] = model

    runs = pd.DataFrame(rows)
    order = ["1", "3", "8", "inf"]
    cols = ["accuracy", "breach_recall", "specificity", "f1", "roc_auc"]
    summ_u = runs.groupby("epsilon")[cols + ["noise_multiplier", "weight_norm", "train_sec"]].agg(["mean", "std"]).reindex(order)

    print("\n" + "=" * 100 + f"\nPart A - DP-SGD utility, TEST, mean +- std over seeds {a.seeds}\n" + "=" * 100)
    for e in order:
        s = summ_u.loc[e]
        print(f"  eps={e:3s}  " + "  ".join(f"{k}={s[(k,'mean')]:.3f}+-{s[(k,'std')]:.3f}" for k in cols)
              + f"  noise x{s[('noise_multiplier','mean')]:.2f}")

    piv = runs.pivot(index="seed", columns="epsilon", values=cols).reindex(a.seeds)
    print("\nPaired DP cost per seed (eps=inf minus eps=k, in points; + = DP costs)")
    cost_rows = []
    for e in ["1", "3", "8"]:
        for k in cols:
            diff = 100 * (piv[(k, "inf")] - piv[(k, e)])
            cost_rows.append({"epsilon": e, "metric": k, "mean": round(diff.mean(), 1), "std": round(diff.std(), 1)})
            print(f"  eps={e}  {k:13s} mean={diff.mean():+6.1f}  std={diff.std():5.1f}  "
                  f"per-seed=[{', '.join(f'{x:+.1f}' for x in diff)}]")
    print("  (Phase 1, single run: breach recall 83.3% -> 47.5% at eps=1; specificity slightly UP)")

    DP_OUT.mkdir(parents=True, exist_ok=True); EVAL_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(DP_OUT / f"dp_runs{tag}.csv", index=False)
    summ_u.to_csv(EVAL_DIR / f"dp_utility_summary{tag}.csv")
    pd.DataFrame(cost_rows).to_csv(EVAL_DIR / f"dp_paired_cost{tag}.csv", index=False)

    # ---------------------------------------------------------------- Part B
    print("\n" + "=" * 100 + "\nPart B - leakage vs eps (seed-42 centralized models, 12 victims, gradients recomputed)\n" + "=" * 100)
    V = torch.load(VICTIMS_PT)
    VX, Vy = V["victims_X"], V["victims_y"]
    names = [str(n) for n in load_features()["feature_names"]]
    blocks = load_feature_blocks(names); input_dim = len(names)

    if BASELINE_TUNED:
        models = keep42                      # eps=inf = the tuned non-private seed-42 model trained above
    else:
        inf_model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
        inf_model.load_state_dict(torch.load(MODEL_DIR / "central_mlp_state_dict.pt")); inf_model.eval()
        models = {**keep42, "inf": inf_model}

    init_norms = []
    for sd in range(20):
        torch.manual_seed(sd); init_norms.append(weight_norm(SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)))
    print(f"Reference: UNTRAINED model weight norm = {np.mean(init_norms):.3f} +- {np.std(init_norms):.3f}  "
          f"(a trained model's norm close to this barely moved from its random start)")
    leak, leak_rows = {}, []
    for e in order:
        mdl = models[e]; mdl.eval()
        ig, cf = attack(mdl, VX, Vy, blocks, input_dim)
        si, sc = summ(ig), summ(cf)
        wn = weight_norm(mdl)
        leak[e] = {"weight_norm": wn, "inverting_gradients": {"per_victim": ig, "summary": si},
                   "closed_form_first_layer": {"per_victim": cf, "summary": sc,
                                               "max_abs_error": max(r["max_abs_error"] for r in cf)}}
        leak_rows.append({"epsilon": e, "weight_norm": round(wn, 3),
                          "IG_exact": f"{si['exact']}/{si['n']}", "IG_median_mse": round(si["median_mse"], 4),
                          "IG_cat_acc": round(si["cat_acc"], 3), "IG_tool_P": round(si["tool_precision"], 3),
                          "IG_tool_R": round(si["tool_recall"], 3),
                          "CF_exact": f"{sc['exact']}/{sc['n']}", "CF_cat_acc": round(sc["cat_acc"], 3),
                          "CF_exact_nonzero_grad": f"{sc['exact_of_nonzero']}/{sc['n'] - sc['zero_gradient']}",
                          "zero_grad_victims": sc["zero_gradient"], "max_abs_logit": round(sc["max_abs_logit"], 1),
                          "CF_max_abs_error": f"{leak[e]['closed_form_first_layer']['max_abs_error']:.1e}",
                          "label_acc": round(si["label_acc"], 3)})
    leak_df = pd.DataFrame(leak_rows)
    print(leak_df.to_string(index=False))
    for e in order:
        cf = leak[e]["closed_form_first_layer"]["per_victim"]
        bad = [(i, r) for i, r in enumerate(cf) if r["overall_mse"] >= EXACT_MSE]
        if bad:
            print(f"\n  eps={e}: closed-form NOT exact for {len(bad)} victim(s):")
            for i, r in bad:
                print(f"    victim {i:2d}  true_label={int(r['true_label'])}  logit={r['logit']:+8.2f}  "
                      f"zero_gradient={r['zero_gradient']}  (sigmoid(logit) rounds to exactly "
                      f"{'1' if r['logit'] > 0 else '0'} in float32 when |logit| is large)")
    (DP_OUT / f"leakage_vs_epsilon{tag}.json").write_text(json.dumps(leak, indent=2, default=float), encoding="utf-8")
    leak_df.to_csv(EVAL_DIR / f"dp_leakage_vs_epsilon{tag}.csv", index=False)

    # ---------------------------------------------------------------- Part C
    master = []
    for e in order:
        s = summ_u.loc[e]; l = leak[e]
        master.append({"epsilon": e,
                       "accuracy": f"{s[('accuracy','mean')]:.3f}+-{s[('accuracy','std')]:.3f}",
                       "breach_recall": f"{s[('breach_recall','mean')]:.3f}+-{s[('breach_recall','std')]:.3f}",
                       "specificity": f"{s[('specificity','mean')]:.3f}+-{s[('specificity','std')]:.3f}",
                       "roc_auc": f"{s[('roc_auc','mean')]:.3f}+-{s[('roc_auc','std')]:.3f}",
                       "IG_exact": f"{l['inverting_gradients']['summary']['exact']}/12",
                       "IG_cat_acc": round(l["inverting_gradients"]["summary"]["cat_acc"], 3),
                       "CF_exact": f"{l['closed_form_first_layer']['summary']['exact']}/12",
                       "weight_norm_s42": round(l["weight_norm"], 2),
                       "train_sec_mean": round(s[("train_sec", "mean")], 2)})
    master = pd.DataFrame(master)
    master.to_csv(EVAL_DIR / f"dp_master_tradeoff{tag}.csv", index=False)
    print("\n" + "=" * 100 + "\nPart C - master tradeoff (utility = mean+-std over seeds; leakage = seed 42)\n" + "=" * 100)
    print(master.to_string(index=False))

    # Phase 1 side-by-side
    p1_util = P1 / "evaluation" / "phase6" / "dp_test_comparison.csv"
    p1_leak = P1 / "outputs" / "phase7" / "01_leakage_vs_epsilon_results.json"
    if p1_util.exists() and p1_leak.exists():
        u1 = pd.read_csv(p1_util)
        u1["eps"] = u1["epsilon"].apply(lambda v: "inf" if str(v).lower() in ("inf", "infinity") or v == float("inf")
                                        else str(int(float(v))))
        l1 = json.loads(p1_leak.read_text(encoding="utf-8"))
        side = []
        for e in order:
            r1 = u1[u1["eps"] == e].iloc[0]
            ig1 = l1[e]["inverting_gradients"]["per_victim"]; cf1 = l1[e]["closed_form_first_layer"]["per_victim"]
            s2 = summ_u.loc[e]; L2 = leak[e]
            side.append({"epsilon": e,
                         "P1_acc": r1["accuracy"], "P2_acc": round(s2[("accuracy", "mean")], 3),
                         "P1_breach_recall": r1["minority_class_recall (security=True)"],
                         "P2_breach_recall": round(s2[("breach_recall", "mean")], 3),
                         "P1_specificity": r1["majority_class_specificity (security=False)"],
                         "P2_specificity": round(s2[("specificity", "mean")], 3),
                         "P1_IG_exact": f"{sum(r['overall_mse'] < EXACT_MSE for r in ig1)}/{len(ig1)}",
                         "P2_IG_exact": f"{L2['inverting_gradients']['summary']['exact']}/12",
                         "P1_CF_exact": f"{sum(r['overall_mse'] < EXACT_MSE for r in cf1)}/{len(cf1)}",
                         "P2_CF_exact": f"{L2['closed_form_first_layer']['summary']['exact']}/12",
                         "P1_wnorm": l1[e]["weight_norm"], "P2_wnorm": round(L2["weight_norm"], 2)})
        side = pd.DataFrame(side)
        print("\nPhase 1 (AgentDojo, single run) vs Phase 2 (synthetic, mean over seeds):")
        print(side.to_string(index=False))
        side.to_csv(EVAL_DIR / f"dp_phase1_vs_phase2{tag}.csv", index=False)
    else:
        print(f"\n(Phase 1 DP outputs not found under {P1} - skipping side-by-side)")

    # figure
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xl = ["ε=1", "ε=3", "ε=8", "ε=∞"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, lab in [("accuracy", "accuracy"), ("breach_recall", "breach recall"),
                   ("specificity", "specificity"), ("roc_auc", "ROC-AUC")]:
        ax[0].errorbar(xl, summ_u[(k, "mean")], yerr=summ_u[(k, "std")], marker="o", capsize=3, label=lab)
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8)
    ax[0].set_title(f"A. Utility vs privacy budget\n(test, mean ± std over {len(a.seeds)} seeds)")
    ig_ex = [leak[e]["inverting_gradients"]["summary"]["exact"] for e in order]
    cf_ex = [leak[e]["closed_form_first_layer"]["summary"]["exact"] for e in order]
    ax[1].plot(xl, ig_ex, marker="o", color="tab:red", label="Inverting Gradients")
    ax[1].plot(xl, cf_ex, marker="s", color="tab:blue", label="Closed-form")
    ax[1].set_ylim(-0.5, 12.5); ax[1].set_ylabel("victims reconstructed exactly (of 12)")
    ax[1].set_title("B. Exact reconstructions vs privacy budget\n(seed-42 models)"); ax[1].legend(fontsize=8)
    ax[2].plot(xl, [leak[e]["weight_norm"] for e in order], marker="o", color="tab:green")
    ax[2].set_title("C. Model weight norm vs privacy budget\n(seed-42 models)")
    fig.suptitle("Phase 2 (synthetic) - DP-SGD privacy / utility / leakage")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"dp_master_tradeoff{tag}.png", dpi=150); plt.close(fig)

    print(f"\nSaved: {DP_MODEL_DIR}/ (seed-42 DP models)\n       {DP_OUT}/\n       {EVAL_DIR}/dp_*{tag}.csv"
          f"\n       {FIG_DIR / f'dp_master_tradeoff{tag}.png'}")


if __name__ == "__main__":
    main()