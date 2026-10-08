"""
phase2_attacks.py  -  Phase 2 gradient leakage attacks (mirrors Phase 1's src/phase5/ 01-05)
--------------------------------------------------------------------------------------------
Runs, on the Phase 2 FEDERATED model (seed 42, models/phase2_mlp/federated_mlp_state_dict.pt):

  step 1  victims      12 victims = 3 per suite client, drawn from that client's TRAIN rows
                       with np.random.default_rng(42), suites in Phase 1's order.
                       True gradient = ONE example, ONE forward+backward pass, model in .eval()
                       (Phase 1's "FedSGD, batch size 1" threat model - see the module docstring
                       of phase1_agentdojo/src/common/gradient_leakage.py).
                       Null baselines: random-noise guess and training-mean guess.
  step 2  DLG          L2 loss, label OPTIMISED,   LBFGS lr 1.0, 150 iters
  step 3  iDLG         L2 loss, label CLOSED-FORM, LBFGS lr 1.0, 150 iters
  step 4  IG           cosine,  label CLOSED-FORM, Adam  lr 0.05, 500 iters
  step 5  comparison   table + figure + sensitive-tool case study
  extra   CLOSED-FORM  x = dW1[j] / db1[j]  (Phase 1 found this in its Phase 7; it is the
                       exact upper bound on single-gradient leakage, so it belongs in this table)

All attack code is Phase 1's src/common/gradient_leakage.py, imported UNCHANGED:
  compute_gradient, reconstruct, infer_label_from_final_bias_grad,
  closed_form_first_layer_attack.
Only the scoring LAYOUT is Phase-2-specific (Phase 1's load_feature_blocks /
score_reconstruction are hard-wired to its 176 columns incl. is_dos_attack). The
scoring FORMULAS below are copied verbatim from Phase 1; only the DoS flag is absent.

Additions vs Phase 1 (clearly separate, do not change Phase 1's metrics):
  * every metric is also computed for the training-mean guess, so categorical/tool
    numbers have a "blind guess" reference too, not just MSE
  * sensitive-tool leakage is scored over ALL 12 victims (Phase 1 used one case study)
  * if Phase 1's results JSONs are present, a side-by-side Phase 1 vs Phase 2 table

Run from phase2_synthetic/:   python phase2_attacks.py
"""

import json
from dataclasses import dataclass

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from phase2_common import (EVAL_DIR, FIG_DIR, MODEL_DIR, OUT_DIR, P1_SRC, ROOT, SUITES,
                           TABLE_CSV, SecurityMLP, load_features)
from phase2_features import CATEGORICAL_COLS, NUMERIC_COLS
from common.gradient_leakage import (closed_form_first_layer_attack, compute_gradient,  # Phase 1, unchanged
                                     infer_label_from_final_bias_grad, reconstruct)

matplotlib.use("Agg")

# ---- identical to Phase 1's src/phase5/
VICTIMS_PER_SUITE = 3
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.2
RANDOM_STATE = 42
DLG_ITERS, DLG_LR = 150, 1.0
IG_ITERS, IG_LR = 500, 0.05

MODEL_IN = MODEL_DIR / "federated_mlp_state_dict.pt"
ENC_DIR = ROOT / "models" / "phase2_encoders"
ATT_DIR = OUT_DIR / "attacks"
P1_PHASE5 = ROOT / "phase1_agentdojo" / "outputs" / "phase5"

EXACT_MSE = 1e-4   # a victim counts as "exactly reconstructed" if MSE < this

# Tools that MOVE money or data out = the attacker's channels in each suite
SENSITIVE_TOOLS = ["send_money", "send_email", "send_dm", "post_message", "invite_user"]


# ============================================================ scoring (Phase 2 layout)
@dataclass
class FeatureBlocks:
    categorical_fields: dict
    tool_range: tuple
    numeric_range: tuple
    feature_names: list


def load_feature_blocks(feature_names):
    """Same idea as Phase 1's load_feature_blocks: block boundaries derived from the
    fitted encoders, not hard-coded."""
    ohe = joblib.load(ENC_DIR / "onehot.joblib")
    vec = joblib.load(ENC_DIR / "tool_vectorizer.joblib")
    fields, idx = {}, 0
    for field, cats in zip(CATEGORICAL_COLS, ohe.categories_):
        fields[field] = (idx, idx + len(cats)); idx += len(cats)
    n_tool = len(vec.get_feature_names_out())
    blocks = FeatureBlocks(fields, (idx, idx + n_tool), (idx + n_tool, idx + n_tool + len(NUMERIC_COLS)),
                           list(feature_names))
    assert blocks.numeric_range[1] == len(feature_names), "feature layout does not match encoders"
    return blocks


def score_reconstruction(x_true, x_recon, b):
    """Phase 1's score_reconstruction, formulas unchanged; only is_dos_attack removed."""
    r = {"overall_mse": float(np.mean((x_true - x_recon) ** 2))}
    ok = 0
    for field, (s, e) in b.categorical_fields.items():
        c = int(np.argmax(x_true[s:e])) == int(np.argmax(x_recon[s:e]))
        r[f"categorical_{field}_correct"] = bool(c); ok += int(c)
    r["categorical_accuracy"] = ok / len(b.categorical_fields)
    ts, te = b.tool_range
    t, p = (x_true[ts:te] > 0.5).astype(int), (x_recon[ts:te] > 0.5).astype(int)
    tp, fp, fn = int(((t == 1) & (p == 1)).sum()), int(((t == 0) & (p == 1)).sum()), int(((t == 1) & (p == 0)).sum())
    r["tool_precision"] = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if t.sum() == 0 else 0.0)
    r["tool_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    r["tool_true_count"] = int(t.sum())
    ns, ne = b.numeric_range
    r["numeric_mae_std_units"] = float(np.mean(np.abs(x_true[ns:ne] - x_recon[ns:ne])))
    return r


def agg(rows, key):
    v = [r[key] for r in rows]
    return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}   # np.std, as Phase 1


def aggregate(rows, label_key):
    return {**{k: agg(rows, k) for k in ["overall_mse", "categorical_accuracy", "tool_precision",
                                          "tool_recall", "numeric_mae_std_units"]},
            "label_accuracy": round(float(np.mean([r[label_key] for r in rows])), 4),
            "exact_recoveries": int(sum(r["overall_mse"] < EXACT_MSE for r in rows)),
            "median_mse": round(float(np.median([r["overall_mse"] for r in rows])), 4)}


def tools_on(x, b):
    ts, te = b.tool_range
    return [b.feature_names[ts + j].replace("tool_", "", 1) for j in range(te - ts) if x[ts + j] > 0.5]


# ================================================================== main
def main():
    torch.manual_seed(RANDOM_STATE)
    d = load_features()
    names = [str(n) for n in d["feature_names"]]
    b = load_feature_blocks(names)
    input_dim = len(names)
    table = pd.read_csv(TABLE_CSV, keep_default_na=False).set_index("run_id")   # empty tool_sequence stays ""

    lo, hi = float(d["X_train"].min()), float(d["X_train"].max())
    print(f"Feature range in train: [{lo:.2f}, {hi:.2f}]  (Phase 1 clamp (-15, 25) must contain it)")
    assert -15 <= lo and hi <= 25, "features fall outside Phase 1's clamp range - clamp would distort reconstructions"

    model = SecurityMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model.load_state_dict(torch.load(MODEL_IN))
    model.eval()

    # ------------------------------------------------------------ step 1: victims
    rng = np.random.default_rng(RANDOM_STATE)
    V = []
    for s in SUITES:
        m = d["suite_train"] == s
        Xs, ys, ids = d["X_train"][m], d["y_train"][m], d["run_id_train"][m]
        for i in rng.choice(len(Xs), size=VICTIMS_PER_SUITE, replace=False):
            x = torch.tensor(Xs[i]).unsqueeze(0); y = torch.tensor([ys[i]])
            with torch.no_grad():
                logit = float(model(x).item())
            V.append({"suite": s, "row_index": int(i), "run_id": str(ids[i]), "x": Xs[i], "y": float(ys[i]),
                      "g": compute_gradient(model, x, y), "logit": logit,
                      "tool_sequence": str(table.loc[str(ids[i]), "tool_sequence"]),
                      "injection_style": str(table.loc[str(ids[i]), "injection_style"])})

    mean_vec = d["X_train"].mean(axis=0)
    torch.manual_seed(RANDOM_STATE)
    noise = [torch.randn(input_dim).numpy() for _ in V]
    null = {"random_noise_guess_mean": round(float(np.mean([np.mean((v["x"] - n) ** 2) for v, n in zip(V, noise)])), 4),
            "training_mean_vector_guess_mean": round(float(np.mean([np.mean((v["x"] - mean_vec) ** 2) for v in V])), 4)}
    mean_rows = [score_reconstruction(v["x"], mean_vec, b) for v in V]
    for r in mean_rows:
        r["label_ok"] = False
    mean_agg = aggregate(mean_rows, "label_ok")

    print(f"\nStep 1: {len(V)} victims ({VICTIMS_PER_SUITE} per suite) from the federated model's clients")
    for i, v in enumerate(V):
        print(f"  victim {i:2d} {v['suite']:10s} breach={int(v['y'])}  logit={v['logit']:+7.2f}  "
              f"{v['injection_style']:22s} tools: {v['tool_sequence'] or '(none)'}")
    print(f"Null baseline MSE - random noise: {null['random_noise_guess_mean']}  |  training mean: {null['training_mean_vector_guess_mean']}")

    ATT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"victims_X": torch.tensor(np.stack([v["x"] for v in V])),
                "victims_y": torch.tensor([v["y"] for v in V]),
                "victim_grads": [v["g"] for v in V],
                "victim_meta": [{k: v[k] for k in ("suite", "row_index", "run_id", "logit")} for v in V],
                "input_dim": input_dim, "mean_vector": torch.tensor(mean_vec)}, ATT_DIR / "victims.pt")

    # ------------------------------------------------------------ steps 2-4 + closed form
    attacks = {
        "DLG": dict(mode="l2", know_label=False, n_iters=DLG_ITERS, lr=DLG_LR, optimizer_name="lbfgs"),
        "iDLG": dict(mode="l2", know_label=True, n_iters=DLG_ITERS, lr=DLG_LR, optimizer_name="lbfgs"),
        "Inverting Gradients": dict(mode="cosine", know_label=True, n_iters=IG_ITERS, lr=IG_LR, optimizer_name="adam"),
    }
    results, recon = {}, {}
    for name, cfg in attacks.items():
        print(f"\nStep: {name}  ({cfg['mode']} loss, {'closed-form' if cfg['know_label'] else 'optimised'} label, "
              f"{cfg['optimizer_name']} lr={cfg['lr']}, {cfg['n_iters']} iters)")
        rows, X_r = [], []
        for i, v in enumerate(V):
            y_true = torch.tensor([v["y"]])
            if cfg["know_label"]:
                y_inf = infer_label_from_final_bias_grad(v["g"][-1])
                label_ok = int(y_inf.item()) == int(v["y"])
                x_r, _, hist = reconstruct(model, v["g"], input_dim, mode=cfg["mode"], know_label=True, true_y=y_inf,
                                           n_iters=cfg["n_iters"], lr=cfg["lr"], seed=RANDOM_STATE + i,
                                           optimizer_name=cfg["optimizer_name"])
                label_info = {"inferred_label_closed_form": float(y_inf.item())}
            else:
                x_r, y_r, hist = reconstruct(model, v["g"], input_dim, mode=cfg["mode"], know_label=False,
                                             n_iters=cfg["n_iters"], lr=cfg["lr"], seed=RANDOM_STATE + i,
                                             optimizer_name=cfg["optimizer_name"])
                label_ok = round(float(y_r.item())) == int(v["y"])
                label_info = {"recovered_label_soft": float(y_r.item())}
            xr = x_r.squeeze(0).numpy(); X_r.append(xr)
            sc = score_reconstruction(v["x"], xr, b)
            rows.append({"victim_index": i, "suite": v["suite"], "run_id": v["run_id"], "true_label": v["y"],
                         **label_info, "label_ok": bool(label_ok), "best_gradient_matching_loss": min(hist), **sc})
            print(f"  [{name[:4]:4s} victim {i:2d} / {v['suite']:10s}] best_loss={min(hist):.6f}  "
                  f"mse={sc['overall_mse']:.4f}  cat_acc={sc['categorical_accuracy']:.2f}  "
                  f"tool_P={sc['tool_precision']:.2f} tool_R={sc['tool_recall']:.2f}  label_ok={label_ok}")
        results[name] = {"attack": name, "config": cfg, "per_victim": rows,
                         "aggregate": aggregate(rows, "label_ok")}
        recon[name] = np.stack(X_r)

    # closed-form (exact) attack
    rows, X_r = [], []
    for i, v in enumerate(V):
        xr = closed_form_first_layer_attack(v["g"]).squeeze(0).numpy(); X_r.append(xr)
        y_inf = infer_label_from_final_bias_grad(v["g"][-1])
        sc = score_reconstruction(v["x"], xr, b)
        rows.append({"victim_index": i, "suite": v["suite"], "run_id": v["run_id"], "true_label": v["y"],
                     "inferred_label_closed_form": float(y_inf.item()),
                     "label_ok": int(y_inf.item()) == int(v["y"]),
                     "max_abs_error": float(np.max(np.abs(v["x"] - xr))), **sc})
    results["Closed-form"] = {"attack": "Closed-form", "config": {"method": "x = dW1[j]/db1[j], j = argmax|db1|"},
                              "per_victim": rows, "aggregate": aggregate(rows, "label_ok")}
    recon["Closed-form"] = np.stack(X_r)
    print(f"\nStep: Closed-form  max abs error over all victims = {max(r['max_abs_error'] for r in rows):.2e}")

    for name, r in results.items():
        r["null_baseline_mse"] = null
        r["mse_improvement_vs_mean_guess"] = round(1 - r["aggregate"]["overall_mse"]["mean"] / null["training_mean_vector_guess_mean"], 4)
        fname = name.lower().replace(" ", "_").replace("-", "_")
        (ATT_DIR / f"{fname}_results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    np.savez(ATT_DIR / "reconstructions.npz", X_true=np.stack([v["x"] for v in V]),
             **{k.replace(" ", "_").replace("-", "_"): x for k, x in recon.items()})

    # ------------------------------------------------------------ step 5: comparison
    def row(name, a, imp):
        return {"attack": name, "exact_recoveries": f"{a['exact_recoveries']}/{len(V)}",
                "overall_mse": a["overall_mse"]["mean"], "median_mse": a["median_mse"],
                "mse_improvement_vs_mean_guess": imp,
                "categorical_accuracy": a["categorical_accuracy"]["mean"],
                "tool_precision": a["tool_precision"]["mean"], "tool_recall": a["tool_recall"]["mean"],
                "numeric_mae_std_units": a["numeric_mae_std_units"]["mean"],
                "label_accuracy": a["label_accuracy"]}
    comp = pd.DataFrame([row("(training-mean guess)", mean_agg, 0.0)] +
                        [row(n, r["aggregate"], r["mse_improvement_vs_mean_guess"]) for n, r in results.items()])
    comp.loc[0, "label_accuracy"] = np.nan
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(EVAL_DIR / "attack_comparison.csv", index=False)

    print("\n" + "=" * 100 + "\nPhase 2 attack comparison (n=12 victims, mean)\n" + "=" * 100)
    print(comp.round(4).to_string(index=False))

    # per-suite categorical accuracy (indicative: 3 victims each)
    ps = pd.DataFrame([{"suite": r["suite"], "attack": n, "cat_acc": r["categorical_accuracy"], "tool_R": r["tool_recall"]}
                       for n, res in results.items() for r in res["per_victim"]])
    print("\nPer-suite categorical accuracy (3 victims each - indicative only):")
    print(ps.pivot_table(index="suite", columns="attack", values="cat_acc").round(2).to_string())

    # sensitive tools over ALL victims
    print(f"\nSensitive-tool leakage over all 12 victims  (sensitive = {', '.join(SENSITIVE_TOOLS)})")
    sens_rows = []
    for name, X_r in recon.items():
        hit = miss = false_alarm = 0
        for v, xr in zip(V, X_r):
            true_s = set(tools_on(v["x"], b)) & set(SENSITIVE_TOOLS)
            rec_s = set(tools_on(xr, b)) & set(SENSITIVE_TOOLS)
            hit += len(true_s & rec_s); miss += len(true_s - rec_s); false_alarm += len(rec_s - true_s)
        sens_rows.append({"attack": name, "sensitive_recovered": hit, "sensitive_missed": miss,
                          "false_sensitive_alarms": false_alarm,
                          "sensitive_recall": round(hit / (hit + miss), 3) if hit + miss else None})
    sens = pd.DataFrame(sens_rows); print(sens.to_string(index=False))
    sens.to_csv(EVAL_DIR / "attack_sensitive_tools.csv", index=False)

    # case study: first BREACH victim whose true tools include a sensitive one
    cs = next((i for i, v in enumerate(V) if v["y"] == 1 and set(tools_on(v["x"], b)) & set(SENSITIVE_TOOLS)), None)
    if cs is not None:
        v = V[cs]
        print(f"\nCase study - victim {cs} ({v['suite']}, breach, {v['injection_style']}), run_id {v['run_id']}")
        print(f"  true tool sequence : {v['tool_sequence']}")
        for name, X_r in recon.items():
            rt = tools_on(X_r[cs], b)
            print(f"  {name:20s}: {', '.join(rt) if rt else '(none)'}  [{len(rt)} tools flagged]")

    # Phase 1 side-by-side (if its outputs are present locally)
    p1_files = {"DLG": "02_dlg_results.json", "iDLG": "03_idlg_results.json", "Inverting Gradients": "04_ig_results.json"}
    if all((P1_PHASE5 / f).exists() for f in p1_files.values()):
        side = []
        for name, f in p1_files.items():
            p1 = json.loads((P1_PHASE5 / f).read_text(encoding="utf-8"))
            a1, a2 = p1["aggregate"], results[name]["aggregate"]
            n1 = len(p1["per_victim"])
            ex1 = sum(r["overall_mse"] < EXACT_MSE for r in p1["per_victim"])
            side.append({"attack": name,
                         "P1_exact": f"{ex1}/{n1}", "P2_exact": f"{a2['exact_recoveries']}/{len(V)}",
                         "P1_mse_vs_mean_guess": p1.get("mse_improvement_vs_mean_guess"),
                         "P2_mse_vs_mean_guess": results[name]["mse_improvement_vs_mean_guess"],
                         "P1_cat_acc": a1["categorical_accuracy"]["mean"], "P2_cat_acc": a2["categorical_accuracy"]["mean"],
                         "P1_tool_R": a1["tool_recall"]["mean"], "P2_tool_R": a2["tool_recall"]["mean"],
                         "P1_label": a1.get("label_recovery_rate", a1.get("label_inference_accuracy")),
                         "P2_label": a2["label_accuracy"]})
        side = pd.DataFrame(side)
        print("\nPhase 1 (AgentDojo, 176 features) vs Phase 2 (synthetic, 67 features):")
        print(side.to_string(index=False))
        side.to_csv(EVAL_DIR / "attack_phase1_vs_phase2.csv", index=False)
    else:
        print(f"\n(Phase 1 results not found under {P1_PHASE5} - skipping side-by-side table)")

    # figure
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    labels = ["mean\nguess"] + [n.replace("Inverting Gradients", "IG") for n in results]
    ax[0].bar(labels, comp["overall_mse"].clip(lower=1e-12)); ax[0].set_yscale("log")
    ax[0].set_title("Reconstruction MSE (log)\nlower = more leaked")
    for k, (col, t) in enumerate([("categorical_accuracy", "Categorical field accuracy"),
                                  ("tool_recall", "Tool-usage recall"), ("tool_precision", "Tool-usage precision")], 1):
        ax[k].bar(labels, comp[col]); ax[k].set_ylim(0, 1.05); ax[k].set_title(t)
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle("Phase 2 (synthetic) - gradient leakage on the federated model, 12 victims")
    fig.tight_layout(); fig.savefig(FIG_DIR / "attack_comparison.png", dpi=150); plt.close(fig)

    print(f"\nSaved: {ATT_DIR}/ (victims.pt, *_results.json, reconstructions.npz)")
    print(f"       {EVAL_DIR / 'attack_comparison.csv'}, attack_sensitive_tools.csv")
    print(f"       {FIG_DIR / 'attack_comparison.png'}")


if __name__ == "__main__":
    main()