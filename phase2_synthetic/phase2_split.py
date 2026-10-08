""" phase2_split.py  -  Phase 2 (synthetic LangChain logs), Path 1
--------------------------------------------------------------------
Splits the 563 usable records into train / val / test in the SAME WAY as Phase 1's src/phase1/03_grouped_split.py:

  StratifiedGroupKFold(n_splits=5) applied twice
    step 1: hold out 1 of 5 folds as TEST           -> ~20%
    step 2: on what's left, hold out 1 of 5 as VAL  -> ~16%
            the rest is TRAIN                       -> ~64%

  GROUP     = scenario  (suite + task_id + injection_style)
              Phase 1 grouped on user_prompt. Our 3 repeats of a scenario
              are near-identical runs, so they MUST land in the same split;
              otherwise the model sees a run in training and is tested on
              its twin, which inflates the score (leakage).
  STRATIFY  = breach    (Phase 1 stratified on its security label)
              keeps the ~50/50 breach balance similar in every split.

Output: data/processed/phase2_splits.csv  (run_id -> train/val/test)
Next:   python phase2_features.py

Place this file next to runner.py and phase2_features.py.
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from phase2_features import build_table, PROC_DIR, SPLITS_CSV

RANDOM_STATE = 42
N_SPLITS = 5


def one_fold(df, seed):
    """Return (rest_idx, held_out_idx) using the first fold of a StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(df, y=df["breach"], groups=df["scenario_id"]))
    return df.iloc[rest_idx], df.iloc[held_idx]


def main():
    df = build_table()

    # ---- step 1: carve out TEST
    trainval, test = one_fold(df, RANDOM_STATE)
    # ---- step 2: carve VAL out of what's left
    train, val = one_fold(trainval.reset_index(drop=True), RANDOM_STATE)

    # ---- hard leakage asserts (same spirit as Phase 1)
    tr, va, te = (set(d["scenario_id"]) for d in (train, val, test))
    assert not tr & va, "scenario shared between train and val"
    assert not tr & te, "scenario shared between train and test"
    assert not va & te, "scenario shared between val and test"
    ids = [set(d["run_id"]) for d in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "record in two splits"
    assert sum(len(i) for i in ids) == len(df), "some records were lost"
    # every repeat of a scenario stays together
    assert (df.groupby("scenario_id")["run_id"].count()
            == pd.concat([train, val, test]).groupby("scenario_id")["run_id"].count()).all()

    # ---- save
    out = pd.concat([train.assign(split="train"),
                     val.assign(split="val"),
                     test.assign(split="test")], ignore_index=True)
    out = out[["run_id", "scenario_id", "suite", "injection_style", "breach", "split"]]
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out.sort_values(["split", "scenario_id"]).to_csv(SPLITS_CSV, index=False)

    # ---- report
    n = len(df)
    print(f"\nSplit of {n} records / {df['scenario_id'].nunique()} scenarios  (seed={RANDOM_STATE})")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:<5} {len(d):>4} records ({len(d)/n:5.1%})  "
              f"{d['scenario_id'].nunique():>3} scenarios  breach={d['breach'].mean():.2f}")

    print("\nRecords per suite (= FedAvg client) and split")
    print(pd.crosstab(out["suite"], out["split"])[["train", "val", "test"]].to_string())

    print("\nBreach rate per suite and split")
    print(out.pivot_table(index="suite", columns="split", values="breach", aggfunc="mean")
             [["train", "val", "test"]].round(2).to_string())

    missing = [(s, sp) for s in df["suite"].unique() for sp in ("train", "val", "test")
               if not ((out["suite"] == s) & (out["split"] == sp)).any()]
    if missing:
        print(f"\n  WARNING: some suite has no records in a split: {missing}")

    print(f"\nSaved: {SPLITS_CSV}\nNext:  python phase2_features.py")


if __name__ == "__main__":
    main()