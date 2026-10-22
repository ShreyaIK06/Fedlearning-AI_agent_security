"""
rebuild_phase1_train.py  -  recreate Phase 1's ORIGINAL 03_train.pkl from Niharika's committed split
------------------------------------------------------------------------------------------------------
Her 03_val.pkl and 03_test.pkl define the split exactly. The training set is every scoped row that is
in neither, so it can be rebuilt without re-running 03_grouped_split.py (whose output changed under
a newer pandas). Checks the result against her recorded numbers before saving.

Needs (in phase1_agentdojo/outputs/phase1/): 02_scoped.pkl (re-run src/phase1/01 and 02 first),
and HER original 03_val.pkl, 03_test.pkl (git restore phase1_agentdojo).

Run from phase2_synthetic/:   python rebuild_phase1_train.py
Then run phase1_agentdojo's src/phase1/04_feature_encoding.py and 05_validate_processed.py.
"""

import json

import pandas as pd

from phase2_common import ROOT

P1 = ROOT / "phase1_agentdojo" / "outputs" / "phase1"
EXPECTED = {"train": 19052, "val": 6601, "test": 7258, "train_security_pct": 15.24}

scoped = pd.read_pickle(P1 / "02_scoped.pkl")
val = pd.read_pickle(P1 / "03_val.pkl")
test = pd.read_pickle(P1 / "03_test.pkl")
print(f"scoped {len(scoped)}  val {len(val)}  test {len(test)}")

assert len(val) == EXPECTED["val"] and len(test) == EXPECTED["test"], \
    "03_val/03_test are not Niharika's originals - run `git restore phase1_agentdojo` first"
for name, part in [("val", val), ("test", test)]:
    assert part.index.isin(scoped.index).all(), f"{name} rows not found in 02_scoped.pkl"
    same = scoped.loc[part.index, "user_prompt"].astype(str).values == part["user_prompt"].astype(str).values
    assert same.all(), f"{name}: row indices point to different records in 02_scoped.pkl"

train = scoped.drop(index=val.index.union(test.index))
pct = round(100 * train["security"].astype(bool).mean(), 2)
print(f"rebuilt train {len(train)} rows, security {pct}%  (expected {EXPECTED['train']}, {EXPECTED['train_security_pct']}%)")
assert len(train) == EXPECTED["train"], "train size differs from Niharika's split"
assert abs(pct - EXPECTED["train_security_pct"]) < 0.01, "train breach rate differs from Niharika's split"
p_tr, p_va, p_te = (set(x["user_prompt"]) for x in (train, val, test))
assert not (p_tr & p_va) and not (p_tr & p_te) and not (p_va & p_te), "user_prompt overlap between splits"
print(f"prompts: train {len(p_tr)}  val {len(p_va)}  test {len(p_te)}  (expected 67 / 25 / 29) - no overlap")

# 04_feature_encoding.py reads rows in the saved order; keep the scoped order, as 03_grouped_split did
train.to_pickle(P1 / "03_train.pkl")
print(f"Saved {P1 / '03_train.pkl'}")
y_old = pd.read_pickle(P1 / "04_y_train.pkl") if (P1 / "04_y_train.pkl").exists() else None
if y_old is not None:
    ok = len(y_old) == len(train) and (y_old.index == train.index).all()
    print(f"matches her committed 04_y_train.pkl row order: {ok}")