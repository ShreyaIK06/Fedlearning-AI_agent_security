"""Validate the whole Phase 2 dataset across all suites. 
Usage:  python validate_all.py"""

import json
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SUITES = ["banking", "slack", "workspace", "travel"]
EXCLUDE = {"empty_response", "generation_error", "not_exposed"}

all_records = []
print("== Per-suite ==")
for s in SUITES:
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in (RAW / s).glob("*.json")]
    for r in recs:
        r["_suite"] = s
    all_records += recs
    usable = [r for r in recs if r["outcome"] not in EXCLUDE]
    b = sum(r["breach"] for r in usable)
    print(f"{s:<11} total={len(recs):>3}  usable={len(usable):>3}  "
          f"breach={b:>3}  clean={len(usable)-b:>3}  excluded={len(recs)-len(usable)}")

total = len(all_records)
usable = [r for r in all_records if r["outcome"] not in EXCLUDE]
print(f"\nTOTAL records : {total}")
print(f"Usable        : {len(usable)}")
print(f"Excluded      : {total - len(usable)}")

# Overall breach balance
b = sum(r["breach"] for r in usable)
print(f"\n== Breach balance (usable) ==")
print(f"breach=1: {b} ({b/len(usable):.0%})   breach=0: {len(usable)-b} ({(len(usable)-b)/len(usable):.0%})")

# Excluded breakdown
print(f"\n== Excluded outcomes ==")
for k, n in Counter(r["outcome"] for r in all_records if r["outcome"] in EXCLUDE).most_common():
    print(f"{k:<16}{n}")

# Integrity checks
print(f"\n== Integrity checks ==")
dupe_ids = [rid for rid, n in Counter(r["run_id"] for r in all_records).items() if n > 1]
print(f"Duplicate run_ids           : {len(dupe_ids)}")

bad_breach = [r for r in usable if r["breach"] not in (0, 1)]
print(f"Usable records with bad label: {len(bad_breach)}")

# Breach recorded but attacker not actually in any action (catch mislabels)
def attacker_in_actions(r):
    tgt = str(r.get("attacker_target") or "").lower()
    return any(tgt and tgt in str(a.get("to", "")).lower() for a in r.get("actions", []))
mism = [r for r in usable if r["breach"] == 1 and not attacker_in_actions(r)]
print(f"breach=1 but attacker absent : {len(mism)}  (should be 0)")

# Exposure: every usable injected record must have seen the injection
unexposed = [r for r in usable if r["injection_present"] and not r.get("injection_exposed")]
print(f"Injected but not exposed     : {len(unexposed)}  (should be 0)")

# Scenario grouping (for train/test split by scenario)
scenarios = defaultdict(list)
for r in all_records:
    scenarios[(r["_suite"], r["task_id"], r["injection_style"])].append(r)
print(f"\n== Scenario grouping ==")
print(f"Distinct scenarios : {len(scenarios)}  (expected 192 = 4 suites x 8 tasks x 6 styles)")
reps = Counter(len(v) for v in scenarios.values())
print(f"Repeats per scenario: {dict(reps)}")