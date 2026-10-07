"""Summarize all saved records for a suite. Usage:  python summary.py banking"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

suite = sys.argv[1] if len(sys.argv) > 1 else "banking"
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / suite
records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(data_dir.glob("*.json"))]
total = len(records)
print(f"Suite: {suite}   Records: {total}\n")

# 1. Outcome counts
print("== Outcomes ==")
for k, n in Counter(r["outcome"] for r in records).most_common():
    print(f"{k:<16}{n:>5}  ({n/total:.0%})")

# 2. Breach label balance (empty responses excluded)
usable = [r for r in records if r["outcome"] not in ("empty_response", "generation_error")]
b = sum(r["breach"] for r in usable)
print(f"\n== Breach label ({len(usable)} usable records) ==")
print(f"breach=1: {b}   breach=0: {len(usable) - b}")

# 3. Breach rate per injection style and per task
def rate_table(title, key):
    groups = defaultdict(list)
    for r in usable:
        if r["injection_present"]:
            groups[r[key]].append(r["breach"])
    print(f"\n== {title} ==")
    for k, v in sorted(groups.items()):
        print(f"{k:<24}{sum(v)}/{len(v)}  ({sum(v)/len(v):.0%})")

rate_table("Breach rate by injection style", "injection_style")
rate_table("Breach rate by task", "task_id")

# 4. Consistency: did all repeats of a scenario give the same outcome?
scen = defaultdict(set)
for r in records:
    scen[(r["task_id"], r["injection_style"])].add(r["outcome"])
consistent = sum(1 for s in scen.values() if len(s) == 1)
print(f"\n== Consistency across repeats ==")
print(f"{consistent}/{len(scen)} scenarios gave the same outcome every repeat")