import json
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
rows = []
for s in ["banking", "slack", "workspace", "travel"]:
    for f in (RAW / s).glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append((s, r["breach"], r.get("data_leaked", 0)))

print("== data_leaked by suite ==")
for s in ["banking", "slack", "workspace", "travel"]:
    sr = [r for r in rows if r[0] == s]
    print(f"{s:<11} leaked={sum(r[2] for r in sr)}/{len(sr)}")

print("\n== breach vs data_leaked ==")
print(f"breach=1 & leaked=1 : {sum(1 for r in rows if r[1]==1 and r[2]==1)}")
print(f"breach=0 & leaked=1 : {sum(1 for r in rows if r[1]==0 and r[2]==1)}  <- collateral leaks")
print(f"breach=1 & leaked=0 : {sum(1 for r in rows if r[1]==1 and r[2]==0)}  <- money diversions, no doc")