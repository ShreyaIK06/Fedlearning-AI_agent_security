"""Print the final agent message for every record with a given outcome.
Usage:  python show.py refusal
"""
import json, sys
from pathlib import Path

suite = sys.argv[1] if len(sys.argv) > 1 else "banking"
outcome = sys.argv[2] if len(sys.argv) > 2 else "no_action"
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / suite

for f in sorted(data_dir.glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    if r["outcome"] == outcome:
        print(f"--- {r['task_id']} / {r['injection_style']}")
        print("ACTIONS:", r["actions"])
        print("FINAL:", r["messages"][-1]["content"][:300])
        print()