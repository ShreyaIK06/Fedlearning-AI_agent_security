"""Print a random sample of records for manual review.
Usage:  python spotcheck.py          (10 records)
        python spotcheck.py 15       (15 records)
"""
import json, sys, random
from pathlib import Path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

files = [f for s in ["banking", "slack", "workspace", "travel"]
         for f in (RAW / s).glob("*.json")]
random.seed()                      # different sample each run
sample = random.sample(files, n)

for i, f in enumerate(sample, 1):
    r = json.loads(f.read_text(encoding="utf-8"))
    print(f"\n{'='*70}")
    print(f"[{i}] {r['suite']} / {r['task_id']} / injection={r['injection_style']}")
    print(f"TASK: {r['user_task']}")
    if r.get("injection_text"):
        print(f"INJECTION: {r['injection_text'][:150]}")
    print(f"ACTIONS: {r['actions']}")
    print(f"LABELS: breach={r['breach']}  task_completed={r['task_completed']}  outcome={r['outcome']}")