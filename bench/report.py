import json
import os
import time
from datetime import datetime

hier_file = "bench/results/hierarchical_run.jsonl"
base_file = "bench/results/baseline_run.jsonl"
start_time_str = "2026-07-24T03:44:43"

def load_results(path, mode):
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                results[data["instance_id"]] = data[f"{mode}_patch"]
    return results

hier_results = load_results(hier_file, "hier")
base_results = load_results(base_file, "base")

hier_count = len(hier_results)
base_count = len(base_results)

total_count = hier_count + base_count
avg_count = total_count / 2
percent = (avg_count / 300) * 100

# Compute Win/Loss/Tie for common instances
common_ids = set(hier_results.keys()).intersection(base_results.keys())
wins = 0
losses = 0
ties = 0

for iid in common_ids:
    h = hier_results[iid]
    b = base_results[iid]
    if h and not b: wins += 1
    elif b and not h: losses += 1
    else: ties += 1

# Calculate ETA dynamically based on current time and start time
start_time = datetime.fromisoformat(start_time_str)
now = datetime.now()
elapsed_mins = (now - start_time).total_seconds() / 60.0

if avg_count > 0 and elapsed_mins > 0:
    rate = avg_count / elapsed_mins # instances per minute
    remaining = 300 - avg_count
    eta_mins = remaining / rate if rate > 0 else 0
else:
    eta_mins = 0

print("### Hourly Benchmark Progress Report")
print(f"**Hierarchical Run:** {hier_count}/300 instances completed")
print(f"**Baseline Run:** {base_count}/300 instances completed")
print(f"**Overall Progress:** {percent:.1f}%")
print(f"**Elapsed Time (since restart):** {int(elapsed_mins)} mins")
print(f"**Estimated ETA:** {eta_mins/60:.1f} hours ({int(eta_mins)} mins remaining)")
print("\n#### Score (on instances finished by both)")
print(f"- Total Overlapping Instances: {len(common_ids)}")
print(f"- Hierarchical Wins (Patch generated, Baseline failed): {wins}")
print(f"- Baseline Wins (Patch generated, Hierarchical failed): {losses}")
print(f"- Ties (Both generated or both failed): {ties}")
