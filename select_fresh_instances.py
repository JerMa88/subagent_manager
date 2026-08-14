import json
from datasets import load_dataset
completed_ids = set()
try:
    with open('bench/results/ablation_qwen35_300.jsonl') as f:
        for line in f:
            completed_ids.add(json.loads(line)['instance_id'])
except Exception as e:
    pass

ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
fresh_ids = []
for instance in ds:
    if instance['instance_id'] not in completed_ids:
        fresh_ids.append(instance['instance_id'])
    if len(fresh_ids) == 10:
        break

print(" ".join(fresh_ids))
