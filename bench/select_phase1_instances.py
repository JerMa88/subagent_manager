"""
bench/select_phase1_instances.py — Select stratified 50-instance benchmark subset.

Selects a representative 50-instance subset across 7 major repositories from
SWE-bench Lite to evaluate Phase 1 without consuming unnecessary GPU hours.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.swe_bench_harness import load_swe_bench_instances

REPO_TARGETS = {
    "django/django": 18,
    "sympy/sympy": 10,  # Explicitly included to verify clone & checkout resolution
    "scikit-learn/scikit-learn": 8,
    "sphinx-doc/sphinx": 4,
    "pytest-dev/pytest": 4,
    "matplotlib/matplotlib": 4,
    "astropy/astropy": 2,
}

def main():
    instances = load_swe_bench_instances("princeton-nlp/SWE-bench_Lite")
    by_repo = defaultdict(list)
    for inst in instances:
        by_repo[inst.repo].append(inst)

    selected = []
    for repo, count in REPO_TARGETS.items():
        pool = by_repo.get(repo, [])
        selected.extend(pool[:count])

    out_file = Path(__file__).resolve().parent / "phase1_instances.json"
    data = [
        {
            "instance_id": inst.instance_id,
            "repo": inst.repo,
            "base_commit": inst.base_commit,
            "problem_statement": inst.problem_statement,
            "patch": inst.patch,
            "hints_text": getattr(inst, "hints_text", ""),
        }
        for inst in selected
    ]
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} stratified instances saved to {out_file}")

if __name__ == "__main__":
    main()
