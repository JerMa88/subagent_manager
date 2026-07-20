"""
bench/eval/compare.py — Hierarchical vs Baseline comparison for SWE-bench.

Runs the same set of instances through two modes and prints a side-by-side table:
  1. Hierarchical (our 6-agent SubAgentManager pipeline)
  2. Baseline     (single-agent: direct LLM call with the issue + file content, no orchestration)

Usage:
    python bench/eval/compare.py \\
        --model ollama/ornith \\
        --instance-ids sympy__sympy-15345 django__django-11099 astropy__astropy-12907 \\
        --work-dir bench/repos \\
        --output bench/results/comparison.json

    # Or use a random sample:
    python bench/eval/compare.py --model ollama/ornith --sample 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from subagent_manager import SubAgentManager, SubAgentConfig, configure_logging
from subagent_manager.logging_config import VERBOSE1

from bench.swe_bench_harness import (
    RunResult,
    SWEBenchInstance,
    clone_and_checkout,
    extract_patch,
    load_swe_bench_instances,
    run_instance,
)
from bench.swe_bench_agents import build_swe_bench_agents
from bench.swe_bench_tools import ShellExecTool, FileWriterTool, StrReplaceTool, ViewFileTool

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline agent: single-agent, no orchestration
# ---------------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """You are an expert software engineer fixing a GitHub bug.

You will be given:
1. A bug description (GitHub issue)
2. The content of the relevant source file(s) with line numbers

Your job: call str_replace to fix the bug. Then call shell_exec to run the reproduce script.

## RULES
- Call str_replace FIRST with old_str = the exact broken lines, new_str = the fix
- Make MINIMAL changes — only the lines necessary to fix the bug
- old_str must match EXACTLY (whitespace, indentation)
- After the fix, run: PYTHONPATH=/tmp/repo python /tmp/reproduce.py
"""

def build_baseline_agent(repo_dir: str, prompt_repo_dir: str) -> SubAgentConfig:
    """Single-agent baseline: patch_writer only, no orchestration."""
    str_replace = StrReplaceTool(working_dir=repo_dir)
    view_file = ViewFileTool(working_dir=repo_dir)
    shell_exec = ShellExecTool(working_dir=repo_dir)

    return SubAgentConfig(
        name="baseline_patcher",
        description="Direct bug fixer — reads file, calls str_replace, verifies.",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        tools=[str_replace, view_file, shell_exec],
        max_tool_iterations=8,
        max_answer_tokens=2048,
        max_history_chars=25_000,
        temperature=0.1,
    )


async def run_baseline(
    instance: SWEBenchInstance,
    model: str,
    work_dir: str,
    verbosity: int = 0,
) -> RunResult:
    """
    Run a single-agent (non-hierarchical) baseline on one instance.

    The baseline gets the issue text + pre-loaded file content in a single
    prompt and must call str_replace directly — no planning, no subagents.
    """
    t0 = time.monotonic()

    try:
        repo_dir = clone_and_checkout(instance.repo, instance.base_commit, work_dir)
    except Exception as e:
        return RunResult(
            instance_id=instance.instance_id,
            success=False,
            patch_generated=False,
            error=f"Clone failed: {e}",
            elapsed_seconds=time.monotonic() - t0,
        )

    # Short symlink
    short_repo = "/tmp/repo_baseline"
    try:
        if os.path.islink(short_repo) or os.path.exists(short_repo):
            os.remove(short_repo)
        os.symlink(repo_dir, short_repo)
        prompt_repo_dir = short_repo
    except Exception:
        prompt_repo_dir = repo_dir

    # Pre-load relevant files (same logic as harness)
    import re
    py_pat = re.compile(r'[\w/.-]+\.py')
    file_content_block = ""
    for m in py_pat.finditer(instance.problem_statement):
        rel = m.group(0)
        abs_path = Path(repo_dir) / rel
        if not abs_path.is_file():
            # Try stripping prefixes
            parts = Path(rel).parts
            for start in range(1, len(parts)):
                cand = Path(repo_dir) / Path(*parts[start:])
                if cand.is_file():
                    abs_path = cand
                    rel = str(Path(*parts[start:]))
                    break
            else:
                continue
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            numbered = "\n".join(f"{i+1:4d}: {ln}" for i, ln in enumerate(lines))
            if len(numbered) > 8000:
                numbered = numbered[:8000] + "\n... [truncated]"
            file_content_block += f"\n\n=== FILE: {rel} ===\n{numbered}\n"
            if len(file_content_block) > 16000:
                break
        except Exception:
            pass

    agent = build_baseline_agent(repo_dir, prompt_repo_dir)
    manager = SubAgentManager(
        model=model,
        subagents=[agent],
        strategy="sequential",
        max_subtasks=1,
        verbose=verbosity,
    )

    issue_text = instance.problem_statement
    full_prompt = (
        f"## GITHUB ISSUE\n\n{issue_text}\n\n"
        f"## REPOSITORY\n{instance.repo}\n"
        f"## FILE CONTENT\n{file_content_block}"
    )

    try:
        await manager.run(full_prompt)
    except Exception as e:
        return RunResult(
            instance_id=instance.instance_id,
            success=False,
            patch_generated=False,
            error=f"Baseline failed: {e}",
            elapsed_seconds=time.monotonic() - t0,
        )

    patch = extract_patch(repo_dir)
    return RunResult(
        instance_id=instance.instance_id,
        success=True,
        patch_generated=bool(patch),
        patch=patch,
        elapsed_seconds=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    instance_id: str
    hier_patch: bool = False
    hier_time: float = 0.0
    hier_tokens: int = 0
    hier_error: str = ""
    base_patch: bool = False
    base_time: float = 0.0
    base_tokens: int = 0
    base_error: str = ""


def print_table(rows: list[ComparisonRow]) -> None:
    header = (
        f"{'Instance':<35} {'Hier✓':>6} {'BTime':>7} {'Base✓':>6} {'BTime':>7}  Delta"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        h = "✅" if r.hier_patch else "❌"
        b = "✅" if r.base_patch else "❌"
        delta = ""
        if r.hier_patch and not r.base_patch:
            delta = "← HIER WINS"
        elif r.base_patch and not r.hier_patch:
            delta = "← BASE WINS"
        elif r.hier_patch and r.base_patch:
            delta = "BOTH"
        print(
            f"{r.instance_id:<35} {h:>6} {r.hier_time:>6.0f}s "
            f"{b:>6} {r.base_time:>6.0f}s  {delta}"
        )
    print("=" * len(header))

    hier_total = sum(1 for r in rows if r.hier_patch)
    base_total = sum(1 for r in rows if r.base_patch)
    n = len(rows)
    print(f"\nHierarchical: {hier_total}/{n} patches ({100*hier_total//max(n,1)}%)")
    print(f"Baseline:     {base_total}/{n} patches ({100*base_total//max(n,1)}%)")
    if hier_total > base_total:
        print(f"\n✅ Hierarchical beats baseline by {hier_total - base_total} instance(s)")
    elif base_total > hier_total:
        print(f"\n⚠️  Baseline beats hierarchical by {base_total - hier_total} instance(s)")
    else:
        print("\n— Tie")


async def compare(
    model: str,
    instance_ids: list[str],
    work_dir: str,
    output: str,
    verbosity: int,
) -> None:
    configure_logging(verbosity)
    instances = load_swe_bench_instances(instance_ids=instance_ids)
    rows: list[ComparisonRow] = []

    for inst in instances:
        row = ComparisonRow(instance_id=inst.instance_id)
        print(f"\n{'─'*60}")
        print(f"  Instance: {inst.instance_id}")
        print(f"{'─'*60}")

        # --- Hierarchical ---
        print("  [1/2] Running HIERARCHICAL pipeline…")
        h_result = await run_instance(inst, model=model, work_dir=work_dir, verbosity=verbosity)
        row.hier_patch = h_result.patch_generated
        row.hier_time = h_result.elapsed_seconds
        row.hier_error = h_result.error or ""
        status = "✅ PATCH" if h_result.patch_generated else "❌ NO PATCH"
        print(f"         {status}  ({h_result.elapsed_seconds:.0f}s)")

        # Reset repo for baseline
        import subprocess
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(Path(work_dir) / inst.repo.replace("/", "__")), capture_output=True)

        # --- Baseline ---
        print("  [2/2] Running BASELINE (single-agent)…")
        b_result = await run_baseline(inst, model=model, work_dir=work_dir, verbosity=0)
        row.base_patch = b_result.patch_generated
        row.base_time = b_result.elapsed_seconds
        row.base_error = b_result.error or ""
        status = "✅ PATCH" if b_result.patch_generated else "❌ NO PATCH"
        print(f"         {status}  ({b_result.elapsed_seconds:.0f}s)")

        rows.append(row)

    print_table(rows)

    # Write JSON output
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"\nResults saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hierarchical vs baseline SWE-bench performance"
    )
    parser.add_argument("--model", default="ollama/ornith")
    parser.add_argument(
        "--instance-ids", nargs="+",
        help="Specific instance IDs to evaluate"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Random sample N instances from SWE-bench Lite"
    )
    parser.add_argument("--work-dir", default="bench/repos")
    parser.add_argument("--output", default="bench/results/comparison.json")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    if not args.instance_ids and not args.sample:
        parser.error("Provide --instance-ids or --sample N")

    instance_ids = args.instance_ids or []
    if args.sample:
        all_instances = load_swe_bench_instances()
        import random
        sample = random.sample(all_instances, min(args.sample, len(all_instances)))
        instance_ids = [i.instance_id for i in sample]

    asyncio.run(compare(
        model=args.model,
        instance_ids=instance_ids,
        work_dir=args.work_dir,
        output=args.output,
        verbosity=args.verbose,
    ))


if __name__ == "__main__":
    main()
