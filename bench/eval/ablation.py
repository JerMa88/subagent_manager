"""
bench/eval/ablation.py — Multi-level hierarchy ablation experiment.

Runs the same SWE-bench instance through 5 conditions:
  Level 0: One-shot  — single LLM call, no tool loop, no agents
  Level 1: Baseline  — single agent with tools (current baseline condition)
  Level 2: Flat      — orchestrator + flat workers, no Domain Managers
  Level 3: Hierarchy — full 3-level architecture (current H condition)
  Level 4: Deep      — Phase Manager → Domain Managers → Sub-Domain → Workers

Usage:
    python bench/eval/ablation.py \\
        --model openai/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \\
        --api-base http://localhost:8000/v1 \\
        --instance-ids astropy__astropy-12907 django__django-11099 \\
        --levels 0 1 2 3 \\
        --output bench/results/ablation_smoke.jsonl

    # Full ablation (all levels, all 300 instances — WARNING: very long runtime):
    python bench/eval/ablation.py --all --levels 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from subagent_manager import SubAgentManager, SubAgentConfig, configure_logging
from subagent_manager.llm_client import LLMClient
from subagent_manager.logging_config import VERBOSE1

from bench.swe_bench_harness import (
    RunResult,
    SWEBenchInstance,
    SWEBenchPrediction,
    clone_and_checkout,
    extract_patch,
    load_swe_bench_instances,
    _inject_file_content_for_patch_writer,
    _try_extract_and_write_code,
)
from bench.swe_bench_agents import build_swe_bench_agents
from bench.swe_bench_tools import ShellExecTool, FileWriterTool, StrReplaceTool, ViewFileTool
from bench.swe_bench_prompts import build_swe_bench_orchestrator_prompt
from bench.swe_bench_hierarchy import (
    HierarchicalSWEBenchManager,
    run_hierarchical_instance,
)
from bench.eval.compare import run_baseline, BASELINE_SYSTEM_PROMPT, build_baseline_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level 0: One-shot (no tools, no agents — pure single LLM call)
# ---------------------------------------------------------------------------

ONE_SHOT_SYSTEM = """You are an expert software engineer.
You will be given a GitHub issue. Output ONLY a valid unified diff (git diff format)
that fixes the issue. No explanations, no markdown, just the diff."""


async def run_oneshot(
    instance: SWEBenchInstance,
    model: str,
    work_dir: str,
    api_base: str | None = None,
    api_key: str | None = None,
    verbosity: int = 0,
) -> RunResult:
    """
    Level 0: Single LLM call, no tool loop, no agents.

    WHY THIS EXISTS: Establishes the true lower bound — what does the model know
    without any iterative exploration? The gap between L0 and L1 measures the
    value of tool-use alone (independent of agent orchestration).

    The diff is written directly to the repo via `git apply` if it parses.
    No verify step; if the diff doesn't apply, patch_generated=False.
    """
    t0 = time.monotonic()
    try:
        repo_dir = clone_and_checkout(instance.repo, instance.base_commit, work_dir)
    except Exception as e:
        return RunResult(
            instance_id=instance.instance_id, success=False, patch_generated=False,
            error=f"Clone failed: {e}", elapsed_seconds=time.monotonic() - t0,
            effective_depth=0,
        )

    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    messages = [
        {"role": "system", "content": ONE_SHOT_SYSTEM},
        {"role": "user", "content": f"## GITHUB ISSUE\n\n{instance.problem_statement}"},
    ]
    try:
        result = await llm.complete(messages=messages, max_tokens=2048)
    except Exception as e:
        return RunResult(
            instance_id=instance.instance_id, success=False, patch_generated=False,
            error=f"LLM call failed: {e}", elapsed_seconds=time.monotonic() - t0,
            effective_depth=0,
        )

    # Try to apply the raw diff output
    diff_text = result.content.strip()
    patch_applied = False
    if diff_text.startswith("diff ") or diff_text.startswith("---"):
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as tmp:
            tmp.write(diff_text)
            tmp_path = tmp.name
        proc = subprocess.run(
            ["git", "apply", "--check", tmp_path],
            cwd=repo_dir, capture_output=True,
        )
        if proc.returncode == 0:
            subprocess.run(["git", "apply", tmp_path], cwd=repo_dir, capture_output=True)
            patch_applied = True
        os.unlink(tmp_path)

    patch = extract_patch(repo_dir)
    elapsed = time.monotonic() - t0
    return RunResult(
        instance_id=instance.instance_id,
        success=True,
        patch_generated=bool(patch),
        prediction=SWEBenchPrediction(
            instance_id=instance.instance_id,
            model_name_or_path=model,
            model_patch=patch or "",
        ) if patch else None,
        elapsed_seconds=elapsed,
        total_tokens=result.usage.get("total_tokens", 0),
        effective_depth=0,
        schema_version=2,
    )


# ---------------------------------------------------------------------------
# Level 2: Flat (orchestrator + flat workers, no Domain Managers)
# ---------------------------------------------------------------------------

async def run_flat(
    instance: SWEBenchInstance,
    model: str,
    work_dir: str,
    api_base: str | None = None,
    api_key: str | None = None,
    verbosity: int = 0,
) -> RunResult:
    """
    Level 2: Orchestrator → flat workers (no Domain Managers, no verification loop).

    WHY THIS EXISTS: Tests whether adding a planning orchestrator above flat workers
    (without the full 3-level domain separation) improves over the baseline.
    The gap between L2 and L3 isolates the value of Domain Manager abstraction.

    Uses the existing SubAgentManager with all SWE-bench workers but without the
    HierarchicalSWEBenchManager's 4-phase verification loop.
    """
    t0 = time.monotonic()
    try:
        repo_dir = clone_and_checkout(instance.repo, instance.base_commit, work_dir)
    except Exception as e:
        return RunResult(
            instance_id=instance.instance_id, success=False, patch_generated=False,
            error=f"Clone failed: {e}", elapsed_seconds=time.monotonic() - t0,
            effective_depth=2,
        )

    sanitized_id = instance.instance_id.replace("/", "_").replace("__", "_")
    short_repo = f"/tmp/repo_flat_{sanitized_id}"
    abs_repo_dir = os.path.abspath(repo_dir)
    try:
        if os.path.islink(short_repo) or os.path.exists(short_repo):
            os.remove(short_repo)
        os.symlink(abs_repo_dir, short_repo)
        prompt_repo_dir = short_repo
    except Exception:
        prompt_repo_dir = abs_repo_dir

    agents = build_swe_bench_agents(abs_repo_dir, prompt_repo_dir=prompt_repo_dir)
    manager = SubAgentManager(
        model=model,
        subagents=agents,
        strategy="adaptive",
        max_subtasks=6,
        api_base=api_base,
        api_key=api_key,
        verbose=verbosity,
    )

    # Inject file content for patch_writer same as hierarchical harness
    original_plan = manager._plan

    async def patched_plan(goal: str, context: str = "") -> tuple:
        plan, plan_data = await original_plan(goal, context=context)
        plan.subtasks = _inject_file_content_for_patch_writer(
            subtasks=plan.subtasks,
            repo_dir=abs_repo_dir,
            prompt_repo_dir=prompt_repo_dir,
            goal=goal,
            logger=logger,
        )
        return plan, plan_data

    manager._plan = patched_plan

    goal = instance.problem_statement
    context = f"Repository: {instance.repo}\nWorking directory: {abs_repo_dir}"
    if instance.hints_text:
        context = f"Hints:\n{instance.hints_text}\n\n{context}"

    try:
        result = await manager.run(goal, context=context)
    except Exception as e:
        return RunResult(
            instance_id=instance.instance_id, success=False, patch_generated=False,
            error=f"Pipeline failed: {e}", elapsed_seconds=time.monotonic() - t0,
            effective_depth=2,
        )

    patch = extract_patch(abs_repo_dir)
    if not patch:
        _try_extract_and_write_code(result, abs_repo_dir)
        patch = extract_patch(abs_repo_dir)

    elapsed = time.monotonic() - t0
    return RunResult(
        instance_id=instance.instance_id,
        success=True,
        patch_generated=bool(patch),
        prediction=SWEBenchPrediction(
            instance_id=instance.instance_id,
            model_name_or_path=model,
            model_patch=patch or "",
        ) if patch else None,
        elapsed_seconds=elapsed,
        total_tokens=result.total_tokens,
        total_tool_calls=result.total_tool_calls,
        subtasks_count=len(result.subtask_results),
        effective_depth=2,
        schema_version=2,
    )


# ---------------------------------------------------------------------------
# Level 4: Deep hierarchy (Phase Manager → Domain Managers → Sub-Domain → Workers)
# ---------------------------------------------------------------------------

async def run_deep(
    instance: SWEBenchInstance,
    model: str,
    work_dir: str,
    api_base: str | None = None,
    api_key: str | None = None,
    verbosity: int = 0,
) -> RunResult:
    """
    Level 4: Phase Manager → Domain Managers → Sub-Domain Managers → Workers.

    WHY THIS EXISTS: Tests whether adding a 4th level (decomposing domain managers
    into sub-domain managers) improves over 3 levels. The gap between L3 and L4
    measures whether deeper hierarchy helps or hurts for this task type.

    Implementation: Wraps HierarchicalSWEBenchManager (L1→L3) in an additional
    PhaseManager (L4→L1) that pre-plans which phases to execute based on a quick
    issue triage. For the current SWE-bench task structure, this extra level is
    unlikely to help because the 4 phases are already well-defined. This run
    is expected to show L4 ≈ L3 or slightly worse (extra overhead with no gain).

    NOTE: Full L4 implementation is a stub until smoke tests confirm L3 > L2 > L1.
    The L4 result is computed as L3 + a lightweight issue-triage pre-call.
    This is the conservative implementation — replace with a true 4-level manager
    once the ablation data confirms whether depth >3 is beneficial.
    """
    # Stub: L4 = L3 + triage pre-call overhead
    # We delegate to run_hierarchical_instance and prepend a triage call.
    t0 = time.monotonic()
    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)

    # L4 triage: classify difficulty before deciding which phases to run
    triage_messages = [
        {"role": "system", "content": (
            "You are a software engineering issue triager. "
            "Read the issue and output JSON: "
            "{\"complexity\": \"trivial|moderate|complex\", \"files_affected\": [\"...\"], \"confidence\": 0.0-1.0}"
        )},
        {"role": "user", "content": f"## ISSUE\n{instance.problem_statement[:2000]}"},
    ]
    triage_tokens = 0
    try:
        triage_result = await llm.complete(messages=triage_messages, max_tokens=256)
        triage_tokens = triage_result.usage.get("total_tokens", 0)
        logger.log(VERBOSE1, f"[L4_TRIAGE] Triage result: {triage_result.content[:200]}")
    except Exception as e:
        logger.warning(f"[L4_TRIAGE] Triage failed (non-fatal): {e}")

    # Delegate to L3 for actual execution
    l3_result = await run_hierarchical_instance(
        instance=instance,
        model=model,
        work_dir=work_dir,
        api_base=api_base,
        api_key=api_key,
        verbosity=verbosity,
    )

    elapsed = time.monotonic() - t0
    # Override metadata to reflect L4 depth and extra triage tokens
    return RunResult(
        instance_id=l3_result.instance_id,
        success=l3_result.success,
        patch_generated=l3_result.patch_generated,
        prediction=l3_result.prediction,
        error=l3_result.error,
        elapsed_seconds=elapsed,
        total_tokens=l3_result.total_tokens + triage_tokens,
        total_tool_calls=l3_result.total_tool_calls,
        subtasks_count=getattr(l3_result, "subtasks_count", 0),
        effective_depth=4,
        retry_count=getattr(l3_result, "retry_count", 0),
        domain_trace=getattr(l3_result, "domain_trace", []),
        schema_version=2,
    )


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

LEVEL_RUNNERS = {
    0: run_oneshot,
    1: run_baseline,
    2: run_flat,
    3: run_hierarchical_instance,
    4: run_deep,
}

LEVEL_NAMES = {
    0: "one-shot",
    1: "baseline (flat-1)",
    2: "flat-2 (orch+workers)",
    3: "hierarchy-3 (current)",
    4: "deep-4 (phase+domain+sub+worker)",
}


async def run_ablation(
    model: str,
    api_base: str | None,
    api_key: str | None,
    instance_ids: list[str],
    levels: list[int],
    work_dir: str,
    output: str,
    verbosity: int,
) -> None:
    configure_logging(verbosity)
    instances = load_swe_bench_instances(instance_ids=instance_ids)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for inst in instances:
        print(f"\n{'═'*60}")
        print(f"  Instance: {inst.instance_id}")
        print(f"{'═'*60}")
        row: dict[str, Any] = {"instance_id": inst.instance_id, "schema_version": 2}

        for level in levels:
            runner = LEVEL_RUNNERS.get(level)
            if runner is None:
                logger.warning(f"Unknown level {level}, skipping")
                continue

            level_name = LEVEL_NAMES[level]
            print(f"  [L{level}: {level_name}]")

            try:
                # Level 1 and 2 use different call signatures
                if level == 1:
                    result = await run_baseline(
                        inst, model=model, work_dir=work_dir, verbosity=verbosity
                    )
                elif level in (0, 2, 4):
                    result = await runner(
                        inst, model=model, work_dir=work_dir,
                        api_base=api_base, api_key=api_key, verbosity=verbosity,
                    )
                else:  # level == 3
                    result = await run_hierarchical_instance(
                        instance=inst, model=model, work_dir=work_dir,
                        api_base=api_base, api_key=api_key, verbosity=verbosity,
                    )
            except Exception as e:
                logger.error(f"  L{level} failed: {e}", exc_info=True)
                result = RunResult(
                    instance_id=inst.instance_id, success=False,
                    patch_generated=False, error=str(e),
                    effective_depth=level,
                )

            status = "✅ PATCH" if result.patch_generated else "❌ NO PATCH"
            print(
                f"    {status}  {result.elapsed_seconds:.0f}s  "
                f"tokens={result.total_tokens}  depth={getattr(result, 'effective_depth', level)}"
            )

            # Record per-level results
            row[f"l{level}_patch"] = result.patch_generated
            row[f"l{level}_time"] = round(result.elapsed_seconds, 2)
            row[f"l{level}_tokens"] = result.total_tokens
            row[f"l{level}_subtasks"] = getattr(result, "subtasks_count", 0)
            row[f"l{level}_retry"] = getattr(result, "retry_count", 0)
            row[f"l{level}_depth"] = getattr(result, "effective_depth", level)
            row[f"l{level}_error"] = result.error or ""
            row[f"l{level}_domain_trace"] = getattr(result, "domain_trace", [])

            # Reset repo between conditions so each starts from the same commit
            import subprocess
            repo_subdir = inst.repo.replace("/", "__")
            for wd in [work_dir, "bench/repos_base", "bench/repos_hier"]:
                repo_path = Path(wd) / repo_subdir
                if repo_path.exists():
                    subprocess.run(
                        ["git", "reset", "--hard", "HEAD"],
                        cwd=str(repo_path), capture_output=True,
                    )

        rows.append(row)

        # Incremental write — safe against crashes mid-run
        with open(output, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  Row saved → {output}")

    # Summary
    print(f"\n{'═'*60}\nAblation Summary ({len(instances)} instances)\n{'═'*60}")
    for level in levels:
        key = f"l{level}_patch"
        count = sum(1 for r in rows if r.get(key, False))
        pct = 100 * count / max(len(rows), 1)
        print(f"  L{level} ({LEVEL_NAMES[level]}): {count}/{len(rows)} patches ({pct:.1f}%)")
    print(f"{'═'*60}\nFull results: {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-level hierarchy ablation for SWE-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (5 instances, levels 0,1,3 — fast):
  python bench/eval/ablation.py --sample 5 --levels 0 1 3

  # Full comparison of all levels (smoke first!):
  python bench/eval/ablation.py --sample 20 --levels 0 1 2 3 4
        """,
    )
    parser.add_argument("--model", default="openai/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--instance-ids", nargs="+", default=None)
    parser.add_argument("--sample", type=int, default=0,
                        help="Random sample N instances instead of --instance-ids")
    parser.add_argument("--all", action="store_true", help="Run all 300 instances")
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 3],
                        help="Levels to run (0=one-shot 1=baseline 2=flat 3=hierarchy 4=deep)")
    parser.add_argument("--work-dir", default="bench/repos")
    parser.add_argument("--output", default="bench/results/ablation.jsonl")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    instance_ids = args.instance_ids or []
    if args.sample:
        import random
        all_insts = load_swe_bench_instances()
        instance_ids = [i.instance_id for i in random.sample(all_insts, min(args.sample, len(all_insts)))]
    elif args.all:
        instance_ids = None  # load_swe_bench_instances() loads all when None

    asyncio.run(run_ablation(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        instance_ids=instance_ids,
        levels=args.levels,
        work_dir=args.work_dir,
        output=args.output,
        verbosity=args.verbose,
    ))


if __name__ == "__main__":
    main()
