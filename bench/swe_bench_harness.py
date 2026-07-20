"""
SWE-bench evaluation harness for subagent_manager.

Loads SWE-bench Lite instances from HuggingFace, runs each through the
multi-agent orchestration pipeline, and produces prediction JSONL files
compatible with the official SWE-bench evaluation harness.

Usage:
    python bench/swe_bench_harness.py \
        --model ollama/ornith \
        --max-instances 5 \
        --output bench/results/predictions.jsonl \
        --verbose 1

    # With specific instance IDs:
    python bench/swe_bench_harness.py \
        --model ollama/ornith \
        --instance-ids astropy__astropy-12907 django__django-11099 \
        --verbose 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path for bench imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subagent_manager import SubAgentManager, configure_logging
from subagent_manager.logging_config import VERBOSE1, VERBOSE2

from bench.swe_bench_agents import build_swe_bench_agents
from bench.swe_bench_prompts import (
    build_swe_bench_orchestrator_prompt,
    build_swe_bench_synthesis_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchInstance:
    """A single SWE-bench task instance."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str  # gold patch (for reference only)
    hints_text: str = ""
    version: str = ""


@dataclass
class SWEBenchPrediction:
    """A prediction in the SWE-bench submission format."""
    instance_id: str
    model_name_or_path: str
    model_patch: str


@dataclass
class RunResult:
    """Result of running the pipeline on a single instance."""
    instance_id: str
    success: bool
    patch_generated: bool
    prediction: SWEBenchPrediction | None = None
    error: str = ""
    elapsed_seconds: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    diagnosis: str = ""


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------

def clone_and_checkout(
    repo: str, base_commit: str, work_dir: str
) -> str:
    """
    Clone a repo and checkout the base commit.

    Args:
        repo: GitHub repo in 'owner/name' format.
        base_commit: Commit hash to checkout.
        work_dir: Parent directory for the clone.

    Returns:
        Absolute path to the cloned repo directory.
    """
    repo_url = f"https://github.com/{repo}.git"
    repo_name = repo.replace("/", "__")
    repo_dir = os.path.join(work_dir, repo_name)

    if os.path.exists(repo_dir):
        logger.log(
            VERBOSE1,
            f"[HARNESS] Repo already cloned at {repo_dir}, resetting to {base_commit[:8]}",
        )
        # Reset existing clone
        subprocess.run(
            ["git", "checkout", "-f", base_commit],
            cwd=repo_dir, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=repo_dir, capture_output=True, check=True,
        )
    else:
        logger.log(
            VERBOSE1,
            f"[HARNESS] Cloning {repo_url} → {repo_dir}",
        )
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, repo_dir],
            capture_output=True, check=True, timeout=300,
        )
        logger.log(VERBOSE1, f"[HARNESS] Checking out {base_commit[:8]}")
        subprocess.run(
            ["git", "checkout", "-f", base_commit],
            cwd=repo_dir, capture_output=True, check=True,
        )

    return repo_dir


def extract_patch(repo_dir: str) -> str:
    """
    Extract the git diff (uncommitted changes) as a patch string.

    Args:
        repo_dir: Path to the repository.

    Returns:
        The diff string, or empty string if no changes.
    """
    result = subprocess.run(
        ["git", "diff"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    patch = result.stdout.strip()

    if not patch:
        # Also check for new untracked files
        result_untracked = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if result_untracked.stdout.strip():
            # Stage everything and get diff
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repo_dir, capture_output=True,
            )
            result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=repo_dir, capture_output=True, text=True,
            )
            patch = result.stdout.strip()
            # Unstage
            subprocess.run(
                ["git", "reset", "HEAD"],
                cwd=repo_dir, capture_output=True,
            )

    logger.log(VERBOSE1, f"[HARNESS] Extracted patch: {len(patch)} chars")
    return patch


# ---------------------------------------------------------------------------
# Fallback code extraction
# ---------------------------------------------------------------------------

import re


def _try_extract_and_write_code(
    result: Any,
    repo_dir: str,
) -> bool:
    """
    Fallback: extract and apply a surgical fix from the patch_writer's text answer.

    Strategy (in priority order):
    1. Look for explicit old_str / new_str markers in the prose (most precise).
    2. Look for a before/after code block pair (--- / +++ style or explicit labels).
    3. Last resort: write the largest code block as the entire file.

    Returns True if any change was successfully applied.
    """
    # Find the patch_writer's result
    patch_writer_answer = ""
    for sr in result.subtask_results:
        if sr.agent_name == "patch_writer" and sr.success:
            patch_writer_answer = sr.answer or ""
            break

    if not patch_writer_answer:
        logger.log(VERBOSE1, "[HARNESS] No patch_writer answer to extract from")
        return False

    logger.log(
        VERBOSE1,
        f"[HARNESS] Attempting code extraction from {len(patch_writer_answer)}-char answer",
    )

    # --- Strategy 1: find old_str / new_str blocks -----------------------
    # Handles patterns like:
    #   old_str: ```\n...\n```   new_str: ```\n...\n```
    #   "Replace:\n...\nWith:\n..."
    old_new_patterns = [
        # old_str / new_str with code fences
        (
            r'old_str[:\s]*```(?:\w+)?\n(.*?)```.*?new_str[:\s]*```(?:\w+)?\n(.*?)```',
            re.DOTALL,
        ),
        # Replace X with Y (code fences)
        (
            r'[Rr]eplace[:\s]*```(?:\w+)?\n(.*?)```.*?[Ww]ith[:\s]*```(?:\w+)?\n(.*?)```',
            re.DOTALL,
        ),
        # Before / After labels
        (
            r'[Bb]efore[:\s]*```(?:\w+)?\n(.*?)```.*?[Aa]fter[:\s]*```(?:\w+)?\n(.*?)```',
            re.DOTALL,
        ),
    ]

    old_str: str | None = None
    new_str: str | None = None
    for pattern, flags in old_new_patterns:
        m = re.search(pattern, patch_writer_answer, flags)
        if m:
            old_str = m.group(1)
            new_str = m.group(2)
            logger.log(
                VERBOSE1,
                f"[HARNESS] Found old/new pair via pattern (old={len(old_str)} chars, "
                f"new={len(new_str)} chars)",
            )
            break

    # --- Identify the target file in all cases ---------------------------
    file_patterns = re.findall(
        r'(?:(?:in|file|modify|fix|update|change)\s*[:=]?\s*)?'
        r'[`\'"]*([a-zA-Z_][\w/.-]*\.py)[`\'"]*',
        patch_writer_answer,
    )
    if not file_patterns:
        full_text = result.answer or ""
        for sr in result.subtask_results:
            full_text += " " + (sr.answer or "")
        file_patterns = re.findall(r'([a-zA-Z_][\w/.-]*\.py)', full_text)

    target_path: str | None = None
    for fp in file_patterns:
        # Strip leading path prefixes the model may have hallucinated
        for prefix in [f"{repo_dir}/", "/tmp/repo/", "/testbed/"]:
            if fp.startswith(prefix):
                fp = fp[len(prefix):]
        full_path = os.path.join(repo_dir, fp)
        if os.path.exists(full_path):
            target_path = full_path
            break

    if not target_path:
        logger.warning("[HARNESS] Fallback: no target file path found")
        return False

    # --- Apply Strategy 1 if we have old/new pair -------------------------
    if old_str is not None and new_str is not None:
        try:
            content = Path(target_path).read_text(encoding="utf-8", errors="replace")
            if old_str in content:
                new_content = content.replace(old_str, new_str, 1)
                Path(target_path).write_text(new_content, encoding="utf-8")
                logger.log(
                    VERBOSE1,
                    f"[HARNESS] FALLBACK str_replace applied to {target_path} "
                    f"({len(old_str)} → {len(new_str)} chars)",
                )
                return True
            else:
                logger.warning(
                    f"[HARNESS] old_str not found in {target_path} — "
                    "falling through to code-block write"
                )
        except Exception as e:
            logger.error(f"[HARNESS] Fallback str_replace failed: {e}")

    # --- Strategy 2 / 3: extract code blocks and write largest -----------
    code_blocks = re.findall(
        r'```(?:python)?\s*\n(.*?)```',
        patch_writer_answer,
        re.DOTALL,
    )
    if not code_blocks:
        logger.log(VERBOSE1, "[HARNESS] No code blocks found in patch_writer answer")
        return False

    largest_block = max(code_blocks, key=len)
    logger.log(
        VERBOSE1,
        f"[HARNESS] FALLBACK: Writing {len(largest_block)} chars to {target_path}",
    )
    try:
        Path(target_path).write_text(largest_block, encoding="utf-8")
        logger.log(VERBOSE1, f"[HARNESS] Fallback write successful: {target_path}")
        return True
    except Exception as e:
        logger.error(f"[HARNESS] Fallback write failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Harness-level file pre-loading helper (P0 fix)
# ---------------------------------------------------------------------------

def _inject_file_content_for_patch_writer(
    subtasks: list,
    repo_dir: str,
    prompt_repo_dir: str,
    goal: str,
    logger,
) -> list:
    """
    Deterministically pre-load relevant source files into the patch_writer subtask.

    The patch_writer is supposed to call str_replace immediately, but can only do
    so if it already has the file content. We read the files from disk here (in the
    harness process) and inject them into the subtask context field — no agent tool
    calls required.

    Strategy:
    1. Parse the patch_writer task description for .py file paths (relative or absolute).
    2. Try those paths against repo_dir.
    3. Fallback: grep repo for Python files whose names appear in the issue text.
    4. Read up to 3 files (max 8000 chars each), prepend line numbers, and append
       to the patch_writer subtask's context.
    """
    import re

    from subagent_manager.strategies.base import SubtaskDef  # local import avoids circular

    result_subtasks = []
    for subtask in subtasks:
        if subtask.agent_name != "patch_writer":
            result_subtasks.append(subtask)
            continue

        # --- Collect candidate file paths ---
        candidates: list[str] = []

        # 1. Parse .py paths from the task description
        py_pat = re.compile(r'[\w/.-]+\.py')
        for m in py_pat.finditer(subtask.task):
            raw = m.group(0)
            # Strip common wrong prefixes that the orchestrator may write
            for prefix in [f"{prompt_repo_dir}/", f"{repo_dir}/", "bench/repos/", "/testbed/"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
            candidates.append(raw)

        # 2. Fallback: extract .py filenames from the issue text
        if not candidates:
            for m in py_pat.finditer(goal):
                candidates.append(m.group(0))

        # 3. Deduplicate preserving order
        seen: set[str] = set()
        unique_candidates: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        # --- Read files from disk ---
        file_blocks: list[str] = []
        for rel_path in unique_candidates[:5]:  # try up to 5 candidates
            if len(file_blocks) >= 3:
                break
            abs_path = Path(repo_dir) / rel_path
            if not abs_path.is_file():
                # try stripping leading path components
                parts = Path(rel_path).parts
                for start in range(1, len(parts)):
                    candidate_abs = Path(repo_dir) / Path(*parts[start:])
                    if candidate_abs.is_file():
                        abs_path = candidate_abs
                        rel_path = str(Path(*parts[start:]))
                        break
                else:
                    continue

            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                # Add line numbers so old_str can be copied exactly
                lines = content.splitlines()
                numbered = "\n".join(f"{i+1:4d}: {ln}" for i, ln in enumerate(lines))
                # Truncate if enormous
                MAX_FILE_CHARS = 8000
                if len(numbered) > MAX_FILE_CHARS:
                    numbered = numbered[:MAX_FILE_CHARS] + "\n... [truncated]"
                file_blocks.append(
                    f"=== FILE: {rel_path} ({len(lines)} lines) ===\n{numbered}\n"
                )
                logger.log(
                    VERBOSE1,
                    f"[HARNESS] Pre-loaded {rel_path} ({len(lines)} lines) "
                    f"into patch_writer context",
                )
            except Exception as exc:
                logger.warning(f"[HARNESS] Could not pre-load {abs_path}: {exc}")

        if file_blocks:
            injection = (
                "\n\n## PRE-LOADED FILE CONTENT\n\n"
                "The following files are already loaded for you. "
                "Use the line-numbered content to write old_str for str_replace.\n\n"
                + "\n".join(file_blocks)
            )
            new_context = (subtask.context or "") + injection
            subtask = SubtaskDef(
                id=subtask.id,
                task=subtask.task,
                agent_name=subtask.agent_name,
                depends_on=subtask.depends_on,
                context=new_context,
            )
            logger.log(
                VERBOSE1,
                f"[HARNESS] patch_writer context enriched: "
                f"{len(file_blocks)} file(s) pre-loaded "
                f"({len(injection):,} chars injected)",
            )
        else:
            logger.warning("[HARNESS] patch_writer: no source files found to pre-load")

        result_subtasks.append(subtask)

    return result_subtasks


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

async def run_instance(
    instance: SWEBenchInstance,
    model: str,
    work_dir: str,
    verbosity: int = 1,
    max_subtasks: int = 6,
) -> RunResult:
    """
    Run the subagent_manager pipeline on a single SWE-bench instance.

    Args:
        instance: The SWE-bench instance to solve.
        model: LLM model identifier (e.g., 'ollama/ornith').
        work_dir: Directory for cloned repos.
        verbosity: Logging verbosity (0, 1, or 2).
        max_subtasks: Maximum subtasks for the orchestrator.

    Returns:
        RunResult with the outcome and generated patch.
    """
    logger.log(
        VERBOSE1,
        f"\n[HARNESS] ════════════════════════════════════════\n"
        f"[HARNESS] Instance: {instance.instance_id}\n"
        f"[HARNESS] Repo: {instance.repo}\n"
        f"[HARNESS] Commit: {instance.base_commit[:12]}\n"
        f"[HARNESS] ════════════════════════════════════════",
    )

    t0 = time.monotonic()

    # Step 1: Clone and checkout
    try:
        repo_dir = clone_and_checkout(
            instance.repo, instance.base_commit, work_dir
        )
    except Exception as e:
        logger.error(f"[HARNESS] Failed to clone/checkout: {e}")
        return RunResult(
            instance_id=instance.instance_id,
            success=False,
            patch_generated=False,
            error=f"Clone failed: {e}",
            elapsed_seconds=time.monotonic() - t0,
        )

    # Step 2: Build agents scoped to the repo.
    # Use a short symlink /tmp/repo -> repo_dir so agent prompts never hit token-
    # truncation on deeply-nested absolute paths like
    # /Users/zma/Documents/programs/subagent_manager/bench/repos/sympy__sympy/...
    # Models see only "/tmp/repo/sympy/printing/mathematica.py" which is 40 chars
    # instead of 100+, well within any token budget.
    short_repo = "/tmp/repo"
    abs_repo_dir = os.path.abspath(repo_dir)  # must be absolute for symlink to resolve from /tmp
    try:
        if os.path.islink(short_repo) or os.path.exists(short_repo):
            os.remove(short_repo)
        os.symlink(abs_repo_dir, short_repo)
        prompt_repo_dir = short_repo
        logger.log(VERBOSE1, f"[HARNESS] Symlink: {short_repo} → {abs_repo_dir}")
    except Exception as e:
        # Symlink creation failed (permissions, etc.) — fall back to real path
        logger.warning(f"[HARNESS] Could not create /tmp/repo symlink: {e}. Using full path.")
        abs_repo_dir = repo_dir
        prompt_repo_dir = abs_repo_dir

    agents = build_swe_bench_agents(repo_dir, prompt_repo_dir=prompt_repo_dir)

    # Step 3: Create the manager with SWE-bench prompts
    manager = SubAgentManager(
        model=model,
        subagents=agents,
        strategy="adaptive",
        max_subtasks=max_subtasks,
        verbose=verbosity,
    )

    # Override the orchestrator prompt builder
    # We monkey-patch _plan to use our custom prompt
    original_plan = manager._plan

    async def custom_plan(goal: str, context: str = "") -> tuple:
        """Use SWE-bench-specific prompts for planning."""
        agent_descriptions = [
            {"name": c.name, "description": c.description}
            for c in manager.agent_configs
        ]
        system_prompt = build_swe_bench_orchestrator_prompt(
            available_agents=agent_descriptions,
            max_subtasks=manager.max_subtasks,
            repo_dir=prompt_repo_dir,
        )

        user_content = f"## GITHUB ISSUE\n\n{goal}"
        if context:
            user_content = f"## ADDITIONAL CONTEXT\n\n{context}\n\n{user_content}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.log(VERBOSE2, f"[HARNESS] SWE-bench orchestrator prompt:\n{system_prompt[:500]}...")

        from subagent_manager.logging_config import format_tokens
        from subagent_manager.strategies.base import ExecutionPlan, SubtaskDef

        result = await manager.orchestrator_client.complete(
            messages=messages,
            max_tokens=2048,
        )

        logger.log(
            VERBOSE1,
            f"[PLAN] SWE-bench orchestrator response: {len(result.content)} chars, "
            f"{format_tokens(result.usage)}",
        )
        logger.log(VERBOSE2, f"[PLAN] Raw response:\n{result.content}")

        plan_data = manager._parse_plan_json(result.content)

        subtasks = []
        for item in plan_data:
            subtasks.append(
                SubtaskDef(
                    id=item.get("id", len(subtasks) + 1),
                    task=item.get("task", ""),
                    agent_name=item.get("agent", "issue_analyzer"),
                    depends_on=item.get("depends_on", []),
                    context=item.get("context", ""),
                )
            )

        # ── Harness-level file pre-loading (deterministic, P0 fix) ─────────────
        # The patch_writer agent must call str_replace immediately — it cannot
        # waste iterations reading the file. We pre-load relevant source files
        # from disk here and inject them into the patch_writer subtask context.
        # This is done by the harness (not the agent) for determinism.
        subtasks = _inject_file_content_for_patch_writer(
            subtasks=subtasks,
            repo_dir=repo_dir,
            prompt_repo_dir=prompt_repo_dir,
            goal=goal,
            logger=logger,
        )

        return ExecutionPlan(subtasks=subtasks), plan_data

    manager._plan = custom_plan

    # Step 4: Run the pipeline
    goal = instance.problem_statement
    context = ""
    if instance.hints_text:
        context = f"Hints from the issue:\n{instance.hints_text}"

    # Add repo structure context
    context += f"\n\nRepository: {instance.repo}\nWorking directory: {repo_dir}"

    try:
        result = await manager.run(goal, context=context)
    except Exception as e:
        logger.error(f"[HARNESS] Pipeline failed: {e}", exc_info=True)
        return RunResult(
            instance_id=instance.instance_id,
            success=False,
            patch_generated=False,
            error=f"Pipeline failed: {e}",
            elapsed_seconds=time.monotonic() - t0,
        )

    # Step 5: Fallback code extraction
    # If the patch_writer described the fix in text (with code blocks)
    # instead of calling write_file, extract and apply the code.
    patch = extract_patch(repo_dir)
    if not patch:
        logger.log(
            VERBOSE1,
            "[HARNESS] No git diff detected — attempting fallback code extraction "
            "from patch_writer text answer",
        )
        _try_extract_and_write_code(result, repo_dir)
        patch = extract_patch(repo_dir)

    # Step 6: Independent reproduce.py validation (skeptical senior programmer check).
    # The harness itself — NOT the test_runner subagent — re-runs the reproduction
    # script and records the objective signal. This cannot be faked by a subagent
    # self-report.
    diagnosis = result.answer[:500] if result.answer else ""
    reproduce_script = "/tmp/reproduce.py"
    if os.path.exists(reproduce_script):
        logger.log(
            VERBOSE1,
            f"[HARNESS] Running independent reproduce validation: {reproduce_script}",
        )
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_dir + (
                os.pathsep + env.get("PYTHONPATH", "")
            )
            rep_proc = subprocess.run(
                ["python", reproduce_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=repo_dir,
                env=env,
            )
            stdout = rep_proc.stdout.strip()
            stderr = rep_proc.stderr.strip()
            combined = stdout + ("\n" + stderr if stderr else "")

            if "BUG FIXED" in stdout:
                diagnosis = f"REPRODUCE_PASS: {combined[:300]}"
                logger.log(
                    VERBOSE1,
                    f"[HARNESS] ✅ REPRODUCE_PASS — /tmp/reproduce.py output: {stdout[:200]}",
                )
            elif "BUG REPRODUCED" in stdout:
                # Script ran fine but fix wasn't applied
                diagnosis = f"REPRODUCE_FAIL (patch did not fix bug): {combined[:300]}"
                logger.warning(
                    f"[HARNESS] ❌ REPRODUCE_FAIL — patch did not fix bug. "
                    f"reproduce.py output: {stdout[:200]}"
                )
            else:
                # Script ran but printed neither marker — treat as inconclusive
                diagnosis = f"REPRODUCE_INCONCLUSIVE: {combined[:300]}"
                logger.warning(
                    f"[HARNESS] ⚠ REPRODUCE_INCONCLUSIVE — reproduce.py output: {combined[:200]}"
                )
        except subprocess.TimeoutExpired:
            diagnosis = "REPRODUCE_TIMEOUT: /tmp/reproduce.py timed out after 30s"
            logger.warning("[HARNESS] reproduce.py timed out after 30s")
        except Exception as e:
            diagnosis = f"REPRODUCE_ERROR: {e}"
            logger.warning(f"[HARNESS] reproduce.py execution failed: {e}")
    else:
        logger.log(
            VERBOSE1,
            "[HARNESS] /tmp/reproduce.py not found — reproducer agent may not have run",
        )

    elapsed = time.monotonic() - t0

    prediction = SWEBenchPrediction(
        instance_id=instance.instance_id,
        model_name_or_path=model,
        model_patch=patch if patch else "",
    )

    run_result = RunResult(
        instance_id=instance.instance_id,
        success=True,
        patch_generated=bool(patch),
        prediction=prediction,
        elapsed_seconds=elapsed,
        total_tokens=result.total_tokens,
        total_tool_calls=result.total_tool_calls,
        diagnosis=diagnosis,
    )

    logger.log(
        VERBOSE1,
        f"[HARNESS] Instance {instance.instance_id} completed in {elapsed:.1f}s: "
        f"patch={'YES' if patch else 'NO'} ({len(patch)} chars), "
        f"tokens={result.total_tokens:,}, tool_calls={result.total_tool_calls}, "
        f"diagnosis={diagnosis[:80]}",
    )

    return run_result


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_swe_bench_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    max_instances: int | None = None,
    instance_ids: list[str] | None = None,
) -> list[SWEBenchInstance]:
    """
    Load SWE-bench instances from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split to use.
        max_instances: Maximum number of instances to load.
        instance_ids: Specific instance IDs to load (overrides max_instances).

    Returns:
        List of SWEBenchInstance objects.
    """
    from datasets import load_dataset

    logger.log(VERBOSE1, f"[HARNESS] Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    instances = []
    for item in ds:
        inst = SWEBenchInstance(
            instance_id=item["instance_id"],
            repo=item["repo"],
            base_commit=item["base_commit"],
            problem_statement=item["problem_statement"],
            patch=item.get("patch", ""),
            hints_text=item.get("hints_text", ""),
            version=item.get("version", ""),
        )

        if instance_ids:
            if inst.instance_id in instance_ids:
                instances.append(inst)
        else:
            instances.append(inst)

        if not instance_ids and max_instances and len(instances) >= max_instances:
            break

    logger.log(
        VERBOSE1,
        f"[HARNESS] Loaded {len(instances)} instances from {dataset_name}",
    )
    return instances


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_harness(
    model: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    max_instances: int = 5,
    instance_ids: list[str] | None = None,
    output_path: str = "bench/results/predictions.jsonl",
    verbosity: int = 1,
    work_dir: str | None = None,
    max_subtasks: int = 6,
) -> list[RunResult]:
    """
    Run the full SWE-bench evaluation harness.

    Args:
        model: LLM model identifier.
        dataset_name: HuggingFace dataset name.
        max_instances: Maximum instances to process.
        instance_ids: Specific instance IDs (overrides max_instances).
        output_path: Path for the predictions JSONL file.
        verbosity: Logging verbosity level.
        work_dir: Working directory for repo clones. Uses temp dir if None.
        max_subtasks: Maximum subtasks per instance.

    Returns:
        List of RunResult objects.
    """
    # Setup logging
    configure_logging(verbosity=verbosity)

    # Load instances
    instances = load_swe_bench_instances(
        dataset_name=dataset_name,
        max_instances=max_instances,
        instance_ids=instance_ids,
    )

    if not instances:
        logger.error("[HARNESS] No instances loaded!")
        return []

    # Setup working directory
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="swebench_")
    os.makedirs(work_dir, exist_ok=True)
    # Always use absolute path — prevents git diff and Path.resolve() mismatches
    # when the process cwd differs during async execution or between instances.
    work_dir = os.path.abspath(work_dir)

    logger.log(
        VERBOSE1,
        f"[HARNESS] ══════════════════════════════════════════\n"
        f"[HARNESS] SWE-bench Harness Starting\n"
        f"[HARNESS]   Model: {model}\n"
        f"[HARNESS]   Dataset: {dataset_name}\n"
        f"[HARNESS]   Instances: {len(instances)}\n"
        f"[HARNESS]   Output: {output_path}\n"
        f"[HARNESS]   Work dir: {work_dir}\n"
        f"[HARNESS] ══════════════════════════════════════════",
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run instances sequentially
    results: list[RunResult] = []
    predictions: list[dict] = []

    for i, instance in enumerate(instances, 1):
        logger.info(
            f"\n{'='*60}\n"
            f"Instance {i}/{len(instances)}: {instance.instance_id}\n"
            f"{'='*60}"
        )

        run_result = await run_instance(
            instance=instance,
            model=model,
            work_dir=work_dir,
            verbosity=verbosity,
            max_subtasks=max_subtasks,
        )
        results.append(run_result)

        # Write prediction
        if run_result.prediction:
            pred_dict = {
                "instance_id": run_result.prediction.instance_id,
                "model_name_or_path": run_result.prediction.model_name_or_path,
                "model_patch": run_result.prediction.model_patch,
            }
            predictions.append(pred_dict)

            # Append to JSONL file incrementally
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pred_dict) + "\n")

    # Print summary
    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    patches_generated = sum(1 for r in results if r.patch_generated)
    total_tokens = sum(r.total_tokens for r in results)
    total_time = sum(r.elapsed_seconds for r in results)

    summary = (
        f"\n{'='*60}\n"
        f"SWE-bench Harness Summary\n"
        f"{'='*60}\n"
        f"  Instances attempted: {total}\n"
        f"  Pipeline succeeded:  {succeeded}/{total}\n"
        f"  Patches generated:   {patches_generated}/{total}\n"
        f"  Total tokens:        {total_tokens:,}\n"
        f"  Total time:          {total_time:.1f}s\n"
        f"  Avg time/instance:   {total_time/max(total,1):.1f}s\n"
        f"  Predictions file:    {output_path}\n"
        f"{'='*60}"
    )
    print(summary)
    logger.info(summary)

    # Write detailed results
    results_path = str(Path(output_path).with_suffix(".results.json"))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "instance_id": r.instance_id,
                    "success": r.success,
                    "patch_generated": r.patch_generated,
                    "error": r.error,
                    "elapsed_seconds": r.elapsed_seconds,
                    "total_tokens": r.total_tokens,
                    "total_tool_calls": r.total_tool_calls,
                    "diagnosis": r.diagnosis,
                    "patch_length": len(r.prediction.model_patch) if r.prediction else 0,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    logger.info(f"Detailed results saved to: {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench evaluation harness for subagent_manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bench/swe_bench_harness.py --model ollama/ornith --max-instances 5
  python bench/swe_bench_harness.py --model ollama/ornith --instance-ids astropy__astropy-12907
  python bench/swe_bench_harness.py --model ollama/ornith --verbose 2 --max-instances 1
        """,
    )
    parser.add_argument(
        "--model", default="ollama/ornith",
        help="LLM model identifier (default: ollama/ornith)",
    )
    parser.add_argument(
        "--dataset", default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--max-instances", type=int, default=5,
        help="Maximum instances to process (default: 5)",
    )
    parser.add_argument(
        "--instance-ids", nargs="+", default=None,
        help="Specific instance IDs to process",
    )
    parser.add_argument(
        "--output", default="bench/results/predictions.jsonl",
        help="Output path for predictions JSONL",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level: 0=quiet, 1=decisions, 2=full detail",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Working directory for repo clones (default: temp dir)",
    )
    parser.add_argument(
        "--max-subtasks", type=int, default=6,
        help="Maximum subtasks per instance (default: 6)",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run_harness(
            model=args.model,
            dataset_name=args.dataset,
            max_instances=args.max_instances,
            instance_ids=args.instance_ids,
            output_path=args.output,
            verbosity=args.verbose,
            work_dir=args.work_dir,
            max_subtasks=args.max_subtasks,
        )
    )

    # Exit code: 0 if any patches generated, 1 otherwise
    sys.exit(0 if any(r.patch_generated for r in results) else 1)


if __name__ == "__main__":
    main()
