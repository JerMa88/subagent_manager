"""
bench/swe_bench_hierarchy.py — 3-Level Hierarchical Multi-Agent Pipeline for SWE-bench.

Architecture:
  Level 1: Singleton High Manager (HierarchicalSWEBenchManager)
    - Responsible for global orchestration across domain phases (Analysis -> Reproduction -> Patching -> Verification).
    - Enforces skeptical senior programmer verification loop: patch is re-evaluated by Level 2 Testing Manager until green.

  Level 2: Mid-Level Domain Managers (DomainManager)
    1. AnalysisDomainManager     (Worker subagents: issue_analyzer, code_explorer)
    2. TestingDomainManager      (Worker subagents: reproducer, test_generator, test_runner)
    3. PatchDomainManager        (Worker subagents: patch_writer)

  Level 3: Grounded Worker Subagents (with specialized tools)
    - issue_analyzer, code_explorer, patch_writer, reproducer, test_generator, test_runner
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from bench.swe_bench_agents import build_swe_bench_agents
from bench.swe_bench_harness import _inject_file_content_for_patch_writer
from subagent_manager.domain_manager import DomainManager, DomainManagerConfig, DomainResult
from subagent_manager.events import EventBus
from subagent_manager.llm_client import LLMClient
from subagent_manager.logging_config import VERBOSE1, VERBOSE2, truncate_for_log
from subagent_manager.subagent import SubAgentConfig, SubAgentResult

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalResult:
    """Result of the 3-Level Hierarchical execution on an issue."""

    answer: str
    patch: str = ""
    domain_results: list[DomainResult] = field(default_factory=list)
    verified_success: bool = False
    total_tokens: int = 0
    total_tool_calls: int = 0
    subtasks_count: int = 0
    elapsed_seconds: float = 0.0

    # ── Ablation / telemetry fields (added for multi-level comparison) ──────
    # These fields are ALWAYS populated even when their values are trivial
    # (e.g., effective_depth is always 3 in the current 3-level architecture).
    # They exist so that ablation conditions with different depths produce
    # structurally identical JSONL rows that can be compared without schema
    # changes between runs.
    #
    # WHY KEPT: Critical for the planned 0/1/2/3/4-level ablation experiment.
    # Removing them would make past and future runs incomparable.

    schema_version: int = 2
    """Incremented when new fields are added; lets downstream scripts detect stale rows."""

    effective_depth: int = 3
    """Actual hierarchy levels traversed (0=one-shot, 1=flat, 2=orchestrator+workers, 3=full)."""

    retry_count: int = 0
    """How many patch-verify retries occurred (0 = first attempt succeeded or no patch)."""

    domain_trace: list[dict] = field(default_factory=list)
    """
    Per-domain breakdown for budget attribution. Each entry:
      {"domain": str, "phase": int, "tokens": int, "tool_calls": int,
       "subtasks": int, "time_s": float, "success": bool}
    Allows isolating the cost of each Level-2 domain manager independently.
    """

    planning_tokens: int = 0
    """
    Tokens spent on Level-2 orchestration overhead (planning + synthesis calls
    inside each DomainManager's SubAgentManager). Computed as:
      total_tokens - work_tokens
    where work_tokens = sum of Level-3 worker agent tokens.
    Currently approximated as 0 (full split requires per-agent token logging
    inside manager._plan and manager._synthesize, tracked in Gap-5 of the
    ablation readiness doc). Reserved field — will be populated once
    planning-token split is implemented.
    """


class HierarchicalSWEBenchManager:
    """
    Level 1 Singleton High Manager for SWE-bench.

    Orchestrates Level 2 domain managers through a 4-phase verification loop:
      Phase 1: Analysis Domain (issue_analyzer + code_explorer)
      Phase 2: Reproduction Domain (reproducer + test_generator)
      Phase 3: Patch Modification Domain (patch_writer)
      Phase 4: Independent Verification Domain (test_runner)
    """

    def __init__(
        self,
        model: str,
        repo_dir: str,
        prompt_repo_dir: str | None = None,
        max_patch_retries: int = 2,
        api_key: str | None = None,
        api_base: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.repo_dir = repo_dir
        self.prompt_repo_dir = prompt_repo_dir or repo_dir
        self.max_patch_retries = max_patch_retries
        self.verbose = verbose

        # Build all Level 3 worker subagents
        all_workers = build_swe_bench_agents(repo_dir, prompt_repo_dir=self.prompt_repo_dir)
        worker_dict = {w.name: w for w in all_workers}

        # Level 2 Domain Manager 1: Analysis Domain
        analysis_workers = [worker_dict["issue_analyzer"], worker_dict["code_explorer"]]
        self.analysis_mgr = DomainManager(
            config=DomainManagerConfig(
                name="AnalysisDomainManager",
                domain_description="Analyzes GitHub issue, diagnoses bug, explores codebase",
                worker_agents=analysis_workers,
                strategy="sequential",
            ),
            model=model,
            api_key=api_key,
            api_base=api_base,
            verbose=verbose,
        )

        # Level 2 Domain Manager 2: Testing & Reproduction Domain
        testing_workers = [worker_dict["reproducer"], worker_dict["test_generator"], worker_dict["test_runner"]]
        self.testing_mgr = DomainManager(
            config=DomainManagerConfig(
                name="TestingDomainManager",
                domain_description="Creates bug reproduce scripts, targeted pytests, and verifies patches",
                worker_agents=testing_workers,
                strategy="sequential",
            ),
            model=model,
            api_key=api_key,
            api_base=api_base,
            verbose=verbose,
        )

        # Level 2 Domain Manager 3: Patch Modification Domain
        patch_workers = [worker_dict["patch_writer"]]
        self.patch_mgr = DomainManager(
            config=DomainManagerConfig(
                name="PatchDomainManager",
                domain_description="Applies surgical code patch using str_replace",
                worker_agents=patch_workers,
                strategy="sequential",
            ),
            model=model,
            api_key=api_key,
            api_base=api_base,
            verbose=verbose,
        )

        # Hook patch_mgr manager plan to inject pre-loaded file content for patch_writer
        orig_patch_plan = self.patch_mgr.manager._plan

        async def custom_patch_plan(goal: str, context: str = "") -> tuple:
            plan, plan_data = await orig_patch_plan(goal, context=context)
            plan.subtasks = _inject_file_content_for_patch_writer(
                subtasks=plan.subtasks,
                repo_dir=self.repo_dir,
                prompt_repo_dir=self.prompt_repo_dir,
                goal=goal,
                logger=logger,
            )
            return plan, plan_data

        self.patch_mgr.manager._plan = custom_patch_plan

    async def run(
        self,
        problem_statement: str,
        context: str = "",
        event_bus: EventBus | None = None,
    ) -> HierarchicalResult:
        """
        Run 3-Level Hierarchical execution pipeline on a SWE-bench problem statement.
        """
        t0 = time.monotonic()
        logger.log(VERBOSE1, f"[LEVEL1_MANAGER] ═════ Starting 3-Level Hierarchy Pipeline ═════")
        logger.log(VERBOSE1, f"[LEVEL1_MANAGER] Issue: {truncate_for_log(problem_statement, 150)}")

        domain_results: list[DomainResult] = []
        domain_trace: list[dict] = []
        total_tokens = 0
        total_tool_calls = 0
        subtasks_count = 0

        def _record_domain(phase: int, res: DomainResult) -> None:
            """Append a domain result to the trace list."""
            domain_trace.append({
                "domain": res.domain_name,
                "phase": phase,
                "tokens": res.total_tokens,
                "tool_calls": res.total_tool_calls,
                "subtasks": len(res.worker_results),
                "time_s": round(res.elapsed_seconds, 2),
                "success": res.success,
            })

        # ------------------------------------------------------------------
        # Phase 1 (Level 2 AnalysisDomainManager): Diagnose bug & explore repo
        # ------------------------------------------------------------------
        analysis_goal = (
            f"Analyze the following GitHub issue and inspect relevant code files.\n\n"
            f"Issue Description:\n{problem_statement}\n\n"
            f"Determine:\n1. What is the root cause?\n2. Which files and functions are involved?"
        )
        res_analysis = await self.analysis_mgr.run_domain(analysis_goal, context=context, event_bus=event_bus)
        domain_results.append(res_analysis)
        _record_domain(1, res_analysis)
        total_tokens += res_analysis.total_tokens
        total_tool_calls += res_analysis.total_tool_calls
        subtasks_count += len(res_analysis.worker_results)
        diagnosis_text = res_analysis.summary

        # ------------------------------------------------------------------
        # Phase 2 (Level 2 TestingDomainManager): Reproduce bug
        # ------------------------------------------------------------------
        repro_goal = (
            f"Write and execute /tmp/reproduce.py to confirm the bug described in the issue.\n\n"
            f"Issue:\n{problem_statement}\n\n"
            f"Diagnosis:\n{diagnosis_text}"
        )
        res_repro = await self.testing_mgr.run_domain(repro_goal, context=context, event_bus=event_bus)
        domain_results.append(res_repro)
        _record_domain(2, res_repro)
        total_tokens += res_repro.total_tokens
        total_tool_calls += res_repro.total_tool_calls
        subtasks_count += len(res_repro.worker_results)
        repro_summary = res_repro.summary

        # ------------------------------------------------------------------
        # Phase 3 & 4 (Level 2 PatchDomainManager -> Level 2 TestingDomainManager):
        # Patch & Verification Loop
        # ------------------------------------------------------------------
        verified_pass = False
        retry_count = 0
        latest_verification_feedback = ""

        while retry_count <= self.max_patch_retries and not verified_pass:
            attempt_label = f"Attempt {retry_count + 1}/{self.max_patch_retries + 1}"
            logger.log(VERBOSE1, f"[LEVEL1_MANAGER] {attempt_label}: Invoking PatchDomainManager")

            patch_goal = (
                f"Apply surgical str_replace fix for the diagnosed issue.\n\n"
                f"Issue:\n{problem_statement}\n\n"
                f"Diagnosis:\n{diagnosis_text}\n\n"
                f"Reproduction Output:\n{repro_summary}"
            )
            if latest_verification_feedback:
                patch_goal += f"\n\nPrevious Verification Feedback (Fix failed! Retry required):\n{latest_verification_feedback}"

            res_patch = await self.patch_mgr.run_domain(patch_goal, context=context, event_bus=event_bus)
            domain_results.append(res_patch)
            _record_domain(3, res_patch)
            total_tokens += res_patch.total_tokens
            total_tool_calls += res_patch.total_tool_calls
            subtasks_count += len(res_patch.worker_results)

            # Level 1 Skeptical Senior Programmer Verification Gate
            verify_goal = (
                f"Independently verify the code fix by re-running /tmp/reproduce.py or running pytest.\n"
                f"Check if the output contains 'BUG FIXED' or green test pass."
            )
            res_verify = await self.testing_mgr.run_domain(verify_goal, context=context, event_bus=event_bus)
            domain_results.append(res_verify)
            _record_domain(4, res_verify)
            total_tokens += res_verify.total_tokens
            total_tool_calls += res_verify.total_tool_calls
            subtasks_count += len(res_verify.worker_results)

            verify_text = res_verify.summary.upper()
            if "BUG FIXED" in verify_text or "PASSED" in verify_text or "GREEN" in verify_text:
                logger.log(VERBOSE1, f"[LEVEL1_MANAGER] Verification SUCCESS on {attempt_label}!")
                verified_pass = True
            else:
                latest_verification_feedback = res_verify.summary
                logger.warning(f"[LEVEL1_MANAGER] Verification FAILED on {attempt_label}. Feedback: {truncate_for_log(latest_verification_feedback, 200)}")
                retry_count += 1

        elapsed = time.monotonic() - t0
        summary_answer = (
            f"3-Level Hierarchy Completed in {elapsed:.1f}s.\n"
            f"Verified Success: {verified_pass}\n"
            f"Retries: {retry_count}\n"
            f"Final Verification Output:\n{latest_verification_feedback if not verified_pass else 'BUG FIXED'}"
        )
        logger.log(
            VERBOSE1,
            f"[LEVEL1_MANAGER] ═════ Pipeline Done ═════  "
            f"elapsed={elapsed:.1f}s  tokens={total_tokens:,}  "
            f"retries={retry_count}  subtasks={subtasks_count}  verified={verified_pass}",
        )

        return HierarchicalResult(
            answer=summary_answer,
            domain_results=domain_results,
            verified_success=verified_pass,
            total_tokens=total_tokens,
            total_tool_calls=total_tool_calls,
            subtasks_count=subtasks_count,
            elapsed_seconds=elapsed,
            effective_depth=3,
            retry_count=retry_count,
            domain_trace=domain_trace,
        )

async def run_hierarchical_instance(
    instance: Any,
    model: str,
    work_dir: str,
    api_base: str | None = None,
    api_key: str | None = None,
    verbosity: int = 0,
) -> Any:
    """
    Run 3-Level Hierarchical Multi-Agent pipeline on a SWE-bench instance.
    """
    from bench.swe_bench_harness import (
        RunResult,
        SWEBenchPrediction,
        clone_and_checkout,
        extract_patch,
        _try_extract_and_write_code,
    )

    t0 = time.monotonic()
    try:
        repo_dir = clone_and_checkout(instance.repo, instance.base_commit, work_dir)
    except Exception as e:
        logger.error(f"[HARNESS:HIER] Clone failed: {e}")
        return RunResult(
            instance_id=instance.instance_id,
            success=False,
            patch_generated=False,
            error=f"Clone failed: {e}",
            elapsed_seconds=time.monotonic() - t0,
        )

    sanitized_id = instance.instance_id.replace("/", "_").replace("__", "_")
    short_repo = f"/tmp/repo_{sanitized_id}"
    abs_repo_dir = os.path.abspath(repo_dir)
    try:
        if os.path.islink(short_repo) or os.path.exists(short_repo):
            os.remove(short_repo)
        os.symlink(abs_repo_dir, short_repo)
        prompt_repo_dir = short_repo
    except Exception as e:
        logger.warning(f"[HARNESS:HIER] Could not create {short_repo} symlink: {e}")
        prompt_repo_dir = abs_repo_dir

    h_mgr = HierarchicalSWEBenchManager(
        model=model,
        repo_dir=abs_repo_dir,
        prompt_repo_dir=prompt_repo_dir,
        api_key=api_key,
        api_base=api_base,
        verbose=verbosity > 0,
    )

    goal = instance.problem_statement
    context = f"Hints from issue:\n{instance.hints_text}\n" if getattr(instance, "hints_text", None) else ""
    context += f"Repository: {instance.repo}\nWorking directory: {abs_repo_dir}"

    try:
        res = await h_mgr.run(goal, context=context)
    except Exception as e:
        logger.error(f"[HARNESS:HIER] Pipeline failed: {e}", exc_info=True)
        return RunResult(
            instance_id=instance.instance_id,
            success=False,
            patch_generated=False,
            error=f"Pipeline failed: {e}",
            elapsed_seconds=time.monotonic() - t0,
        )

    patch = extract_patch(abs_repo_dir)
    if not patch:
        logger.log(
            VERBOSE1,
            "[HARNESS:HIER] No git diff detected — attempting fallback code extraction",
        )
        class PseudoResult:
            def __init__(self, answer, domain_results):
                self.answer = answer
                self.subtask_results = []
                for dr in domain_results:
                    self.subtask_results.extend(dr.worker_results)

        pseudo_res = PseudoResult(res.answer, res.domain_results)
        _try_extract_and_write_code(pseudo_res, abs_repo_dir)
        patch = extract_patch(abs_repo_dir)

    return RunResult(
        instance_id=instance.instance_id,
        success=res.verified_success or bool(patch),
        patch_generated=bool(patch),
        prediction=SWEBenchPrediction(
            instance_id=instance.instance_id,
            model_name_or_path=model,
            model_patch=patch or "",
        )
        if patch
        else None,
        elapsed_seconds=time.monotonic() - t0,
        total_tokens=res.total_tokens,
        total_tool_calls=res.total_tool_calls,
        subtasks_count=res.subtasks_count,
        diagnosis=res.answer[:500] if res.answer else "",
        # Ablation telemetry — propagated from HierarchicalResult
        effective_depth=res.effective_depth,
        retry_count=res.retry_count,
        domain_trace=res.domain_trace,
        schema_version=res.schema_version,
    )

