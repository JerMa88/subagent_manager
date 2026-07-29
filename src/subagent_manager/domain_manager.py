"""
DomainManager — Mid-level (Level 2) Orchestrator in the 3-Level Hierarchy.

A DomainManager is responsible for a specific domain (e.g., Analysis, Code Modification, Testing).
It manages a group of Level 3 worker subagents, plans domain-specific subtasks,
executes them, and synthesizes a concise domain report back to the Level 1 High Manager.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from subagent_manager.events import Event, EventBus, EventType
from subagent_manager.llm_client import LLMClient
from subagent_manager.logging_config import (
    format_tokens,
    truncate_for_log,
    VERBOSE1,
    VERBOSE2,
)
from subagent_manager.manager import ManagerResult, SubAgentManager
from subagent_manager.subagent import SubAgentConfig, SubAgentResult

logger = logging.getLogger(__name__)


@dataclass
class DomainManagerConfig:
    """Configuration for a Level 2 Domain Manager."""

    name: str
    """Unique domain manager name (e.g. 'analysis_manager', 'testing_manager')."""

    domain_description: str
    """Description of the domain area handled by this manager."""

    worker_agents: list[SubAgentConfig]
    """Level 3 worker subagents managed by this domain manager."""

    max_subtasks: int = 5
    strategy: str = "adaptive"
    system_prompt: str | None = None

    bypass_planning: bool = False
    """
    When True AND the domain has exactly 1 worker, skip the SubAgentManager
    planning + synthesis calls and invoke the worker directly.

    WHY THIS EXISTS: Each DomainManager wraps a full SubAgentManager, which
    always runs a planning LLM call (_plan) and a synthesis LLM call
    (_synthesize) even when there is only one possible worker to delegate to.
    For PatchDomainManager (1 worker: patch_writer), this wastes ~2000 tokens
    and ~5–10s of wall time producing a trivial plan ["delegate to patch_writer"].

    SMOKE TEST REQUIRED before enabling in full ablation: confirm that
    bypassing the orchestrator prompt does not degrade patch quality.
    Results documented in bench/results/smoke_bypass_*.jsonl.
    Default is False (disabled) — must be explicitly opted into per-domain.

    WHY NOT ALWAYS ON: Multi-worker domains (analysis: 2 workers, testing: 3
    workers) NEED planning to dynamically order their workers. Only enable
    bypass_planning for domains with a single, fixed-purpose worker.
    """


@dataclass
class DomainResult:
    """Result returned by a Level 2 Domain Manager to Level 1 High Manager."""

    domain_name: str
    summary: str
    worker_results: list[SubAgentResult] = field(default_factory=list)
    success: bool = True
    total_tokens: int = 0
    total_tool_calls: int = 0
    elapsed_seconds: float = 0.0


class DomainManager:
    """
    Level 2 Domain Manager in 3-Level Hierarchy.

    Wrapping a SubAgentManager to execute domain-specific planning,
    delegation to Level 3 subagents, and synthesis.
    """

    def __init__(
        self,
        config: DomainManagerConfig,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.manager = SubAgentManager(
            model=model,
            subagents=config.worker_agents,
            strategy=config.strategy,
            max_subtasks=config.max_subtasks,
            api_key=api_key,
            api_base=api_base,
            verbose=verbose,
        )

    async def run_domain(
        self,
        domain_goal: str,
        context: str = "",
        event_bus: EventBus | None = None,
    ) -> DomainResult:
        """
        Execute domain task via Level 3 workers.

        If bypass_planning=True and only 1 worker exists, skips the
        SubAgentManager plan+synthesize overhead and calls the worker directly.
        """
        t0 = time.monotonic()
        logger.log(
            VERBOSE1,
            f"[DOMAIN:{self.config.name}] Starting domain goal: {truncate_for_log(domain_goal, 150)}",
        )

        # ── Fast path: single-worker domain with planning bypass ─────────────────
        # Bypasses the SubAgentManager orchestrator prompt + synthesis call.
        # Only active when bypass_planning=True and exactly 1 worker is configured.
        # See DomainManagerConfig.bypass_planning docstring for rationale.
        if self.config.bypass_planning and len(self.config.worker_agents) == 1:
            worker_config = self.config.worker_agents[0]
            from subagent_manager.subagent import SubAgent
            worker = SubAgent(config=worker_config, llm_client=self.manager.llm_client)
            logger.log(
                VERBOSE1,
                f"[DOMAIN:{self.config.name}] bypass_planning=True: "
                f"invoking '{worker_config.name}' directly (skipping plan+synthesize)",
            )
            worker_result = await worker.execute(
                task=domain_goal,
                context=context,
                event_bus=event_bus,
            )
            elapsed = time.monotonic() - t0
            logger.log(
                VERBOSE1,
                f"[DOMAIN:{self.config.name}] bypass complete in {elapsed:.1f}s "
                f"tokens={worker_result.tokens_used}",
            )
            return DomainResult(
                domain_name=self.config.name,
                summary=worker_result.answer,
                worker_results=[worker_result],
                success=worker_result.success,
                total_tokens=worker_result.tokens_used,
                total_tool_calls=worker_result.tool_calls_made,
                elapsed_seconds=elapsed,
            )

        # ── Normal path: full SubAgentManager orchestration ───────────────────
        res: ManagerResult = await self.manager.run(
            goal=domain_goal,
            context=context,
            event_bus=event_bus,
        )

        elapsed = time.monotonic() - t0
        domain_success = all(r.success for r in res.subtask_results) if res.subtask_results else True

        logger.log(
            VERBOSE1,
            f"[DOMAIN:{self.config.name}] Completed in {elapsed:.1f}s — success={domain_success}, tokens={res.total_tokens}",
        )

        return DomainResult(
            domain_name=self.config.name,
            summary=res.answer,
            worker_results=res.subtask_results,
            success=domain_success,
            total_tokens=res.total_tokens,
            total_tool_calls=res.total_tool_calls,
            elapsed_seconds=elapsed,
        )
