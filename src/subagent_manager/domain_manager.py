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
        """
        t0 = time.monotonic()
        logger.log(
            VERBOSE1,
            f"[DOMAIN:{self.config.name}] Starting domain goal: {truncate_for_log(domain_goal, 150)}",
        )

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
