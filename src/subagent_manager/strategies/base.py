"""
Base strategy for orchestration.

Defines the common data structures and interface for all
execution strategies (parallel, sequential, adaptive).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from subagent_manager.events import EventBus
from subagent_manager.subagent import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


@dataclass
class SubtaskDef:
    """A single subtask as defined by the orchestrator's plan."""

    id: int
    """Unique ID within the plan."""

    task: str
    """The task description for the subagent."""

    agent_name: str
    """Which agent should handle this."""

    depends_on: list[int] = field(default_factory=list)
    """IDs of subtasks that must complete before this one."""

    context: str = ""
    """Additional context for the subagent."""


@dataclass
class ExecutionPlan:
    """A parsed execution plan from the orchestrator."""

    subtasks: list[SubtaskDef]
    """Ordered list of subtasks."""

    @property
    def has_dependencies(self) -> bool:
        """Whether any subtask depends on another."""
        return any(s.depends_on for s in self.subtasks)

    @property
    def independent_tasks(self) -> list[SubtaskDef]:
        """Subtasks that can run immediately (no dependencies)."""
        return [s for s in self.subtasks if not s.depends_on]

    def get_ready_tasks(self, completed_ids: set[int]) -> list[SubtaskDef]:
        """Get subtasks whose dependencies are all satisfied."""
        return [
            s
            for s in self.subtasks
            if s.id not in completed_ids
            and all(dep in completed_ids for dep in s.depends_on)
        ]


class BaseStrategy(ABC):
    """
    Abstract base for orchestration strategies.

    A strategy decides HOW to execute a plan — in parallel,
    sequentially, or with adaptive routing.
    """

    @abstractmethod
    async def execute(
        self,
        plan: ExecutionPlan,
        agents: dict[str, SubAgent],
        completed_results: dict[int, SubAgentResult] | None = None,
        event_bus: EventBus | None = None,
        pause_events: dict[int, asyncio.Event] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> list[SubAgentResult]:
        """
        Execute the plan using the available agents.

        Args:
            plan: The execution plan with subtask definitions.
            agents: Map of agent name -> SubAgent instance.
            completed_results: Previously completed results (for resumption).
            event_bus: Optional event bus for GUI streaming.
            pause_events: Per-subtask asyncio.Event for pause/resume control.
            cancel_event: Global cancel signal for entire orchestration.

        Returns:
            List of SubAgentResult, one per subtask.
        """
        ...

    def _build_dependency_context(
        self,
        subtask: SubtaskDef,
        results: dict[int, SubAgentResult],
    ) -> str:
        """
        Build context string from dependency results.

        When a subtask depends on others, inject the dependency
        results as context. Keep it minimal to preserve the
        short-horizon constraint.
        """
        if not subtask.depends_on:
            return subtask.context

        parts = []
        if subtask.context:
            parts.append(subtask.context)

        for dep_id in subtask.depends_on:
            dep_result = results.get(dep_id)
            if dep_result and dep_result.success:
                parts.append(
                    f"[Result from previous subtask '{dep_result.agent_name}']: "
                    f"{dep_result.answer}"
                )

        return "\n\n".join(parts)

    async def _execute_subtask(
        self,
        subtask: SubtaskDef,
        agents: dict[str, SubAgent],
        results: dict[int, SubAgentResult],
        event_bus: EventBus | None = None,
        pause_events: dict[int, asyncio.Event] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> SubAgentResult:
        """Execute a single subtask with the appropriate agent."""
        agent = agents.get(subtask.agent_name)

        if agent is None:
            # Fallback: use the first available agent
            logger.warning(
                f"Agent '{subtask.agent_name}' not found for subtask {subtask.id}. "
                f"Using fallback agent."
            )
            agent = next(iter(agents.values()))

        context = self._build_dependency_context(subtask, results)
        pause_event = pause_events.get(subtask.id) if pause_events else None
        return await agent.execute(
            task=subtask.task,
            context=context,
            event_bus=event_bus,
            subtask_id=subtask.id,
            pause_event=pause_event,
            cancel_event=cancel_event,
        )
