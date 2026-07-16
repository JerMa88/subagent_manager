"""
Sequential execution strategy.

Executes subtasks one at a time in order. Each subtask's result
is available as context for subsequent tasks. Useful when every
task depends on the previous one's output.
"""

from __future__ import annotations

import asyncio
import logging
import time

from subagent_manager.events import EventBus
from subagent_manager.logging_config import VERBOSE1
from subagent_manager.strategies.base import BaseStrategy, ExecutionPlan
from subagent_manager.subagent import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class SequentialStrategy(BaseStrategy):
    """
    Execute subtasks sequentially, one at a time.

    Each subtask can access results from all previously completed
    subtasks via dependency context injection. Simple and predictable.
    """

    async def execute(
        self,
        plan: ExecutionPlan,
        agents: dict[str, SubAgent],
        completed_results: dict[int, SubAgentResult] | None = None,
        event_bus: EventBus | None = None,
        pause_events: dict[int, asyncio.Event] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> list[SubAgentResult]:
        """Execute subtasks one by one in order."""
        results: dict[int, SubAgentResult] = dict(completed_results or {})
        all_results: list[SubAgentResult] = list(results.values())

        for subtask in plan.subtasks:
            if subtask.id in results:
                continue  # Already completed

            logger.log(
                VERBOSE1,
                f"[STRATEGY] Sequential: executing subtask {subtask.id}/{len(plan.subtasks)} "
                f"(agent={subtask.agent_name})",
            )

            subtask_t0 = time.monotonic()
            result = await self._execute_subtask(
                subtask, agents, results,
                event_bus=event_bus,
                pause_events=pause_events,
                cancel_event=cancel_event,
            )
            results[subtask.id] = result
            all_results.append(result)
            subtask_duration = time.monotonic() - subtask_t0

            logger.log(
                VERBOSE1,
                f"[STRATEGY] Sequential: subtask {subtask.id} "
                f"{'completed' if result.success else 'FAILED'} "
                f"in {subtask_duration:.1f}s",
            )

            if not result.success:
                logger.warning(
                    f"[STRATEGY] Subtask {subtask.id} failed: {result.error}. "
                    f"Continuing with remaining tasks."
                )

        return all_results
