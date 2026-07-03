"""
Sequential execution strategy.

Executes subtasks one at a time in order. Each subtask's result
is available as context for subsequent tasks. Useful when every
task depends on the previous one's output.
"""

from __future__ import annotations

import logging

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
    ) -> list[SubAgentResult]:
        """Execute subtasks one by one in order."""
        results: dict[int, SubAgentResult] = dict(completed_results or {})
        all_results: list[SubAgentResult] = list(results.values())

        for subtask in plan.subtasks:
            if subtask.id in results:
                continue  # Already completed

            logger.info(
                f"Sequential: executing subtask {subtask.id} "
                f"({subtask.agent_name})"
            )

            result = await self._execute_subtask(subtask, agents, results)
            results[subtask.id] = result
            all_results.append(result)

            if not result.success:
                logger.warning(
                    f"Subtask {subtask.id} failed: {result.error}. "
                    f"Continuing with remaining tasks."
                )

        return all_results
