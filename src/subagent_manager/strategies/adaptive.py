"""
Adaptive execution strategy.

Analyzes the dependency graph from the orchestrator's plan and
automatically chooses between parallel and sequential execution.
Independent tasks run in parallel; dependent chains run sequentially.

This is the default strategy — it gives the best of both worlds.
"""

from __future__ import annotations

import logging

from subagent_manager.strategies.base import BaseStrategy, ExecutionPlan
from subagent_manager.strategies.parallel import ParallelStrategy
from subagent_manager.strategies.sequential import SequentialStrategy
from subagent_manager.subagent import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class AdaptiveStrategy(BaseStrategy):
    """
    Automatically choose between parallel and sequential execution.

    Analyzes the dependency structure of the plan:
    - If no dependencies exist → use parallel strategy
    - If all tasks are sequential → use sequential strategy
    - If mixed → use parallel strategy (which handles dependencies via waves)

    This is the recommended default strategy.
    """

    def __init__(self) -> None:
        self._parallel = ParallelStrategy()
        self._sequential = SequentialStrategy()

    async def execute(
        self,
        plan: ExecutionPlan,
        agents: dict[str, SubAgent],
        completed_results: dict[int, SubAgentResult] | None = None,
    ) -> list[SubAgentResult]:
        """Execute the plan with the best strategy for its structure."""
        if not plan.subtasks:
            return []

        if len(plan.subtasks) == 1:
            # Single task — no strategy needed
            logger.info("Adaptive: single task, executing directly")
            return await self._sequential.execute(plan, agents, completed_results)

        # Analyze dependency structure
        has_deps = plan.has_dependencies
        independent_count = len(plan.independent_tasks)
        total_count = len(plan.subtasks)

        # Check if it's a pure chain (each depends on the previous)
        is_pure_chain = all(
            s.depends_on == [s.id - 1] for s in plan.subtasks if s.depends_on
        ) and independent_count == 1

        if not has_deps:
            # All independent — full parallel
            logger.info(
                f"Adaptive: all {total_count} tasks are independent, "
                f"using parallel strategy"
            )
            return await self._parallel.execute(plan, agents, completed_results)

        elif is_pure_chain:
            # Pure sequential chain
            logger.info(
                f"Adaptive: pure sequential chain of {total_count} tasks, "
                f"using sequential strategy"
            )
            return await self._sequential.execute(plan, agents, completed_results)

        else:
            # Mixed dependencies — parallel with waves
            logger.info(
                f"Adaptive: mixed dependencies ({independent_count} independent "
                f"of {total_count} total), using parallel wave strategy"
            )
            return await self._parallel.execute(plan, agents, completed_results)
