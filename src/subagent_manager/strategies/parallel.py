"""
Parallel execution strategy.

Executes independent subtasks concurrently using asyncio.gather().
Subtasks with dependencies wait until their dependencies complete.
Maximizes throughput for tasks that can run independently.
"""

from __future__ import annotations

import asyncio
import logging

from subagent_manager.strategies.base import BaseStrategy, ExecutionPlan
from subagent_manager.subagent import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class ParallelStrategy(BaseStrategy):
    """
    Execute subtasks in parallel waves.

    Groups subtasks by dependency layer:
    - Wave 0: All tasks with no dependencies (run in parallel)
    - Wave 1: All tasks that depend only on Wave 0 tasks
    - etc.

    This maximizes parallelism while respecting dependencies.
    """

    async def execute(
        self,
        plan: ExecutionPlan,
        agents: dict[str, SubAgent],
        completed_results: dict[int, SubAgentResult] | None = None,
    ) -> list[SubAgentResult]:
        """Execute the plan in parallel waves."""
        results: dict[int, SubAgentResult] = dict(completed_results or {})
        completed_ids: set[int] = set(results.keys())
        all_results: list[SubAgentResult] = list(results.values())

        # Execute in waves until all tasks are done
        max_waves = len(plan.subtasks)  # Safety limit
        for wave_num in range(max_waves):
            ready = plan.get_ready_tasks(completed_ids)

            if not ready:
                # Check if everything is done
                remaining = [s for s in plan.subtasks if s.id not in completed_ids]
                if remaining:
                    logger.warning(
                        f"No ready tasks but {len(remaining)} tasks remain. "
                        f"Possible circular dependency."
                    )
                    # Force-execute remaining tasks without dependencies
                    for task in remaining:
                        result = await self._execute_subtask(task, agents, results)
                        results[task.id] = result
                        completed_ids.add(task.id)
                        all_results.append(result)
                break

            logger.info(
                f"Wave {wave_num + 1}: executing {len(ready)} tasks in parallel"
            )

            # Execute all ready tasks in parallel
            coros = [
                self._execute_subtask(subtask, agents, results) for subtask in ready
            ]
            wave_results = await asyncio.gather(*coros, return_exceptions=True)

            for subtask, result in zip(ready, wave_results):
                if isinstance(result, Exception):
                    logger.error(f"Subtask {subtask.id} raised exception: {result}")
                    result = SubAgentResult(
                        agent_name=subtask.agent_name,
                        task=subtask.task,
                        answer="",
                        success=False,
                        error=str(result),
                    )
                results[subtask.id] = result
                completed_ids.add(subtask.id)
                all_results.append(result)

        return all_results
