"""
Parallel execution strategy.

Executes independent subtasks concurrently using asyncio.gather().
Subtasks with dependencies wait until their dependencies complete.
Maximizes throughput for tasks that can run independently.
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
        event_bus: EventBus | None = None,
        pause_events: dict[int, asyncio.Event] | None = None,
        cancel_event: asyncio.Event | None = None,
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
                        f"[STRATEGY] No ready tasks but {len(remaining)} tasks remain. "
                        f"Possible circular dependency. Remaining IDs: "
                        f"{[s.id for s in remaining]}, deps: "
                        f"{[(s.id, s.depends_on) for s in remaining]}"
                    )
                    # Force-execute remaining tasks without dependencies
                    for task in remaining:
                        result = await self._execute_subtask(
                            task, agents, results,
                            event_bus=event_bus,
                            pause_events=pause_events,
                            cancel_event=cancel_event,
                        )
                        results[task.id] = result
                        completed_ids.add(task.id)
                        all_results.append(result)
                break

            logger.log(
                VERBOSE1,
                f"[STRATEGY] Wave {wave_num + 1}: executing {len(ready)} tasks in parallel "
                f"(IDs: {[s.id for s in ready]}, agents: {[s.agent_name for s in ready]})",
            )
            wave_t0 = time.monotonic()

            # Execute all ready tasks in parallel
            coros = [
                self._execute_subtask(
                    subtask, agents, results,
                    event_bus=event_bus,
                    pause_events=pause_events,
                    cancel_event=cancel_event,
                )
                for subtask in ready
            ]
            wave_results = await asyncio.gather(*coros, return_exceptions=True)
            wave_duration = time.monotonic() - wave_t0

            for subtask, result in zip(ready, wave_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"[STRATEGY] Subtask {subtask.id} (agent={subtask.agent_name}) "
                        f"raised exception: {result}"
                    )
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
