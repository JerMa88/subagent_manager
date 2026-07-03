"""Tests for execution strategies."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from subagent_manager.strategies.base import (
    BaseStrategy,
    ExecutionPlan,
    SubtaskDef,
)
from subagent_manager.strategies.parallel import ParallelStrategy
from subagent_manager.strategies.sequential import SequentialStrategy
from subagent_manager.strategies.adaptive import AdaptiveStrategy
from subagent_manager.subagent import SubAgent, SubAgentConfig, SubAgentResult
from subagent_manager.llm_client import LLMClient


def make_mock_agent(name: str) -> SubAgent:
    """Create a mock subagent that returns a canned result."""
    config = SubAgentConfig(name=name, description=f"{name} agent")
    client = LLMClient(model="test-model")
    agent = SubAgent(config=config, llm_client=client)

    async def mock_execute(task: str, context: str = "") -> SubAgentResult:
        return SubAgentResult(
            agent_name=name,
            task=task,
            answer=f"Answer from {name}: {task[:50]}",
            success=True,
        )

    agent.execute = mock_execute  # type: ignore
    return agent


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_has_dependencies(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="a"),
            SubtaskDef(id=2, task="Task 2", agent_name="b", depends_on=[1]),
        ])
        assert plan.has_dependencies is True

    def test_no_dependencies(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="a"),
            SubtaskDef(id=2, task="Task 2", agent_name="b"),
        ])
        assert plan.has_dependencies is False

    def test_independent_tasks(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="a"),
            SubtaskDef(id=2, task="Task 2", agent_name="b"),
            SubtaskDef(id=3, task="Task 3", agent_name="c", depends_on=[1]),
        ])
        assert len(plan.independent_tasks) == 2

    def test_get_ready_tasks(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="a"),
            SubtaskDef(id=2, task="Task 2", agent_name="b", depends_on=[1]),
            SubtaskDef(id=3, task="Task 3", agent_name="c", depends_on=[1, 2]),
        ])

        # Initially, only task 1 is ready
        ready = plan.get_ready_tasks(set())
        assert len(ready) == 1
        assert ready[0].id == 1

        # After task 1 completes, task 2 is ready
        ready = plan.get_ready_tasks({1})
        assert len(ready) == 1
        assert ready[0].id == 2

        # After tasks 1 and 2 complete, task 3 is ready
        ready = plan.get_ready_tasks({1, 2})
        assert len(ready) == 1
        assert ready[0].id == 3


class TestParallelStrategy:
    """Tests for the parallel execution strategy."""

    @pytest.mark.asyncio
    async def test_all_independent_tasks(self):
        """All independent tasks should run in a single wave."""
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="agent_a"),
            SubtaskDef(id=2, task="Task 2", agent_name="agent_b"),
        ])
        agents = {
            "agent_a": make_mock_agent("agent_a"),
            "agent_b": make_mock_agent("agent_b"),
        }

        strategy = ParallelStrategy()
        results = await strategy.execute(plan, agents)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dependent_tasks(self):
        """Dependent tasks should wait for their dependencies."""
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Research", agent_name="agent_a"),
            SubtaskDef(
                id=2, task="Analyze", agent_name="agent_b", depends_on=[1]
            ),
        ])
        agents = {
            "agent_a": make_mock_agent("agent_a"),
            "agent_b": make_mock_agent("agent_b"),
        }

        strategy = ParallelStrategy()
        results = await strategy.execute(plan, agents)

        assert len(results) == 2
        assert results[0].agent_name == "agent_a"
        assert results[1].agent_name == "agent_b"


class TestSequentialStrategy:
    """Tests for the sequential execution strategy."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="agent_a"),
            SubtaskDef(id=2, task="Task 2", agent_name="agent_a"),
        ])
        agents = {"agent_a": make_mock_agent("agent_a")}

        strategy = SequentialStrategy()
        results = await strategy.execute(plan, agents)

        assert len(results) == 2
        assert results[0].task == "Task 1"
        assert results[1].task == "Task 2"


class TestAdaptiveStrategy:
    """Tests for the adaptive execution strategy."""

    @pytest.mark.asyncio
    async def test_single_task(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Only task", agent_name="agent_a"),
        ])
        agents = {"agent_a": make_mock_agent("agent_a")}

        strategy = AdaptiveStrategy()
        results = await strategy.execute(plan, agents)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_empty_plan(self):
        plan = ExecutionPlan(subtasks=[])
        agents = {"agent_a": make_mock_agent("agent_a")}

        strategy = AdaptiveStrategy()
        results = await strategy.execute(plan, agents)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_all_independent_uses_parallel(self):
        plan = ExecutionPlan(subtasks=[
            SubtaskDef(id=1, task="Task 1", agent_name="agent_a"),
            SubtaskDef(id=2, task="Task 2", agent_name="agent_b"),
            SubtaskDef(id=3, task="Task 3", agent_name="agent_a"),
        ])
        agents = {
            "agent_a": make_mock_agent("agent_a"),
            "agent_b": make_mock_agent("agent_b"),
        }

        strategy = AdaptiveStrategy()
        results = await strategy.execute(plan, agents)

        assert len(results) == 3
        assert all(r.success for r in results)
