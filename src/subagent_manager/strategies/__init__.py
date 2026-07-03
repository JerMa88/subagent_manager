"""Orchestration strategies for subagent execution."""

from subagent_manager.strategies.adaptive import AdaptiveStrategy
from subagent_manager.strategies.base import BaseStrategy, ExecutionPlan, SubtaskDef
from subagent_manager.strategies.parallel import ParallelStrategy
from subagent_manager.strategies.sequential import SequentialStrategy

__all__ = [
    "BaseStrategy",
    "ExecutionPlan",
    "SubtaskDef",
    "ParallelStrategy",
    "SequentialStrategy",
    "AdaptiveStrategy",
]
