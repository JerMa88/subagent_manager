"""System prompts for orchestrator and subagent roles."""

from subagent_manager.prompts.orchestrator import (
    build_orchestrator_system_prompt,
    build_synthesis_prompt,
)
from subagent_manager.prompts.subagent import build_subagent_system_prompt

__all__ = [
    "build_orchestrator_system_prompt",
    "build_synthesis_prompt",
    "build_subagent_system_prompt",
]
