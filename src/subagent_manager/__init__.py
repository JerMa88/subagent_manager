"""
subagent_manager — Universal short-horizon reasoning framework for LLMs.

Enforces single-step reasoning chains through subagent delegation to reduce
hallucinations and "lost-in-the-middle" information loss.

Usage:
    from subagent_manager import SubAgentManager, SubAgentConfig

    manager = SubAgentManager(model="ollama/qwen3")
    result = manager.run_sync("Research the latest advances in quantum computing")
"""

from subagent_manager.events import Event, EventBus, EventType
from subagent_manager.llm_client import LLMClient
from subagent_manager.logging_config import configure_logging
from subagent_manager.manager import ManagerResult, SubAgentManager
from subagent_manager.subagent import SubAgent, SubAgentConfig, SubAgentResult

__all__ = [
    "SubAgentManager",
    "ManagerResult",
    "SubAgent",
    "SubAgentConfig",
    "SubAgentResult",
    "LLMClient",
    "EventBus",
    "EventType",
    "Event",
    "configure_logging",
]

__version__ = "0.1.0"
