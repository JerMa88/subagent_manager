"""
SubAgent — an isolated worker that executes a single task.

Each SubAgent runs in a fresh context window with its own tools.
It performs ONE task, uses tools to ground its answer, and returns
a concise result. The context is discarded after execution.

This isolation is the key to preventing "lost-in-the-middle"
information loss and multi-step reasoning hallucinations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from subagent_manager.llm_client import LLMClient
from subagent_manager.prompts.subagent import build_subagent_system_prompt
from subagent_manager.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class SubAgentConfig:
    """
    Configuration for a subagent.

    Defines the agent's identity, capabilities, and constraints.
    The orchestrator uses `name` and `description` to decide which
    agent to assign each subtask to.
    """

    name: str
    """Unique identifier for this agent (e.g., 'researcher', 'analyzer')."""

    description: str
    """What this agent specializes in. Used by the orchestrator for routing."""

    tools: list[BaseTool] = field(default_factory=list)
    """Tools available to this agent."""

    model: str | None = None
    """Override model for this agent. If None, uses the manager's model."""

    system_prompt: str | None = None
    """Custom system prompt. If None, auto-generated from name/description."""

    max_tool_iterations: int = 5
    """Max rounds of tool calls before forcing a final answer."""

    max_answer_tokens: int = 512
    """Max tokens for the final answer. Forces concise responses."""

    temperature: float = 0.0
    """Temperature for this agent's LLM calls."""


@dataclass
class SubAgentResult:
    """
    The result from a subagent execution.

    Designed to be small and focused — just the answer, sources,
    and metadata. This is what flows back to the orchestrator.
    """

    agent_name: str
    """Which agent produced this result."""

    task: str
    """The task that was assigned."""

    answer: str
    """The concise, grounded answer."""

    sources: list[str] = field(default_factory=list)
    """URLs or references used by the agent."""

    tool_calls_made: int = 0
    """How many tool calls were used."""

    success: bool = True
    """Whether the agent completed successfully."""

    error: str | None = None
    """Error message if success is False."""

    tokens_used: int = 0
    """Total tokens consumed by this agent."""


class SubAgent:
    """
    An isolated worker that executes a single task.

    The key design principle: each execution creates a FRESH context.
    No state is carried between executions. This enforces short-horizon
    reasoning by ensuring the agent can only work with the immediate
    task and its tool results.
    """

    def __init__(
        self,
        config: SubAgentConfig,
        llm_client: LLMClient,
    ) -> None:
        """
        Initialize a subagent.

        Args:
            config: The agent's configuration.
            llm_client: The LLM client to use (may be overridden by config.model).
        """
        self.config = config

        # Use model override if specified
        if config.model and config.model != llm_client.model:
            self.llm_client = LLMClient(
                model=config.model,
                api_key=llm_client.api_key,
                api_base=llm_client.api_base,
                default_temperature=config.temperature,
                default_max_tokens=config.max_answer_tokens,
            )
        else:
            self.llm_client = llm_client

    async def execute(self, task: str, context: str = "") -> SubAgentResult:
        """
        Execute a single task in a fresh, isolated context.

        This is the core of the short-horizon reasoning constraint:
        1. A FRESH context is created (no history)
        2. The agent receives ONLY the task + minimal context
        3. It uses tools to gather information
        4. It returns a SHORT, grounded answer
        5. The context is DISCARDED

        Args:
            task: The specific task to accomplish.
            context: Optional minimal context (e.g., results from dependency tasks).

        Returns:
            SubAgentResult with the concise answer and metadata.
        """
        logger.info(f"SubAgent '{self.config.name}' executing: {task[:100]}...")

        # Build system prompt
        system_prompt = self.config.system_prompt or build_subagent_system_prompt(
            agent_name=self.config.name,
            agent_description=self.config.description,
        )

        # Build user message
        user_content = f"## YOUR TASK\n\n{task}"
        if context:
            user_content = f"## CONTEXT\n\n{context}\n\n{user_content}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            if self.config.tools:
                # Run with tool loop
                result = await self.llm_client.complete_with_tool_loop(
                    messages=messages,
                    tools=self.config.tools,
                    max_iterations=self.config.max_tool_iterations,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_answer_tokens,
                )

                return SubAgentResult(
                    agent_name=self.config.name,
                    task=task,
                    answer=result.final_answer,
                    sources=result.sources,
                    tool_calls_made=result.tool_calls_made,
                    success=True,
                    tokens_used=result.total_tokens,
                )
            else:
                # No tools — direct completion
                result = await self.llm_client.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_answer_tokens,
                )

                # Try to extract sources from the text
                sources = self._extract_sources(result.content)

                return SubAgentResult(
                    agent_name=self.config.name,
                    task=task,
                    answer=result.content,
                    sources=sources,
                    tool_calls_made=0,
                    success=True,
                    tokens_used=result.usage.get("total_tokens", 0),
                )

        except Exception as e:
            logger.error(f"SubAgent '{self.config.name}' failed: {e}")
            return SubAgentResult(
                agent_name=self.config.name,
                task=task,
                answer="",
                success=False,
                error=str(e),
            )

    @staticmethod
    def _extract_sources(text: str) -> list[str]:
        """Extract URLs from text as potential sources."""
        url_pattern = r"https?://[^\s\)\]\}\"'>]+"
        return list(set(re.findall(url_pattern, text)))

    def __repr__(self) -> str:
        tools_str = ", ".join(t.name for t in self.config.tools)
        return (
            f"<SubAgent name={self.config.name!r} "
            f"tools=[{tools_str}]>"
        )
