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

from subagent_manager.events import Event, EventBus, EventType
from subagent_manager.llm_client import LLMClient
from subagent_manager.logging_config import (
    format_tokens,
    truncate_for_log,
    VERBOSE1,
    VERBOSE2,
)
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

    max_history_chars: int | None = None
    """Override for sliding-window context pruning threshold.
    If None, uses the LLM client's class-level MAX_HISTORY_CHARS default (10,000).
    Set higher for agents that need to retain large file contents (e.g., patch_writer).
    """

    mandatory_tool_call: bool = False
    """If True, the agent MUST call a tool on each iteration until budget is exhausted.
    Removes the 'respond directly if no tool needed' escape hatch from tool instructions.
    Use for agents like patch_writer (must call str_replace) and reproducer (must call write_file).
    """


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

    async def execute(
        self,
        task: str,
        context: str = "",
        event_bus: EventBus | None = None,
        subtask_id: int | None = None,
        pause_event: Any | None = None,
        cancel_event: Any | None = None,
    ) -> SubAgentResult:
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
            event_bus: Optional event bus for GUI streaming.
            subtask_id: Subtask ID for event tagging.
            pause_event: asyncio.Event — cleared=paused, set=running.
            cancel_event: asyncio.Event — set=cancel requested.

        Returns:
            SubAgentResult with the concise answer and metadata.
        """
        logger.log(
            VERBOSE1,
            f"[AGENT:{self.config.name}] Executing task: {truncate_for_log(task, 200)}",
        )
        if context:
            logger.log(VERBOSE1, f"[AGENT:{self.config.name}] Context provided ({len(context)} chars)")
            logger.log(VERBOSE2, f"[AGENT:{self.config.name}] Context: {truncate_for_log(context, 500)}")

        # Emit subtask_started event
        if event_bus:
            event_bus.emit(Event(
                type=EventType.SUBTASK_STARTED,
                subtask_id=subtask_id,
                agent_name=self.config.name,
                data={
                    "task": task,
                    "context": context[:500] if context else "",
                    "tools": [t.name for t in self.config.tools],
                },
            ))

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

        logger.log(VERBOSE2, f"[AGENT:{self.config.name}] System prompt ({len(system_prompt)} chars):\n{system_prompt}")
        logger.log(VERBOSE2, f"[AGENT:{self.config.name}] User message ({len(user_content)} chars):\n{user_content}")

        try:
            if self.config.tools:
                # Run with tool loop
                logger.log(
                    VERBOSE1,
                    f"[AGENT:{self.config.name}] Using tool loop "
                    f"({len(self.config.tools)} tools, max_iter={self.config.max_tool_iterations})",
                )
                result = await self.llm_client.complete_with_tool_loop(
                    messages=messages,
                    tools=self.config.tools,
                    max_iterations=self.config.max_tool_iterations,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_answer_tokens,
                    max_history_chars=self.config.max_history_chars,
                    mandatory_tool_call=self.config.mandatory_tool_call,
                    event_bus=event_bus,
                    subtask_id=subtask_id,
                    agent_name=self.config.name,
                    pause_event=pause_event,
                    cancel_event=cancel_event,
                )

                sub_result = SubAgentResult(
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
                logger.log(
                    VERBOSE1,
                    f"[AGENT:{self.config.name}] Direct completion (no tools)",
                )
                result = await self.llm_client.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_answer_tokens,
                )

                # Try to extract sources from the text
                sources = self._extract_sources(result.content)

                sub_result = SubAgentResult(
                    agent_name=self.config.name,
                    task=task,
                    answer=result.content,
                    sources=sources,
                    tool_calls_made=0,
                    success=True,
                    tokens_used=result.usage.get("total_tokens", 0),
                )

            # Log result summary
            logger.log(
                VERBOSE1,
                f"[AGENT:{self.config.name}] Completed: "
                f"success={sub_result.success}, "
                f"tokens={sub_result.tokens_used:,}, "
                f"tool_calls={sub_result.tool_calls_made}, "
                f"answer={len(sub_result.answer)} chars, "
                f"sources={len(sub_result.sources)}",
            )
            logger.log(
                VERBOSE2,
                f"[AGENT:{self.config.name}] Answer:\n{truncate_for_log(sub_result.answer, 1000)}",
            )

            # Emit subtask_completed event
            if event_bus:
                event_bus.emit(Event(
                    type=EventType.SUBTASK_COMPLETED,
                    subtask_id=subtask_id,
                    agent_name=self.config.name,
                    data={
                        "answer": sub_result.answer[:500] if sub_result.answer else "",
                        "sources": sub_result.sources,
                        "tokens_used": sub_result.tokens_used,
                        "tool_calls_made": sub_result.tool_calls_made,
                    },
                ))
            return sub_result

        except Exception as e:
            logger.error(
                f"[AGENT:{self.config.name}] Failed: {e}",
                exc_info=True,
            )
            err_result = SubAgentResult(
                agent_name=self.config.name,
                task=task,
                answer="",
                success=False,
                error=str(e),
            )
            if event_bus:
                event_bus.emit(Event(
                    type=EventType.SUBTASK_FAILED,
                    subtask_id=subtask_id,
                    agent_name=self.config.name,
                    data={"error": str(e)},
                ))
            return err_result


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
