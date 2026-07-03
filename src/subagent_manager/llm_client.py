"""
Universal LLM client wrapping LiteLLM.

Provides a stateless, provider-agnostic interface for LLM calls
with tool execution loops. Each call gets a fresh context — this
is the foundation of the short-horizon reasoning constraint.

Supports 100+ providers via LiteLLM:
- OpenAI: model="gpt-4o-mini"
- Anthropic: model="anthropic/claude-sonnet-4"
- Google: model="gemini/gemini-2.5-flash"
- Ollama (local): model="ollama/qwen3"
- Any OpenAI-compatible API: model="openai/my-model" + api_base
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from subagent_manager.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result from a single LLM completion call."""

    content: str
    """The text content of the response."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Any tool calls requested by the model."""

    finish_reason: str = "stop"
    """Why the model stopped generating."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage statistics."""


@dataclass
class ToolLoopResult:
    """Result from a multi-turn tool execution loop."""

    final_answer: str
    """The model's final text response after all tool calls."""

    tool_calls_made: int = 0
    """Total number of tool calls across all iterations."""

    sources: list[str] = field(default_factory=list)
    """URLs and references discovered during tool use."""

    total_tokens: int = 0
    """Total tokens consumed across all iterations."""


class LLMClient:
    """
    Stateless LLM client. Each call gets a fresh context.

    This is intentionally stateless — no conversation history accumulation.
    Every call to `complete()` or `complete_with_tool_loop()` is independent,
    which enforces the short-horizon reasoning constraint.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model: Model identifier in LiteLLM format.
                Examples: "gpt-4o-mini", "ollama/qwen3", "gemini/gemini-2.5-flash"
            api_key: API key (optional, can use env vars).
            api_base: Custom API base URL (for self-hosted / Ollama).
            default_temperature: Default temperature for completions.
            default_max_tokens: Default max tokens for completions.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """
        Make a single LLM completion call.

        This is stateless — messages must contain the full conversation context.

        Args:
            messages: List of message dicts (role/content).
            tools: Optional list of tool schemas in OpenAI format.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            CompletionResult with the model's response.
        """
        try:
            import litellm

            # Suppress litellm's verbose logging
            litellm.suppress_debug_info = True
        except ImportError:
            raise ImportError(
                "litellm is required. Install it with: pip install litellm"
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls_data = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        # Extract usage
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return CompletionResult(
            content=message.content or "",
            tool_calls=tool_calls_data,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    async def complete_with_tool_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[BaseTool],
        max_iterations: int = 5,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ToolLoopResult:
        """
        Run LLM with iterative tool execution until a final answer.

        The model can call tools, receive results, and call more tools
        up to `max_iterations` rounds. This hard cap prevents runaway
        reasoning chains — a core part of the short-horizon constraint.

        Args:
            messages: Initial messages (system + user).
            tools: List of BaseTool instances available to the model.
            max_iterations: Hard cap on tool call rounds.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            ToolLoopResult with the final answer and metadata.
        """
        # Build tool schemas
        tool_schemas = [t.to_openai_schema() for t in tools]
        tool_map = {t.name: t for t in tools}

        # Working copy of messages
        conversation = list(messages)
        total_tool_calls = 0
        total_tokens = 0
        sources: list[str] = []

        for iteration in range(max_iterations):
            logger.debug(f"Tool loop iteration {iteration + 1}/{max_iterations}")

            result = await self.complete(
                messages=conversation,
                tools=tool_schemas if tool_schemas else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            total_tokens += result.usage.get("total_tokens", 0)

            # If no tool calls, we have a final answer
            if not result.tool_calls:
                return ToolLoopResult(
                    final_answer=result.content,
                    tool_calls_made=total_tool_calls,
                    sources=sources,
                    total_tokens=total_tokens,
                )

            # Add assistant message with tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in result.tool_calls
                ],
            }
            conversation.append(assistant_msg)

            # Execute each tool call
            for tc in result.tool_calls:
                func_name = tc["function"]["name"]
                func_args_raw = tc["function"]["arguments"]
                total_tool_calls += 1

                tool = tool_map.get(func_name)
                if tool is None:
                    tool_result = f"Error: Unknown tool '{func_name}'"
                else:
                    try:
                        parsed_args = tool.parse_arguments(func_args_raw)
                        tool_result = await tool.safe_execute(**parsed_args)

                        # Extract URLs from search results as sources
                        if func_name == "web_search" and "URL:" in tool_result:
                            for line in tool_result.split("\n"):
                                if line.strip().startswith("URL:"):
                                    url = line.strip().replace("URL: ", "").strip()
                                    if url and url not in sources:
                                        sources.append(url)
                        elif func_name == "read_url":
                            url_arg = parsed_args.get("url", "")
                            if url_arg and url_arg not in sources:
                                sources.append(url_arg)

                    except Exception as e:
                        tool_result = f"Error executing {func_name}: {str(e)}"
                        logger.warning(f"Tool execution error: {e}")

                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        # Exhausted iterations — ask for final answer
        conversation.append({
            "role": "user",
            "content": (
                "You have used all available tool calls. Please provide your "
                "final answer based on the information you have gathered so far."
            ),
        })

        final_result = await self.complete(
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_tokens += final_result.usage.get("total_tokens", 0)

        return ToolLoopResult(
            final_answer=final_result.content,
            tool_calls_made=total_tool_calls,
            sources=sources,
            total_tokens=total_tokens,
        )
