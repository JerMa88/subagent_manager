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

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from subagent_manager.events import Event, EventBus, EventType
from subagent_manager.logging_config import (
    format_tokens,
    truncate_for_log,
    VERBOSE1,
    VERBOSE2,
)
from subagent_manager.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Providers where LiteLLM's native tool support works correctly.
# For all others, we use prompt-based tool calling to avoid
# LiteLLM injecting broken JSON mode / format directives.
_NATIVE_TOOL_PROVIDERS = frozenset({
    "openai", "anthropic", "gemini", "azure", "cohere",
    "mistral", "groq", "together_ai", "deepseek",
})


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

    # Maximum total chars across all conversation messages before pruning.
    # Keeps the prompt-based tool loop inside small-model context windows.
    # Override per-instance if needed: client.MAX_HISTORY_CHARS = 20_000
    MAX_HISTORY_CHARS: int = 10_000

    @property
    def _supports_native_tools(self) -> bool:
        """Check if this model's provider supports LiteLLM native tool calling."""
        provider = self.model.split("/")[0] if "/" in self.model else "openai"
        return provider in _NATIVE_TOOL_PROVIDERS

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

        # For local/Ollama models (especially thinking models like Gemma 4),
        # the thinking tokens count against max_tokens. Boost the budget
        # so there's room for both thinking and the actual response.
        if not self._supports_native_tools:
            kwargs["max_tokens"] = kwargs["max_tokens"] * 4

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # Only pass tools to LiteLLM for providers with working native support.
        use_native_tools = tools and self._supports_native_tools
        if use_native_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        total_msg_chars = sum(len(m.get("content", "") or "") for m in messages)
        logger.log(
            VERBOSE1,
            f"[LLM] → Call: model={self.model}, msgs={len(messages)}, "
            f"chars={total_msg_chars:,}, temp={kwargs['temperature']}, "
            f"max_tokens={kwargs['max_tokens']}, "
            f"native_tools={'yes' if use_native_tools else 'no'}"
            f"{f' ({len(tools)} tool schemas)' if use_native_tools else ''}",
        )
        logger.log(VERBOSE2, f"[LLM] Messages payload:")
        for i, msg in enumerate(messages):
            content_preview = truncate_for_log(msg.get('content', '') or '', 300)
            logger.log(VERBOSE2, f"[LLM]   [{i}] role={msg['role']}: {content_preview}")

        # Retry logic: thinking models (Gemma 4, etc.) can exhaust max_tokens
        # on internal <think> tokens, returning empty content. LiteLLM throws
        # "model output error" in this case. We catch it and retry with a
        # larger token budget.
        max_attempts = 3
        response = None
        call_t0 = time.monotonic()
        for attempt in range(max_attempts):
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                is_empty_output = (
                    "model output" in error_msg
                    or "cannot both be empty" in error_msg
                )
                if is_empty_output and attempt < max_attempts - 1:
                    old_max = kwargs["max_tokens"]
                    kwargs["max_tokens"] = old_max * 2
                    logger.warning(
                        f"[LLM] Empty model output (thinking model likely exhausted "
                        f"token budget). Retrying with max_tokens="
                        f"{kwargs['max_tokens']} (was {old_max}), "
                        f"attempt {attempt + 2}/{max_attempts}"
                    )
                    await asyncio.sleep(0.5)
                    continue
                logger.error(f"[LLM] Completion failed: {e}")
                raise

            content_text = response.choices[0].message.content or ""
            has_tool_calls = (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            )

            if content_text.strip() or has_tool_calls:
                break

            # Got a response but content is empty (thinking ate all tokens)
            if attempt < max_attempts - 1:
                old_max = kwargs["max_tokens"]
                kwargs["max_tokens"] = old_max * 2
                logger.warning(
                    f"[LLM] Empty response with "
                    f"{response.usage.completion_tokens if response.usage else '?'} "
                    f"completion tokens. Retrying with max_tokens="
                    f"{kwargs['max_tokens']} (was {old_max})"
                )
                await asyncio.sleep(0.5)

        call_duration = time.monotonic() - call_t0

        choice = response.choices[0]
        message = choice.message

        # Extract tool calls (native mode only)
        tool_calls_data = []
        if use_native_tools and hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        content = message.content or ""

        # Extract usage
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        result = CompletionResult(
            content=content,
            tool_calls=tool_calls_data,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

        logger.log(
            VERBOSE1,
            f"[LLM] ← Response: {format_tokens(usage)}, "
            f"finish={result.finish_reason}, "
            f"content={len(content)} chars, "
            f"tool_calls={len(tool_calls_data)}, "
            f"latency={call_duration:.2f}s",
        )
        if tool_calls_data:
            for tc in tool_calls_data:
                logger.log(
                    VERBOSE1,
                    f"[LLM]   tool_call: {tc['function']['name']}"
                    f"({truncate_for_log(tc['function']['arguments'], 200)})",
                )
        logger.log(VERBOSE2, f"[LLM] Response content: {truncate_for_log(content, 1000)}")

        return result

    async def complete_with_tool_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[BaseTool],
        max_iterations: int = 5,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_history_chars: int | None = None,
        mandatory_tool_call: bool = False,
        event_bus: EventBus | None = None,
        subtask_id: int | None = None,
        agent_name: str | None = None,
        pause_event: asyncio.Event | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ToolLoopResult:
        """
        Run LLM with iterative tool execution until a final answer.

        The model can call tools, receive results, and call more tools
        up to `max_iterations` rounds. This hard cap prevents runaway
        reasoning chains — a core part of the short-horizon constraint.

        Supports two modes:
        - **Native**: For providers like OpenAI/Anthropic/Gemini that support
          structured tool calling via the API.
        - **Prompt-based**: For Ollama and other providers where native tool
          support is broken. Tool descriptions are embedded in the prompt,
          and tool calls are parsed from the model's text output.

        Args:
            messages: Initial messages (system + user).
            tools: List of BaseTool instances available to the model.
            max_iterations: Hard cap on tool call rounds.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            max_history_chars: Per-call override for sliding-window pruning threshold.
                If None, uses the class-level MAX_HISTORY_CHARS (default 10,000).
                Set higher for agents that hold large file contents in context.
            mandatory_tool_call: If True, the agent MUST call a tool on each iteration
                until it has exhausted its budget. Removes the 'respond directly' escape
                hatch from the tool instructions. Use for patch_writer / reproducer.
            event_bus: Optional event bus for GUI streaming.
            subtask_id: Subtask ID for event tagging.
            agent_name: Agent name for event tagging.
            pause_event: If set, the loop will wait at each checkpoint
                when this event is *cleared* (paused state = event not set).
            cancel_event: If set, the loop will abort when this event is set.

        Returns:
            ToolLoopResult with the final answer and metadata.
        """
        if self._supports_native_tools:
            return await self._tool_loop_native(
                messages, tools, max_iterations, temperature, max_tokens,
                event_bus=event_bus, subtask_id=subtask_id, agent_name=agent_name,
                pause_event=pause_event, cancel_event=cancel_event,
            )
        else:
            return await self._tool_loop_prompt_based(
                messages, tools, max_iterations, temperature, max_tokens,
                max_history_chars=max_history_chars,
                mandatory_tool_call=mandatory_tool_call,
                event_bus=event_bus, subtask_id=subtask_id, agent_name=agent_name,
                pause_event=pause_event, cancel_event=cancel_event,
            )

    async def _tool_loop_native(
        self,
        messages: list[dict[str, Any]],
        tools: list[BaseTool],
        max_iterations: int,
        temperature: float | None,
        max_tokens: int | None,
        event_bus: EventBus | None = None,
        subtask_id: int | None = None,
        agent_name: str | None = None,
        pause_event: asyncio.Event | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ToolLoopResult:
        logger.log(
            VERBOSE1,
            f"[LLM] Native tool loop: {len(tools)} tools, max_iter={max_iterations}",
        )
        tool_schemas = [t.to_openai_schema() for t in tools]
        tool_map = {t.name: t for t in tools}

        conversation = list(messages)
        total_tool_calls = 0
        total_tokens = 0
        sources: list[str] = []

        for iteration in range(max_iterations):
            logger.log(
                VERBOSE1,
                f"[LLM] Native loop iter {iteration + 1}/{max_iterations} "
                f"(cumulative: {total_tool_calls} tool calls, {total_tokens:,} tokens)",
            )

            # --- Checkpoint: pause/cancel between iterations ---
            if cancel_event and cancel_event.is_set():
                logger.info(f"Native tool loop cancelled at iteration {iteration + 1}")
                raise asyncio.CancelledError("Agent cancelled by user")
            if pause_event:
                # pause_event is SET when running, CLEARED when paused
                if not pause_event.is_set():
                    if event_bus:
                        event_bus.emit(Event(
                            type=EventType.SUBTASK_PAUSED,
                            subtask_id=subtask_id, agent_name=agent_name,
                            data={"iteration": iteration},
                        ))
                    logger.info(f"Native tool loop paused at iteration {iteration + 1}")
                    await pause_event.wait()  # blocks until user resumes
                    if event_bus:
                        event_bus.emit(Event(
                            type=EventType.SUBTASK_RESUMED,
                            subtask_id=subtask_id, agent_name=agent_name,
                            data={"iteration": iteration},
                        ))

            t_start = time.monotonic()
            result = await self.complete(
                messages=conversation,
                tools=tool_schemas,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration_ms = int((time.monotonic() - t_start) * 1000)

            total_tokens += result.usage.get("total_tokens", 0)

            if event_bus:
                event_bus.emit(Event(
                    type=EventType.LLM_CALL_COMPLETED,
                    subtask_id=subtask_id, agent_name=agent_name,
                    data={
                        "iteration": iteration,
                        "tokens": getattr(result.usage, "total_tokens", 0) or 0,
                        "prompt_tokens": getattr(result.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(result.usage, "completion_tokens", 0) or 0,
                        "finish_reason": result.finish_reason,
                        "duration_ms": duration_ms,
                        "has_tool_calls": bool(result.tool_calls),
                        "message_content": result.content,
                    },
                ))

            # If no tool calls, we have a final answer
            if not result.tool_calls:
                logger.log(
                    VERBOSE1,
                    f"[LLM] Native loop: final answer at iter {iteration + 1} "
                    f"({len(result.content)} chars, {total_tokens:,} total tokens)",
                )
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

                if event_bus:
                    event_bus.emit(Event(
                        type=EventType.TOOL_CALL_STARTED,
                        subtask_id=subtask_id, agent_name=agent_name,
                        data={"tool_name": func_name, "arguments": func_args_raw},
                    ))

                t_tool_start = time.monotonic()
                tool_result = await self._execute_tool(
                    func_name, func_args_raw, tool_map, sources
                )
                tool_duration_ms = int((time.monotonic() - t_tool_start) * 1000)

                if event_bus:
                    event_bus.emit(Event(
                        type=EventType.TOOL_CALL_COMPLETED,
                        subtask_id=subtask_id, agent_name=agent_name,
                        data={
                            "tool_name": func_name,
                            "result": tool_result[:500] if len(tool_result) > 500 else tool_result,
                            "duration_ms": tool_duration_ms,
                        },
                    ))

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        # Exhausted iterations — force a final answer
        logger.warning(
            f"[LLM] Native tool loop exhausted {max_iterations} iterations. "
            f"Forcing final answer. ({total_tool_calls} tool calls, {total_tokens:,} tokens)"
        )
        conversation.append({
            "role": "user",
            "content": (
                "You have used all available tool calls. Please provide your "
                "final answer based on the information you have gathered so far."
            ),
        })

        final_result = await self.complete(
            messages=conversation,
            tools=None,
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

    async def _tool_loop_prompt_based(
        self,
        messages: list[dict[str, Any]],
        tools: list[BaseTool],
        max_iterations: int,
        temperature: float | None,
        max_tokens: int | None,
        max_history_chars: int | None = None,
        mandatory_tool_call: bool = False,
        event_bus: EventBus | None = None,
        subtask_id: int | None = None,
        agent_name: str | None = None,
        pause_event: asyncio.Event | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ToolLoopResult:
        """
        Tool loop for models without native function calling (e.g., Ollama).

        Instead of passing tools to LiteLLM (which injects broken JSON mode),
        we embed tool descriptions in the system prompt and parse tool calls
        from the model's text output.
        """
        logger.log(
            VERBOSE1,
            f"[LLM] Prompt-based tool loop: {len(tools)} tools, max_iter={max_iterations}",
        )
        tool_map = {t.name: t for t in tools}

        # Build tool description block for the prompt
        tool_desc_parts = []
        for t in tools:
            schema = t.to_openai_schema()
            func = schema["function"]
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            param_lines = []
            for pname, pinfo in params.items():
                req_mark = " (required)" if pname in required else " (optional)"
                param_lines.append(
                    f"    - {pname}{req_mark}: {pinfo.get('description', '')}"
                )

            tool_desc_parts.append(
                f"- **{func['name']}**: {func.get('description', '')}\n"
                f"  Parameters:\n" + "\n".join(param_lines)
            )

        if mandatory_tool_call:
            tool_instructions = (
                "\n\n## AVAILABLE TOOLS\n\n"
                "You MUST call one of these tools. Do NOT respond with plain text. "
                "Respond ONLY with a JSON block:\n\n"
                '```json\n{"name": "tool_name", "arguments": {"param": "value"}}\n```\n\n'
                "After the tool result is returned, you may call another tool or "
                "give your final summary.\n\n"
                + "\n".join(tool_desc_parts)
            )
        else:
            tool_instructions = (
                "\n\n## AVAILABLE TOOLS\n\n"
                "You have access to these tools. To call a tool, respond with ONLY "
                "a JSON block like this:\n\n"
                '```json\n{"name": "tool_name", "arguments": {"param": "value"}}\n```\n\n'
                "After calling a tool, you will receive the result and can then "
                "answer the question. If you do NOT need a tool, respond with your "
                "answer directly as plain text (no JSON).\n\n"
                + "\n".join(tool_desc_parts)
            )

        logger.log(VERBOSE2, f"[LLM] Tool instruction block ({len(tool_instructions)} chars):\n{tool_instructions}")

        # Inject tool descriptions into the system prompt
        conversation = list(messages)
        if conversation and conversation[0]["role"] == "system":
            conversation[0] = {
                "role": "system",
                "content": conversation[0]["content"] + tool_instructions,
            }
        else:
            conversation.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant." + tool_instructions,
            })

        total_tool_calls = 0
        total_tokens = 0
        sources: list[str] = []

        for iteration in range(max_iterations):
            logger.log(
                VERBOSE1,
                f"[LLM] Prompt-based loop iter {iteration + 1}/{max_iterations} "
                f"(cumulative: {total_tool_calls} tool calls, {total_tokens:,} tokens)",
            )

            # --- Checkpoint: pause/cancel between iterations ---
            if cancel_event and cancel_event.is_set():
                logger.info(f"Prompt-based tool loop cancelled at iteration {iteration + 1}")
                raise asyncio.CancelledError("Agent cancelled by user")
            if pause_event:
                if not pause_event.is_set():
                    if event_bus:
                        event_bus.emit(Event(
                            type=EventType.SUBTASK_PAUSED,
                            subtask_id=subtask_id, agent_name=agent_name,
                            data={"iteration": iteration},
                        ))
                    logger.info(f"Prompt-based tool loop paused at iteration {iteration + 1}")
                    await pause_event.wait()
                    if event_bus:
                        event_bus.emit(Event(
                            type=EventType.SUBTASK_RESUMED,
                            subtask_id=subtask_id, agent_name=agent_name,
                            data={"iteration": iteration},
                        ))

            # --- 80% budget warning ---
            if iteration > 0 and iteration >= int(max_iterations * 0.8):
                logger.warning(
                    f"[LLM] Agent '{agent_name}' at {iteration + 1}/{max_iterations} iterations "
                    f"({100 * (iteration + 1) // max_iterations}% of budget consumed). "
                    "Consider increasing max_tool_iterations if the task is complex."
                )

            # --- Sliding-window context pruning ---
            # Prevent context explosion: if conversation history exceeds
            # the threshold, keep the system prompt + a summary marker +
            # the last 6 messages (3 tool-call + result pairs).
            history_limit = max_history_chars if max_history_chars is not None else self.MAX_HISTORY_CHARS
            total_chars = sum(len(m.get("content", "")) for m in conversation)
            if total_chars > history_limit and len(conversation) > 6:
                pruned = conversation[1:-6]
                pruned_summary = (
                    f"[Context pruned: {len(pruned)} older messages summarized to save "
                    "space. Key findings from prior tool calls are in the most recent messages.]"
                )
                conversation = (
                    [conversation[0], {"role": "user", "content": pruned_summary}]
                    + conversation[-6:]
                )
                new_chars = sum(len(m.get("content", "")) for m in conversation)
                logger.log(
                    VERBOSE1,
                    f"[LLM] Context pruned: {total_chars:,} \u2192 {new_chars:,} chars "
                    f"({len(pruned)} messages summarized)",
                )

            # No tools passed to LiteLLM — we handle it ourselves
            t_start = time.monotonic()
            result = await self.complete(
                messages=conversation,
                tools=None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration_ms = int((time.monotonic() - t_start) * 1000)

            total_tokens += result.usage.get("total_tokens", 0)
            content = result.content.strip()

            logger.log(
                VERBOSE2,
                f"[LLM] Prompt-based loop raw content "
                f"(len={len(content)}): {truncate_for_log(content, 500)}",
            )

            if event_bus:
                event_bus.emit(Event(
                    type=EventType.LLM_CALL_COMPLETED,
                    subtask_id=subtask_id, agent_name=agent_name,
                    data={
                        "iteration": iteration,
                        "tokens": getattr(result.usage, "total_tokens", 0) or 0,
                        "prompt_tokens": getattr(result.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(result.usage, "completion_tokens", 0) or 0,
                        "finish_reason": result.finish_reason,
                        "duration_ms": duration_ms,
                        "message_content": content,
                    },
                ))

            # Try to parse a tool call from the response
            tool_schemas = [t.to_openai_schema() for t in tools]
            parsed_calls = self._parse_tool_calls_from_content(
                content, tool_schemas
            )

            logger.log(
                VERBOSE2,
                f"[PARSE] Parsed {len(parsed_calls)} tool call(s) from model content",
            )

            if not parsed_calls:
                if mandatory_tool_call and iteration < max_iterations - 1:
                    # Re-prompt: the model gave a text answer when it must call a tool
                    logger.log(
                        VERBOSE1,
                        f"[LLM] Agent '{agent_name}': text response when tool call required. "
                        f"Re-prompting (iter {iteration + 1}/{max_iterations}).",
                    )
                    conversation.append({"role": "assistant", "content": content})
                    # Trim context before re-prompt: keep system + last 2 exchanges.
                    # This prevents the 4k-token wall where the model can only output
                    # 24 completion tokens (not enough for a JSON tool call).
                    if len(conversation) > 5:
                        system_msg = conversation[0]
                        recent = conversation[-4:]  # last 2 user+assistant pairs
                        conversation = [system_msg] + recent
                        logger.log(
                            VERBOSE1,
                            f"[LLM] Context trimmed for re-prompt: {len(conversation)} msgs kept",
                        )
                    # Build a concrete re-prompt using a generic example so the model doesn't
                    # blindly copy placeholder arguments for a real tool.
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You must call a tool. Respond ONLY with a JSON block, for example:\n"
                            '```json\n{"name": "some_tool", "arguments": {"arg1": "value"}}\n```\n'
                            "Fill in the correct tool name and arguments for your task."
                        ),
                    })
                    continue
                # No tool call — this is the final answer
                logger.log(
                    VERBOSE1,
                    f"[LLM] Prompt-based loop: final answer at iter {iteration + 1} "
                    f"({len(content)} chars, {total_tokens:,} total tokens)",
                )
                return ToolLoopResult(
                    final_answer=content,
                    tool_calls_made=total_tool_calls,
                    sources=sources,
                    total_tokens=total_tokens,
                )

            # Execute the tool call(s)
            logger.log(
                VERBOSE1,
                f"[LLM] Parsed {len(parsed_calls)} tool call(s) from model output "
                f"(prompt-based): {[tc['function']['name'] for tc in parsed_calls]}",
            )

            tool_results_text: list[str] = []
            for tc in parsed_calls:
                func_name = tc["function"]["name"]
                func_args_raw = tc["function"]["arguments"]
                total_tool_calls += 1

                if event_bus:
                    event_bus.emit(Event(
                        type=EventType.TOOL_CALL_STARTED,
                        subtask_id=subtask_id, agent_name=agent_name,
                        data={"tool_name": func_name, "arguments": func_args_raw},
                    ))

                t_tool_start = time.monotonic()
                tool_result = await self._execute_tool(
                    func_name, func_args_raw, tool_map, sources
                )
                tool_duration_ms = int((time.monotonic() - t_tool_start) * 1000)

                if event_bus:
                    event_bus.emit(Event(
                        type=EventType.TOOL_CALL_COMPLETED,
                        subtask_id=subtask_id, agent_name=agent_name,
                        data={
                            "tool_name": func_name,
                            "result": tool_result[:500] if len(tool_result) > 500 else tool_result,
                            "duration_ms": tool_duration_ms,
                        },
                    ))

                tool_results_text.append(
                    f"[Tool: {func_name}] Result:\n{tool_result}"
                )

            # Feed tool results back as a user message
            conversation.append({
                "role": "assistant",
                "content": content,
            })
            conversation.append({
                "role": "user",
                "content": (
                    "Here are the tool results:\n\n"
                    + "\n\n".join(tool_results_text)
                    + "\n\nContinue with your task. You may call another tool "
                    "if needed, or provide your final answer as plain text "
                    "(no JSON) if you are done."
                ),
            })

        # Exhausted iterations — force a final answer
        logger.warning(
            f"[LLM] Prompt-based tool loop exhausted {max_iterations} iterations. "
            f"Forcing final answer. ({total_tool_calls} tool calls, {total_tokens:,} tokens)"
        )
        conversation.append({
            "role": "user",
            "content": (
                "You have used all available tool calls. Please provide your "
                "final answer as plain text based on the information gathered."
            ),
        })

        final_result = await self.complete(
            messages=conversation,
            tools=None,
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

    async def _execute_tool(
        self,
        func_name: str,
        func_args_raw: str,
        tool_map: dict[str, BaseTool],
        sources: list[str],
    ) -> str:
        """Execute a single tool call and extract sources."""
        tool = tool_map.get(func_name)
        if tool is None:
            logger.warning(f"[TOOL:{func_name}] Unknown tool requested")
            return f"Error: Unknown tool '{func_name}'"

        try:
            parsed_args = tool.parse_arguments(func_args_raw)
            logger.log(
                VERBOSE1,
                f"[TOOL:{func_name}] Executing with args: {truncate_for_log(str(parsed_args), 300)}",
            )
            t0 = time.monotonic()
            tool_result = await tool.safe_execute(**parsed_args)
            duration = time.monotonic() - t0

            logger.log(
                VERBOSE1,
                f"[TOOL:{func_name}] Completed in {duration:.2f}s, "
                f"result={len(tool_result)} chars",
            )
            logger.log(
                VERBOSE2,
                f"[TOOL:{func_name}] Result: {truncate_for_log(tool_result, 500)}",
            )

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

            return tool_result

        except Exception as e:
            logger.warning(f"[TOOL:{func_name}] Execution error: {e}", exc_info=True)
            return f"Error executing {func_name}: {str(e)}"

    @staticmethod
    def _parse_tool_calls_from_content(
        content: str,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Parse tool calls from model text output.

        Used for models that don't support native function calling.
        Detects JSON tool call patterns in the content like:
        {"name": "web_search", "arguments": {"query": "..."}}
        """
        # Collect valid tool names for validation
        valid_names = set()
        for t in tools:
            if isinstance(t, dict) and "function" in t:
                valid_names.add(t["function"]["name"])

        if not valid_names:
            return []

        candidates = []

        # Strategy 1: Content is pure JSON
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                candidates.append(data)
            elif isinstance(data, list):
                candidates.extend(data)
        except json.JSONDecodeError:
            pass

        # Strategy 2: JSON inside a code block
        if not candidates:
            blocks = re.findall(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", content)
            for block in blocks:
                try:
                    data = json.loads(block.strip())
                    if isinstance(data, dict):
                        candidates.append(data)
                    elif isinstance(data, list):
                        candidates.extend(data)
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Find JSON objects embedded in text
        if not candidates:
            for match in re.finditer(r"\{[^{}]*\}", content):
                try:
                    data = json.loads(match.group())
                    if isinstance(data, dict):
                        candidates.append(data)
                except json.JSONDecodeError:
                    continue

        # Validate and convert candidates to tool call format
        tool_calls = []
        for obj in candidates:
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})

            if name in valid_names:
                call_id = f"fallback_{uuid.uuid4().hex[:8]}"
                tool_calls.append({
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": (
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    },
                })

        return tool_calls
