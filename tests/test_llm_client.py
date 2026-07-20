"""Tests for the LLM client."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from subagent_manager.llm_client import LLMClient, CompletionResult, ToolLoopResult
from subagent_manager.tools.base import BaseTool, ToolParameter


class MockTool(BaseTool):
    """A simple mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"
    parameters = [
        ToolParameter(name="input", type="string", description="Test input"),
    ]

    async def execute(self, **kwargs):
        return f"Mock result for: {kwargs.get('input', '')}"


class TestLLMClient:
    """Tests for the LLMClient."""

    def test_init_defaults(self):
        client = LLMClient(model="test-model")
        assert client.model == "test-model"
        assert client.api_key is None
        assert client.api_base is None
        assert client.default_temperature == 0.0
        assert client.default_max_tokens == 1024

    def test_init_with_overrides(self):
        client = LLMClient(
            model="custom-model",
            api_key="test-key",
            api_base="http://localhost:11434",
            default_temperature=0.5,
            default_max_tokens=2048,
        )
        assert client.model == "custom-model"
        assert client.api_key == "test-key"
        assert client.api_base == "http://localhost:11434"
        assert client.default_temperature == 0.5
        assert client.default_max_tokens == 2048


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""

    def test_defaults(self):
        result = CompletionResult(content="hello")
        assert result.content == "hello"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage == {}


class TestToolLoopResult:
    """Tests for ToolLoopResult dataclass."""

    def test_defaults(self):
        result = ToolLoopResult(final_answer="done")
        assert result.final_answer == "done"
        assert result.tool_calls_made == 0
        assert result.sources == []
        assert result.total_tokens == 0


class TestMockTool:
    """Tests verifying the mock tool works."""

    def test_schema(self):
        tool = MockTool()
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "mock_tool"

    @pytest.mark.asyncio
    async def test_execute(self):
        tool = MockTool()
        result = await tool.execute(input="hello")
        assert result == "Mock result for: hello"


class TestContextPruning:
    """
    Unit tests for the sliding-window context pruning logic in LLMClient.

    These tests verify the pruning invariants without actually calling an LLM.
    The pruning logic in run_tool_loop_prompt_based is:
        if total_chars > MAX_HISTORY_CHARS and len(conversation) > 6:
            pruned = conversation[1:-6]
            conversation = [sys_msg, summary_msg] + conversation[-6:]
    """

    def _make_message(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def _apply_pruning(self, conversation: list[dict], max_chars: int) -> list[dict]:
        """
        Replicate the pruning logic from run_tool_loop_prompt_based
        so we can test it independently.
        """
        total_chars = sum(len(m.get("content", "")) for m in conversation)
        if total_chars > max_chars and len(conversation) > 6:
            pruned = conversation[1:-6]
            pruned_summary = (
                f"[Context pruned: {len(pruned)} older messages summarized to save "
                "space. Key findings from prior tool calls are in the most recent messages.]"
            )
            conversation = (
                [conversation[0], {"role": "user", "content": pruned_summary}]
                + conversation[-6:]
            )
        return conversation

    def test_max_history_chars_attribute_exists(self):
        """LLMClient must expose MAX_HISTORY_CHARS as a class attribute."""
        client = LLMClient(model="test-model")
        assert hasattr(client, "MAX_HISTORY_CHARS")
        assert isinstance(client.MAX_HISTORY_CHARS, int)
        assert client.MAX_HISTORY_CHARS > 0

    def test_pruning_fires_when_over_threshold(self):
        """When total chars > MAX_HISTORY_CHARS and len > 6, pruning activates."""
        # Build a conversation with a large middle section
        sys_msg = self._make_message("system", "system prompt " + "x" * 100)
        # 10 big middle messages (will be pruned)
        middle = [self._make_message("user" if i % 2 == 0 else "assistant", "y" * 1500)
                  for i in range(10)]
        # 6 recent messages (must be kept)
        recent = [self._make_message("user" if i % 2 == 0 else "assistant", f"recent{i}")
                  for i in range(6)]
        conversation = [sys_msg] + middle + recent

        total = sum(len(m["content"]) for m in conversation)
        pruned = self._apply_pruning(conversation, max_chars=100)  # tiny threshold

        # System message always at index 0
        assert pruned[0]["role"] == "system"
        assert pruned[0]["content"] == sys_msg["content"]

        # Summary marker at index 1
        assert "Context pruned" in pruned[1]["content"]

        # Last 6 (recent) messages preserved
        assert len(pruned) == 8  # sys + summary + 6 recent
        assert pruned[2]["content"] == "recent0"
        assert pruned[7]["content"] == "recent5"

    def test_pruning_does_not_fire_for_short_conversations(self):
        """Conversations under the threshold are left untouched."""
        conversation = [
            self._make_message("system", "sys"),
            self._make_message("user", "hi"),
            self._make_message("assistant", "hello"),
        ]
        result = self._apply_pruning(conversation, max_chars=10_000)
        assert result == conversation  # unchanged

    def test_pruning_does_not_fire_for_short_length(self):
        """Conversations with <= 6 messages are never pruned even if large."""
        big_content = "z" * 50_000
        conversation = [
            self._make_message("system", big_content),
            self._make_message("user", big_content),
            self._make_message("assistant", big_content),
        ]  # only 3 messages — < 6 guard
        result = self._apply_pruning(conversation, max_chars=1)
        assert result == conversation  # not pruned (len not > 6)

    def test_system_message_content_unchanged_after_pruning(self):
        """System message content is NEVER modified or truncated."""
        sys_content = "You are a specialized SWE-bench agent. " * 20
        sys_msg = self._make_message("system", sys_content)
        filler = [self._make_message("user" if i % 2 == 0 else "assistant", "fill " * 500)
                  for i in range(14)]
        conversation = [sys_msg] + filler

        result = self._apply_pruning(conversation, max_chars=100)
        assert result[0]["content"] == sys_content

    def test_max_history_chars_is_overridable(self):
        """MAX_HISTORY_CHARS can be overridden per-instance without affecting the class."""
        client = LLMClient(model="test-model")
        original = LLMClient.MAX_HISTORY_CHARS
        client.MAX_HISTORY_CHARS = 500
        assert client.MAX_HISTORY_CHARS == 500
        # Class default unchanged
        assert LLMClient.MAX_HISTORY_CHARS == original

