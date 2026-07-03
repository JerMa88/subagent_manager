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
