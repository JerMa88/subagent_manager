"""Tests for the SubAgent."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from subagent_manager.subagent import SubAgent, SubAgentConfig, SubAgentResult
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
        return f"Result: {kwargs.get('input', '')}"


class TestSubAgentConfig:
    """Tests for SubAgentConfig."""

    def test_defaults(self):
        config = SubAgentConfig(
            name="test_agent",
            description="A test agent",
        )
        assert config.name == "test_agent"
        assert config.description == "A test agent"
        assert config.tools == []
        assert config.model is None
        assert config.system_prompt is None
        assert config.max_tool_iterations == 5
        assert config.max_answer_tokens == 512
        assert config.temperature == 0.0

    def test_custom_config(self):
        tool = MockTool()
        config = SubAgentConfig(
            name="custom",
            description="Custom agent",
            tools=[tool],
            model="gpt-4o-mini",
            max_tool_iterations=3,
            max_answer_tokens=256,
            temperature=0.5,
        )
        assert len(config.tools) == 1
        assert config.model == "gpt-4o-mini"
        assert config.max_tool_iterations == 3


class TestSubAgentResult:
    """Tests for SubAgentResult."""

    def test_success_result(self):
        result = SubAgentResult(
            agent_name="researcher",
            task="Find information about X",
            answer="X is a thing",
            sources=["https://example.com"],
            tool_calls_made=2,
        )
        assert result.success is True
        assert result.error is None
        assert len(result.sources) == 1

    def test_failure_result(self):
        result = SubAgentResult(
            agent_name="researcher",
            task="Find information",
            answer="",
            success=False,
            error="API timeout",
        )
        assert result.success is False
        assert result.error == "API timeout"


class TestSubAgent:
    """Tests for SubAgent."""

    def test_init_without_model_override(self):
        config = SubAgentConfig(name="test", description="Test")
        client = LLMClient(model="base-model")
        agent = SubAgent(config=config, llm_client=client)
        assert agent.llm_client.model == "base-model"

    def test_init_with_model_override(self):
        config = SubAgentConfig(
            name="test", description="Test", model="override-model"
        )
        client = LLMClient(model="base-model")
        agent = SubAgent(config=config, llm_client=client)
        assert agent.llm_client.model == "override-model"

    def test_repr(self):
        tool = MockTool()
        config = SubAgentConfig(
            name="researcher", description="Research agent", tools=[tool]
        )
        client = LLMClient(model="test-model")
        agent = SubAgent(config=config, llm_client=client)
        repr_str = repr(agent)
        assert "researcher" in repr_str
        assert "mock_tool" in repr_str

    def test_extract_sources(self):
        sources = SubAgent._extract_sources(
            "Check https://example.com and https://test.org for details"
        )
        assert "https://example.com" in sources
        assert "https://test.org" in sources

    def test_extract_sources_no_urls(self):
        sources = SubAgent._extract_sources("No URLs here")
        assert sources == []
