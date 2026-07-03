"""Tests for the SubAgentManager."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from subagent_manager.manager import SubAgentManager, ManagerResult, DEFAULT_AGENTS
from subagent_manager.subagent import SubAgentConfig


class TestSubAgentManagerInit:
    """Tests for SubAgentManager initialization."""

    def test_default_init(self):
        manager = SubAgentManager()
        assert manager.model == "ollama/qwen3"
        assert len(manager.agents) == len(DEFAULT_AGENTS)
        assert "researcher" in manager.agents
        assert "analyzer" in manager.agents
        assert "coder" in manager.agents
        assert "verifier" in manager.agents

    def test_custom_model(self):
        manager = SubAgentManager(model="gpt-4o-mini")
        assert manager.model == "gpt-4o-mini"
        assert manager.llm_client.model == "gpt-4o-mini"

    def test_custom_agents(self):
        custom = [
            SubAgentConfig(name="my_agent", description="Custom agent"),
        ]
        manager = SubAgentManager(subagents=custom)
        assert len(manager.agents) == 1
        assert "my_agent" in manager.agents

    def test_orchestrator_model_override(self):
        manager = SubAgentManager(
            model="ollama/qwen3",
            orchestrator_model="gpt-4o",
        )
        assert manager.llm_client.model == "ollama/qwen3"
        assert manager.orchestrator_client.model == "gpt-4o"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            SubAgentManager(strategy="nonexistent")

    def test_valid_strategies(self):
        for strategy in ["parallel", "sequential", "adaptive"]:
            manager = SubAgentManager(strategy=strategy)
            assert manager is not None


class TestPlanParsing:
    """Tests for the JSON plan parser."""

    def test_parse_pure_json(self):
        manager = SubAgentManager()
        plan_json = json.dumps({
            "plan": [
                {"id": 1, "task": "Search for X", "agent": "researcher"},
                {"id": 2, "task": "Analyze results", "agent": "analyzer", "depends_on": [1]},
            ]
        })
        result = manager._parse_plan_json(plan_json)
        assert len(result) == 2
        assert result[0]["task"] == "Search for X"
        assert result[1]["depends_on"] == [1]

    def test_parse_json_in_code_block(self):
        manager = SubAgentManager()
        text = """Here is the plan:

```json
{
  "plan": [
    {"id": 1, "task": "Research topic", "agent": "researcher"}
  ]
}
```
"""
        result = manager._parse_plan_json(text)
        assert len(result) == 1
        assert result[0]["agent"] == "researcher"

    def test_parse_json_list(self):
        manager = SubAgentManager()
        plan_json = json.dumps([
            {"id": 1, "task": "Task 1", "agent": "researcher"},
        ])
        result = manager._parse_plan_json(plan_json)
        assert len(result) == 1

    def test_parse_invalid_json_returns_empty(self):
        manager = SubAgentManager()
        result = manager._parse_plan_json("This is not JSON at all")
        assert result == []

    def test_parse_json_embedded_in_text(self):
        manager = SubAgentManager()
        text = 'I will create a plan: {"plan": [{"id": 1, "task": "Do X", "agent": "researcher"}]}'
        result = manager._parse_plan_json(text)
        assert len(result) == 1


class TestManagerResult:
    """Tests for ManagerResult."""

    def test_defaults(self):
        result = ManagerResult(answer="Final answer")
        assert result.answer == "Final answer"
        assert result.subtask_results == []
        assert result.plan == []
        assert result.total_tokens == 0
        assert result.total_tool_calls == 0
        assert result.sources == []


class TestDefaultAgents:
    """Tests for the default agent configurations."""

    def test_researcher_has_web_tools(self):
        researcher = next(a for a in DEFAULT_AGENTS if a.name == "researcher")
        tool_names = [t.name for t in researcher.tools]
        assert "web_search" in tool_names
        assert "read_url" in tool_names

    def test_analyzer_has_no_tools(self):
        analyzer = next(a for a in DEFAULT_AGENTS if a.name == "analyzer")
        assert len(analyzer.tools) == 0

    def test_coder_has_exec_tools(self):
        coder = next(a for a in DEFAULT_AGENTS if a.name == "coder")
        tool_names = [t.name for t in coder.tools]
        assert "python_exec" in tool_names
        assert "read_file" in tool_names

    def test_verifier_has_web_tools(self):
        verifier = next(a for a in DEFAULT_AGENTS if a.name == "verifier")
        tool_names = [t.name for t in verifier.tools]
        assert "web_search" in tool_names
