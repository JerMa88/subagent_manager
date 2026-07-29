"""
tests/test_hierarchy.py — Unit tests for 3-Level Hierarchical Planning.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bench.swe_bench_hierarchy import HierarchicalResult, HierarchicalSWEBenchManager
from subagent_manager.domain_manager import DomainManager, DomainManagerConfig, DomainResult
from subagent_manager.subagent import SubAgentConfig, SubAgentResult


def test_domain_manager_init():
    worker = SubAgentConfig(name="worker1", description="Test worker")
    cfg = DomainManagerConfig(
        name="test_domain",
        domain_description="Testing domain",
        worker_agents=[worker],
    )
    dm = DomainManager(config=cfg, model="test-model")
    assert dm.config.name == "test_domain"
    assert len(dm.config.worker_agents) == 1


@pytest.mark.asyncio
async def test_domain_manager_run():
    worker = SubAgentConfig(name="worker1", description="Test worker")
    cfg = DomainManagerConfig(
        name="test_domain",
        domain_description="Testing domain",
        worker_agents=[worker],
    )
    dm = DomainManager(config=cfg, model="test-model")

    mock_res = MagicMock()
    mock_res.answer = "Domain complete"
    mock_res.subtask_results = [
        SubAgentResult(agent_name="worker1", task="t1", answer="a1", success=True)
    ]
    mock_res.total_tokens = 100
    mock_res.total_tool_calls = 2

    dm.manager.run = AsyncMock(return_value=mock_res)

    res = await dm.run_domain("Test goal")
    assert res.domain_name == "test_domain"
    assert res.summary == "Domain complete"
    assert res.success is True
    assert res.total_tokens == 100


@pytest.mark.asyncio
async def test_hierarchical_swe_bench_manager():
    with patch("bench.swe_bench_hierarchy.build_swe_bench_agents") as mock_build:
        mock_build.return_value = [
            SubAgentConfig(name="issue_analyzer", description="d1"),
            SubAgentConfig(name="code_explorer", description="d2"),
            SubAgentConfig(name="reproducer", description="d3"),
            SubAgentConfig(name="test_generator", description="d4"),
            SubAgentConfig(name="patch_writer", description="d5"),
            SubAgentConfig(name="test_runner", description="d6"),
        ]

        h_mgr = HierarchicalSWEBenchManager(
            model="test-model",
            repo_dir="/tmp/fake_repo",
        )

        h_mgr.analysis_mgr.run_domain = AsyncMock(
            return_value=DomainResult(domain_name="Analysis", summary="Root cause diagnosed")
        )
        h_mgr.testing_mgr.run_domain = AsyncMock(
            side_effect=[
                DomainResult(domain_name="Testing", summary="BUG REPRODUCED"),
                DomainResult(domain_name="Testing", summary="BUG FIXED - GREEN"),
            ]
        )
        h_mgr.patch_mgr.run_domain = AsyncMock(
            return_value=DomainResult(domain_name="Patch", summary="str_replace applied")
        )

        res = await h_mgr.run("Fix bug in math function")
        assert isinstance(res, HierarchicalResult)
        assert res.verified_success is True
        assert h_mgr.analysis_mgr.run_domain.called
        assert h_mgr.patch_mgr.run_domain.called
        assert h_mgr.testing_mgr.run_domain.call_count == 2
