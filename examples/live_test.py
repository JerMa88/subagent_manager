"""
Live integration test for subagent_manager.

Runs a real query against Ollama and logs every step of the
orchestration pipeline with timestamps.
"""

import asyncio
import json
import logging
import time
import sys

# Set up detailed logging BEFORE imports
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from subagent_manager import SubAgentManager, SubAgentConfig
from subagent_manager.tools import WebSearchTool


async def main():
    print("=" * 70)
    print("SUBAGENT MANAGER — LIVE INTEGRATION TEST")
    print("=" * 70)

    # Use a simple question that doesn't require web search
    # so we can test the core plan→delegate→synthesize loop
    manager = SubAgentManager(
        model="ollama/gemma4:e2b-mlx",
        subagents=[
            SubAgentConfig(
                name="researcher",
                description=(
                    "Searches the web for factual information, current data, "
                    "and statistics. Use for any task requiring real-world facts."
                ),
                tools=[WebSearchTool()],
                max_tool_iterations=2,
                max_answer_tokens=300,
            ),
            SubAgentConfig(
                name="analyzer",
                description=(
                    "Analyzes information, compares options, and draws conclusions. "
                    "Use for reasoning about provided data — NOT for gathering new info."
                ),
                tools=[],
                max_answer_tokens=300,
            ),
        ],
        strategy="adaptive",
        max_subtasks=4,
        verbose=True,
    )

    print(f"\nModel: {manager.model}")
    print(f"Agents: {list(manager.agents.keys())}")
    print(f"Strategy: {type(manager.strategy).__name__}")

    query = "What are 3 benefits of using Python for machine learning?"

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    result = await manager.run(query)
    elapsed = time.time() - t0

    # ---- Print Results ----
    print(f"\n{'=' * 70}")
    print("ORCHESTRATION PLAN")
    print(f"{'=' * 70}")
    print(json.dumps(result.plan, indent=2))

    print(f"\n{'=' * 70}")
    print("SUBTASK RESULTS")
    print(f"{'=' * 70}")
    for i, r in enumerate(result.subtask_results, 1):
        status = "✓" if r.success else "✗ FAILED"
        print(f"\n--- Subtask {i} [{status}] (Agent: {r.agent_name}) ---")
        print(f"Task: {r.task}")
        print(f"Tool calls: {r.tool_calls_made}")
        print(f"Tokens: {r.tokens_used}")
        if r.sources:
            print(f"Sources: {r.sources}")
        print(f"Answer:\n{r.answer[:500]}")
        if r.error:
            print(f"Error: {r.error}")

    print(f"\n{'=' * 70}")
    print("FINAL SYNTHESIZED ANSWER")
    print(f"{'=' * 70}")
    print(result.answer)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Subtasks: {len(result.subtask_results)}")
    print(f"Tool calls: {result.total_tool_calls}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Sources: {len(result.sources)}")


if __name__ == "__main__":
    asyncio.run(main())
