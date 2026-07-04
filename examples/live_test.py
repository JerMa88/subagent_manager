"""
Live integration test for subagent_manager.

Runs a real query against Ollama and logs every step of the
orchestration pipeline with timestamps.

Usage:
    python examples/live_test.py

Output is saved to examples/live_test_output.log
"""

import asyncio
import json
import logging
import time
import sys

# Only show INFO from our code, suppress all the noisy library debug output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Suppress noisy libraries
for noisy in [
    "httpcore", "httpx", "LiteLLM", "litellm",
    "primp", "hickory_net", "hickory_resolver",
    "h2", "hyper_util", "rustls",
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from subagent_manager import SubAgentManager, SubAgentConfig
from subagent_manager.tools import WebSearchTool


async def main():
    header = "=" * 70
    output_lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log(header)
    log("SUBAGENT MANAGER — LIVE INTEGRATION TEST")
    log(header)

    manager = SubAgentManager(
        model="ollama/ornith",
        subagents=[
            SubAgentConfig(
                name="researcher",
                description=(
                    "Searches the web for factual information, current data, "
                    "and statistics. Use for any task requiring real-world facts."
                ),
                tools=[WebSearchTool()],
                max_tool_iterations=3,
                max_answer_tokens=512,
            ),
            SubAgentConfig(
                name="analyzer",
                description=(
                    "Analyzes information, compares options, and draws conclusions. "
                    "Use for reasoning about provided data — NOT for gathering new info."
                ),
                tools=[],
                max_answer_tokens=512,
            ),
        ],
        strategy="adaptive",
        max_subtasks=4,
        verbose=True,
    )

    log(f"\nModel: {manager.model}")
    log(f"Agents: {list(manager.agents.keys())}")
    log(f"Strategy: {type(manager.strategy).__name__}")

    query = "What are 3 benefits of using Python for machine learning?"

    log(f"\n{header}")
    log(f"QUERY: {query}")
    log(f"{header}\n")

    t0 = time.time()
    result = await manager.run(query)
    elapsed = time.time() - t0

    # ---- Print Results ----
    log(f"\n{header}")
    log("ORCHESTRATION PLAN")
    log(header)
    log(json.dumps(result.plan, indent=2))

    log(f"\n{header}")
    log("SUBTASK RESULTS")
    log(header)
    for i, r in enumerate(result.subtask_results, 1):
        status = "✓" if r.success else "✗ FAILED"
        log(f"\n--- Subtask {i} [{status}] (Agent: {r.agent_name}) ---")
        log(f"Task: {r.task}")
        log(f"Tool calls: {r.tool_calls_made}")
        log(f"Tokens: {r.tokens_used}")
        if r.sources:
            log(f"Sources: {r.sources}")
        log(f"Answer:\n{r.answer[:800]}")
        if r.error:
            log(f"Error: {r.error}")

    log(f"\n{header}")
    log("FINAL SYNTHESIZED ANSWER")
    log(header)
    log(result.answer)

    log(f"\n{header}")
    log("SUMMARY")
    log(header)
    log(f"Total time: {elapsed:.1f}s")
    log(f"Subtasks: {len(result.subtask_results)}")
    log(f"Tool calls: {result.total_tool_calls}")
    log(f"Tokens: {result.total_tokens}")
    log(f"Sources: {len(result.sources)}")

    # Save to file
    import pathlib
    out_path = pathlib.Path(__file__).parent / "live_test_output.log"
    out_path.write_text("\n".join(output_lines))
    log(f"\n📄 Output saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
