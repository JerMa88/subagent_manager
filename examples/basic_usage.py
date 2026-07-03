"""
Basic usage example for subagent_manager.

This example shows the simplest way to use the framework.
It creates a manager with default settings and runs a query.

Prerequisites:
    pip install -e .
    # Plus one of:
    # - Ollama running locally (default)
    # - OPENAI_API_KEY env var set
    # - GEMINI_API_KEY env var set
"""

import os
from subagent_manager import SubAgentManager


def main():
    # Option 1: Use with Ollama (local, free)
    # Make sure Ollama is running: ollama serve
    # And a model is pulled: ollama pull qwen3
    manager = SubAgentManager(
        model="ollama/qwen3",
        verbose=True,
    )

    # Option 2: Use with OpenAI
    # manager = SubAgentManager(
    #     model="gpt-4o-mini",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     verbose=True,
    # )

    # Option 3: Use with Google Gemini
    # manager = SubAgentManager(
    #     model="gemini/gemini-2.5-flash",
    #     api_key=os.getenv("GEMINI_API_KEY"),
    #     verbose=True,
    # )

    # Run a query
    result = manager.run_sync(
        "What are the three most significant breakthroughs in quantum computing "
        "from 2024-2025, and what are their practical implications?"
    )

    # Print the result
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.answer)

    print("\n" + "=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"Subtasks executed: {len(result.subtask_results)}")
    print(f"Total tool calls: {result.total_tool_calls}")
    print(f"Total tokens used: {result.total_tokens}")
    print(f"Sources found: {len(result.sources)}")
    for source in result.sources:
        print(f"  - {source}")

    print("\nSubtask details:")
    for r in result.subtask_results:
        status = "✓" if r.success else "✗"
        print(f"  [{status}] {r.agent_name}: {r.task[:60]}...")


if __name__ == "__main__":
    main()
