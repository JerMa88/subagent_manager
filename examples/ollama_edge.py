"""
Edge deployment example using Ollama.

This example shows how to run subagent_manager entirely locally
using Ollama, with no API keys or cloud services required.

This is the key differentiator of subagent_manager:
- Small models (3B-8B) struggle with multi-step reasoning
- By decomposing into single-step subtasks, even small models
  produce reliable, grounded results
- Each subagent only needs to handle ONE thing at a time

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull a model: ollama pull qwen3
    3. Start Ollama: ollama serve
    4. Install subagent_manager: pip install -e .

Recommended models for edge deployment:
    - qwen3 (default, good balance of speed and quality)
    - llama3.1:8b (strong reasoning)
    - mistral-small (good tool calling)
    - nemotron-mini (optimized for function calling)

For the orchestrator, you can use a slightly larger model:
    - Use qwen3 for subagents (fast)
    - Use a 14B model for the orchestrator (smarter planning)
"""

from subagent_manager import SubAgentManager, SubAgentConfig
from subagent_manager.tools import WebSearchTool, URLReaderTool, PythonExecTool


def basic_edge_example():
    """Simplest possible edge deployment."""
    print("=" * 60)
    print("BASIC EDGE EXAMPLE")
    print("=" * 60)

    manager = SubAgentManager(
        model="ollama/qwen3",
        verbose=True,
    )

    result = manager.run_sync(
        "What is the current population of the top 5 most populous countries?"
    )

    print(f"\nAnswer:\n{result.answer}")
    print(f"\nTool calls: {result.total_tool_calls}")
    print(f"Sources: {result.sources}")


def hybrid_model_example():
    """
    Use different models for orchestrator vs subagents.

    Strategy: smarter model plans, faster model executes.
    This gives you the best of both worlds on edge hardware.
    """
    print("\n" + "=" * 60)
    print("HYBRID MODEL EXAMPLE")
    print("=" * 60)

    manager = SubAgentManager(
        model="ollama/qwen3",  # Fast model for subagents
        orchestrator_model="ollama/qwen3",  # Can use a larger model here
        api_base="http://localhost:11434",
        verbose=True,
    )

    result = manager.run_sync(
        "Explain the key differences between REST and GraphQL APIs, "
        "with examples of when to use each."
    )

    print(f"\nAnswer:\n{result.answer}")


def custom_edge_agents():
    """
    Define minimal agents optimized for edge deployment.

    Key optimizations:
    - Fewer tools per agent (reduces function schema complexity)
    - Lower max_tool_iterations (faster completion)
    - Lower max_answer_tokens (less generation overhead)
    """
    print("\n" + "=" * 60)
    print("CUSTOM EDGE AGENTS EXAMPLE")
    print("=" * 60)

    manager = SubAgentManager(
        model="ollama/qwen3",
        subagents=[
            SubAgentConfig(
                name="searcher",
                description="Searches the web for facts and data",
                tools=[WebSearchTool()],
                max_tool_iterations=3,  # Fewer iterations for speed
                max_answer_tokens=256,  # Shorter answers
            ),
            SubAgentConfig(
                name="calculator",
                description="Performs calculations and data analysis",
                tools=[PythonExecTool(timeout=5.0)],
                max_tool_iterations=2,
                max_answer_tokens=256,
            ),
            SubAgentConfig(
                name="thinker",
                description="Analyzes and reasons about information",
                tools=[],  # Pure reasoning
                max_answer_tokens=384,
            ),
        ],
        max_subtasks=5,  # Limit subtasks for efficiency
        verbose=True,
    )

    result = manager.run_sync(
        "Calculate compound interest on $10,000 at 7% annual rate "
        "over 10 years, compounded monthly. Compare this with a "
        "typical savings account rate."
    )

    print(f"\nAnswer:\n{result.answer}")
    print(f"\nSubtasks: {len(result.subtask_results)}")
    print(f"Total tool calls: {result.total_tool_calls}")


if __name__ == "__main__":
    basic_edge_example()
    # Uncomment these for more examples:
    # hybrid_model_example()
    # custom_edge_agents()
