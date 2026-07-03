"""
Web research example with grounded answers.

This example demonstrates how subagent_manager uses web search
and URL reading to produce grounded, evidence-based answers.

The key difference from a regular LLM call:
- A regular LLM might hallucinate facts
- subagent_manager forces subagents to search the web first,
  ground their answers in real sources, and cite everything

Prerequisites:
    pip install -e .
    # Plus one of: Ollama running, OPENAI_API_KEY, GEMINI_API_KEY
"""

import os
from subagent_manager import SubAgentManager, SubAgentConfig
from subagent_manager.tools import WebSearchTool, URLReaderTool


def main():
    # Create a manager with custom research-focused agents
    manager = SubAgentManager(
        model="gpt-4o-mini",  # or "ollama/qwen3" for local
        api_key=os.getenv("OPENAI_API_KEY"),
        subagents=[
            SubAgentConfig(
                name="web_researcher",
                description=(
                    "Expert at finding specific facts and data from the web. "
                    "Always searches before answering."
                ),
                tools=[WebSearchTool(), URLReaderTool()],
            ),
            SubAgentConfig(
                name="deep_reader",
                description=(
                    "Reads and summarizes long web articles or documentation. "
                    "Use when you need detailed information from a specific URL."
                ),
                tools=[URLReaderTool(max_content_length=8000)],
            ),
            SubAgentConfig(
                name="synthesizer",
                description=(
                    "Combines information from multiple sources into a clear, "
                    "structured summary. Does NOT search — only analyzes."
                ),
                tools=[],
            ),
        ],
        strategy="adaptive",
        verbose=True,
    )

    # Research query — this will automatically:
    # 1. Decompose into subtasks (search, read, verify)
    # 2. Execute each in an isolated context
    # 3. Synthesize into a grounded answer
    result = manager.run_sync(
        "Compare the latest features and pricing of Cursor, GitHub Copilot, "
        "and Windsurf as AI coding assistants. Which is best for a Python developer?"
    )

    print("\n" + "=" * 60)
    print("GROUNDED RESEARCH RESULT")
    print("=" * 60)
    print(result.answer)

    if result.sources:
        print("\n" + "-" * 40)
        print("SOURCES")
        print("-" * 40)
        for source in result.sources:
            print(f"  📎 {source}")


if __name__ == "__main__":
    main()
