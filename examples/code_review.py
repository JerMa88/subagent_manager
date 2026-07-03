"""
Code review example using subagent_manager.

This example demonstrates using subagents for code analysis tasks.
The coder agent reads files and executes analysis code, while the
analyzer agent reasons about code quality.

Prerequisites:
    pip install -e .
    # Plus an LLM provider
"""

import os
from subagent_manager import SubAgentManager, SubAgentConfig
from subagent_manager.tools import FileReaderTool, PythonExecTool, WebSearchTool


def main():
    manager = SubAgentManager(
        model="gpt-4o-mini",  # or "ollama/qwen3" for local
        api_key=os.getenv("OPENAI_API_KEY"),
        subagents=[
            SubAgentConfig(
                name="code_reader",
                description=(
                    "Reads and understands source code files. Use when you "
                    "need to examine the contents of a specific file."
                ),
                tools=[FileReaderTool()],
            ),
            SubAgentConfig(
                name="code_analyzer",
                description=(
                    "Analyzes code for bugs, security issues, performance "
                    "problems, and style violations. Can run linting scripts."
                ),
                tools=[PythonExecTool(), FileReaderTool()],
            ),
            SubAgentConfig(
                name="best_practices_checker",
                description=(
                    "Searches for current best practices and compares code "
                    "against community standards and documentation."
                ),
                tools=[WebSearchTool()],
            ),
            SubAgentConfig(
                name="reviewer",
                description=(
                    "Synthesizes findings from code reading and analysis into "
                    "a structured code review with actionable recommendations."
                ),
                tools=[],
            ),
        ],
        strategy="adaptive",
        verbose=True,
    )

    # Review a file (replace with your actual file path)
    result = manager.run_sync(
        "Review the file at src/subagent_manager/manager.py for: "
        "1) Code quality and maintainability "
        "2) Error handling robustness "
        "3) Performance considerations "
        "4) Any security concerns"
    )

    print("\n" + "=" * 60)
    print("CODE REVIEW RESULT")
    print("=" * 60)
    print(result.answer)


if __name__ == "__main__":
    main()
