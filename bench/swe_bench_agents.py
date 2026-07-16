"""
SWE-bench specialized agent configurations.

Defines four agents optimized for the SWE-bench code debugging workflow:
1. issue_analyzer  — Understands the bug from the problem statement + code
2. code_explorer   — Navigates the repo to find relevant files
3. patch_writer    — Generates the actual code fix
4. test_runner     — Runs tests to verify the patch

Each agent is pre-configured with appropriate tools scoped to the
target repository's working directory.
"""

from __future__ import annotations

from subagent_manager import SubAgentConfig
from subagent_manager.tools.file_reader import FileReaderTool

from bench.swe_bench_tools import (
    DirectoryListTool,
    FileWriterTool,
    GrepTool,
    ShellExecTool,
)


def build_swe_bench_agents(repo_dir: str) -> list[SubAgentConfig]:
    """
    Build SWE-bench agent configurations scoped to a repository directory.

    Args:
        repo_dir: Absolute path to the checked-out repository.

    Returns:
        List of SubAgentConfig instances for the SWE-bench pipeline.
    """
    # Shared tool instances scoped to the repo
    file_reader = FileReaderTool(allowed_dirs=[repo_dir], working_dir=repo_dir)
    shell_exec = ShellExecTool(working_dir=repo_dir)
    file_writer = FileWriterTool(working_dir=repo_dir)
    dir_list = DirectoryListTool(working_dir=repo_dir)
    grep = GrepTool(working_dir=repo_dir)

    return [
        SubAgentConfig(
            name="issue_analyzer",
            description=(
                "Analyzes a GitHub issue to understand the bug or feature request. "
                "Reads the problem statement and relevant source code to identify "
                "what behavior is expected vs. what is actually happening. Produces "
                "a clear diagnosis: what file(s) are involved, what the root cause "
                "likely is, and what kind of fix is needed."
            ),
            tools=[file_reader, grep, dir_list],
            max_tool_iterations=5,
            max_answer_tokens=1024,
            temperature=0.3,
        ),
        SubAgentConfig(
            name="code_explorer",
            description=(
                "Explores the repository structure to find files relevant to a bug. "
                "Navigates directories, reads file contents, and searches for "
                "function definitions, class declarations, imports, and test files. "
                "Returns a map of the relevant files and their roles."
            ),
            tools=[file_reader, grep, dir_list, shell_exec],
            max_tool_iterations=8,
            max_answer_tokens=1024,
            temperature=0.2,
        ),
        SubAgentConfig(
            name="patch_writer",
            description=(
                "Generates and APPLIES a code fix for a diagnosed bug. You MUST "
                "use the read_file tool to read the current file content, then "
                "use the write_file tool to write the corrected version. Simply "
                "describing the fix in text is NOT sufficient — you must actually "
                "call write_file with the complete corrected file content. The fix "
                "should be minimal — change only what is necessary to resolve the "
                "issue without altering unrelated code."
            ),
            tools=[file_reader, file_writer, grep, shell_exec],
            max_tool_iterations=10,
            max_answer_tokens=2048,
            temperature=0.2,
        ),
        SubAgentConfig(
            name="test_runner",
            description=(
                "Runs the test suite (or specific tests) to verify that a patch "
                "resolves the reported issue without breaking existing functionality. "
                "Executes test commands and interprets the results."
            ),
            tools=[shell_exec, file_reader],
            max_tool_iterations=5,
            max_answer_tokens=1024,
            temperature=0.1,
        ),
    ]
