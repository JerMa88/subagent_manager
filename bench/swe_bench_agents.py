"""
SWE-bench specialized agent configurations.

Defines six agents optimized for the SWE-bench code debugging workflow:
1. issue_analyzer  — Understands the bug from the problem statement + code
2. reproducer      — Writes and runs a minimal script to confirm BUG REPRODUCED
3. code_explorer   — Navigates the repo to find relevant files
4. patch_writer    — Generates the actual code fix (surgical str_replace only)
5. test_generator  — Writes a targeted pytest that is RED before fix, GREEN after
6. test_runner     — Independently verifies the fix by re-running the reproduce script

Each agent is pre-configured with appropriate tools scoped to the
target repository's working directory.

The orchestrator embodies the SKEPTICAL SENIOR PROGRAMMER principle:
it never trusts a subagent's self-report. After every patch_writer,
test_runner independently verifies the outcome. No plan ends without
a green verification signal.
"""

from __future__ import annotations

from subagent_manager import SubAgentConfig
from subagent_manager.tools.file_reader import FileReaderTool

from bench.swe_bench_tools import (
    DirectoryListTool,
    FileWriterTool,
    GrepTool,
    ShellExecTool,
    StrReplaceTool,
    ViewFileTool,
)


def build_swe_bench_agents(repo_dir: str) -> list[SubAgentConfig]:
    """
    Build SWE-bench agent configurations scoped to a repository directory.

    Args:
        repo_dir: Absolute path to the checked-out repository.

    Returns:
        List of SubAgentConfig instances for the SWE-bench pipeline.
    """
    # ------------------------------------------------------------------
    # Shared tool instances scoped to the repo
    # ------------------------------------------------------------------
    file_reader = FileReaderTool(
        allowed_dirs=[repo_dir],
        working_dir=repo_dir,
        max_content_length=12000,
    )
    # Override truncation limits for SWE-bench (source files can be large)
    file_reader.max_result_length = 12000

    shell_exec = ShellExecTool(working_dir=repo_dir)
    shell_exec.max_result_length = 8000  # Test output can be verbose

    # file_writer kept for reproducer/test_generator (writing /tmp scripts)
    file_writer = FileWriterTool(working_dir=repo_dir)

    dir_list = DirectoryListTool(working_dir=repo_dir)
    grep = GrepTool(working_dir=repo_dir)

    # New surgical-edit tools (P1)
    str_replace = StrReplaceTool(working_dir=repo_dir)
    view_file = ViewFileTool(working_dir=repo_dir)

    # ------------------------------------------------------------------
    # System prompts — explicit workflows override the "be concise" default
    # ------------------------------------------------------------------

    reproducer_system_prompt = f"""You are **reproducer**, a bug reproduction agent.

Your job is to write and run a minimal Python script that demonstrates the bug.

## ENVIRONMENT

The repository is at: `{repo_dir}`
**File paths for tools:**
- Relative to repo root: `sympy/printing/mathematica.py` (preferred for view_file/grep_search)
- Full absolute: `{repo_dir}/sympy/printing/mathematica.py`
- NEVER use `/testbed`, `/workspace`, or any other path
- shell_exec cwd is `{repo_dir}` — use relative paths or absolute paths, NOT `bench/repos/...`

## WORKFLOW — FOLLOW EXACTLY

1. Call **view_file** or **grep_search** to understand the relevant API
2. Call **write_file** to write /tmp/reproduce.py that:
   - Imports the relevant module from the repo:
     ```python
     import sys
     sys.path.insert(0, '{repo_dir}')
     ```
   - Calls the buggy function
   - Prints 'BUG REPRODUCED' if the bug is present
   - Prints 'BUG FIXED' if the behavior is correct
3. Call **shell_exec** with: `PYTHONPATH={repo_dir} python /tmp/reproduce.py`
4. Report the exact output

## RULES

- You MUST call write_file to create /tmp/reproduce.py — describing it is not enough
- You MUST call shell_exec to run it — assuming it works is not evidence
- Keep the script minimal (< 30 lines); the goal is to isolate the bug, not test everything
- If the import fails, use shell_exec to install the package or add it to PYTHONPATH
"""

    patch_writer_system_prompt = f"""You are **patch_writer**, a surgical code patching agent.

Your job is to FIX a software bug by replacing ONLY the broken lines.

## ENVIRONMENT

The repository is at: `{repo_dir}`
**File paths for tools:**
- Relative to repo root: `sympy/printing/mathematica.py` (preferred)
- Full absolute: `{repo_dir}/sympy/printing/mathematica.py`
- NEVER use `/testbed`, `/workspace`, or `bench/repos/...` as relative paths
- shell_exec cwd is `{repo_dir}`

## CRITICAL WORKFLOW — YOU MUST FOLLOW THESE STEPS:

1. Call **view_file** to read the relevant section (use start_line/end_line to zoom in)
2. Identify the EXACT lines that need changing
3. Call **str_replace** with:
   - path = the file to modify (relative to repo root OR absolute)
   - old_str = the EXACT current code (copy from view_file output, character-perfect)
   - new_str = the corrected code

## RULES

- You MUST call str_replace — describing the fix in text is USELESS
- Do NOT rewrite the entire file. Only change what is broken.
- old_str must match EXACTLY including all whitespace and indentation
- Copy old_str directly from view_file output — do not rephrase or re-indent
- If str_replace returns "not found", call view_file again to get the exact text
- Make MINIMAL changes — only modify the lines necessary to fix the bug

## COMMON MISTAKE

If str_replace returns "Error: old_str not found", it means your old_str has
different whitespace, tabs vs spaces, or missing/extra lines. Call view_file
with the exact line numbers and copy the text verbatim.

After writing the fix, briefly summarize: what file, what lines, what changed.
"""

    test_generator_system_prompt = f"""You are **test_generator**, an automated test-writing agent.

Your job is to write a targeted pytest that FAILS (RED) on the unpatched code
and will PASS (GREEN) after a correct fix is applied.

## ENVIRONMENT

The repository is at: `{repo_dir}`
- Use relative paths from repo root: `sympy/printing/mathematica.py`
- NEVER use `/testbed`, `/workspace`, or `bench/repos/...` as a relative prefix

## WORKFLOW

1. Read the bug description and understand what behaviour is broken
2. Call **view_file** or **grep_search** to find the relevant code and existing tests
3. Call **write_file** to write a new pytest file at /tmp/test_bug.py:
   - Use pytest conventions (function names start with test_)
   - Test EXACTLY the broken behaviour described in the issue
   - The test must FAIL on the current (broken) code
   - Use sys.path.insert(0, '{repo_dir}') at the top
4. Call **shell_exec** to run: `PYTHONPATH={repo_dir} python -m pytest /tmp/test_bug.py -v`
5. Confirm the test FAILS (exit code 1) — this is the RED state
6. Report the test file contents and the failure output

## RULES

- You MUST write and run the test — do not just describe it
- The test must be self-contained and not require repo installation
- Keep it focused: one function, one assertion per bug
"""

    # ------------------------------------------------------------------
    # Agent definitions
    # ------------------------------------------------------------------

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
            max_answer_tokens=2048,
            temperature=0.3,
        ),
        SubAgentConfig(
            name="reproducer",
            description=(
                "Creates and runs a minimal Python script (/tmp/reproduce.py) that "
                "demonstrates the bug. The script prints 'BUG REPRODUCED' when the "
                "bug is present and 'BUG FIXED' when the fix is applied. "
                "Returns the exact shell output — not a verbal confirmation."
            ),
            system_prompt=reproducer_system_prompt,
            tools=[file_writer, shell_exec, view_file, grep],
            max_tool_iterations=6,
            max_answer_tokens=2048,
            temperature=0.2,
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
            max_answer_tokens=2048,
            temperature=0.2,
        ),
        SubAgentConfig(
            name="patch_writer",
            description=(
                "Applies a SURGICAL code fix for a diagnosed bug using str_replace. "
                "Reads the exact lines to change with view_file, then calls str_replace "
                "with old_str=<exact current code> and new_str=<corrected code>. "
                "NEVER rewrites the entire file. NEVER describes the fix in text only. "
                "The fix is only complete when str_replace returns a success confirmation."
            ),
            system_prompt=patch_writer_system_prompt,
            tools=[str_replace, view_file, grep, shell_exec],
            max_tool_iterations=10,
            max_answer_tokens=2048,
            temperature=0.2,
        ),
        SubAgentConfig(
            name="test_generator",
            description=(
                "Writes a targeted pytest (/tmp/test_bug.py) that FAILS on the "
                "current broken code and will PASS after a correct fix. "
                "Runs the test immediately to confirm it is RED. "
                "Provides a machine-checkable contract for the fix."
            ),
            system_prompt=test_generator_system_prompt,
            tools=[file_writer, shell_exec, view_file, grep],
            max_tool_iterations=5,
            max_answer_tokens=2048,
            temperature=0.2,
        ),
        SubAgentConfig(
            name="test_runner",
            description=(
                "Independently verifies a patch by re-running /tmp/reproduce.py "
                "and/or the relevant test suite. Does NOT rely on patch_writer's "
                "own report. Returns the ACTUAL shell output — pass or fail. "
                "If /tmp/test_bug.py exists, also runs it and reports RED/GREEN. "
                "This is the final verification gate; the orchestrator must not "
                "accept any fix without this agent's independent confirmation."
            ),
            tools=[shell_exec, file_reader],
            max_tool_iterations=5,
            max_answer_tokens=2048,
            temperature=0.1,
        ),
    ]
