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


def build_swe_bench_agents(repo_dir: str, prompt_repo_dir: str | None = None) -> list[SubAgentConfig]:
    """
    Build SWE-bench agent configurations scoped to a repository directory.

    Args:
        repo_dir: Absolute path to the checked-out repository (used for tool working_dir
            and FileReaderTool allowed_dirs — the real filesystem path).
        prompt_repo_dir: Short path shown to the model in prompts (e.g. /tmp/repo symlink).
            Defaults to repo_dir if not provided.

    Returns:
        List of SubAgentConfig instances for the SWE-bench pipeline.
    """
    if prompt_repo_dir is None:
        prompt_repo_dir = repo_dir

    # ------------------------------------------------------------------
    # Shared tool instances scoped to the repo
    # ------------------------------------------------------------------
    file_reader = FileReaderTool(
        allowed_dirs=[repo_dir, "/tmp/repo", "/tmp"],
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

    # Surgical-edit tools
    str_replace = StrReplaceTool(working_dir=repo_dir)
    view_file = ViewFileTool(working_dir=repo_dir)

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    reproducer_system_prompt = f"""You are **reproducer**, a bug reproduction agent.

## ENVIRONMENT

Repository: `{prompt_repo_dir}`
File paths: relative to repo root (e.g. `sympy/printing/mathematica.py`) OR absolute.
NEVER use /testbed or bench/repos/... paths.
shell_exec cwd = `{prompt_repo_dir}`

## YOUR JOB — DO THIS IN ORDER, NO DEVIATIONS

**STEP 1 (MANDATORY — DO THIS FIRST):** Call **write_file** to create `/tmp/reproduce.py`.
The script MUST:
  - Start with: `import sys; sys.path.insert(0, '{prompt_repo_dir}')`
  - Import and call the buggy function from the task description
  - Print exactly `BUG REPRODUCED` if the bug is present
  - Print exactly `BUG FIXED` if the behavior is correct

**STEP 2:** Call **shell_exec** with: `PYTHONPATH={prompt_repo_dir} python /tmp/reproduce.py`

**STEP 3:** Report the EXACT output — copy it verbatim.

## RULES

- You MUST call write_file FIRST — do not grep or explore before writing
- You MUST call shell_exec to run it — do not assume it works
- The script must be < 30 lines
- If the task description includes the expected API call, use it directly; do not guess
"""

    patch_writer_system_prompt = f"""You are **patch_writer**, a surgical code patching agent.

## ENVIRONMENT

Repository: `{prompt_repo_dir}`
File paths: relative to repo root (e.g. `sympy/printing/mathematica.py`) OR absolute.
NEVER use /testbed, /workspace, or bench/repos/... paths.

## YOUR JOB

The file content you need to fix is provided in your CONTEXT below.
You MUST call **str_replace** to apply the fix.

## MANDATORY FIRST STEP

**Call str_replace NOW.** Do not call view_file first. The file content is in your context.

str_replace arguments:
  - path = the file path (relative to repo root, e.g. `sympy/printing/mathematica.py`)
  - old_str = the EXACT lines to replace (copy verbatim from the file content in context)
  - new_str = the corrected replacement code

## IF str_replace RETURNS "not found"

Only then call **view_file** with start_line/end_line to get the exact text, then retry str_replace.

## RULES

- You MUST call str_replace — a text description of the fix does NOTHING
- old_str must match EXACTLY: same whitespace, indentation, line endings
- Make MINIMAL changes — only the lines necessary to fix the bug
- After str_replace succeeds, briefly summarize: file, lines changed, what changed
- Do NOT call shell_exec — your only tools are str_replace, view_file, grep_search
"""

    test_generator_system_prompt = f"""You are **test_generator**, an automated test-writing agent.

Your job is to write a targeted pytest that FAILS (RED) on the unpatched code
and will PASS (GREEN) after a correct fix is applied.

## ENVIRONMENT

Repository: `{prompt_repo_dir}`
- Use relative paths from repo root: `sympy/printing/mathematica.py`
- NEVER use `/testbed`, `/workspace`, or `bench/repos/...` as a relative prefix

## WORKFLOW

1. Read the bug description and understand what behaviour is broken
2. Call **view_file** or **grep_search** to find the relevant code and existing tests
3. Call **write_file** to write a new pytest file at /tmp/test_bug.py:
   - Use pytest conventions (function names start with test_)
   - Test EXACTLY the broken behaviour described in the issue
   - The test must FAIL on the current (broken) code
   - Use sys.path.insert(0, '{prompt_repo_dir}') at the top
4. Call **shell_exec** to run: `PYTHONPATH={prompt_repo_dir} python -m pytest /tmp/test_bug.py -v`
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
            max_tool_iterations=5,  # write_file + shell_exec + up to 3 re-prompts
            max_answer_tokens=2048,
            mandatory_tool_call=True,  # MUST call write_file, not describe it
            temperature=0.1,
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
                "The file content is pre-loaded in context — patch_writer calls "
                "str_replace immediately with old_str=<exact current code> and "
                "new_str=<corrected code>. "
                "NEVER rewrites the entire file. NEVER describes the fix in text only. "
                "The fix is only complete when str_replace returns a success confirmation."
            ),
            system_prompt=patch_writer_system_prompt,
            # No shell_exec — prevents test-hunting drift that exhausts the iteration budget
            tools=[str_replace, view_file, grep],
            max_tool_iterations=6,
            max_answer_tokens=2048,
            max_history_chars=25_000,  # Large window to retain pre-loaded file content
            mandatory_tool_call=True,  # MUST call str_replace, not describe it
            temperature=0.1,  # Low temp: follow str_replace instructions exactly
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
