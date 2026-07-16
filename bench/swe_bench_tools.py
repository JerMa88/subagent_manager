"""
SWE-bench specific tools for code debugging and patching.

These tools extend the base toolkit with capabilities needed to
navigate repositories, run shell commands, write files, and search
codebases — the core operations for resolving GitHub issues.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from subagent_manager.logging_config import VERBOSE1, VERBOSE2, truncate_for_log
from subagent_manager.tools.base import BaseTool, ToolParameter

logger = logging.getLogger(__name__)


class ShellExecTool(BaseTool):
    """
    Execute shell commands within a constrained working directory.

    Used for running tests, git operations, grep, find, and other
    shell utilities needed during code debugging.
    """

    name = "shell_exec"
    description = (
        "Execute a shell command and return stdout/stderr. Use this for running "
        "tests (pytest, unittest), git commands (git diff, git log), or shell "
        "utilities (grep, find, cat, head, tail). Commands run in the repository "
        "root directory."
    )
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description=(
                "The shell command to execute. Examples: 'pytest tests/ -x', "
                "'git diff', 'grep -rn \"def foo\" src/', 'find . -name \"*.py\"'"
            ),
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds (default: 60, max: 120).",
            required=False,
        ),
    ]

    def __init__(
        self,
        working_dir: str | None = None,
        default_timeout: float = 60.0,
        max_timeout: float = 120.0,
    ) -> None:
        """
        Initialize the shell executor.

        Args:
            working_dir: Working directory for commands. If None, uses cwd.
            default_timeout: Default command timeout in seconds.
            max_timeout: Maximum allowed timeout.
        """
        self.working_dir = working_dir
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout

    async def execute(self, **kwargs: Any) -> str:
        """Execute a shell command."""
        command = kwargs.get("command", "")
        timeout = min(
            float(kwargs.get("timeout", self.default_timeout)),
            self.max_timeout,
        )

        if not command:
            return "Error: No command provided."

        cwd = self.working_dir or os.getcwd()
        logger.log(
            VERBOSE1,
            f"[TOOL:shell_exec] Running: {command} (cwd={cwd}, timeout={timeout}s)",
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning(f"[TOOL:shell_exec] Timed out after {timeout}s: {command}")
                return f"Error: Command timed out after {timeout} seconds."

            exit_code = proc.returncode
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            logger.log(
                VERBOSE1,
                f"[TOOL:shell_exec] exit_code={exit_code}, "
                f"stdout={len(stdout_text)} chars, stderr={len(stderr_text)} chars",
            )

            parts = []
            parts.append(f"Exit code: {exit_code}")
            if stdout_text.strip():
                parts.append(f"STDOUT:\n{stdout_text}")
            if stderr_text.strip():
                parts.append(f"STDERR:\n{stderr_text}")

            if not stdout_text.strip() and not stderr_text.strip():
                parts.append("(no output)")

            result = "\n".join(parts)

            # Truncate very long output
            max_len = 6000
            if len(result) > max_len:
                result = result[:max_len] + "\n\n[... output truncated]"

            return result

        except Exception as e:
            logger.error(f"[TOOL:shell_exec] Failed: {e}", exc_info=True)
            return f"Error executing command: {str(e)}"


class FileWriterTool(BaseTool):
    """
    Write content to a file, creating directories as needed.

    Used for generating patches by writing modified file content.
    """

    name = "write_file"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Creates parent directories automatically. "
        "Use this to apply code fixes by writing the corrected file content."
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="The file path to write to (relative to repo root or absolute).",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The complete file content to write.",
        ),
    ]

    def __init__(self, working_dir: str | None = None) -> None:
        """
        Initialize the file writer.

        Args:
            working_dir: Base directory for relative paths.
        """
        self.working_dir = working_dir

    async def execute(self, **kwargs: Any) -> str:
        """Write content to a file."""
        path_str = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not path_str:
            return "Error: No file path provided."

        # Resolve relative paths against working_dir
        path = Path(path_str)
        if not path.is_absolute() and self.working_dir:
            path = Path(self.working_dir) / path

        path = path.resolve()

        logger.log(
            VERBOSE1,
            f"[TOOL:write_file] Writing {len(content)} chars to {path}",
        )

        try:
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            path.write_text(content, encoding="utf-8")

            logger.log(VERBOSE1, f"[TOOL:write_file] Successfully wrote {path}")
            return f"Successfully wrote {len(content)} characters to {path}"

        except Exception as e:
            logger.error(f"[TOOL:write_file] Failed to write {path}: {e}")
            return f"Error writing to {path}: {str(e)}"


class DirectoryListTool(BaseTool):
    """
    List directory contents with optional recursive traversal.

    Used for exploring repository structure to find relevant files.
    """

    name = "list_directory"
    description = (
        "List the contents of a directory. Returns file names, sizes, and types. "
        "Use this to explore repository structure and find relevant source files, "
        "test files, or configuration files."
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="The directory path to list (relative to repo root or absolute).",
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="If true, list recursively up to max_depth. Default: false.",
            required=False,
        ),
        ToolParameter(
            name="max_depth",
            type="integer",
            description="Maximum recursion depth (default: 3). Only used if recursive=true.",
            required=False,
        ),
        ToolParameter(
            name="pattern",
            type="string",
            description="Glob pattern to filter results (e.g., '*.py'). Default: '*'.",
            required=False,
        ),
    ]

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir

    async def execute(self, **kwargs: Any) -> str:
        """List directory contents."""
        path_str = kwargs.get("path", ".")
        recursive = kwargs.get("recursive", False)
        max_depth = int(kwargs.get("max_depth", 3))
        pattern = kwargs.get("pattern", "*")

        path = Path(path_str)
        if not path.is_absolute() and self.working_dir:
            path = Path(self.working_dir) / path
        path = path.resolve()

        if not path.exists():
            return f"Error: Directory not found: {path}"
        if not path.is_dir():
            return f"Error: {path} is not a directory."

        logger.log(
            VERBOSE1,
            f"[TOOL:list_directory] Listing {path} "
            f"(recursive={recursive}, pattern={pattern})",
        )

        try:
            entries = []
            if recursive:
                entries = self._list_recursive(path, pattern, max_depth, 0)
            else:
                for item in sorted(path.glob(pattern)):
                    entry_type = "DIR " if item.is_dir() else "FILE"
                    size = ""
                    if item.is_file():
                        size = f" ({item.stat().st_size:,} bytes)"
                    rel = item.relative_to(path)
                    entries.append(f"  {entry_type} {rel}{size}")

            if not entries:
                return f"Directory {path} is empty (or no matches for '{pattern}')"

            header = f"Contents of {path} ({len(entries)} items):\n"
            result = header + "\n".join(entries)

            # Truncate
            if len(result) > 5000:
                result = result[:5000] + f"\n\n[... truncated, {len(entries)} items total]"

            return result

        except Exception as e:
            logger.error(f"[TOOL:list_directory] Failed: {e}")
            return f"Error listing directory: {str(e)}"

    def _list_recursive(
        self, base: Path, pattern: str, max_depth: int, current_depth: int
    ) -> list[str]:
        """Recursively list directory contents."""
        entries = []
        if current_depth > max_depth:
            return entries

        try:
            for item in sorted(base.iterdir()):
                # Skip hidden dirs and __pycache__
                if item.name.startswith(".") or item.name == "__pycache__":
                    continue

                indent = "  " * (current_depth + 1)
                if item.is_dir():
                    entries.append(f"{indent}DIR  {item.name}/")
                    entries.extend(
                        self._list_recursive(item, pattern, max_depth, current_depth + 1)
                    )
                elif item.is_file():
                    from fnmatch import fnmatch
                    if pattern == "*" or fnmatch(item.name, pattern):
                        size = item.stat().st_size
                        entries.append(f"{indent}FILE {item.name} ({size:,} bytes)")
        except PermissionError:
            entries.append(f"{'  ' * (current_depth + 1)}[permission denied]")

        return entries


class GrepTool(BaseTool):
    """
    Search for patterns across files in a directory.

    Wraps grep -rn for fast codebase searching.
    """

    name = "grep_search"
    description = (
        "Search for a text pattern across files in the repository. Returns "
        "matching lines with file paths and line numbers. Use this to find "
        "function definitions, class declarations, variable usage, error "
        "messages, or any text pattern in the codebase."
    )
    parameters = [
        ToolParameter(
            name="pattern",
            type="string",
            description="The search pattern (supports basic regex).",
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Directory or file to search in (default: repo root).",
            required=False,
        ),
        ToolParameter(
            name="include",
            type="string",
            description="File pattern to include (e.g., '*.py'). Default: all files.",
            required=False,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of matching lines to return (default: 50).",
            required=False,
        ),
    ]

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir

    async def execute(self, **kwargs: Any) -> str:
        """Search for a pattern in files."""
        pattern = kwargs.get("pattern", "")
        search_path = kwargs.get("path", ".")
        include = kwargs.get("include", "")
        max_results = int(kwargs.get("max_results", 50))

        if not pattern:
            return "Error: No search pattern provided."

        # Build grep command
        cmd_parts = ["grep", "-rn", "--color=never"]
        if include:
            cmd_parts.extend(["--include", include])
        cmd_parts.append("--")
        cmd_parts.append(pattern)
        cmd_parts.append(search_path)

        # Join carefully for shell execution
        import shlex
        command = " ".join(shlex.quote(p) for p in cmd_parts)

        cwd = self.working_dir or os.getcwd()
        logger.log(
            VERBOSE1,
            f"[TOOL:grep_search] pattern='{pattern}', path={search_path}, "
            f"include={include or '*'}",
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0,
            )

            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            if not output.strip():
                return f"No matches found for pattern: {pattern}"

            # Limit results
            lines = output.strip().split("\n")
            total = len(lines)
            if total > max_results:
                lines = lines[:max_results]

            result = f"Found {total} matches for '{pattern}':\n\n" + "\n".join(lines)
            if total > max_results:
                result += f"\n\n[... showing {max_results} of {total} matches]"

            logger.log(VERBOSE1, f"[TOOL:grep_search] Found {total} matches")

            return result

        except asyncio.TimeoutError:
            return "Error: Search timed out after 30 seconds."
        except Exception as e:
            logger.error(f"[TOOL:grep_search] Failed: {e}")
            return f"Error during search: {str(e)}"
