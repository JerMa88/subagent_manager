"""
File reader tool.

Reads local files for codebase analysis, data inspection, and
document processing workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from subagent_manager.tools.base import BaseTool, ToolParameter

logger = logging.getLogger(__name__)


class FileReaderTool(BaseTool):
    """
    Read the contents of a local file.

    Supports text files of any type. Binary files are rejected.
    Content is truncated for very large files.
    """

    name = "read_file"
    description = (
        "Read the contents of a local file. Returns the file content as text. "
        "Use this to inspect code, configuration files, data files, or documents."
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="The file path to read (absolute or relative).",
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Optional: start reading from this line number (1-indexed).",
            required=False,
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="Optional: stop reading at this line number (inclusive).",
            required=False,
        ),
    ]

    def __init__(
        self,
        max_content_length: int = 8000,
        allowed_dirs: list[str] | None = None,
    ) -> None:
        """
        Initialize the file reader.

        Args:
            max_content_length: Maximum characters to return.
            allowed_dirs: If set, only files within these directories can be read.
                Provides basic sandboxing for security.
        """
        self.max_content_length = max_content_length
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or [])]

    async def execute(self, **kwargs: Any) -> str:
        """Read a file and return its contents."""
        path_str = kwargs.get("path", "")

        if not path_str:
            return "Error: No file path provided."

        path = Path(path_str).resolve()

        # Security check
        if self.allowed_dirs:
            if not any(self._is_subpath(path, d) for d in self.allowed_dirs):
                return (
                    f"Error: Access denied. File {path} is outside allowed directories."
                )

        if not path.exists():
            return f"Error: File not found: {path}"

        if not path.is_file():
            return f"Error: {path} is not a file."

        # Check if binary
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: {path} appears to be a binary file and cannot be read as text."

        # Handle line range
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")

        if start_line is not None or end_line is not None:
            lines = content.split("\n")
            start = max(1, int(start_line or 1)) - 1  # Convert to 0-indexed
            end = min(len(lines), int(end_line or len(lines)))
            selected_lines = lines[start:end]
            content = "\n".join(
                f"{i + start + 1}: {line}" for i, line in enumerate(selected_lines)
            )
            header = f"File: {path} (lines {start + 1}-{end} of {len(lines)})\n\n"
        else:
            total_lines = content.count("\n") + 1
            header = f"File: {path} ({total_lines} lines)\n\n"

        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "\n\n[... content truncated]"

        return header + content

    @staticmethod
    def _is_subpath(path: Path, parent: Path) -> bool:
        """Check if path is within parent directory."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
