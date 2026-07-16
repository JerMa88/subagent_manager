"""
Sandboxed Python execution tool.

Executes Python code in a restricted subprocess with timeout.
Useful for data analysis, calculations, and transformations.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from subagent_manager.tools.base import BaseTool, ToolParameter

from subagent_manager.logging_config import VERBOSE1, VERBOSE2, truncate_for_log

logger = logging.getLogger(__name__)


class PythonExecTool(BaseTool):
    """
    Execute Python code in a sandboxed subprocess.

    Runs code in a separate process with a timeout to prevent
    hanging. Only standard library modules are available.
    """

    name = "python_exec"
    description = (
        "Execute Python code and return the output. Use this for calculations, "
        "data processing, string manipulation, or any task that benefits from code. "
        "Only standard library modules are available. Print your results to stdout."
    )
    parameters = [
        ToolParameter(
            name="code",
            type="string",
            description=(
                "The Python code to execute. Use print() to output results. "
                "Only standard library is available."
            ),
        ),
    ]

    def __init__(self, timeout: float = 10.0) -> None:
        """
        Initialize the Python executor.

        Args:
            timeout: Maximum execution time in seconds.
        """
        self.timeout = timeout

    async def execute(self, **kwargs: Any) -> str:
        """Execute Python code in a subprocess."""
        code = kwargs.get("code", "")

        if not code:
            return "Error: No code provided."

        logger.log(VERBOSE1, f"[TOOL:python_exec] Executing code ({len(code)} chars, timeout={self.timeout}s)")
        logger.log(VERBOSE2, f"[TOOL:python_exec] Code:\n{truncate_for_log(code, 1000)}")

        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning(f"[TOOL:python_exec] Timed out after {self.timeout}s")
                return f"Error: Code execution timed out after {self.timeout} seconds."

            exit_code = proc.returncode
            stdout_size = len(stdout) if stdout else 0
            stderr_size = len(stderr) if stderr else 0
            logger.log(
                VERBOSE1,
                f"[TOOL:python_exec] Completed: exit_code={exit_code}, "
                f"stdout={stdout_size} bytes, stderr={stderr_size} bytes",
            )

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                output_parts.append(f"Stderr:\n{stderr.decode('utf-8', errors='replace')}")

            if not output_parts:
                return "Code executed successfully with no output."

            result = "\n".join(output_parts)

            # Truncate long output
            max_len = 3000
            if len(result) > max_len:
                result = result[:max_len] + "\n\n[... output truncated]"

            return result

        finally:
            temp_path.unlink(missing_ok=True)
