"""
Structured, colorized logging for the subagent_manager pipeline.

Provides graduated verbosity levels:
- Level 0 (default): Warnings and errors only
- Level 1 (verbose=1): Decision-level logging — plans, strategy choices,
  agent assignments, tool call summaries, timing, token totals
- Level 2 (verbose=2): Full prompt/response logging — every LLM message,
  complete tool arguments/results, raw JSON parsing attempts

Usage:
    from subagent_manager.logging_config import configure_logging

    configure_logging(verbosity=2, log_file="run.log")
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator


# ---------------------------------------------------------------------------
# ANSI color codes for terminal output
# ---------------------------------------------------------------------------

class _Colors:
    """ANSI escape codes for colorized terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Bright foreground
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# Map component prefixes to colors for visual differentiation
_PREFIX_COLORS: dict[str, str] = {
    "ORCHESTRATOR": _Colors.BRIGHT_CYAN,
    "PLAN": _Colors.MAGENTA,
    "STRATEGY": _Colors.BLUE,
    "AGENT": _Colors.GREEN,
    "TOOL": _Colors.YELLOW,
    "SYNTHESIS": _Colors.CYAN,
    "LLM": _Colors.BRIGHT_WHITE,
    "PARSE": _Colors.GRAY,
    "HARNESS": _Colors.BRIGHT_GREEN,
}


# ---------------------------------------------------------------------------
# Custom log levels for graduated verbosity
# ---------------------------------------------------------------------------

# Level 1: Decision-level (between INFO=20 and DEBUG=10)
VERBOSE1 = 15
# Level 2: Full prompt/response detail
VERBOSE2 = 12

logging.addLevelName(VERBOSE1, "VERBOSE1")
logging.addLevelName(VERBOSE2, "VERBOSE2")


def _verbose1(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log at verbosity level 1 (decision-level)."""
    if self.isEnabledFor(VERBOSE1):
        self._log(VERBOSE1, message, args, **kwargs)


def _verbose2(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log at verbosity level 2 (full detail)."""
    if self.isEnabledFor(VERBOSE2):
        self._log(VERBOSE2, message, args, **kwargs)


# Monkey-patch Logger class to add verbose1/verbose2 methods
logging.Logger.verbose1 = _verbose1  # type: ignore[attr-defined]
logging.Logger.verbose2 = _verbose2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class VerboseFormatter(logging.Formatter):
    """
    Color-coded, structured formatter for pipeline logging.

    Produces output like:
        10:32:15 [ORCHESTRATOR] Planning phase started for goal: "Fix the bug..."
        10:32:16 [LLM] ← Response: 1,250 tokens (950 prompt / 300 completion) in 1.2s
        10:32:16 [AGENT:researcher] Executing subtask 1: "Search for..."
        10:32:17 [TOOL:web_search] query="python bug fix" → 5 results (0.3s)
    """

    LEVEL_COLORS = {
        logging.WARNING: _Colors.YELLOW,
        logging.ERROR: _Colors.RED,
        logging.CRITICAL: _Colors.RED + _Colors.BOLD,
    }

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        ms = int((record.created % 1) * 1000)
        timestamp = f"{timestamp}.{ms:03d}"

        # Level indicator
        if record.levelno >= logging.ERROR:
            level_str = "ERROR"
        elif record.levelno >= logging.WARNING:
            level_str = "WARN "
        elif record.levelno >= logging.INFO:
            level_str = "INFO "
        elif record.levelno >= VERBOSE1:
            level_str = "  V1 "
        elif record.levelno >= VERBOSE2:
            level_str = "  V2 "
        else:
            level_str = "DEBUG"

        message = record.getMessage()

        if self.use_color:
            # Color the level
            level_color = self.LEVEL_COLORS.get(record.levelno, _Colors.DIM)
            colored_level = f"{level_color}{level_str}{_Colors.RESET}"

            # Color any [PREFIX] or [PREFIX:detail] tags in the message
            colored_message = message
            for prefix, color in _PREFIX_COLORS.items():
                # Match [PREFIX] and [PREFIX:anything]
                tag_simple = f"[{prefix}]"
                if tag_simple in colored_message:
                    colored_message = colored_message.replace(
                        tag_simple,
                        f"{color}{_Colors.BOLD}[{prefix}]{_Colors.RESET}",
                    )
                # Match [PREFIX:detail] patterns
                import re
                pattern = rf"\[{prefix}:([^\]]+)\]"
                def _colorize(m: re.Match) -> str:  # noqa: E501
                    return f"{color}{_Colors.BOLD}[{prefix}:{m.group(1)}]{_Colors.RESET}"
                colored_message = re.sub(pattern, _colorize, colored_message)

            return f"{_Colors.DIM}{timestamp}{_Colors.RESET} {colored_level} {colored_message}"
        else:
            return f"{timestamp} {level_str} {message}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Noisy libraries to always suppress
_NOISY_LOGGERS = [
    "httpcore", "httpx", "LiteLLM", "litellm",
    "primp", "hickory_net", "hickory_resolver",
    "h2", "hyper_util", "rustls", "urllib3",
    "asyncio", "hpack",
]


def configure_logging(
    verbosity: int = 0,
    log_file: str | None = None,
) -> None:
    """
    Configure the subagent_manager logging hierarchy.

    Args:
        verbosity: Logging verbosity level.
            0 = WARNING only (quiet, default)
            1 = Decision-level logging (plans, strategies, agent assignments,
                timing, token summaries)
            2 = Full detail (complete LLM prompts/responses, tool arguments,
                raw JSON parsing, full tracebacks)
        log_file: Optional path to write logs to a file (always uncolored).
    """
    # Determine the effective log level
    if verbosity >= 2:
        level = VERBOSE2
    elif verbosity >= 1:
        level = VERBOSE1
    else:
        level = logging.WARNING

    # Configure the root subagent_manager logger
    pkg_logger = logging.getLogger("subagent_manager")
    pkg_logger.setLevel(level)
    pkg_logger.handlers.clear()
    pkg_logger.propagate = False

    # Console handler (colorized)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(VerboseFormatter(use_color=True))
    pkg_logger.addHandler(console_handler)

    # File handler (uncolored)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(VerboseFormatter(use_color=False))
        pkg_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

@contextmanager
def log_phase(
    logger: logging.Logger,
    phase_name: str,
    prefix: str = "",
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager that logs the start and end of a phase with timing.

    Usage:
        with log_phase(logger, "Planning", "[ORCHESTRATOR]") as ctx:
            # do work
        # automatically logs: "[ORCHESTRATOR] Planning completed in 1.23s"

    The yielded dict can be used to attach metadata:
        with log_phase(logger, "Planning", "[ORCHESTRATOR]") as ctx:
            ctx["tokens"] = 1234
        # logs: "[ORCHESTRATOR] Planning completed in 1.23s (tokens=1234)"
    """
    meta: dict[str, Any] = {}
    tag = f"{prefix} " if prefix else ""
    logger.log(VERBOSE1, f"{tag}{phase_name} started")
    t0 = time.monotonic()
    try:
        yield meta
    finally:
        elapsed = time.monotonic() - t0
        meta_str = ""
        if meta:
            parts = [f"{k}={v}" for k, v in meta.items()]
            meta_str = f" ({', '.join(parts)})"
        logger.log(VERBOSE1, f"{tag}{phase_name} completed in {elapsed:.2f}s{meta_str}")


def format_tokens(usage: dict[str, int]) -> str:
    """Format token usage into a readable string."""
    total = usage.get("total_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return f"{total:,} tokens ({prompt:,} prompt / {completion:,} completion)"


def truncate_for_log(text: str, max_len: int = 500) -> str:
    """Truncate text for logging, indicating if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text) - max_len} more chars]"
