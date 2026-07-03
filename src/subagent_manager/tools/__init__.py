"""
Built-in tools for subagents.

Provides web search, URL reading, Python execution, and file reading
tools that work out of the box with no API keys required.
"""

from subagent_manager.tools.base import BaseTool
from subagent_manager.tools.file_reader import FileReaderTool
from subagent_manager.tools.python_exec import PythonExecTool
from subagent_manager.tools.url_reader import URLReaderTool
from subagent_manager.tools.web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "WebSearchTool",
    "URLReaderTool",
    "PythonExecTool",
    "FileReaderTool",
]
