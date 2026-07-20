"""Tests for the tool implementations."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Ensure bench package is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subagent_manager.tools.base import BaseTool, ToolParameter
from subagent_manager.tools.web_search import WebSearchTool
from subagent_manager.tools.url_reader import URLReaderTool
from subagent_manager.tools.python_exec import PythonExecTool
from subagent_manager.tools.file_reader import FileReaderTool
from bench.swe_bench_tools import StrReplaceTool, ViewFileTool


class TestBaseTool:
    """Tests for the BaseTool abstract class."""

    def test_openai_schema_generation(self):
        """Test that tools generate valid OpenAI function schemas."""
        tool = WebSearchTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "web_search"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_parse_arguments_dict(self):
        """Test parsing arguments from a dict."""
        tool = WebSearchTool()
        args = tool.parse_arguments({"query": "test", "max_results": 3})
        assert args["query"] == "test"
        assert args["max_results"] == 3

    def test_parse_arguments_json_string(self):
        """Test parsing arguments from a JSON string."""
        tool = WebSearchTool()
        args = tool.parse_arguments('{"query": "test query"}')
        assert args["query"] == "test query"

    def test_parse_arguments_plain_string_single_param(self):
        """Test parsing a plain string for a single-param tool."""

        class SingleParamTool(BaseTool):
            name = "test"
            description = "test"
            parameters = [
                ToolParameter(
                    name="input", type="string", description="test"
                )
            ]

            async def execute(self, **kwargs):
                return "ok"

        tool = SingleParamTool()
        args = tool.parse_arguments("hello world")
        assert args["input"] == "hello world"


class TestWebSearchTool:
    """Tests for the WebSearchTool."""

    def test_schema_has_required_params(self):
        tool = WebSearchTool()
        schema = tool.to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]

        assert "query" in props
        assert "query" in required
        assert "max_results" in props

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        tool = WebSearchTool()
        result = await tool.execute(query="")
        assert "Error" in result


class TestURLReaderTool:
    """Tests for the URLReaderTool."""

    def test_init_with_custom_max_length(self):
        tool = URLReaderTool(max_content_length=1000)
        assert tool.max_content_length == 1000

    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self):
        tool = URLReaderTool()
        result = await tool.execute(url="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_invalid_url_returns_error(self):
        tool = URLReaderTool()
        result = await tool.execute(url="not-a-url")
        assert "Error" in result


class TestPythonExecTool:
    """Tests for the PythonExecTool."""

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        tool = PythonExecTool(timeout=5.0)
        result = await tool.execute(code="print(2 + 2)")
        assert "4" in result

    @pytest.mark.asyncio
    async def test_empty_code_returns_error(self):
        tool = PythonExecTool()
        result = await tool.execute(code="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        tool = PythonExecTool(timeout=1.0)
        result = await tool.execute(code="import time; time.sleep(10)")
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self):
        tool = PythonExecTool(timeout=5.0)
        result = await tool.execute(code="def broken(")
        assert "Stderr" in result or "SyntaxError" in result


class TestFileReaderTool:
    """Tests for the FileReaderTool."""

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!\nLine 2\nLine 3")

        tool = FileReaderTool()
        result = await tool.execute(path=str(test_file))
        assert "Hello, world!" in result
        assert "3 lines" in result

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        tool = FileReaderTool()
        result = await tool.execute(
            path=str(test_file), start_line=2, end_line=4
        )
        assert "Line 2" in result
        assert "Line 4" in result
        assert "lines 2-4" in result

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self):
        tool = FileReaderTool()
        result = await tool.execute(path="/nonexistent/path/file.txt")
        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_path_returns_error(self):
        tool = FileReaderTool()
        result = await tool.execute(path="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_allowed_dirs_enforcement(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("secret data")

        tool = FileReaderTool(allowed_dirs=["/some/other/dir"])
        result = await tool.execute(path=str(test_file))
        assert "Access denied" in result

    @pytest.mark.asyncio
    async def test_truncation(self, tmp_path):
        test_file = tmp_path / "big.txt"
        test_file.write_text("x" * 20000)

        tool = FileReaderTool(max_content_length=100)
        result = await tool.execute(path=str(test_file))
        assert "truncated" in result.lower()


class TestStrReplaceTool:
    """Tests for the StrReplaceTool (surgical file edit)."""

    @pytest.mark.asyncio
    async def test_basic_replacement(self, tmp_path):
        """Happy path: old_str found exactly once, gets replaced."""
        f = tmp_path / "hello.py"
        f.write_text("def foo():\n    return 1\n")

        tool = StrReplaceTool()
        result = await tool.execute(
            path=str(f),
            old_str="    return 1\n",
            new_str="    return 42\n",
        )

        assert "Replaced 1 occurrence" in result
        assert "hello.py" in result
        assert f.read_text() == "def foo():\n    return 42\n"

    @pytest.mark.asyncio
    async def test_zero_matches_returns_error(self, tmp_path):
        """old_str not in file → descriptive error, file unchanged."""
        f = tmp_path / "code.py"
        original = "x = 1\n"
        f.write_text(original)

        tool = StrReplaceTool()
        result = await tool.execute(
            path=str(f),
            old_str="this string does not exist",
            new_str="y = 2\n",
        )

        assert "Error" in result
        assert "not found" in result.lower()
        # File must be unchanged
        assert f.read_text() == original

    @pytest.mark.asyncio
    async def test_multiple_matches_returns_error(self, tmp_path):
        """old_str appears twice → ambiguity error, file unchanged."""
        f = tmp_path / "dup.py"
        original = "pass\n# comment\npass\n"
        f.write_text(original)

        tool = StrReplaceTool()
        result = await tool.execute(
            path=str(f),
            old_str="pass\n",
            new_str="return\n",
        )

        assert "Error" in result
        assert "2 times" in result or "ambiguous" in result.lower()
        # File must be unchanged
        assert f.read_text() == original

    @pytest.mark.asyncio
    async def test_relative_path_resolution(self, tmp_path):
        """Relative path is resolved against working_dir correctly."""
        sub = tmp_path / "src"
        sub.mkdir()
        f = sub / "mod.py"
        f.write_text("value = 0\n")

        tool = StrReplaceTool(working_dir=str(tmp_path))
        result = await tool.execute(
            path="src/mod.py",
            old_str="value = 0\n",
            new_str="value = 99\n",
        )

        assert "Replaced 1 occurrence" in result
        assert f.read_text() == "value = 99\n"

    @pytest.mark.asyncio
    async def test_no_path_returns_error(self):
        """Missing path argument → error."""
        tool = StrReplaceTool()
        result = await tool.execute(old_str="x", new_str="y")
        assert "Error" in result
        assert "path" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_old_str_returns_error(self, tmp_path):
        """Empty old_str is rejected before file access."""
        f = tmp_path / "file.py"
        f.write_text("content\n")

        tool = StrReplaceTool()
        result = await tool.execute(path=str(f), old_str="", new_str="replacement")
        assert "Error" in result
        assert "empty" in result.lower() or "old_str" in result.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self, tmp_path):
        """File that doesn't exist → clean error message."""
        tool = StrReplaceTool()
        result = await tool.execute(
            path=str(tmp_path / "nonexistent.py"),
            old_str="x",
            new_str="y",
        )
        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_multiline_replacement(self, tmp_path):
        """Multi-line old_str is replaced correctly."""
        f = tmp_path / "multi.py"
        f.write_text("def bad():\n    x = 1\n    return x\n")

        tool = StrReplaceTool()
        result = await tool.execute(
            path=str(f),
            old_str="    x = 1\n    return x\n",
            new_str="    return 42\n",
        )

        assert "Replaced 1 occurrence" in result
        assert f.read_text() == "def bad():\n    return 42\n"


class TestViewFileTool:
    """Tests for the ViewFileTool (file viewer with line-range support)."""

    @pytest.mark.asyncio
    async def test_view_full_file(self, tmp_path):
        """Reading without line range returns all lines with numbers."""
        f = tmp_path / "full.py"
        f.write_text("line1\nline2\nline3\n")

        tool = ViewFileTool()
        result = await tool.execute(path=str(f))

        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        # Line numbers should be present
        assert "1 |" in result or "    1 |" in result

    @pytest.mark.asyncio
    async def test_view_line_range(self, tmp_path):
        """start_line/end_line slices correctly."""
        lines = [f"line{i}" for i in range(1, 11)]
        f = tmp_path / "range.py"
        f.write_text("\n".join(lines) + "\n")

        tool = ViewFileTool()
        result = await tool.execute(path=str(f), start_line=3, end_line=5)

        assert "line3" in result
        assert "line4" in result
        assert "line5" in result
        assert "line1" not in result
        assert "line9" not in result

    @pytest.mark.asyncio
    async def test_view_shows_total_lines(self, tmp_path):
        """Header reports total line count."""
        f = tmp_path / "info.py"
        f.write_text("a\nb\nc\n")

        tool = ViewFileTool()
        result = await tool.execute(path=str(f))

        # Total line count should appear in the header
        assert "3" in result

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self, tmp_path):
        """Missing file → error without crashing."""
        tool = ViewFileTool()
        result = await tool.execute(path=str(tmp_path / "ghost.py"))

        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_no_path_returns_error(self):
        """Missing path argument → error."""
        tool = ViewFileTool()
        result = await tool.execute()
        assert "Error" in result
        assert "path" in result.lower()

    @pytest.mark.asyncio
    async def test_relative_path_resolution(self, tmp_path):
        """Relative path is resolved against working_dir."""
        sub = tmp_path / "pkg"
        sub.mkdir()
        f = sub / "mod.py"
        f.write_text("hello\n")

        tool = ViewFileTool(working_dir=str(tmp_path))
        result = await tool.execute(path="pkg/mod.py")

        assert "hello" in result

    @pytest.mark.asyncio
    async def test_start_line_beyond_file_returns_error(self, tmp_path):
        """start_line past EOF → clean error."""
        f = tmp_path / "short.py"
        f.write_text("a\nb\n")

        tool = ViewFileTool()
        result = await tool.execute(path=str(f), start_line=999)

        assert "Error" in result

