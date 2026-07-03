"""Tests for the tool implementations."""

from __future__ import annotations

import asyncio
import json
import pytest

from subagent_manager.tools.base import BaseTool, ToolParameter
from subagent_manager.tools.web_search import WebSearchTool
from subagent_manager.tools.url_reader import URLReaderTool
from subagent_manager.tools.python_exec import PythonExecTool
from subagent_manager.tools.file_reader import FileReaderTool


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
