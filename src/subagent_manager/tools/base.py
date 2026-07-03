"""
Base tool abstraction.

All tools inherit from BaseTool and implement the `execute` method.
Tools auto-generate OpenAI-compatible function schemas from their
Pydantic model definitions.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Definition of a single tool parameter."""

    name: str
    type: str  # "string", "integer", "boolean", "number"
    description: str
    required: bool = True
    enum: list[str] | None = None


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Subclasses must define `name`, `description`, `parameters`,
    and implement the `execute` method.

    The tool auto-generates an OpenAI-compatible function schema
    that LiteLLM can use with any provider.
    """

    name: str
    description: str
    parameters: list[ToolParameter]

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert this tool to an OpenAI function-calling schema.

        This format is understood by LiteLLM and works with all
        supported providers (OpenAI, Anthropic, Gemini, Ollama, etc.).

        Returns:
            Dict in OpenAI tool format.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments matching the parameter definitions.

        Returns:
            A string result that will be passed back to the LLM.
        """
        ...

    async def safe_execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with error handling.

        Returns:
            The tool result, or an error message if execution fails.
        """
        try:
            result = await self.execute(**kwargs)
            # Truncate very long results to prevent context explosion
            max_len = 4000
            if len(result) > max_len:
                result = result[:max_len] + "\n\n[... truncated — result too long]"
            return result
        except Exception as e:
            logger.warning(f"Tool {self.name} failed: {e}")
            return f"Error executing {self.name}: {str(e)}"

    def parse_arguments(self, arguments: str | dict[str, Any]) -> dict[str, Any]:
        """
        Parse tool arguments from LLM output.

        Handles both string (JSON) and dict formats since different
        providers return tool arguments differently.

        Args:
            arguments: Either a JSON string or a dict of arguments.

        Returns:
            Parsed dict of arguments.
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                # Some models return plain strings for single-param tools
                if len(self.parameters) == 1:
                    return {self.parameters[0].name: arguments}
                raise ValueError(f"Could not parse arguments for {self.name}: {arguments}")
        return arguments

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
