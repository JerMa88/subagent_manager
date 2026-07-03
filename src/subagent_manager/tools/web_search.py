"""
Web search tool using DuckDuckGo.

No API key required — uses the duckduckgo-search library for
free, unrestricted web searches. Ideal for edge deployments.
"""

from __future__ import annotations

import logging
from typing import Any

from subagent_manager.tools.base import BaseTool, ToolParameter

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Search the web using DuckDuckGo.

    Returns a summary of top search results including titles,
    URLs, and snippets. No API key required.
    """

    name = "web_search"
    description = (
        "Search the web for information. Returns titles, URLs, and snippets "
        "from top results. Use this to find current information, facts, and data."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query. Be specific for better results.",
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results to return (default: 5, max: 10).",
            required=False,
        ),
    ]

    async def execute(self, **kwargs: Any) -> str:
        """Execute a web search and return formatted results."""
        query = kwargs.get("query", "")
        max_results = min(int(kwargs.get("max_results", 5)), 10)

        if not query:
            return "Error: No search query provided."

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return (
                "Error: duckduckgo-search is not installed. "
                "Install it with: pip install duckduckgo-search"
            )

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)

            if not results:
                return f"No results found for query: {query}"

            output_lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("href", r.get("link", "No URL"))
                snippet = r.get("body", r.get("snippet", "No description"))
                output_lines.append(f"{i}. **{title}**")
                output_lines.append(f"   URL: {url}")
                output_lines.append(f"   {snippet}\n")

            return "\n".join(output_lines)

        except Exception as e:
            logger.warning(f"Web search failed for query '{query}': {e}")
            return f"Web search failed: {str(e)}"
