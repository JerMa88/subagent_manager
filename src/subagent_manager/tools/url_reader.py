"""
URL reader tool.

Fetches web pages and extracts readable content by converting
HTML to markdown. Handles timeouts and content truncation.
"""

from __future__ import annotations

import logging
from typing import Any

from subagent_manager.tools.base import BaseTool, ToolParameter

logger = logging.getLogger(__name__)


class URLReaderTool(BaseTool):
    """
    Read the content of a web page.

    Fetches the URL, strips HTML, and returns readable markdown
    content. Useful for reading articles, documentation, and
    other web content discovered via web search.
    """

    name = "read_url"
    description = (
        "Fetch and read the content of a web page URL. Returns the page content "
        "as readable text. Use this after web_search to read full articles or pages."
    )
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="The full URL to read (must start with http:// or https://).",
        ),
    ]

    def __init__(self, max_content_length: int = 6000) -> None:
        """
        Initialize the URL reader.

        Args:
            max_content_length: Maximum characters to return. Content is
                truncated beyond this to prevent context explosion.
        """
        self.max_content_length = max_content_length

    async def execute(self, **kwargs: Any) -> str:
        """Fetch a URL and return its content as markdown."""
        url = kwargs.get("url", "")

        if not url:
            return "Error: No URL provided."

        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"

        try:
            import httpx
        except ImportError:
            return "Error: httpx is not installed. Install it with: pip install httpx"

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=15.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; SubAgentManager/0.1; "
                        "+https://github.com/JerMa88/subagent_manager)"
                    ),
                },
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                return self._parse_html(response.text, url)
            elif "text/plain" in content_type or "application/json" in content_type:
                text = response.text[: self.max_content_length]
                if len(response.text) > self.max_content_length:
                    text += "\n\n[... content truncated]"
                return f"Content from {url}:\n\n{text}"
            else:
                return f"Unsupported content type: {content_type} at {url}"

        except httpx.TimeoutException:
            return f"Error: Request to {url} timed out after 15 seconds."
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} for {url}"
        except Exception as e:
            logger.warning(f"URL reader failed for {url}: {e}")
            return f"Error reading URL {url}: {str(e)}"

    def _parse_html(self, html: str, url: str) -> str:
        """Parse HTML and return markdown content."""
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify as md
        except ImportError:
            return (
                "Error: beautifulsoup4 and markdownify are required. "
                "Install with: pip install beautifulsoup4 markdownify"
            )

        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Try to find main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main is None:
            main = soup

        # Convert to markdown
        text = md(str(main), heading_style="ATX", strip=["img"])

        # Clean up excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not prev_blank:
                    cleaned_lines.append("")
                prev_blank = True
            else:
                cleaned_lines.append(stripped)
                prev_blank = False

        content = "\n".join(cleaned_lines)

        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "\n\n[... content truncated]"

        return f"Content from {url}:\n\n{content}"
