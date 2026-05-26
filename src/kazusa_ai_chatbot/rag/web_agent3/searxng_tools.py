"""SearXNG-backed web tools used by web_agent3."""

from __future__ import annotations

from langchain_core.tools import tool

from kazusa_ai_chatbot.mcp_client import mcp_manager


@tool
async def web_search(
    query: str,
    pageno: int = 1,
    time_range: str = "",
    language: str = "",
) -> str:
    """Perform a web search through the configured SearXNG MCP facility.

    Args:
        query: Search query string.
        pageno: Search page number starting from 1.
        time_range: Optional SearXNG time range such as ``day`` or ``month``.
        language: Optional result language code.

    Returns:
        Raw text returned by the configured SearXNG search tool.
    """
    return_value = await mcp_manager.call_tool("mcp-searxng__searxng_web_search", {
        "query": query,
        "pageno": pageno,
        "time_range": time_range,
        "language": language,
        "safesearch": 0,
    })
    return return_value


@tool
async def web_url_read(
    url: str,
    startChar: int = 0,
    maxLength: int = 10000,
    section: str = "",
    paragraphRange: str = "",
    readHeadings: bool = False,
) -> str:
    """Read URL content through the configured SearXNG MCP facility.

    Args:
        url: Complete HTTP(S) URL to read.
        startChar: Starting character offset.
        maxLength: Maximum returned character count; 0 means no explicit cap.
        section: Optional heading text for section-scoped reads.
        paragraphRange: Optional paragraph range such as ``1-5``.
        readHeadings: Whether to return headings rather than page text.

    Returns:
        Raw text returned by the configured URL reader tool.
    """
    args = {
        "url": url,
        "startChar": startChar,
        "section": section,
        "paragraphRange": paragraphRange,
        "readHeadings": readHeadings,
    }
    if maxLength > 0:
        args["maxLength"] = maxLength
    return_value = await mcp_manager.call_tool("mcp-searxng__web_url_read", args)
    return return_value


_ALL_TOOLS = [web_search, web_url_read]
_TOOLS_BY_NAME = {tool_item.name: tool_item for tool_item in _ALL_TOOLS}
