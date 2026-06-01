"""LangChain web tools used by web_agent3."""

from __future__ import annotations

from langchain_core.tools import tool

from kazusa_ai_chatbot.rag.web_agent3 import direct_searxng, url_reader


@tool
async def web_search(
    query: str,
    pageno: int = 1,
    time_range: str = "",
    language: str = "",
) -> str:
    """Perform a web search through the configured direct SearXNG endpoint.

    Args:
        query: Search query string.
        pageno: Search page number starting from 1.
        time_range: Optional SearXNG time range such as ``day`` or ``month``.
        language: Optional result language code.

    Returns:
        Bounded direct search output.
    """
    return_value = await direct_searxng.web_search(
        query=query,
        pageno=pageno,
        time_range=time_range,
        language=language,
    )
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
    """Read URL content through the process-local HTTP(S) reader.

    Args:
        url: Complete HTTP(S) URL to read.
        startChar: Starting character offset.
        maxLength: Maximum returned character count; 0 uses the config cap.
        section: Optional heading text for section-scoped reads.
        paragraphRange: Optional paragraph range such as ``1-5``.
        readHeadings: Whether to return headings rather than page text.

    Returns:
        Bounded readable URL content.
    """
    return_value = await url_reader.web_url_read(
        url=url,
        startChar=startChar,
        maxLength=maxLength,
        section=section,
        paragraphRange=paragraphRange,
        readHeadings=readHeadings,
    )
    return return_value


_ALL_TOOLS = [web_search, web_url_read]
_TOOLS_BY_NAME = {tool_item.name: tool_item for tool_item in _ALL_TOOLS}
