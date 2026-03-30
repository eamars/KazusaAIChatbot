"""Tests for MCP client — McpManager and ToolInfo."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bot.mcp_client import McpManager, ToolInfo


@dataclass
class TextBlock:
    """Mimics mcp.types.TextContent returned by a real MCP server."""

    type: str = "text"
    text: str = ""


@dataclass
class CallToolResult:
    """Mimics mcp.types.CallToolResult returned by a real MCP server."""

    content: list[TextBlock]


class FakeSession:
    """Minimal stand-in for mcp.ClientSession that records call_tool invocations."""

    def __init__(self, response_text: str = "ok"):
        self.calls: list[tuple[str, dict]] = []
        self._response_text = response_text

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.calls.append((name, arguments))
        return CallToolResult(content=[TextBlock(text=self._response_text)])


@pytest.mark.asyncio
async def test_call_tool_sends_original_name_not_namespaced():
    """Regression: MCP server receives original tool name, not the namespaced one.

    Previously call_tool sent 'mcp-searxng__searxng_web_search' to the
    server, which rejected it as unknown.  The server only knows the
    original name 'searxng_web_search'.
    """
    session = FakeSession(response_text="search results here")

    manager = McpManager()
    manager._sessions["mcp-searxng"] = session
    manager._tools["mcp-searxng__searxng_web_search"] = ToolInfo(
        name="mcp-searxng__searxng_web_search",
        original_name="searxng_web_search",
        server="mcp-searxng",
        description="Search the web",
        parameters={"properties": {"query": {"type": "string"}}},
    )

    result = await manager.call_tool(
        "mcp-searxng__searxng_web_search",
        {"query": "一之濑明日奈 萌娘百科"},
    )

    assert result == "search results here"
    assert len(session.calls) == 1
    # The critical assertion: server receives the ORIGINAL name
    assert session.calls[0][0] == "searxng_web_search"
    assert session.calls[0][1] == {"query": "一之濑明日奈 萌娘百科"}


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_returns_error():
    """Calling a tool that doesn't exist returns an error string."""
    manager = McpManager()
    result = await manager.call_tool("nonexistent__tool", {})
    assert result.startswith("Error: unknown tool")


@pytest.mark.asyncio
async def test_call_tool_disconnected_server_returns_error():
    """Tool exists but its server session is gone."""
    manager = McpManager()
    manager._tools["srv__do_thing"] = ToolInfo(
        name="srv__do_thing",
        original_name="do_thing",
        server="srv",
        description="",
        parameters={},
    )
    # No session registered for "srv"
    result = await manager.call_tool("srv__do_thing", {})
    assert "not connected" in result


@pytest.mark.asyncio
async def test_call_tool_concatenates_multiple_text_blocks():
    """Tool returning multiple text blocks should be joined with newlines."""
    session = FakeSession()
    # Override with multi-block response
    original_call = session.call_tool

    async def multi_block_call(name, arguments):
        return CallToolResult(content=[
            TextBlock(text="line 1"),
            TextBlock(text="line 2"),
            TextBlock(text="line 3"),
        ])

    session.call_tool = multi_block_call

    manager = McpManager()
    manager._sessions["s"] = session
    manager._tools["s__t"] = ToolInfo(
        name="s__t",
        original_name="t",
        server="s",
        description="",
        parameters={},
    )

    result = await manager.call_tool("s__t", {})
    assert result == "line 1\nline 2\nline 3"


def test_list_tools_returns_all_registered():
    manager = McpManager()
    manager._tools["a__x"] = ToolInfo(
        name="a__x", original_name="x", server="a", description="", parameters={},
    )
    manager._tools["b__y"] = ToolInfo(
        name="b__y", original_name="y", server="b", description="", parameters={},
    )
    tools = manager.list_tools()
    assert len(tools) == 2
    assert {t.name for t in tools} == {"a__x", "b__y"}


def test_get_tool_by_namespaced_name():
    manager = McpManager()
    info = ToolInfo(
        name="srv__func", original_name="func", server="srv",
        description="desc", parameters={},
    )
    manager._tools["srv__func"] = info
    assert manager.get_tool("srv__func") is info
    assert manager.get_tool("func") is None
