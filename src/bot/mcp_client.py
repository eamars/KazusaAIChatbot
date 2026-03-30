"""MCP client — connects to HTTP MCP servers, discovers tools, executes calls.

Each MCP server is a long-lived connection managed as an async context.
The module exposes a singleton ``McpManager`` that is started once at
bot startup and shut down on close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from bot.config import MCP_SERVERS

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Flattened descriptor for a single MCP tool."""

    name: str            # namespaced: {server}__{tool}
    original_name: str   # name as registered on the MCP server
    server: str
    description: str
    parameters: dict  # JSON Schema of the tool's input


@dataclass
class McpManager:
    """Manages connections to all configured MCP servers.

    Lifecycle:
        await manager.start()    # connect & discover tools
        await manager.call(...)  # execute a tool
        await manager.stop()     # tear down connections
    """

    _sessions: dict[str, ClientSession] = field(default_factory=dict)
    _cleanup_fns: list = field(default_factory=list)
    _tools: dict[str, ToolInfo] = field(default_factory=dict)

    # ── Public API ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to every configured MCP server and discover tools."""
        for server_name, server_cfg in MCP_SERVERS.items():
            url = server_cfg.get("url", "")
            if not url:
                logger.warning("MCP server %s has no URL — skipping", server_name)
                continue
            try:
                await self._connect_server(server_name, url)
            except Exception:
                logger.exception("Failed to connect to MCP server %s at %s", server_name, url)

        logger.info(
            "MCP ready — %d tool(s) from %d server(s): %s",
            len(self._tools),
            len(self._sessions),
            ", ".join(self._tools.keys()) or "(none)",
        )

    async def stop(self) -> None:
        """Shut down all MCP connections."""
        for cleanup in reversed(self._cleanup_fns):
            try:
                await cleanup()
            except Exception:
                logger.exception("Error during MCP cleanup")
        self._sessions.clear()
        self._tools.clear()
        self._cleanup_fns.clear()

    def list_tools(self) -> list[ToolInfo]:
        """Return all discovered tools across all servers."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolInfo | None:
        """Look up a tool by its fully-qualified name."""
        return self._tools.get(name)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Execute a tool and return its text result."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"

        session = self._sessions.get(tool.server)
        if session is None:
            return f"Error: server '{tool.server}' not connected"

        try:
            result = await session.call_tool(tool.original_name, arguments or {})
            # Concatenate all text content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else "(no output)"
        except Exception as exc:
            logger.exception("Tool call %s failed", name)
            return f"Error calling {name}: {exc}"

    # ── Internal ─────────────────────────────────────────────────────

    async def _connect_server(self, server_name: str, url: str) -> None:
        """Connect to a single MCP server via streamable HTTP."""
        # streamablehttp_client is an async context manager that yields
        # (read_stream, write_stream, get_url).  We need to manually
        # enter the context and keep it alive for the session's lifetime.
        cm = streamablehttp_client(url)
        read_stream, write_stream, _ = await cm.__aenter__()
        self._cleanup_fns.append(cm.__aexit__)

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        self._cleanup_fns.append(session.__aexit__)

        await session.initialize()
        self._sessions[server_name] = session

        # Discover tools
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            qualified_name = f"{server_name}__{tool.name}"
            self._tools[qualified_name] = ToolInfo(
                name=qualified_name,
                original_name=tool.name,
                server=server_name,
                description=tool.description or "",
                parameters=tool.inputSchema if hasattr(tool, "inputSchema") else {},
            )
        logger.info(
            "Connected to MCP server %s — %d tool(s)",
            server_name,
            len(tools_result.tools),
        )


# Singleton instance
mcp_manager = McpManager()
