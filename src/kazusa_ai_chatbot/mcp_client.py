"""MCP client — connects to HTTP MCP servers, discovers tools, executes calls.

Each MCP server is a long-lived connection managed as an async context.
The module exposes a singleton ``McpManager`` that is started once at
bot startup and shut down on close.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from kazusa_ai_chatbot.config import MCP_CALL_TIMEOUT, MCP_CONNECT_TIMEOUT, MCP_SERVERS

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
class ManagedServerConnection:
    """Runtime bookkeeping for one long-lived MCP server connection.

    Args:
        name: Configured MCP server name.
        url: Streamable HTTP endpoint URL for the server.

    Returns:
        A mutable record used by ``McpManager`` to coordinate startup,
        readiness, and shutdown of the server task.
    """

    name: str
    url: str
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    error: Exception | None = None
    tool_names: list[str] = field(default_factory=list)


@dataclass
class McpManager:
    """Manages connections to all configured MCP servers.

    Lifecycle:
        await manager.start()    # connect & discover tools
        await manager.call(...)  # execute a tool
        await manager.stop()     # tear down connections
    """

    _sessions: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[str, ToolInfo] = field(default_factory=dict)
    _server_connections: dict[str, ManagedServerConnection] = field(default_factory=dict)

    # ── Public API ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to every configured MCP server and discover tools.

        Returns immediately if the manager is already started. Each server is
        hosted inside a dedicated asyncio task so that async context entry and
        exit happen on the same task.
        """
        if self._server_connections:
            logger.debug("MCP manager already started — skipping duplicate start")
            return

        for server_name, server_cfg in MCP_SERVERS.items():
            url = server_cfg.get("url", "")
            if not url:
                logger.warning(f'MCP server {server_name} has no URL — skipping')
                continue

            connection = ManagedServerConnection(name=server_name, url=url)
            connection.task = asyncio.create_task(
                self._serve_server(connection),
                name=f"mcp-server-{server_name}",
            )
            self._server_connections[server_name] = connection

        for connection in self._server_connections.values():
            await connection.ready_event.wait()
            if connection.error is not None:
                logger.error(
                    f"Failed to connect to MCP server {connection.name} "
                    f"at {connection.url}: {connection.error}",
                    exc_info=(
                        type(connection.error),
                        connection.error,
                        connection.error.__traceback__,
                    ),
                )

        logger.info(f'MCP ready — {len(self._tools)} tool(s) from {len(self._sessions)} server(s)')
    async def stop(self) -> None:
        """Shut down all MCP connections.

        Signals each server task to exit its own async contexts, then awaits the
        tasks to finish. This keeps context teardown on the same task that
        created the session and avoids AnyIO cancel-scope ownership errors.
        """
        if not self._server_connections:
            logger.debug("MCP manager already stopped — skipping duplicate stop")
            return

        connections = list(self._server_connections.values())
        for connection in connections:
            connection.stop_event.set()

        for connection in reversed(connections):
            if connection.task is None:
                continue
            try:
                await connection.task
            except asyncio.CancelledError as exc:
                logger.debug(
                    f"MCP server task {connection.name} was cancelled "
                    f"during shutdown: {exc}"
                )
            except Exception as exc:
                logger.exception(f"Error during MCP cleanup: {exc}")

        self._sessions.clear()
        self._tools.clear()
        self._server_connections.clear()

    def list_tools(self) -> list[ToolInfo]:
        """Return all discovered tools across all servers."""
        return_value = list(self._tools.values())
        return return_value

    def get_tool(self, name: str) -> ToolInfo | None:
        """Look up a tool by its fully-qualified name."""
        return_value = self._tools.get(name)
        return return_value

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Execute a tool and return its text result."""
        tool = self._tools.get(name)
        if tool is None:
            return_value = f"Error: unknown tool '{name}'"
            return return_value

        session = self._sessions.get(tool.server)
        if session is None:
            return_value = f"Error: server '{tool.server}' not connected"
            return return_value

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool.original_name, arguments or {}),
                timeout=MCP_CALL_TIMEOUT,
            )
            # Concatenate all text content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return_value = "\n".join(parts) if parts else "(no output)"
            return return_value
        except asyncio.TimeoutError as exc:
            logger.error(
                f"Tool call {name} timed out after "
                f"{MCP_CALL_TIMEOUT:.0f}s: {exc}"
            )
            return_value = f"Error calling {name}: timed out after {MCP_CALL_TIMEOUT:.0f}s"
            return return_value
        except Exception as exc:
            logger.exception(f"Tool call {name} failed: {exc}")
            return_value = f"Error calling {name}: {exc}"
            return return_value

    # ── Internal ─────────────────────────────────────────────────────

    async def _serve_server(self, connection: ManagedServerConnection) -> None:
        """Run one MCP server connection inside its own asyncio task.

        Args:
            connection: Runtime record for the server being connected.

        Returns:
            None. The method updates manager state as the server becomes ready
            and removes that state again during shutdown.
        """
        try:
            async with streamablehttp_client(connection.url) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=MCP_CONNECT_TIMEOUT)
                    self._sessions[connection.name] = session

                    tools_result = await asyncio.wait_for(
                        session.list_tools(), timeout=MCP_CONNECT_TIMEOUT
                    )
                    for tool in tools_result.tools:
                        qualified_name = f"{connection.name}__{tool.name}"
                        connection.tool_names.append(qualified_name)
                        self._tools[qualified_name] = ToolInfo(
                            name=qualified_name,
                            original_name=tool.name,
                            server=connection.name,
                            description=tool.description or "",
                            parameters=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        )

                    connection.ready_event.set()
                    await connection.stop_event.wait()
        except asyncio.TimeoutError as exc:
            connection.error = asyncio.TimeoutError(
                f"MCP server {connection.name!r} did not respond within {MCP_CONNECT_TIMEOUT:.0f}s"
            )
            logger.error(
                f"MCP server {connection.name} timed out during "
                f"connect/discovery (limit {MCP_CONNECT_TIMEOUT:.0f}s): {exc}"
            )
        except Exception as exc:
            connection.error = exc
            if connection.stop_event.is_set():
                logger.debug(
                    f"MCP server {connection.name} exited during shutdown: {exc}"
                )
            else:
                logger.exception(
                    f"MCP server {connection.name} connection loop failed: {exc}"
                )
        finally:
            for tool_name in connection.tool_names:
                self._tools.pop(tool_name, None)
            self._sessions.pop(connection.name, None)
            if not connection.ready_event.is_set():
                connection.ready_event.set()


# Singleton instance
mcp_manager = McpManager()
