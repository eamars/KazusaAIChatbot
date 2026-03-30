"""Tool registry — builds prompt-friendly tool descriptions from MCP tools.

The persona supervisor injects these descriptions into the system prompt
so the LLM knows what tools are available and how to call them.
"""

from __future__ import annotations

from mcp_client import McpManager, ToolInfo


def build_tool_prompt_block(manager: McpManager) -> str:
    """Build a prompt block describing all available tools.

    Returns an empty string if no tools are available.
    """
    tools = manager.list_tools()
    if not tools:
        return ""

    lines = [
        "[Available Tools]",
        "You may call tools when you need real-time information or to perform actions.",
        "To call a tool, output EXACTLY this format on its own line:",
        "",
        '<tool_call>{"name": "tool_name", "args": {"param": "value"}}</tool_call>',
        "",
        "After a tool result is returned to you, use it to formulate your in-character response.",
        "Do NOT call a tool if you can answer from your own knowledge.",
        "Only call ONE tool at a time.",
        "",
    ]

    for tool in tools:
        lines.append(f"- **{tool.name}**: {tool.description}")
        params = _format_params(tool)
        if params:
            lines.append(f"  Parameters: {params}")

    return "\n".join(lines)


def _format_params(tool: ToolInfo) -> str:
    """Format a tool's JSON Schema parameters into a compact string."""
    schema = tool.parameters
    if not schema:
        return "(none)"

    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return "(none)"

    parts = []
    for name, spec in props.items():
        ptype = spec.get("type", "any")
        desc = spec.get("description", "")
        req = " (required)" if name in required else ""
        parts.append(f"{name}: {ptype}{req} — {desc}" if desc else f"{name}: {ptype}{req}")

    return "; ".join(parts)
