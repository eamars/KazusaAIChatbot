"""Web Search Agent — searches the internet via MCP search tools.

Runs in its own LLM context with only the user query and search tool
descriptions.  Executes the tool-calling loop, then summarises the raw
search results into a concise paragraph for the speech agent.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import BaseAgent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_TOOL_ITERATIONS
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import AgentResult, BotState, ToolCall

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Only expose search-related MCP tools to this agent
_SEARCH_TOOL_PREFIXES = ("mcp-searxng__",)


def _is_search_tool(tool_name: str) -> bool:
    """Return True if the tool belongs to this agent's domain."""
    return any(tool_name.startswith(p) for p in _SEARCH_TOOL_PREFIXES)


def _build_tool_block() -> str:
    """Build a prompt block describing only the search tools."""
    tools = [t for t in mcp_manager.list_tools() if _is_search_tool(t.name)]
    if not tools:
        return ""

    lines = [
        "[Available Tools]",
        "To call a tool, output EXACTLY this format:",
        '<tool_call>{"name": "tool_name", "args": {"param": "value"}}</tool_call>',
        "",
    ]
    for tool in tools:
        lines.append(f"- **{tool.name}**: {tool.description}")
        props = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        if props:
            parts = []
            for pname, spec in props.items():
                ptype = spec.get("type", "any")
                desc = spec.get("description", "")
                req = " (required)" if pname in required else ""
                parts.append(f"{pname}: {ptype}{req} — {desc}" if desc else f"{pname}: {ptype}{req}")
            lines.append(f"  Parameters: {'; '.join(parts)}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> dict | None:
    """Extract the first <tool_call> JSON from LLM output."""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, dict) and "name" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse tool_call JSON: %s", match.group(1))
    return None


def _strip_tool_call(text: str) -> str:
    """Remove <tool_call> blocks from text."""
    return _TOOL_CALL_RE.sub("", text).strip()


class WebSearchAgent(BaseAgent):
    """Agent that searches the web and summarises results."""

    @property
    def name(self) -> str:
        return "web_search_agent"

    @property
    def description(self) -> str:
        return (
            "Searches the internet for real-time information using web search tools. "
            "Use when the user asks about current events, weather, news, or anything "
            "that requires up-to-date information not in the bot's knowledge base."
        )

    async def run(self, state: BotState, user_query: str) -> AgentResult:
        """Execute the search tool loop and return a summarised result."""
        tool_history: list[ToolCall] = []

        tool_block = _build_tool_block()
        if not tool_block:
            return AgentResult(
                agent=self.name,
                status="error",
                summary="No search tools available.",
                tool_history=[],
            )

        system_prompt = (
            "You are a web search assistant. Your job is to search the internet "
            "to answer the user's query, then provide a concise factual summary "
            "of what you found. Do NOT role-play or add personality.\n\n"
            f"{tool_block}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Search query: {user_query}"),
        ]

        # Fold system into first human for Qwen compatibility
        messages = _fold_system(messages)

        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=0.3,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
            )

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                result = await llm.ainvoke(messages)
                raw_text = result.content or ""

                tool_req = _parse_tool_call(raw_text)
                if tool_req is None or iteration == MAX_TOOL_ITERATIONS:
                    summary = _strip_tool_call(raw_text) or "No results found."
                    break

                tool_name = tool_req["name"]
                tool_args = tool_req.get("args", {})
                logger.info(
                    "WebSearchAgent tool call [%d/%d]: %s(%s)",
                    iteration + 1, MAX_TOOL_ITERATIONS, tool_name, tool_args,
                )

                tool_result = await mcp_manager.call_tool(tool_name, tool_args)

                tool_history.append(ToolCall(
                    tool=tool_name,
                    args=tool_args,
                    result=tool_result,
                ))

                messages.append(AIMessage(content=raw_text))
                messages.append(HumanMessage(
                    content=f"[Tool result for {tool_name}]:\n{tool_result}\n\n"
                    "Now provide a concise factual summary of the results.",
                ))
            else:
                summary = "Search timed out after maximum iterations."

            return AgentResult(
                agent=self.name,
                status="success",
                summary=summary,
                tool_history=tool_history,
            )

        except Exception as exc:
            logger.exception("WebSearchAgent failed")
            return AgentResult(
                agent=self.name,
                status="error",
                summary=f"Web search failed: {exc}",
                tool_history=tool_history,
            )


def _fold_system(messages: list) -> list:
    """Fold SystemMessage into the first HumanMessage for Qwen compatibility."""
    system_parts = []
    other = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        else:
            other.append(msg)

    if not system_parts or not other:
        return other or messages

    system_text = "\n\n".join(system_parts)
    first = other[0]
    other[0] = HumanMessage(content=f"{system_text}\n\n---\n\n{first.content}")
    return other
