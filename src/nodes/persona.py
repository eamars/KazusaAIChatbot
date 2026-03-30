"""Stage 6 — Persona Agent (LLM supervisor).

Generates an in-character reply, optionally calling MCP tools.

The supervisor loop:
  1. Call the LLM with the assembled messages.
  2. If the response contains a ``<tool_call>`` tag, parse and execute it.
  3. Append the tool result and re-invoke the LLM (up to MAX_TOOL_ITERATIONS).
  4. Return the final text response.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_TOOL_ITERATIONS,
)
from mcp_client import mcp_manager
from state import BotState, ToolCall

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


def _prepare_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Fold SystemMessage content into the first HumanMessage.

    Many local models (e.g. Qwen via LM Studio) have Jinja templates
    that reject or mishandle the ``system`` role.  Merging it into the
    first HumanMessage keeps compatibility with every model template.
    """
    system_parts: list[str] = []
    other: list[BaseMessage] = []
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


def _parse_tool_call(text: str) -> dict | None:
    """Extract the first <tool_call> JSON from the LLM output.

    Returns ``{"name": ..., "args": ...}`` or None.
    """
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
    """Remove <tool_call> blocks from the response text."""
    return _TOOL_CALL_RE.sub("", text).strip()


async def persona_agent(state: BotState) -> BotState:
    """Generate an in-character reply, optionally calling tools."""
    messages: list[BaseMessage] = list(state.get("llm_messages", []))
    if not messages:
        return {**state, "response": "...", "tool_history": []}

    tool_history: list[ToolCall] = []

    try:
        llm = _get_llm()
        prepared = _prepare_messages(messages)

        for iteration in range(MAX_TOOL_ITERATIONS + 1):
            result = await llm.ainvoke(prepared)
            raw_text = result.content or ""

            # Check for a tool call
            tool_req = _parse_tool_call(raw_text)
            if tool_req is None or iteration == MAX_TOOL_ITERATIONS:
                # No tool call or max iterations reached — return final response
                response = _strip_tool_call(raw_text) or "..."
                break

            # Execute the tool
            tool_name = tool_req["name"]
            tool_args = tool_req.get("args", {})
            logger.info("Tool call [%d/%d]: %s(%s)", iteration + 1, MAX_TOOL_ITERATIONS, tool_name, tool_args)

            tool_result = await mcp_manager.call_tool(tool_name, tool_args)

            tool_history.append(ToolCall(
                tool=tool_name,
                args=tool_args,
                result=tool_result,
            ))

            # Append the exchange to the conversation for the next iteration
            prepared.append(AIMessage(content=raw_text))
            prepared.append(HumanMessage(
                content=f"[Tool result for {tool_name}]:\n{tool_result}"
            ))
        else:
            response = "..."

    except Exception:
        logger.exception("Persona agent LLM call failed")
        response = "*stays silent*"

    return {**state, "response": response, "tool_history": tool_history}
