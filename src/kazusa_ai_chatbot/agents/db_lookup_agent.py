from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import BaseAgent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_TOOL_ITERATIONS
from kazusa_ai_chatbot.db import (
    get_affinity,
    get_character_state,
    get_conversation_history,
    get_user_facts,
    search_conversation_history,
    search_users_by_facts,
)
from kazusa_ai_chatbot.state import AgentResult, BotState, ToolCall

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


async def _tool_get_conversation_history(args: dict[str, Any], state: BotState) -> str:
    channel_id = str(args.get("channel_id") or state.get("channel_id") or "").strip()
    if not channel_id:
        raise ValueError("channel_id is required")
    limit = max(1, min(int(args.get("limit", 5)), 20))
    docs = await get_conversation_history(channel_id=channel_id, limit=limit)
    payload = [
        {
            "user_id": doc.get("user_id"),
            "name": doc.get("name"),
            "role": doc.get("role"),
            "content": doc.get("content"),
            "timestamp": doc.get("timestamp"),
        }
        for doc in docs
    ]
    return json.dumps(payload, ensure_ascii=False)


async def _tool_search_conversation_history(args: dict[str, Any], state: BotState) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    channel_id = str(args.get("channel_id") or state.get("channel_id") or "").strip() or None
    user_id = str(args.get("user_id") or state.get("user_id") or "").strip() or None
    limit = max(1, min(int(args.get("limit", 5)), 10))
    method = str(args.get("method", "keyword")).strip().lower() or "keyword"
    if method not in {"keyword", "vector"}:
        method = "keyword"

    rows = await search_conversation_history(
        query=query,
        channel_id=channel_id,
        user_id=user_id,
        limit=limit,
        method=method,
    )
    payload = [
        {
            "score": score,
            "user_id": doc.get("user_id"),
            "name": doc.get("name"),
            "role": doc.get("role"),
            "content": doc.get("content"),
            "timestamp": doc.get("timestamp"),
        }
        for score, doc in rows
    ]
    return json.dumps(payload, ensure_ascii=False)


async def _tool_get_user_facts(args: dict[str, Any], state: BotState) -> str:
    user_id = str(args.get("user_id") or state.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("user_id is required")
    facts = await get_user_facts(user_id)
    return json.dumps({"user_id": user_id, "facts": facts}, ensure_ascii=False)


async def _tool_get_affinity(args: dict[str, Any], state: BotState) -> str:
    user_id = str(args.get("user_id") or state.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("user_id is required")
    affinity = await get_affinity(user_id)
    return json.dumps({"user_id": user_id, "affinity": affinity}, ensure_ascii=False)


async def _tool_get_character_state(args: dict[str, Any], state: BotState) -> str:
    character_state = await get_character_state()
    return json.dumps(character_state, ensure_ascii=False)


async def _tool_search_users_by_facts(args: dict[str, Any], state: BotState) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(args.get("limit", 5)), 10))
    rows = await search_users_by_facts(query=query, limit=limit)
    payload = [
        {
            "score": score,
            "user_id": doc.get("user_id"),
            "facts": doc.get("facts", []),
            "affinity": doc.get("affinity"),
        }
        for score, doc in rows
    ]
    return json.dumps(payload, ensure_ascii=False)


_DB_TOOLS: dict[str, dict[str, Any]] = {
    "get_conversation_history": {
        "description": "Fetch recent messages for the current or specified channel. Good for continuity checks.",
        "parameters": {
            "channel_id": "string, optional; defaults to current channel_id from state",
            "limit": "integer, optional; defaults to 5, max 20",
        },
        "handler": _tool_get_conversation_history,
    },
    "search_conversation_history": {
        "description": "Search recent or historical messages by keyword or vector similarity.",
        "parameters": {
            "query": "string, required",
            "channel_id": "string, optional; defaults to current channel_id from state",
            "user_id": "string, optional; defaults to current user_id from state",
            "limit": "integer, optional; defaults to 5, max 10",
            "method": "string, optional; keyword or vector; default keyword",
        },
        "handler": _tool_search_conversation_history,
    },
    "get_user_facts": {
        "description": "Fetch stored facts and preferences for the current or specified user.",
        "parameters": {
            "user_id": "string, optional; defaults to current user_id from state",
        },
        "handler": _tool_get_user_facts,
    },
    "get_affinity": {
        "description": "Fetch the affinity score for the current or specified user.",
        "parameters": {
            "user_id": "string, optional; defaults to current user_id from state",
        },
        "handler": _tool_get_affinity,
    },
    "get_character_state": {
        "description": "Fetch the current global character state.",
        "parameters": {},
        "handler": _tool_get_character_state,
    },
    "search_users_by_facts": {
        "description": "Search users by remembered facts when the instruction is about identifying which user matches a description.",
        "parameters": {
            "query": "string, required",
            "limit": "integer, optional; defaults to 5, max 10",
        },
        "handler": _tool_search_users_by_facts,
    },
}


def _build_tool_block() -> str:
    lines = [
        "[Available Database Tools]",
        "To call a tool, output EXACTLY this format:",
        '<tool_call>{"name": "tool_name", "args": {"param": "value"}}</tool_call>',
        "",
    ]
    for tool_name, tool_spec in _DB_TOOLS.items():
        lines.append(f"- **{tool_name}**: {tool_spec['description']}")
        params = tool_spec.get("parameters", {})
        if params:
            parts = [f"{name}: {desc}" for name, desc in params.items()]
            lines.append(f"  Parameters: {'; '.join(parts)}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, dict) and "name" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse db tool_call JSON: %s", match.group(1))
    return None


def _strip_tool_call(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text).strip()


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


class DBLookupAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "db_lookup_agent"

    @property
    def description(self) -> str:
        return (
            "Looks up stored conversation history, remembered user facts, affinity, and character state "
            "from the local database. Use when the reply depends on prior chat continuity or user details."
        )

    async def run(
        self,
        state: BotState,
        user_query: str,
        command: str = "",
        expected_response: str = "",
    ) -> AgentResult:
        tool_history: list[ToolCall] = []
        task = command.strip() or user_query
        tool_block = _build_tool_block()

        system_prompt = (
            "You are a database lookup assistant. Your job is to inspect the local chatbot database "
            "using the available read-only tools, then provide a concise factual summary for the supervisor. "
            "Do NOT role-play. Do NOT invent data. Prefer the narrowest lookup that answers the task.\n\n"
            f"{tool_block}"
        )
        if expected_response.strip():
            system_prompt += f"\n\nExpected response: {expected_response.strip()}"
        human_payload = {
            "task": task,
            "expected_response": expected_response,
            "current_user_id": state.get("user_id", ""),
            "current_channel_id": state.get("channel_id", ""),
            "message_text": user_query,
        }
        llm_input_msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(human_payload, ensure_ascii=False)),
        ]

        logger.info(
            "Calling LLM for db lookup agent. Task: %s, Max iterations: %d",
            task[:100] + "..." if len(task) > 100 else task,
            MAX_TOOL_ITERATIONS,
        )

        try:
            llm = _get_llm()

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                logger.debug(
                    "LLM input for DB Lookup Agent (iteration %d):\n%s",
                    iteration,
                    "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in llm_input_msgs),
                )
                result = await llm.ainvoke(llm_input_msgs)
                raw_text = result.content or ""
                tool_req = _parse_tool_call(raw_text)

                if tool_req is None or iteration == MAX_TOOL_ITERATIONS:
                    summary = _strip_tool_call(raw_text) or "No relevant database results found."
                    break

                tool_name = str(tool_req.get("name") or "").strip()
                tool_args = tool_req.get("args", {})
                tool_spec = _DB_TOOLS.get(tool_name)
                if tool_spec is None:
                    tool_result = f"Unknown database tool: {tool_name}"
                else:
                    handler: Callable[[dict[str, Any], BotState], Awaitable[str]] = tool_spec["handler"]
                    tool_result = await handler(tool_args if isinstance(tool_args, dict) else {}, state)

                tool_history.append(ToolCall(
                    tool=tool_name,
                    args=tool_args if isinstance(tool_args, dict) else {},
                    result=tool_result,
                ))

                llm_input_msgs.append(AIMessage(content=raw_text))
                llm_input_msgs.append(HumanMessage(
                    content=(
                        f"[Tool result for {tool_name}]:\n{tool_result}\n\n"
                        f"Now provide a concise factual summary of the findings. {expected_response.strip()}"
                        if expected_response.strip()
                        else f"[Tool result for {tool_name}]:\n{tool_result}\n\nNow provide a concise factual summary of the findings."
                    ),
                ))
            else:
                summary = "Database lookup timed out after maximum iterations."

            return AgentResult(
                agent=self.name,
                status="success",
                summary=summary,
                tool_history=tool_history,
            )
        except Exception as exc:
            logger.exception("DBLookupAgent failed")
            return AgentResult(
                agent=self.name,
                status="error",
                summary=f"Database lookup failed: {exc}",
                tool_history=tool_history,
            )
