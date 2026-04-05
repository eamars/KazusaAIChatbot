from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import BaseAgent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_TOOL_ITERATIONS
from kazusa_ai_chatbot.db import get_conversation_history as get_conversation_history_db, search_conversation_history as search_conversation_history_db
from kazusa_ai_chatbot.state import AgentResult, BotState, ToolCall

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_CONVERSATION_HISTORY_SYSTEM_PROMPT = """\
You are a conversation history lookup assistant. Your job is to inspect only stored past chat history using the available read-only conversation history tools, then provide a concise factual summary for the supervisor.
- Do NOT invent data.
- Prefer the narrowest history lookup that answers the task.
- Use `success` when the history answer is complete, `needs_context` when relevant history could not be found or resolved, and `needs_clarification` when the user must specify what they meant more clearly.

You have access to conversation history search tools. Use them to find relevant past messages.

Output Format (raw JSON text — no markdown wrapping):
{{
  "status": "success|needs_context|needs_clarification",
  "summary": "Key facts from your conversation history lookup based on supervisor instruction."
}}
"""


@tool
async def get_conversation_history(
    channel_id: str = "",
    limit: int = 5
) -> str:
    """Fetch recent messages for the current or specified channel when continuity depends on prior chat turns.
    
    Args:
        channel_id: Channel ID to fetch history from (optional, defaults to current channel)
        limit: Maximum number of messages to return, defaults to 5, max 20 (optional)
    """
    if not channel_id:
        raise ValueError("channel_id is required")
    limit = max(1, min(int(limit), 20))
    
    docs = await get_conversation_history_db(channel_id=channel_id, limit=limit)
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


@tool
async def search_conversation_history(
    query: str,
    channel_id: str = "",
    user_id: str = "",
    limit: int = 5,
    method: str = "vector"
) -> str:
    """Search past chat history by keyword or vector similarity when the task is to remember what was said before.
    
    Args:
        query: Search query to find relevant messages (required)
        channel_id: Channel ID to search in (optional, defaults to current channel)
        user_id: User ID to filter by (optional, defaults to current user)
        limit: Maximum number of results to return, defaults to 5, max 10 (optional)
        method: Search method - 'vector' for semantic search or 'keyword' for text search, defaults to 'vector' (optional)
    """
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(limit), 10))
    if method not in {"keyword", "vector"}:
        method = "vector"

    rows = await search_conversation_history_db(
        query=query,
        channel_id=channel_id or None,
        user_id=user_id or None,
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


def _get_langchain_tools() -> list:
    """Get LangChain tools for this agent."""
    # Note: get_conversation_history is commented out to match the original tool config
    return [search_conversation_history]


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


def _get_llm_with_tools() -> ChatOpenAI:
    """Get LLM with tools bound."""
    llm = _get_llm()
    tools = _get_langchain_tools()
    return llm.bind_tools(tools)


def _parse_final_result(text: str, default_summary: str) -> AgentResult:
    """Parse the final JSON result from the LLM."""
    if not text.strip():
        return AgentResult(status="needs_context", summary=default_summary)
    try:
        # Remove white space
        text = text.strip()

        # Remove JSON markdown guard
        text = text.strip("```").strip("json")
        
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            status = str(parsed.get("status") or "").strip()
            summary = str(parsed.get("summary") or "").strip()
            if status and summary:
                return AgentResult(status=status, summary=summary)  # success path
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse conversation history final JSON: %s", text)
    return AgentResult(status="error", summary=text.strip())


class ConversationHistoryAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "conversation_history_agent"

    @property
    def description(self) -> str:
        return (
            "Looks only at past chat history and message continuity. Use when the supervisor needs to find what was previously said in channel history."
        )

    async def run(
        self,
        state: BotState,
        task: str,
        expected_response: str = "",
    ) -> AgentResult:
        tool_history: list[ToolCall] = []

        # Check if tools are available
        tools = _get_langchain_tools()
        if not tools:
            return AgentResult(
                agent=self.name,
                status="error",
                summary="No conversation history tools available.",
                tool_history=[],
            )

        system_prompt = _CONVERSATION_HISTORY_SYSTEM_PROMPT

        if expected_response:
            task += f"\n\nExpected response: {expected_response}"

        llm_input_msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task),
        ]

        logger.info(
            "Calling LLM for conversation history agent. \nTask: %s\nMax iterations: %d",
            task,
            MAX_TOOL_ITERATIONS
        )

        try:
            llm = _get_llm_with_tools()

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                logger.debug(
                    "LLM input for Conversation History Agent (iteration %d):\n%s",
                    iteration,
                    "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in llm_input_msgs)
                )
                result = await llm.ainvoke(llm_input_msgs)
                
                # Check if there are tool calls
                if result.tool_calls:
                    # Execute tool calls and create tool messages
                    tool_messages = []
                    for tool_call in result.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call["id"]
                        
                        logger.info(
                            "ConversationHistoryAgent tool call [%d/%d]: %s(%s)",
                            iteration + 1, MAX_TOOL_ITERATIONS, tool_name, tool_args,
                        )
                        
                        # Find the corresponding tool and execute it
                        tool = None
                        for t in _get_langchain_tools():
                            if t.name == tool_name:
                                tool = t
                                break
                        
                        if tool:
                            # Add state defaults for channel_id and user_id if not provided
                            if tool_name == "search_conversation_history":
                                final_args = tool_args.copy()
                                if not final_args.get("channel_id") and state.get("channel_id"):
                                    final_args["channel_id"] = state.get("channel_id")
                                if not final_args.get("user_id") and state.get("user_id"):
                                    final_args["user_id"] = state.get("user_id")
                                tool_result = await tool.ainvoke(final_args)
                            else:
                                tool_result = await tool.ainvoke(tool_args)
                        else:
                            tool_result = f"Unknown tool: {tool_name}"
                        
                        # Create ToolMessage for the response
                        tool_message = ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_id
                        )
                        tool_messages.append(tool_message)
                        
                        tool_history.append(ToolCall(
                            tool=tool_name,
                            args=tool_args,
                            result=tool_result,
                        ))
                    
                    # Add tool messages to conversation
                    llm_input_msgs.append(result)  # The AIMessage with tool_calls
                    llm_input_msgs.extend(tool_messages)  # Tool responses
                    llm_input_msgs.append(HumanMessage(
                        content="Now output ONLY valid JSON with `status` and `summary` for the results.",
                    ))
                else:
                    # No tool calls, this should be the final result
                    final_result = _parse_final_result(result.content, "No relevant conversation history results found.")
                    return AgentResult(
                        agent=self.name,
                        status=final_result.get("status"),
                        summary=final_result.get("summary"),
                        tool_history=tool_history,
                    )

                if iteration == MAX_TOOL_ITERATIONS:
                    return AgentResult(
                        agent=self.name,
                        status="error",
                        summary="Conversation history lookup timed out after maximum iterations.",
                        tool_history=tool_history,
                    )

        except Exception as exc:
            logger.exception("ConversationHistoryAgent failed")
            return AgentResult(
                agent=self.name,
                status="error",
                summary=f"Conversation history lookup failed: {exc}",
                tool_history=tool_history,
            )
