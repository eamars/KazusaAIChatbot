from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import BaseAgent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_TOOL_ITERATIONS
from kazusa_ai_chatbot.db import save_memory, search_memory
from kazusa_ai_chatbot.state import AgentResult, BotState, ToolCall

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_MEMORY_SYSTEM_PROMPT = """\
You are a memory assistant. Your job is to actively decide when to recall stored memory and when to save or overwrite memory.
- Do NOT invent facts.
- Before saving new memory, prefer checking whether relevant memory already exists.
- If existing memory already covers the same material, skip storing.
- If the new material is a better or newer version of the same memory, you may overwrite it by saving under the most appropriate memory name.
- Use `success` when the memory result is complete, `needs_context` when relevant memory is missing or insufficient, and `needs_clarification` when the user must specify what they meant more clearly.

You have access to memory recall and storage tools. Use them to manage the bot's long-term memory.

Output Format (strict JSON text — no markdown wrapping):
{{
  "status": "success|needs_context|needs_clarification",
  "summary": "Key facts from your memory retrieval here based on supervisor expected response."
}}
"""

@tool
async def recall_memory(
    query: str,
    limit: int = 5,
    method: str = "vector"
) -> str:
    """Recall previously stored memory relevant to a topic, link, document, or reference the user mentioned before.
    
    Args:
        query: What memory to look up (required)
        limit: Maximum number of results to return, defaults to 5, max 10 (optional)
        method: Search method - 'vector' for semantic search or 'keyword' for text search, defaults to 'vector' (optional)
    """
    if not query:
        raise ValueError("query is required")
    limit = max(1, min(int(limit), 10))
    if method not in {"keyword", "vector"}:
        method = "vector"

    rows = await search_memory(query=query, limit=limit, method=method)
    payload = [
        {
            "score": score,
            "memory_name": doc.get("memory_name"),
            "content": doc.get("content"),
        }
        for score, doc in rows
    ]
    return json.dumps(payload, ensure_ascii=False)


@tool
async def store_memory(
    memory_name: str,
    content: str,
    _timestamp: str = ""
) -> str:
    """Save or overwrite a normalized memory entry when the content should be remembered for later recall.
    
    Args:
        memory_name: Stable short name for the memory entry (required)
        content: Concise normalized content to store (required)
        _timestamp: Internal timestamp parameter (optional)
    """
    if not memory_name:
        raise ValueError("memory_name is required")
    if not content:
        raise ValueError("content is required")

    # Use provided timestamp or generate new one
    timestamp = _timestamp or datetime.now(timezone.utc).isoformat()
    await save_memory(memory_name=memory_name, content=content, timestamp=timestamp)
    return json.dumps(
        {
            "status": "saved",
            "memory_name": memory_name,
            "content": content,
        },
        ensure_ascii=False,
    )


def _get_langchain_tools() -> list:
    """Get LangChain tools for this agent."""
    return [recall_memory, store_memory]


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.5,
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
        logger.warning("Failed to parse memory final JSON: %s", text)
    return AgentResult(status="error", summary=text.strip())


class MemoryAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "memory_agent"

    @property
    def description(self) -> str:
        return (
            "Recalls or stores detailed memory such as shared links, documents, notes, and reference material. "
            "Use when the supervisor needs active long-form memory read/write beyond user facts or character state."
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
                summary="No memory tools available.",
                tool_history=[],
            )

        system_prompt = _MEMORY_SYSTEM_PROMPT

        if expected_response:
            task += f"\n\nExpected response: {expected_response}"

        llm_input_msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task),
        ]

        logger.info(
            "Calling LLM for memory agent. \nTask: %s\nMax iterations: %d",
            task,
            MAX_TOOL_ITERATIONS
        )

        try:
            llm = _get_llm_with_tools()

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                logger.debug(
                    "LLM input for Memory Agent (iteration %d):\n%s",
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
                            "MemoryAgent tool call [%d/%d]: %s(%s)",
                            iteration + 1, MAX_TOOL_ITERATIONS, tool_name, tool_args,
                        )
                        
                        # Find the corresponding tool and execute it
                        tool = None
                        for t in _get_langchain_tools():
                            if t.name == tool_name:
                                tool = t
                                break
                        
                        if tool:
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
                    final_result = _parse_final_result(result.content, "No relevant memory results found.")
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
                        summary="Memory action timed out after maximum iterations.",
                        tool_history=tool_history,
                    )

        except Exception as exc:
            logger.exception("MemoryAgent failed")
            return AgentResult(
                agent=self.name,
                status="error",
                summary=f"Memory action failed: {exc}",
                tool_history=tool_history,
            )
