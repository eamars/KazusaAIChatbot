"""Web Search Agent — searches the internet via MCP search tools.

Runs in its own LLM context with only the user query and search tool
descriptions.  Executes the tool-calling loop, then summarises the raw
search results into a concise paragraph for the speech agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import BaseAgent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_TOOL_ITERATIONS
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import AgentResult, BotState, ToolCall

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_WEB_SEARCH_SYSTEM_PROMPT = """\
You are a web search assistant. Your job is to search the internet to answer the supervisor's query, then provide a concise factual summary of what you found.
- Use `success` when the search answer is complete, `needs_context` when search results are missing or insufficient, and `needs_clarification` when the request is too ambiguous to search properly.
- The current date is {current_date}. When searching for news or recent events, prioritize current information and avoid defaulting to past years unless specifically requested.
- You have access to web search and URL reading tools. Use them to gather information, then provide your final summary.

Output Format (strict JSON text — no markdown wrapping):
{{
  "status": "success|needs_context|needs_clarification",
  "summary": "Key facts from your web search here based on supervisor expected response."
}}
"""


@tool
async def searxng_web_search(
    query: str,
    pageno: int = 1,
    time_range: str = "",
    language: str = ""
) -> str:
    """Performs a web search using the SearXNG API.
    
    Args:
        query: The search query string (required)
        pageno: Search page number starting from 1 (default: 1)
        time_range: Time filter for search results - use 'day', 'month', 'year' or leave empty for all time (default: "")
        language: Language code for results (e.g., 'en', 'zh', 'fr') or leave empty for default (default: "")
    """
    return await mcp_manager.call_tool("mcp-searxng__searxng_web_search", {
        "query": query,
        "pageno": pageno,
        "time_range": time_range,
        "language": language,
        "safesearch": 0  # none
    })


@tool
async def web_url_read(
    url: str,
    startChar: int = 0,
    maxLength: int = 10000,
    section: str = "",
    paragraphRange: str = "",
    readHeadings: bool = False
) -> str:
    """Reads and extracts content from a specific URL.
    
    Args:
        url: The complete URL to read content from (required)
        startChar: Starting character position for content extraction (default: 0)
        maxLength: Maximum number of characters to return, 0 for no limit (default: 10000)
        section: Extract content under a specific heading text (default: "")
        paragraphRange: Return specific paragraph ranges like '1-5', '3', or '10-' (default: "")
        readHeadings: If True, returns only the list of headings instead of full content (default: False)
    """
    # Handle maxLength=0 case by omitting it (MCP tool requires minimum: 1)
    args = {
        "url": url,
        "startChar": startChar,
        "section": section,
        "paragraphRange": paragraphRange,
        "readHeadings": readHeadings
    }
    if maxLength > 0:
        args["maxLength"] = maxLength
    
    return await mcp_manager.call_tool("mcp-searxng__web_url_read", args)


def _get_langchain_tools() -> list:
    """Get LangChain tools for this agent."""
    return [searxng_web_search, web_url_read]


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.3,
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
        logger.warning("Failed to parse web search final JSON: %s", text)
    return AgentResult(status="error", summary=text.strip())


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

    async def run(
        self,
        state: BotState,
        task: str,
        expected_response: str = "",
    ) -> AgentResult:
        """Execute the search tool loop and return a summarised result."""
        tool_history: list[ToolCall] = []

        # Check if tools are available
        tools = _get_langchain_tools()
        if not tools:
            return AgentResult(
                agent=self.name,
                status="error",
                summary="No search tools available.",
                tool_history=[],
            )

        # Use raw timestamp for context
        timestamp = state.get("timestamp", "")
        system_prompt = _WEB_SEARCH_SYSTEM_PROMPT.format(current_date=timestamp)

        if expected_response:
            task += f"\n\nExpected response: {expected_response}"

        llm_input_msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Search query: {task}"),
        ]

        logger.info(
            "Calling LLM for web search agent. \nQuery: %s\nMax iterations: %d",
            task,
            MAX_TOOL_ITERATIONS
        )

        try:
            llm = _get_llm_with_tools()

            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                logger.debug(
                    "LLM input for Web Search Agent (iteration %d):\n%s",
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
                            "WebSearchAgent tool call [%d/%d]: %s(%s)",
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
                    final_result = _parse_final_result(result.content, "No results found.")
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
                        summary="Search timed out after maximum iterations.",
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
