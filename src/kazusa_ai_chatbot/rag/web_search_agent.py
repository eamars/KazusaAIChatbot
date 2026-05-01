"""RAG helper agent: web search via LangGraph tool-call subgraph."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from kazusa_ai_chatbot.config import (
    MAX_WEB_SEARCH_AGENT_RETRY,
    WEB_SEARCH_LLM_API_KEY,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MODEL,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)

_DEFAULT_EXPECTED_RESPONSE = "返回能直接解决当前槽位的来源扎根网页证据。"


@tool
async def web_search(
    query: str,
    pageno: int = 1,
    time_range: str = "",
    language: str = "",
) -> str:
    """Performs a web search using the SearXNG API.

    Args:
        query: The search query string (required)
        pageno: Search page number starting from 1 (default: 1)
        time_range: Time filter for search results - use 'day', 'month', 'year' or leave empty for all time (default: "")
        language: Language code for results (e.g., 'en', 'zh', 'fr') or leave empty for default (default: "")
    """
    return_value = await mcp_manager.call_tool("mcp-searxng__searxng_web_search", {
        "query": query,
        "pageno": pageno,
        "time_range": time_range,
        "language": language,
        "safesearch": 0,
    })
    return return_value


@tool
async def web_url_read(
    url: str,
    startChar: int = 0,
    maxLength: int = 5000,
    section: str = "",
    paragraphRange: str = "",
    readHeadings: bool = False,
) -> str:
    """Reads and extracts content from a specific URL.

    Args:
        url: The complete URL to read content from (required)
        startChar: Starting character position for content extraction (default: 0)
        maxLength: Maximum number of characters to return, 0 for no limit (default: 5000)
        section: Extract content under a specific heading text (default: "")
        paragraphRange: Return specific paragraph ranges like '1-5', '3', or '10-' (default: "")
        readHeadings: If True, returns only the list of headings instead of full content (default: False)
    """
    args = {
        "url": url,
        "startChar": startChar,
        "section": section,
        "paragraphRange": paragraphRange,
        "readHeadings": readHeadings,
    }
    if maxLength > 0:
        args["maxLength"] = maxLength
    return_value = await mcp_manager.call_tool("mcp-searxng__web_url_read", args)
    return return_value


_ALL_TOOLS = [web_search, web_url_read]
_TOOLS_BY_NAME = {t.name: t for t in _ALL_TOOLS}


class WebSearchState(TypedDict):
    """Working state for the web-search LangGraph subgraph."""

    task: str
    context: dict
    next_tool: str
    expected_response: str
    messages: Annotated[list, add_messages]
    should_stop: bool
    retry: int
    timestamp: str
    knowledge_metadata: dict
    final_response: str
    final_status: str
    final_reason: str
    final_is_empty_result: bool


_WEB_SEARCH_GENERATOR_PROMPT = """\
你是一个专家级的网络搜索代理 (Web Search Agent)。你的目标是通过互联网检索最准确、最及时的信息来完成任务。

# 核心准则
- **行为分解**：搜索 (`web_search`) 只是为了寻找线索；阅读 (`web_url_read`) 才是为了获取知识。严禁仅根据搜索结果摘要（Snippets）撰写最终答案。
- **拒绝假设**：严禁猜测未知的 URL 或事实。如果信息不存在，请如实反馈。
- **反馈至上**：评估员 (Evaluator) 的反馈是你的最高指令。如果评估员要求你"深入阅读"，不要再次发起搜索。

# 搜索策略 (Advanced Query Engineering)
1. **多维度搜索**：如果第一次搜索无果，尝试使用近义词、英文翻译（针对技术或国际话题）或特定的日期限定。
2. **搜索语法**：合理利用高级语法，如 `"精确匹配"`，`site:official-website.com`，或 `-排除无关词`。
3. **优先级排序**：优先选择权威来源（政府、大型机构、官方文档）而非博客或社交媒体。

# 任务流程
1. **历史审计**：核查 `messages`。如果上一步已经得到了搜索列表，这一步通常应该使用 `web_url_read` 读取其中最相关的 1-3 个链接。
2. **识别缺口**：对比 `task` 与当前已获取的"正文内容"。
3. **执行行动**：
   - 需要新线索？执行 `web_search`。
   - 已有线索但无详情？执行 `web_url_read`。
   - 无法继续？在回复中说明原因。

# 语言与时间
- **当前时间参考**：{timestamp}。
- **语言匹配**：默认使用任务语言搜索。但对于全球性技术、科学或国际新闻，建议同时尝试英文搜索。

# 输入格式
{{
    "task": "任务描述",
    "context": 辅助搜索信息,
    "messages": [包含评估员反馈的历史记录]
}}
"""
_generator_llm = get_llm(
    temperature=0.3,
    top_p=0.9,
    model=WEB_SEARCH_LLM_MODEL,
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
).bind_tools(_ALL_TOOLS)

_WEB_SEARCH_EVALUATOR_PROMPT = """\
你是一个高级检索评估专家。你的任务是分析检索到的内容与用户任务之间的差距，并决定后续行动。

# 核心任务
1. **决定状态**：
   - 如果检索内容已完全覆盖任务需求，或者已经达到"足够好 (Satisficing)"的程度，设置 `should_stop: true`。
   - 如果关键信息缺失、过时或仅有摘要而无详情，设置 `should_stop: false`。
2. **提供建议**：如果未通过，必须给出具体的"执行指令"：
   - **优化方向**：如果结果太杂，建议增加"双引号"精确匹配或 `site:` 限制。
   - **工具切换**：如果已有相关链接但只有 Snippets，强制建议使用 `web_url_read` 读取正文，禁止重复搜索。
   - **拼写纠错**：观察搜索结果中是否有"您是不是要找..."，如果是，建议修正关键词。

# 停止原则 (Critical)
符合以下任一条件时，必须设置 `should_stop: true`：
1. **足够好原则**：检索内容已覆盖 80% 以上的核心需求，足以构成一个准确、有用的答案，无需为追求 100% 的边缘细节继续浪费 Token。
2. **边际收益递减**：历史记录显示已尝试了 3 种以上不同的搜索策略且结果雷同，没有新信息出现。
3. **确认无果**：已穷尽相关关键词和域名限制（如 search, site: official_site 等）依然无法找到目标信息。

# 消息时效与计算
- **当前时间**：{timestamp}。
- **时间敏感度**：如果任务涉及"最新"、"最近"、"三天内"或特定年份，必须核对结果日期。
- **动态计算**：如果用户要求"过去一周"，请根据当前时间计算出具体的日期范围，并在 `feedback` 中告知 Generator 使用该范围。

# 工具使用约束
- 你只能根据以下工具集给出建议，禁止建议使用不存在的工具：
{agent_tools}
- **警告**：评估专家仅负责逻辑判断，禁止在 response 中生成任何实际的 tool_call。

# 审计步骤
1. 先读取 `task` 和 `expected_response`，确认需要满足的事实范围。
2. 检查 `call_history` 中已搜索和已读取的内容，区分 snippet 与正文。
3. 若已有相关链接但缺正文，建议读取正文；若关键词太泛，建议具体搜索策略。
4. 若信息已足够或继续搜索收益低，设置 `should_stop: true`。

# 输入格式
{{
    "task": "任务描述",
    "expected_response": "用户期待的回复内容和格式，有可能包含更多搜索细节",
    "call_history": [已执行的工具、参数及结果摘要],
    "retry": 当前重试次数 n / MAX_RETRY
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "feedback": "给 Generator 的下一步具体动作建议。如果 should_stop 为 true，此处可留空或总结检索结论。",
    "should_stop": true 或 false
}}
"""
_evaluator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=WEB_SEARCH_LLM_MODEL,
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
)

_WEB_SEARCH_FINALIZER_PROMPT = """\
你是一个信息整理专家。你的任务是将检索到的信息整理成供下游认知使用的证据包，而不是直接替角色回答用户。

# 核心任务
1. **整理信息**：将检索到的关键信息根据**任务描述**与 **expected_response** 整理成来源扎根的证据包。
2. **评估信息**：根据评估者最终反馈评估检索到的信息是否满足任务描述的要求。

# 边界约束
- **不要代替角色回答**：禁止写成直接对用户说话的最终答复。
- **来源优先**：优先保留来源 URL、页面标题、时间或站点名称；若未知则明确写未知。
- **允许压缩，不允许编造**：可以压缩为 3-6 条事实要点，但每条必须来自已有检索内容，禁止补全或猜测。

# 输出说明
- response: 根据 expected_response 整理后的证据包字符串，严禁输出非字符串内容
- score: 评估分数，范围 0-100，表示检索到的信息满足任务描述的程度
- reason: 评估原因（一句话之内概括）
- is_empty_result: 布尔值。仅当最终确认没有任何任务相关外部信息可供下游使用时为 true；只要存在任何任务相关信息，即使不完整，也必须为 false。

# 生成步骤
1. 先读取 `task` 与 `expected_response`，确认下游需要什么证据包。
2. 从 `content` 中提取有来源支撑的事实要点，保留 URL、标题、站点或时间。
3. 结合 `evaluator_feedback` 判断完整度并给出 `score` 与 `reason`。
4. 只有在确认没有任何相关信息时才设置 `is_empty_result: true`。

# 输入格式
{
    "task": "任务描述",
    "content": "收集到的数据",
    "evaluator_feedback": "评估者最终反馈",
    "expected_response": "用户期望的输出内容和格式"
}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
    "response": "string",
    "score": <int: 0-100>,
    "reason": "string",
    "is_empty_result": true or false
}
"""
_finalizer_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=WEB_SEARCH_LLM_MODEL,
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
)


async def _tool_call_executor(state: WebSearchState) -> dict:
    """Execute the tool calls generated by the LLM.

    Args:
        state: Current subgraph state containing messages with tool calls.

    Returns:
        State update with ToolMessage results appended to messages.
    """
    results = []
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            try:
                t = _TOOLS_BY_NAME[tool_call["name"]]
                observation = await t.ainvoke(tool_call["args"])
            except KeyError as exc:
                observation = {"error": f"Incorrect tool was invoked: {tool_call['name']}"}
                logger.error(f'Incorrect tool was invoked: {tool_call["name"]}: {exc}')
            except Exception as exc:
                logger.exception(f'Error executing tool {tool_call["name"]}: {exc}')
                observation = {"error": "tool execution failed"}

            results.append(ToolMessage(
                content=json.dumps(observation, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            ))

    return_value = {"messages": results}
    return return_value


async def _tool_call_generator(state: WebSearchState) -> dict:
    """Ask the LLM to decide the next tool call.

    Args:
        state: Current subgraph state.

    Returns:
        State update with the LLM's next message appended.
    """
    agent_tools = "\n".join([f"- {t.name}: {t.description}" for t in _ALL_TOOLS])
    system_prompt = SystemMessage(
        content=_WEB_SEARCH_GENERATOR_PROMPT.format(agent_tools=agent_tools, timestamp=state["timestamp"])
    )
    user_input = {"task": state["task"], "context": state["context"]}
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    if len(state["messages"]) > 3:
        relevant_history = [state["messages"][0]] + state["messages"][-3:]
    else:
        relevant_history = state["messages"]

    response = await _generator_llm.ainvoke([system_prompt, human_message] + relevant_history)
    return_value = {"messages": [response]}
    return return_value


async def _tool_call_evaluator(state: WebSearchState) -> dict:
    """Evaluate whether the retrieved content is sufficient.

    Args:
        state: Current subgraph state.

    Returns:
        State update with should_stop flag, retry count, and knowledge_metadata.
    """
    retry = state.get("retry", 0) + 1

    call_history = []
    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                result_content = "No result found"
                for next_msg in state["messages"][i + 1:]:
                    if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tc["id"]:
                        result_content = next_msg.content
                        break
                call_history.append({"tool": tc["name"], "arguments": tc["args"], "result": result_content})

    agent_tools = "\n".join([f"- {t.name}: {t.description}" for t in _ALL_TOOLS])
    system_prompt = SystemMessage(
        content=_WEB_SEARCH_EVALUATOR_PROMPT.format(agent_tools=agent_tools, timestamp=state["timestamp"])
    )
    evaluation_input = {
        "task": state["task"],
        "expected_response": state["expected_response"],
        "call_history": call_history,
        "retry": f"{retry}/{MAX_WEB_SEARCH_AGENT_RETRY}",
    }
    response = await _evaluator_llm.ainvoke(
        [system_prompt, HumanMessage(content=json.dumps(evaluation_input, ensure_ascii=False))]
    )
    result = parse_llm_json_output(response.content)
    should_stop = result.get("should_stop", False)
    feedback = result.get("feedback", "")

    knowledge_metadata: dict[str, Any] = {}
    if should_stop and call_history:
        last_tool_call = call_history[-1]
        knowledge_metadata["tool"] = last_tool_call["tool"]
        knowledge_metadata["result"] = last_tool_call["result"]

    if retry >= MAX_WEB_SEARCH_AGENT_RETRY:
        should_stop = True

    final_message = HumanMessage(
        content=json.dumps({"feedback": feedback, "source": "evaluator"}, ensure_ascii=False),
        name="evaluator",
    )
    return_value = {"messages": [final_message], "should_stop": should_stop, "retry": retry, "knowledge_metadata": knowledge_metadata}
    return return_value


async def _tool_call_finalizer(state: WebSearchState) -> dict:
    """Synthesize retrieved content into a compact evidence package.

    Args:
        state: Current subgraph state after the evaluation loop.

    Returns:
        State update with final_response, final_status, final_reason, and final_is_empty_result.
    """
    tool_messages = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    tool_results = "\n".join(tool_messages) if tool_messages else "No information retrieved."

    evaluator_feedback_msgs = [
        m.content for m in state["messages"]
        if isinstance(m, HumanMessage) and m.name == "evaluator"
    ]
    evaluator_feedback = evaluator_feedback_msgs[-1] if evaluator_feedback_msgs else ""

    finalizer_input = {
        "task": state["task"],
        "expected_response": state["expected_response"],
        "content": tool_results,
        "evaluator_feedback": evaluator_feedback,
    }
    response = await _finalizer_llm.ainvoke(
        [SystemMessage(content=_WEB_SEARCH_FINALIZER_PROMPT),
         HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False))]
    )
    result = parse_llm_json_output(response.content)

    score = result.get("score", 0)
    if score > 80:
        status = "success"
    elif score > 50:
        status = "partial"
    else:
        status = "not_found"

    if "response" not in result:
        result["response"] = "No information retrieved."
        result["score"] = 0
        status = "error"
        logger.error(f'Web search finalizer omitted response; raw result: {result}')

    if "reason" not in result:
        result["reason"] = "No reason provided."

    is_empty_result = result.get("is_empty_result")
    if not isinstance(is_empty_result, bool):
        logger.error(f'Web search finalizer omitted is_empty_result; raw result={result}')
        is_empty_result = False

    return_value = {
        "final_response": result.get("response"),
        "final_status": status,
        "final_reason": result.get("reason"),
        "final_is_empty_result": is_empty_result,
    }
    return return_value


async def _run_subgraph(
    task: str,
    context: dict[str, Any],
    expected_response: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build and execute the web-search LangGraph subgraph.

    Args:
        task: Slot description containing the web search request.
        context: Runtime hints from the outer-loop supervisor.
        expected_response: Description of the evidence format the caller expects.
        timestamp: ISO-8601 reference timestamp for the evaluator and generator.

    Returns:
        Dict with status, reason, response, is_empty_result, and knowledge_metadata.
    """
    builder = StateGraph(WebSearchState)
    builder.add_node("executor", _tool_call_executor)
    builder.add_node("generator", _tool_call_generator)
    builder.add_node("evaluator", _tool_call_evaluator)
    builder.add_node("finalizer", _tool_call_finalizer)

    builder.add_edge(START, "generator")
    builder.add_edge("generator", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        lambda s: "finalize" if s.get("should_stop", False) else "loop",
        {"loop": "generator", "finalize": "finalizer"},
    )
    builder.add_edge("finalizer", END)

    sub_graph = builder.compile()
    sub_state: WebSearchState = {
        "task": task,
        "context": context,
        "next_tool": "",
        "expected_response": expected_response,
        "messages": [],
        "should_stop": False,
        "retry": 0,
        "final_status": "error",
        "final_reason": "",
        "final_response": "",
        "final_is_empty_result": False,
        "timestamp": timestamp,
        "knowledge_metadata": {},
    }
    result = await sub_graph.ainvoke(sub_state)
    return_value = {
        "status": result.get("final_status"),
        "reason": result.get("final_reason"),
        "response": result.get("final_response"),
        "is_empty_result": result.get("final_is_empty_result", False),
        "knowledge_metadata": result.get("knowledge_metadata", {}),
    }
    return return_value


class WebSearchAgent(BaseRAGHelperAgent):
    """RAG helper agent that retrieves web evidence via a LangGraph tool-call subgraph.

    Not cached — web search depends on real-time external data.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="web_search_agent",
            cache_name="",
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Retrieve web evidence for a slot description.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Runtime hints; ``current_timestamp`` is used as the reference clock.
            max_attempts: Unused; retry count is controlled by MAX_WEB_SEARCH_AGENT_RETRY.

        Returns:
            Dict with resolved (bool), result (evidence string), and attempts count.
        """
        del max_attempts

        timestamp = str(context.get("current_timestamp") or "").strip()
        if not timestamp:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        raw = await _run_subgraph(
            task=task,
            context=context,
            expected_response=_DEFAULT_EXPECTED_RESPONSE,
            timestamp=timestamp,
        )
        return_value = self.with_cache_status(
            {
                "resolved": not bool(raw.get("is_empty_result", False)),
                "result": str(raw.get("response", "")),
                "attempts": 1,
            },
            hit=False,
            reason="agent_not_cacheable",
        )
        return return_value


async def _test_main() -> None:
    """Run a manual smoke check for WebSearchAgent."""
    try:
        await mcp_manager.start()
    except Exception as exc:
        logger.exception(f"MCP manager failed to start — tools will be unavailable: {exc}")

    agent = WebSearchAgent()
    result = await agent.run(
        task="<active character> 的信息",
        context={},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    await mcp_manager.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
