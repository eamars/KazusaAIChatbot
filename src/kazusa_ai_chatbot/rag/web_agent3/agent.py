"""Public RAG helper wrapper and web_agent3 retrieval loop."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from kazusa_ai_chatbot.config import (
    MAX_WEB_SEARCH_AGENT_RETRY,
    WEB_SEARCH_LLM_API_KEY,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MODEL,
    WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    WEB_SEARCH_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.web_agent3.contracts import (
    _DEFAULT_EXPECTED_RESPONSE,
    _RouterDecision,
    _WebToolResult,
    _limit_status,
    _normalize_router_decision,
    _router_decision_from_state,
    _router_decision_to_dict,
)
from kazusa_ai_chatbot.rag.web_agent3.providers import (
    _execute_source_decision,
)
from kazusa_ai_chatbot.rag.web_agent3.subagent import (
    _SUBAGENT_DESCRIPTIONS,
    _SUBAGENT_NAMES,
    _SUBAGENT_SUPPORTED_ACTIONS,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

def _prompt_timestamp_for_llm(
    local_prompt_timestamp: str,
    context: dict[str, Any],
) -> str:
    """Return the safe local timestamp string used in web-agent prompts.

    Args:
        local_prompt_timestamp: Existing local timestamp string from the helper
            wrapper.
        context: Runtime context that may contain a structured local time block.

    Returns:
        Human-readable local reference time, or an empty string when only
        machine-style time is available.
    """
    local_time_context = context.get("local_time_context") or {}
    local_datetime = str(
        local_time_context.get("current_local_datetime", "")
    ).strip()
    local_weekday = str(
        local_time_context.get("current_local_weekday", "")
    ).strip()
    if local_datetime:
        if local_weekday:
            return_value = f"{local_datetime} ({local_weekday})"
            return return_value
        return_value = local_datetime
        return return_value

    raw_local_prompt_timestamp = str(local_prompt_timestamp or "").strip()
    has_machine_time_marker = (
        "UTC" in raw_local_prompt_timestamp
        or "T" in raw_local_prompt_timestamp
        or "+" in raw_local_prompt_timestamp
        or raw_local_prompt_timestamp.endswith(("Z", "z"))
    )
    if has_machine_time_marker:
        return_value = ""
        return return_value
    return_value = raw_local_prompt_timestamp
    return return_value


def _web_runtime_context_for_llm(context: dict[str, Any]) -> dict[str, Any]:
    """Project only public-web-relevant runtime context into web prompts.

    Args:
        context: Internal RAG context, which may include platform, channel, and
            user identifiers that public web search does not need.

    Returns:
        A small semantic context for web routing/finalization prompts.
    """

    projected: dict[str, Any] = {}
    for key in ("original_query", "current_slot", "channel_topic"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            projected[key] = value.strip()

    local_time_context = context.get("local_time_context")
    if isinstance(local_time_context, dict):
        time_context: dict[str, str] = {}
        for key in ("current_local_datetime", "current_local_weekday"):
            value = local_time_context.get(key)
            if isinstance(value, str) and value.strip():
                time_context[key] = value.strip()
        if time_context:
            projected["time_context"] = time_context

    return_value = projected
    return return_value


class WebAgent3State(TypedDict):
    """Working state for the web_agent3 LangGraph subgraph."""

    task: str
    context: dict[str, Any]
    expected_response: str
    messages: Annotated[list, add_messages]
    router_decision: dict[str, str]
    observations: list[dict[str, Any]]
    evaluator_feedback: str
    should_stop: bool
    retry: int
    prompt_timestamp: str
    knowledge_metadata: dict[str, Any]
    final_response: str
    final_status: str
    final_reason: str
    final_is_empty_result: bool


def _observation_record(
    *,
    decision: _RouterDecision,
    result: Any,
) -> dict[str, Any]:
    """Build the evaluator-visible observation for one source action."""
    record = {
        "action": decision.action,
        "source": decision.source,
        "query": decision.query,
        "result": result,
    }
    return record


_WEB_AGENT3_SOURCE_TOOLS_TEXT = "\n".join(
    "- {source}: {description}".format(
        source=source,
        description=description,
    )
    for source, description in _SUBAGENT_DESCRIPTIONS.items()
)

_WEB_AGENT3_GENERATOR_PROMPT = '''\
你是网络检索路由器。你的任务是为当前 Web evidence 槽位选择下一步检索动作、来源和 query。

# 核心任务
1. 如果需要寻找新线索，设置 `action: "search"`。
2. 如果已有明确目标需要读取，设置 `action: "read"`。
3. 如果已有证据足够，或继续检索没有意义，设置 `action: "stop"`。
4. `source` 必须严格使用来源原则中列出的可用来源名称。
5. `query` 是传给所选来源子代理的唯一 payload。

# 来源原则
{source_tools}

# 职责边界
1. 路由器只判断下一步动作、来源和 query。
2. 来源子代理负责处理各自来源的编号、链接、页面目标和后续读取方式。
3. 对 `search`，`query` 应是可直接用于该来源的搜索文本。
4. 对 `read`，`query` 应保留原始目标字符串，例如 URL、编号、标题或用户给出的目标。
5. 对 `stop`，`query` 必须是空字符串。
6. `query` 必须遵循所选来源描述中的生成规则。

# Query 生成硬规则
1. 不要因为 `reference_time` 自动把当前日期、年份、月份或未来日期加入 search query。
2. 只有当用户任务明确要求某一天、某月、某年或日期范围时，才把日期写入 query。
3. 用户说“当前、最新、最近、现在”时，优先搜索与任务领域相匹配的权威、稳定、可复核来源路径。
4. 如果 `call_history` 显示 No results、查询过窄、包含过多限制词或日期导致失败，下一步 search 必须先移除日期/年份和堆叠约束，再换成更短、更来源导向的 query；不要直接停止。
5. 如果任务要求区分多个来源类别、发布轨道、证据立场或资料类型，应按用户任务中的类别分别尝试；不要把未覆盖的类别写成已确认。
6. 对现实世界当前事实任务，query 应先找到权威来源、候选来源或索引页，再读取具体页面；不要把所有约束一次性塞进一个搜索词。
7. 如果任务点名某个来源表面、官方站点或可推断的规范 URL，而搜索没有结果，可以用 `read` 尝试一个从任务和上下文合理推断的 URL；读取失败才算该 URL 路径失败。
8. 直接 URL 读取只用于发现来源，不等于事实确认；如果读取为空、不可用、跳转无关或内容不含目标事实，最终证据包必须说明该路径未确认。

# 审计步骤
1. 读取 `task`，确认外部证据需求。
2. 读取 `context` 和 `reference_time`，确认地点、时间、URL 或上游约束。
3. 检查 `call_history`，区分 search 摘要和 read 正文。
4. 读取 `evaluator_feedback`，按反馈决定继续搜索、读取目标或停止。
5. 输出一个最小 JSON 对象。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "action": "search|read|stop",
    "source": "string",
    "query": "string"
}}
'''.format(source_tools=_WEB_AGENT3_SOURCE_TOOLS_TEXT)
_llm_interface = LLInterface()
_generator_llm = LLInterface()
_evaluator_llm = LLInterface()
_finalizer_llm = LLInterface()
_generator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=0.3,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=WEB_SEARCH_LLM_THINKING_ENABLED,
    ),
)


async def _tool_call_generator(state: WebAgent3State) -> dict[str, Any]:
    """Ask the router LLM for the next source/action/query decision.

    Args:
        state: Current subgraph state.

    Returns:
        State update containing the validated router decision.
    """
    system_prompt = SystemMessage(content=_WEB_AGENT3_GENERATOR_PROMPT)
    router_input = {
        "task": state["task"],
        "context": state["context"],
        "reference_time": state["prompt_timestamp"],
        "call_history": state["observations"][-3:],
        "evaluator_feedback": state.get("evaluator_feedback", ""),
    }
    human_message = HumanMessage(
        content=json.dumps(router_input, ensure_ascii=False, default=str)
    )
    response = await _generator_llm.ainvoke([system_prompt, human_message], config=_generator_llm_config)
    raw_decision = parse_llm_json_output(response.content)
    decision = _normalize_router_decision(
        raw_decision,
        fallback_query=state["task"],
        valid_sources=_SUBAGENT_NAMES,
        source_actions=_SUBAGENT_SUPPORTED_ACTIONS,
    )
    decision_payload = _router_decision_to_dict(decision)
    decision_message = AIMessage(
        content=json.dumps(decision_payload, ensure_ascii=False)
    )
    return_value = {
        "messages": [decision_message],
        "router_decision": decision_payload,
    }
    return return_value


async def _tool_call_executor(state: WebAgent3State) -> dict[str, Any]:
    """Execute the latest router decision through the selected source subagent.

    Args:
        state: Current subgraph state containing a validated router decision.

    Returns:
        State update with a bounded observation record.
    """
    decision = _router_decision_from_state(state["router_decision"])

    if decision.action == "stop":
        observation = {
            "status": "stopped",
            "source": "web_read",
            "action": "stop",
            "query": "",
            "message": "Router stopped without another web action.",
        }
    else:
        try:
            observation = await _execute_source_decision(decision)
        except Exception as exc:
            logger.exception(f"web_agent3 source execution failed: {exc}")
            observation = {"error": "tool execution failed"}

    record = _observation_record(decision=decision, result=observation)
    observations = list(state.get("observations", []))
    observations.append(record)
    tool_message = ToolMessage(
        content=json.dumps(record, ensure_ascii=False, default=str),
        tool_call_id=f"web-agent3-{len(observations)}",
    )
    return_value = {
        "messages": [tool_message],
        "observations": observations,
    }
    return return_value


_WEB_AGENT3_EVALUATOR_PROMPT = '''\
你是网络检索评估员。你的任务是判断已有检索观察是否足以支持下游认知使用的证据包，并给路由器下一步建议。

# 核心任务
1. 如果观察内容已经覆盖任务需求，或已经足够好，设置 `should_stop: true`。
2. 如果关键信息缺失、过时、只有搜索摘要、或需要读取正文，设置 `should_stop: false`。
3. 如果未停止，`feedback` 必须给出下一步动作建议，例如更具体的 search query，或读取某个已有链接。
4. `feedback` 只描述下一步动作和必要目标，不生成实际 tool_call。

# 停止原则
1. 已有正文或来源信息能支持准确证据包。
2. 多轮尝试没有新增有用信息。
3. 任务相关信息确认缺失，继续检索收益很低。
4. 如果同一 URL 已经 read 过且正文为空或没有新增信息，不要重复读取同一 URL；应停止并说明只有搜索摘要线索，或建议不同 search query。
5. 只有过窄 query 返回 No results 时，不算“事实缺失”。如果还没有尝试过更短 query、官方来源 query 或 source-specific query，应设置 `should_stop: false` 并在 `feedback` 要求改写 query。
6. 如果任务要求区分多个来源类别、发布轨道、证据立场或资料类型，必须确认这些类别已经分别尝试过或已有明确来源；否则不要停止。
7. 如果任务明确要求某个来源表面、官方站点或可推断的规范 URL，而当前历史只有搜索失败、没有任何 `read` 尝试，通常应设置 `should_stop: false`，反馈要求尝试可合理推断的 URL 或更短的来源导向 query。

# 审计步骤
1. 读取 `task` 和 `expected_response`，确认要满足的事实范围。
2. 检查 `call_history`，区分 search 摘要和 read 正文。
3. 涉及时效判断时，用 `reference_time` 核对结果日期。
4. 如果失败原因是 query 过窄、包含自动日期/年份或堆叠约束，反馈里明确要求删除这些限制并改走官方/来源导向 query。
5. 输出 `should_stop` 和给路由器的简短 `feedback`。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
    "feedback": "给路由器的下一步建议。如果 should_stop 为 true，此处可留空或总结检索结论。",
    "should_stop": true or false
}
'''
_evaluator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=WEB_SEARCH_LLM_THINKING_ENABLED,
    ),
)


async def _tool_call_evaluator(state: WebAgent3State) -> dict[str, Any]:
    """Evaluate whether the retrieved content is sufficient.

    Args:
        state: Current subgraph state.

    Returns:
        State update with stop flag, retry count, and evaluator feedback.
    """
    retry = state.get("retry", 0) + 1
    evaluation_input = {
        "task": state["task"],
        "expected_response": state["expected_response"],
        "call_history": state["observations"],
        "reference_time": state["prompt_timestamp"],
    }
    response = await _evaluator_llm.ainvoke([
        SystemMessage(content=_WEB_AGENT3_EVALUATOR_PROMPT),
        HumanMessage(content=json.dumps(evaluation_input, ensure_ascii=False)),
    ], config=_evaluator_llm_config)
    result = parse_llm_json_output(response.content)
    should_stop = bool(result.get("should_stop", False))
    feedback = str(result.get("feedback", "")).strip()

    if retry >= MAX_WEB_SEARCH_AGENT_RETRY:
        should_stop = True

    final_message = HumanMessage(
        content=feedback,
        name="evaluator",
    )
    return_value = {
        "messages": [final_message],
        "should_stop": should_stop,
        "retry": retry,
        "evaluator_feedback": feedback,
        "knowledge_metadata": {},
    }
    return return_value


_WEB_AGENT3_FINALIZER_PROMPT = '''\
你是一个信息整理专家。你的任务是将检索到的信息整理成供下游认知使用的证据包，而不是直接替角色回答用户。

# 核心任务
1. 整理信息：将检索到的关键信息根据任务描述与 `expected_response` 整理成来源扎根的证据包。
2. 评估信息：根据评估者最终反馈评估检索到的信息是否满足任务描述的要求。

# 边界约束
- 不要代替角色回答：禁止写成直接对用户说话的最终答复。
- 来源优先：优先保留来源 URL、页面标题、时间或站点名称；若未知则明确写未知。
- 允许压缩，不允许编造：可以压缩为 3-6 条事实要点，但每条必须来自已有检索内容，禁止补全或猜测。
- 缺失即缺失：如果 `content` 明确说明任务所需事实缺失、未提及、没有相关信息或无法从来源确认，`response` 应记录这个缺失证据，且 `is_empty_result` 必须为 true。
- 时效诚实：如果 `task` 要求最新、最近、当前或现在的信息，用 `reference_time` 对照来源时间；来源过旧或无法确认当前状态时，必须明确写出时效限制。
- 时间边界：不要提及模型知识、训练数据、知识截止或真实世界当前时间；时效判断只能使用 `reference_time`、来源时间和检索内容。
- 失败诚实：只有观察内容明确包含错误、空内容、工具失败或无数据时，才可以声称读取失败或没有检索数据。
- 摘要诚实：如果只有搜索摘要、snippet 或搜索结果线索，或者后续正文读取为空，必须标注为搜索摘要级证据，不要写成已读取正文确认。
- 多次搜索诚实：如果 content 中包含 Search attempts，必须按每次尝试保留哪些 query 有来源线索、
  哪些 query 没有有用结果或工具错误、哪些目标还需要更窄检索；不要把多个尝试压成一个笼统的已找到或未找到结论。
- 跨来源一致性：只有每个被比较的来源类别都有正文或摘要明确包含同一目标事实时，才可以说信息一致；如果某个来源类别读取失败、只有链接、只有邻近页面、或正文没有目标事实，必须写成该来源未确认，不得说没有冲突或信息一致。
- 对象边界：不要把相邻产品、派生轨道、集成说明、镜像、扩展或非目标对象当成目标对象的直接证据；除非任务明确询问这些轨道，否则只能列为邻近线索或排除项。
- 语言策略：提示面向下游认知，优先使用中文整理证据；URL、标题、来源文本、代码名和专有名词保持原文。

# 生成步骤
1. 先读取 `task`、`expected_response` 与 `reference_time`，确认下游需要什么证据包。
2. 从 `content` 中提取有来源支撑的事实要点，保留 URL、标题、站点或时间。
3. 结合 `evaluator_feedback` 判断完整度并给出 `score` 与 `reason`。
4. 如果已有来源明确说目标事实缺失或未提及，将这个缺失作为证据整理，并设置 `is_empty_result: true`。
5. 如果任务要求当前性但来源时间过旧或无法确认当前状态，在证据包中明确说明当前性限制。
6. 只有在已有来源能支持任务所需事实时，才设置 `is_empty_result: false`。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
    "response": "string",
    "score": <int: 0-100>,
    "reason": "string",
    "is_empty_result": true or false
}
'''
_finalizer_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=WEB_SEARCH_LLM_THINKING_ENABLED,
    ),
)


async def _tool_call_finalizer(state: WebAgent3State) -> dict[str, Any]:
    """Synthesize retrieved content into a compact evidence package.

    Args:
        state: Current subgraph state after the evaluation loop.

    Returns:
        State update with final response fields.
    """
    tool_messages = [
        message.content for message in state["messages"]
        if isinstance(message, ToolMessage)
    ]
    if tool_messages:
        tool_results = "\n".join(tool_messages)
    else:
        tool_results = "No information retrieved."

    finalizer_input = {
        "task": state["task"],
        "expected_response": state["expected_response"],
        "reference_time": state["prompt_timestamp"],
        "content": tool_results,
        "evaluator_feedback": state["evaluator_feedback"],
    }
    response = await _finalizer_llm.ainvoke([
        SystemMessage(content=_WEB_AGENT3_FINALIZER_PROMPT),
        HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False)),
    ], config=_finalizer_llm_config)
    result = parse_llm_json_output(response.content)

    raw_score = result.get("score", 0)
    if isinstance(raw_score, int) and not isinstance(raw_score, bool):
        score = raw_score
    else:
        score = 0

    missing_response = "response" not in result
    if "response" not in result:
        result["response"] = "No information retrieved."
        result["score"] = 0
        logger.error(f"web_agent3 finalizer omitted response; raw result: {result}")

    if "reason" not in result:
        result["reason"] = "No reason provided."

    is_empty_result = result.get("is_empty_result")
    if not isinstance(is_empty_result, bool):
        logger.error(f"web_agent3 finalizer omitted is_empty_result; raw result={result}")
        is_empty_result = score <= 0

    if missing_response:
        status = "error"
    else:
        status = _status_from_score(score, is_empty_result)

    return_value = {
        "final_response": result.get("response"),
        "final_status": status,
        "final_reason": result.get("reason"),
        "final_is_empty_result": is_empty_result,
    }
    return return_value


async def _finalize_web_agent3_result(
    *,
    task: str,
    context: dict[str, Any],
    local_prompt_timestamp: str,
    tool_result: _WebToolResult | None = None,
    evaluator_feedback: str = "",
    evidence_limitations: list[str] | None = None,
    max_status: Literal["success", "partial", "not_found"] = "success",
) -> dict[str, Any]:
    """Run the finalizer with a normalized fixture-style tool result.

    Args:
        task: Web evidence task.
        context: Runtime context.
        local_prompt_timestamp: Local reference time.
        tool_result: Optional normalized evidence fixture.
        evaluator_feedback: Evaluation note to pass to the finalizer.
        evidence_limitations: Optional deterministic limitation labels.
        max_status: Highest allowed status after deterministic limitations.

    Returns:
        Public result fields produced by the web evidence finalizer.
    """
    prompt_timestamp = _prompt_timestamp_for_llm(local_prompt_timestamp, context)
    if tool_result is None:
        tool_content = "No information retrieved."
    else:
        tool_content = _tool_result_to_message_content(tool_result)

    state: WebAgent3State = {
        "task": task,
        "context": _web_runtime_context_for_llm(context),
        "expected_response": _DEFAULT_EXPECTED_RESPONSE,
        "messages": [ToolMessage(content=tool_content, tool_call_id="fixture-1")],
        "router_decision": {"action": "stop", "source": "web_read", "query": ""},
        "observations": [],
        "evaluator_feedback": evaluator_feedback,
        "should_stop": True,
        "retry": 0,
        "prompt_timestamp": prompt_timestamp,
        "knowledge_metadata": {},
        "final_response": "",
        "final_status": "error",
        "final_reason": "",
        "final_is_empty_result": False,
    }
    final_state = await _tool_call_finalizer(state)
    status = final_state["final_status"]
    if status in ("success", "partial", "not_found"):
        status = _limit_status(status, max_status)

    result = {
        "status": status,
        "reason": final_state["final_reason"],
        "response": final_state["final_response"],
        "is_empty_result": final_state["final_is_empty_result"],
        "knowledge_metadata": {
            "evidence_limitations": evidence_limitations or [],
            "max_status": max_status,
        },
    }
    return result


def _tool_result_to_message_content(tool_result: _WebToolResult) -> str:
    """Render a normalized fixture result as finalizer-visible tool content."""
    if tool_result.operation == "search":
        items = [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "source": item.source,
            }
            for item in tool_result.items
        ]
        content = json.dumps(items, ensure_ascii=False)
        return content

    payload = {
        "url": tool_result.url,
        "title": tool_result.title,
        "description": tool_result.description,
        "content": tool_result.content,
        "error": tool_result.error,
        "missing_context": tool_result.missing_context,
        "delegation_reason": tool_result.delegation_reason,
    }
    content = json.dumps(payload, ensure_ascii=False)
    return content


def _status_from_score(score: int, is_empty_result: bool) -> str:
    """Map a finalizer score into legacy status labels."""
    if is_empty_result:
        return_value = "not_found"
        return return_value
    if score > 80:
        return_value = "success"
        return return_value
    if score > 50:
        return_value = "partial"
        return return_value
    return_value = "not_found"
    return return_value


async def _run_subgraph(
    task: str,
    context: dict[str, Any],
    expected_response: str,
    local_prompt_timestamp: str,
) -> dict[str, Any]:
    """Build and execute the web_agent3 LangGraph subgraph.

    Args:
        task: Slot description containing the web search request.
        context: Runtime hints from the outer-loop supervisor.
        expected_response: Description of the expected evidence package.
        local_prompt_timestamp: Local reference timestamp for prompt rendering.

    Returns:
        Dict with status, reason, response, empty-result flag, and metadata.
    """
    builder = StateGraph(WebAgent3State)
    builder.add_node("executor", _tool_call_executor)
    builder.add_node("generator", _tool_call_generator)
    builder.add_node("evaluator", _tool_call_evaluator)
    builder.add_node("finalizer", _tool_call_finalizer)

    builder.add_edge(START, "generator")
    builder.add_edge("generator", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        lambda state: "finalize" if state.get("should_stop", False) else "loop",
        {"loop": "generator", "finalize": "finalizer"},
    )
    builder.add_edge("finalizer", END)

    sub_graph = builder.compile()
    llm_context = _web_runtime_context_for_llm(context)
    prompt_timestamp = _prompt_timestamp_for_llm(local_prompt_timestamp, context)
    sub_state: WebAgent3State = {
        "task": task,
        "context": llm_context,
        "expected_response": expected_response,
        "messages": [],
        "router_decision": {"action": "stop", "source": "web_read", "query": ""},
        "observations": [],
        "evaluator_feedback": "",
        "should_stop": False,
        "retry": 0,
        "final_status": "error",
        "final_reason": "",
        "final_response": "",
        "final_is_empty_result": False,
        "prompt_timestamp": prompt_timestamp,
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


class WebAgent3(BaseRAGHelperAgent):
    """RAG helper agent that retrieves external evidence through web_agent3."""

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="web_agent3",
            cache_name="",
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Retrieve web evidence while preserving the helper contract.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Runtime hints including local time context when available.
            max_attempts: Accepted for contract parity; retry count is config-owned.

        Returns:
            Base RAG helper result with text evidence and uncached status.
        """
        del max_attempts

        local_time_context = context.get("local_time_context") or {}
        local_datetime = str(
            local_time_context.get("current_local_datetime", "")
        ).strip()
        local_weekday = str(
            local_time_context.get("current_local_weekday", "")
        ).strip()
        if local_datetime:
            local_prompt_timestamp = (
                f"{local_datetime} ({local_weekday})"
                if local_weekday
                else local_datetime
            )
        else:
            local_prompt_timestamp = ""

        raw = await _run_subgraph(
            task=task,
            context=context,
            expected_response=_DEFAULT_EXPECTED_RESPONSE,
            local_prompt_timestamp=local_prompt_timestamp,
        )
        result = self.with_cache_status(
            {
                "resolved": not bool(raw.get("is_empty_result", False)),
                "result": str(raw.get("response", "")),
                "attempts": 1,
            },
            hit=False,
            reason="agent_not_cacheable",
        )
        return result
