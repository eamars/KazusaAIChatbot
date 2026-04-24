"""Stage 1 / RAG subgraph with semantic cache + depth-based dispatch.

The subgraph has five wrapper phases on top of the existing dispatchers:

* **Phase 0 — Input analysis**: embed ``decontexualized_input`` and initialise
  the unified metadata bundle that accumulates through the whole RAG pass.
* **Phase 1 — Cache check**: probe ``RAGCache`` for ``objective_user_facts``,
  ``character_diary`` and ``external_knowledge`` matches. A strong hit short-
  circuits the rest of the pipeline.
* **Phase 2 — Depth classification**: ``InputDepthClassifier`` selects
  ``SHALLOW`` (no dispatchers; images from profile) or ``DEEP`` (input_context + optional external).
* **Phase 3 — Conditional dispatch**: START edges fan out only to the
  dispatchers permitted by the chosen depth (and the existing affinity gate
  for external).
* **Phase 4 — Cache storage**: the produced results are written back into
  the cache with a TTL matching their semantic type.

Downstream nodes keep reading ``research_facts`` in the same dict shape; the
new ``research_metadata`` key carries the trace from all five phases.
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.agents.web_search_agent2 import web_search_agent
from kazusa_ai_chatbot.agents.memory_retriever_agent import memory_retriever_agent
from kazusa_ai_chatbot.config import (
    AFFINITY_DEFAULT,
    AFFINITY_MIN,
    AFFINITY_MAX,
    RAG_CACHE_MAX_SIZE,
    RAG_CACHE_SIMILARITY_THRESHOLD,
    RAG_CACHE_TTL_SECONDS,
)
from kazusa_ai_chatbot.db import get_text_embedding
from kazusa_ai_chatbot.rag.cache import RAGCache
from kazusa_ai_chatbot.rag.depth_classifier import (
    DEEP,
    InputDepthClassifier,
)
from kazusa_ai_chatbot.utils import get_llm, log_dict_subset, log_preview, parse_llm_json_output

import json
import logging
import operator
from typing import TypedDict, Annotated, Any


logger = logging.getLogger(__name__)


# ── Stage-3 tuning constants ───────────────────────────────────────
# The thresholds below live here until Stage 5 moves them into config.py.

CACHE_HIT_THRESHOLD = 0.82            # similarity required to serve from cache
INTERNAL_RAG_STRONG_THRESHOLD = 0.55  # input_context_rag deemed sufficient for DEEP
EXTERNAL_AFFINITY_SKIP_PERCENT = 40   # skip external_rag below this affinity %
KNOWLEDGE_BASE_PROBE_THRESHOLD = 0.72 # similarity to include knowledge_base results


# ── Lazy singletons ────────────────────────────────────────────────
# One RAGCache and one InputDepthClassifier per process. Both are safe to
# share across concurrent invocations.

_rag_cache: RAGCache | None = None
_depth_classifier: InputDepthClassifier | None = None


async def _get_rag_cache() -> RAGCache:
    """Return the process-wide ``RAGCache``, warm-starting from Mongo on first use.

    Construction parameters (similarity threshold, max size, per-type TTLs) are
    sourced from ``kazusa_ai_chatbot.config`` so they can be overridden via env
    vars without touching code.
    """
    global _rag_cache
    if _rag_cache is None:
        cache = RAGCache(
            max_size=RAG_CACHE_MAX_SIZE,
            similarity_threshold=RAG_CACHE_SIMILARITY_THRESHOLD,
            default_ttl_seconds=RAG_CACHE_TTL_SECONDS,
        )
        await cache.start()
        _rag_cache = cache
    return _rag_cache


def _get_depth_classifier() -> InputDepthClassifier:
    """Return the process-wide ``InputDepthClassifier``."""
    global _depth_classifier
    if _depth_classifier is None:
        _depth_classifier = InputDepthClassifier()
    return _depth_classifier


class RAGState(TypedDict):
    # Inputs
    timestamp: str
    platform: str
    platform_channel_id: str
    platform_message_id: str
    decontexualized_input: str
    channel_topic: str
    input_context_to_timestamp: str

    # Input facts
    user_name: str
    global_user_id: str
    platform_bot_id: str
    character_profile: dict
    user_profile: dict

    # Stage-3 metadata thread (carried through every phase)
    input_embedding: list[float]
    depth: str                         # "SHALLOW" | "DEEP"
    depth_confidence: float
    cache_hit: bool
    trigger_dispatchers: list[str]
    rag_metadata: dict

    # External RAG Dispatcher output
    external_rag_next_action: str
    external_rag_task: str
    external_rag_context: dict
    external_rag_expected_response: str

    # External RAG output
    external_rag_results: Annotated[list[str], operator.add]

    # Input-Context RAG dispatcher output (was: Internal RAG)
    input_context_next_action: str
    input_context_task: str
    input_context_context: dict
    input_context_expected_response: str

    # Input-Context RAG output
    input_context_results: Annotated[list[str], operator.add]


_EXTERNAL_RAG_DISPATCHER_PROMPT = """\
你是角色 {character_name} 的外部感知中枢。你的目标是判断为了让角色做出真实的回应，我们需要检索哪些背景信息。
- 当前的系统时间为 {timestamp}

# 分析逻辑 (Priority)：
1. 外部知识：
   * 触发条件：
     a) 具有强时效性（如：今天的新闻、当下的天气、即时股价）。
     b) 极度专业/冷门（如：某个特定API的报错文档、特定经纬度的地图）。
     c) 认知模型知识截止日期之后发生的事件。
   * "next_action": "web_search_agent"
2. 认知模型处理：
   * 触发条件：大模型通过自身权重即可完美回答。
   * 包括：常识（如：天空颜色、科学定律）、逻辑推理、数学计算、语言翻译、情感安抚、日常寒暄、系统时间查询。
   * "next_action": "end"
3. 内部记忆：
   * 触发条件：涉及用户个人历史、之前的对话约定、角色私有的秘密或特定人际关系。
   * 你的工作不是获取内部记忆，所以你不需要做任何事情。
   * "next_action": "end"

# 任务指派信息
- "task": "具体要检索的任务描述"
- "context": (可选) 在 context 中提取关键实体（人物、时间、地点）。若不提供则默认为空字典 {{}}
- "expected_response":
  * 期望的返回格式（例如表格，长文本，短文本， YY/MM/DD, Yes/No），内容（包含的具体，或者宽泛的信息）和长度（例如<60字）
  * 返回格式应陈述事实，禁止包含第一人称描述。

# 输入格式
{{
    "user_input": "用户给你发送的信息",
    "channel_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）",
}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含但不限于以下字段：
{{
    "next_action": "web_search_agent" | "end",
    "reasoning": "string",
    "task": "string",
    "context": {{
        "target_user_input": "string",  // Transcribe what user says
        "target_user_topic": "string",  // Trasncribe the user topics in your own words
        "key": "value",  // other context
        ...
    }},
    "expected_response": "string"
}}
"""
_external_rag_dispatcher_llm = get_llm(temperature=0, top_p=1.0)
async def external_rag_dispatcher(state: RAGState) -> dict:
    decontexualized_input = state["decontexualized_input"]
    channel_topic = state["channel_topic"]
    timestamp = state["timestamp"]
    character_name=state["character_profile"]["name"],


    system_prompt = SystemMessage(content=_EXTERNAL_RAG_DISPATCHER_PROMPT.format(
        timestamp=timestamp,
        character_name=character_name
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "channel_topic": channel_topic
    }, ensure_ascii=False))

    response = await _external_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    logger.debug(
        "External RAG dispatcher: action=%s task=%s expected=%s context=%s reasoning=%s",
        next_action,
        log_preview(task, max_length=140),
        log_preview(expected_response, max_length=120),
        log_dict_subset(
            context if isinstance(context, dict) else {},
            ["target_user_input", "target_user_topic", "entities", "referenced_event"],
            value_length=80,
        ),
        log_preview(dispatcher_reasoning, max_length=140),
    )

    return {
        "external_rag_next_action": next_action,
        "external_rag_dispatcher_reasoning": dispatcher_reasoning,
        "external_rag_task": task,
        "external_rag_context": context,
        "external_rag_expected_response": expected_response
    }


_INPUT_CONTEXT_RAG_DISPATCHER_PROMPT = """\
你负责判断是否需要检索记忆库，并生成客观的检索任务。
- 你在社交平台的账号为 {platform_bot_id}，角色名为 {character_name}
- 当前的系统时间为 {timestamp}
- 消息 (`user_input`) 发送者为 {user_name}(global_user_id: {global_user_id})

# 分析逻辑 (Priority)：
1. 外部知识：
   * 触发条件：
     a) 具有强时效性（如：今天的新闻、当下的天气、即时股价）。
     b) 极度专业/冷门（如：某个特定API的报错文档、特定经纬度的地图）。
     c) 认知模型知识截止日期之后发生的事件。
   * 你的工作不是获取外部搜索，所以你不需要做任何事情。
   * "next_action": "end"
2. 认知模型处理：
   * 触发条件：大模型通过自身权重即可完美回答。
   * 包括：常识（如：天空颜色、科学定律）、逻辑推理、数学计算、语言翻译、情感安抚、日常寒暄、系统时间查询。
   * "next_action": "end"
3. 内部记忆：
   * 触发条件：用户输入中提到了**具体的名词、未完成的约定、之前的选择、或暗示过往背景的指代**。
   * 核心重心：**不再侧重"用户是个什么样的人"，而侧重"这件事/这个东西我们之前是怎么定的"**。
   * **过滤准则 (CRITICAL)：**
     a) **时间局部性**：优先检索 **3个月内** 创建的事实或承诺。
     b) **有效性过滤**：自动忽略"已完成" 或 "已过期" 的条目。
     c) **状态优先**：重点寻找 "进行中" 、"待定" 或 "未兑现" 的承诺（如：未送出的蛋糕、未完成的缝纫工作）。
   * "next_action": "memory_retriever_agent"

# 任务指派信息
- "task": 用客观语言描述要检索的内容。**不要**带入角色视角或与角色的关系——只描述要找的是什么信息（谁、什么事、何时）。
  * 正确示例："检索关于'啾啾'的身份描述、行为特征及最后提及时间"
  * 错误示例："检索'啾啾'与杏山千纱的关系"（不要加入角色名作为参照物）
- "context": (可选) 在 context 中提取关键实体（人物、时间、地点）。若不提供则默认为空字典 {{}}
- "expected_response": 包括以下
  * 明确要求返回事实细节。例如："具体的口味名称及对话时间"、"关于任务进度的最后一次描述"。
  * 期望的返回格式（例如表格，长文本，短文本， YY/MM/DD, Yes/No），具体内容和长度（例如<60字）
  * 返回格式应陈述客观事实，不要使用角色名或第一人称作为语境锚点。例如："{user_name} 提到..."，而非 "{character_name} 认为..."。

# 输入格式
{{
    "user_input": "用户给你发送的信息",
    "channel_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）"
}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含但不限于以下字段：
{{
    "next_action": "memory_retriever_agent" | "end",
    "reasoning": "string",
    "task": "string",
    "context": {{
        "entities": ["实体关键词"],  // Example
        "time_horizon": "Last 3 Months",  // Example
        "target_user_input": "string",  // Transcribe what user says
        "target_user_topic": "string",  // Trasncribe the user topics in your own words
        "status_filter": {{
            "include": ["pending", "active", "unfulfilled"],  // Example
            "exclude": ["accomplished", "expired", "past_due"],  // Example
            ...
        }},
        "referenced_event": "string"  // Example
    }},
    "expected_response": "string"
}}
"""
_input_context_rag_dispatcher_llm = get_llm(temperature=0, top_p=1.0)
async def input_context_rag_dispatcher(state: RAGState) -> dict:
    decontexualized_input = state["decontexualized_input"]
    channel_topic = state["channel_topic"]
    timestamp = state["timestamp"]
    character_name=state["character_profile"]["name"],
    platform_bot_id = state["platform_bot_id"]
    global_user_id = state["global_user_id"]
    user_name = state["user_name"]

    system_prompt = SystemMessage(content=_INPUT_CONTEXT_RAG_DISPATCHER_PROMPT.format(
        timestamp=timestamp,
        character_name=character_name,
        platform_bot_id=platform_bot_id,
        global_user_id=global_user_id,
        user_name=user_name
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "channel_topic": channel_topic
    }, ensure_ascii=False))

    response = await _input_context_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    logger.debug(
        "Input-context dispatcher: action=%s task=%s expected=%s context=%s reasoning=%s",
        next_action,
        log_preview(task, max_length=140),
        log_preview(expected_response, max_length=120),
        log_dict_subset(
            context if isinstance(context, dict) else {},
            ["target_user_input", "target_user_topic", "entities", "time_horizon", "referenced_event"],
            value_length=80,
        ),
        log_preview(dispatcher_reasoning, max_length=140),
    )

    return {
        "input_context_next_action": next_action,
        "input_context_dispatcher_reasoning": dispatcher_reasoning,
        "input_context_task": task,
        "input_context_context": context,
        "input_context_expected_response": expected_response
    }


async def call_web_search_agent(state: RAGState) -> dict:
    result = await web_search_agent(
        task=state["external_rag_task"],
        context=state["external_rag_context"],
        expected_response=state["external_rag_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    return {
        "external_rag_results": [processed_response]
    }


async def call_memory_retriever_agent_input_context_rag(state: RAGState) -> dict:
    context = dict(state["input_context_context"])
    # Override identity fields with ground-truth from pipeline state to prevent
    # the dispatcher LLM from injecting platform IDs (e.g. bot ID) as user UUID.
    context["target_user_name"] = state["user_name"]
    context["target_global_user_id"] = state["global_user_id"]
    context["target_platform"] = state["platform"]
    context["target_platform_channel_id"] = state["platform_channel_id"]
    context["target_to_timestamp"] = state["input_context_to_timestamp"]
    context["target_platform_bot_id"] = state["platform_bot_id"]

    result = await memory_retriever_agent(
        task=state["input_context_task"],
        context=context,
        expected_response=state["input_context_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    return {
        "input_context_results": [processed_response]
    }


async def _rag_noop(_: RAGState) -> dict:
    return {}


def _input_context_to_timestamp(chat_history_recent: list[dict], current_timestamp: str) -> str:
    """Return the automatic upper timestamp bound for input-context retrieval.

    Args:
        chat_history_recent: Immediate recent-history window already passed directly
            to downstream cognition.
        current_timestamp: Timestamp of the current in-flight turn.

    Returns:
        The earliest timestamp in ``chat_history_recent`` when present, otherwise
        the current turn timestamp. Conversation retrieval uses this as its
        automatic ``to_timestamp`` bound to avoid re-fetching the short-term
        window or the just-written current message.
    """
    if chat_history_recent:
        earliest_recent_timestamp = chat_history_recent[0].get("timestamp", "")
        if earliest_recent_timestamp:
            return earliest_recent_timestamp
    return current_timestamp


# ── Phase helpers ──────────────────────────────────────────────────


def _result_confidence(value: Any) -> float:
    """Estimate how informative a dispatcher's result is.

    A proxy for confidence until dispatchers emit one explicitly — used to
    drive the "internal_rag strong → skip external" early-exit decision and
    populate the metadata bundle.

    Args:
        value: A dispatcher result; may be ``str``, ``list`` of strings, or ``None``.

    Returns:
        ``0.0`` when empty / missing, scaling up with non-whitespace length,
        capped at ``1.0`` once the payload passes ~600 characters.
    """
    if value is None:
        return 0.0
    if isinstance(value, list):
        text = " ".join(str(v) for v in value if v)
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return 0.0
    # Length-based proxy: 120 chars ≈ 0.5, 600+ chars ≈ 1.0
    return min(1.0, len(text) / 600.0 + 0.2)


async def _probe_knowledge_base(
    cache: RAGCache,
    embedding: list[float],
) -> str:
    """Probe the global knowledge_base cache and return any matching distillation.

    This is a supplementary probe for DEEP passes only — it never short-circuits
    the dispatcher pipeline; it only enriches ``research_facts`` with cached
    cross-session topic knowledge.

    Args:
        cache: Process-wide ``RAGCache`` instance.
        embedding: Query embedding of ``decontexualized_input``.

    Returns:
        The ``knowledge_base_results`` string from the best matching cache entry,
        or an empty string if no entry meets the similarity threshold.
    """
    hit = await cache.retrieve_if_similar(
        embedding=embedding,
        cache_type="knowledge_base",
        global_user_id="",
        threshold=KNOWLEDGE_BASE_PROBE_THRESHOLD,
    )
    if hit is None:
        return ""
    results = hit.get("results") or {}
    kb_text = results.get("knowledge_base_results", "")
    logger.debug(
        "Knowledge-base cache hit: similarity=%.3f preview=%s",
        float(hit.get("similarity", 0.0)),
        log_preview(kb_text, max_length=160),
    )
    return kb_text


async def _probe_cache(
    cache: RAGCache,
    embedding: list[float],
    global_user_id: str,
) -> tuple[dict | None, list[dict]]:
    """Probe the three user-scoped cache types plus the global external one.

    Args:
        cache: Process-wide ``RAGCache`` instance.
        embedding: Query embedding of ``decontexualized_input``.
        global_user_id: Current user's internal UUID.

    Returns:
        ``(cached_payload, probe_trace)``. ``cached_payload`` is the best hit's
        ``results`` dict if any probe scored at or above the cache threshold,
        else ``None``. ``probe_trace`` is a per-cache-type record of the probes
        performed (hit/miss + similarity) for the metadata bundle.
    """
    probes = [
        ("objective_user_facts", global_user_id),
        ("character_diary", global_user_id),
        ("external_knowledge", ""),
    ]
    trace: list[dict] = []
    best: tuple[float, dict, str] | None = None
    for cache_type, owner in probes:
        hit = await cache.retrieve_if_similar(
            embedding=embedding,
            cache_type=cache_type,
            global_user_id=owner,
        )
        if hit is None:
            trace.append({"cache_type": cache_type, "hit": False})
            continue
        sim = float(hit["similarity"])
        trace.append({"cache_type": cache_type, "hit": True, "similarity": sim})
        if best is None or sim > best[0]:
            best = (sim, hit["results"], cache_type)
    if best is None:
        return None, trace
    sim, results, cache_type = best
    logger.debug("RAG cache hit: cache_type=%s similarity=%.3f", cache_type, sim)
    return results, trace


def _build_rag_graph(depth: str, affinity_percent: float):
    """Compile a dispatcher graph whose START edges match the chosen depth.

    Args:
        depth: ``"SHALLOW"`` or ``"DEEP"`` from the classifier.
        affinity_percent: User affinity normalised into a 0–100 band; used to
            reproduce the existing affinity-based external_rag skip.

    Returns:
        A compiled langgraph ``StateGraph`` ready for ``ainvoke``.
    """
    builder = StateGraph(RAGState)
    builder.add_node("external_rag_dispatcher", external_rag_dispatcher)
    builder.add_node("input_context_rag_dispatcher", input_context_rag_dispatcher)
    builder.add_node("call_web_search_agent", call_web_search_agent)
    builder.add_node("call_memory_retriever_agent_input_context_rag", call_memory_retriever_agent_input_context_rag)
    builder.add_node("rag_noop", _rag_noop)

    # DEEP: input_context_rag + (optionally) external_rag. SHALLOW: no dispatchers.
    has_entry_edge = depth == DEEP
    if depth == DEEP:
        builder.add_edge(START, "input_context_rag_dispatcher")
        if affinity_percent >= EXTERNAL_AFFINITY_SKIP_PERCENT:
            builder.add_edge(START, "external_rag_dispatcher")
    if not has_entry_edge:
        builder.add_edge(START, "rag_noop")

    # Dispatcher → retriever conditional edges
    builder.add_conditional_edges(
        "external_rag_dispatcher",
        lambda state: state["external_rag_next_action"],
        {"web_search_agent": "call_web_search_agent", "end": END},
    )
    builder.add_conditional_edges(
        "input_context_rag_dispatcher",
        lambda state: state["input_context_next_action"],
        {"memory_retriever_agent": "call_memory_retriever_agent_input_context_rag", "end": END},
    )

    # Fan-in
    builder.add_edge("call_web_search_agent", END)
    builder.add_edge("call_memory_retriever_agent_input_context_rag", END)
    builder.add_edge("rag_noop", END)

    return builder.compile()


async def _store_results_in_cache(
    cache: RAGCache,
    embedding: list[float],
    research_facts: dict,
) -> None:
    """Write the produced RAG results back into the cache.

    Stores external_rag_results as global ``external_knowledge`` so it can
    be invalidated independently of per-user data.

    Args:
        cache: Process-wide cache instance.
        embedding: Query embedding of the current input.
        research_facts: Dict returned to the caller — branch payloads.
    """
    external = research_facts.get("external_rag_results")
    if external:
        await cache.store(
            embedding=embedding,
            results={"external_rag_results": external},
            cache_type="external_knowledge",
            global_user_id="",
            metadata={"source": "rag_subgraph"},
        )


# ── Image assembly helper ──────────────────────────────────────────


def _assemble_image_text(image_doc: dict) -> str:
    """Render a three-tier image document into a compact text block for LLM context.

    Milestone entries are listed first (never compacted), followed by
    ``historical_summary`` (compressed older history), then ``recent_window``
    (most recent session observations).

    Args:
        image_doc: Three-tier image dict with keys ``milestones``,
            ``recent_window``, ``historical_summary``, and ``meta``.

    Returns:
        A formatted text block, or an empty string if the document is empty.
    """
    if not image_doc:
        return ""
    parts: list[str] = []
    milestones = image_doc.get("milestones") or []
    if milestones:
        parts.append("## Milestones")
        for m in milestones:
            cat = m.get("category", m.get("milestone_category", ""))
            desc = m.get("event", m.get("description", ""))
            superseded = m.get("superseded_by", "")
            if superseded:
                parts.append(f"- [{cat}] {desc} (superseded by: {superseded})")
            else:
                parts.append(f"- [{cat}] {desc}")
    historical_summary = image_doc.get("historical_summary", "")
    if historical_summary:
        parts.append("## Historical Summary")
        parts.append(historical_summary)
    recent_window = image_doc.get("recent_window") or []
    if recent_window:
        parts.append("## Recent Observations")
        for obs in recent_window:
            parts.append(f"- {obs}")
    return "\n".join(parts)


# ── Main entry point ───────────────────────────────────────────────


async def call_rag_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """Execute the five-phase RAG pipeline and return research_facts + metadata.

    Phases:
      0. Embed the decontextualised input, initialise the metadata bundle.
      1. Probe the cache — a strong match short-circuits the rest.
      2. Classify depth (SHALLOW vs DEEP) via ``InputDepthClassifier``.
      3. Compile + run the dispatcher graph (routes selected by depth + affinity).
      4. Store the produced results back into the cache.

    Args:
        state: The ``GlobalPersonaState`` carrying the current user, character,
            and pre-computed decontextualised input.

    Returns:
        A partial-state dict with ``research_facts`` (the per-branch payload
        dict consumed by downstream cognition nodes) and ``research_metadata``
        (the unified metadata bundle documenting this pass).
    """
    user_name = state["user_name"]
    global_user_id = state["global_user_id"]
    decontexualized_input = state["decontexualized_input"]
    user_profile = state["user_profile"]
    affinity_score = user_profile.get("affinity", AFFINITY_DEFAULT)
    affinity_percent = ((affinity_score - AFFINITY_MIN) / max(1, AFFINITY_MAX - AFFINITY_MIN)) * 100
    input_context_to_timestamp = _input_context_to_timestamp(
        state.get("chat_history_recent") or [],
        state["timestamp"],
    )

    user_image_text = _assemble_image_text(user_profile.get("user_image") or {})
    objective_facts_text = "\n".join(
        str(item.get("fact", item.get("description", "")))
        for item in (user_profile.get("objective_facts") or [])
        if str(item.get("fact", item.get("description", ""))).strip()
    )
    character_image_text = _assemble_image_text(
        state["character_profile"].get("self_image") or {}
    )

    logger.debug(
        "RAG input: user=%s global_user=%s channel=%s affinity=%s recent_history=%d input=%s",
        user_name,
        global_user_id,
        state["platform_channel_id"] or "<dm>",
        affinity_score,
        len(state.get("chat_history_recent") or []),
        log_preview(decontexualized_input, max_length=180),
    )

    # ── Phase 0: Input analysis ────────────────────────────────
    input_embedding = await get_text_embedding(decontexualized_input)
    metadata: dict = {
        "embedding_dim": len(input_embedding),
        "depth": None,
        "depth_confidence": 0.0,
        "depth_reasoning": "",
        "cache_hit": False,
        "cache_probe": [],
        "trigger_dispatchers": [],
        "rag_sources_used": [],
        "confidence_scores": {},
        "response_confidence": 0.0,
    }

    # ── Phase 1: Cache check ───────────────────────────────────
    cache = await _get_rag_cache()
    cached, probe_trace = await _probe_cache(cache, input_embedding, global_user_id)
    metadata["cache_probe"] = probe_trace
    if cached is not None:
        metadata["cache_hit"] = True
        metadata["rag_sources_used"] = ["cache"]
        metadata["response_confidence"] = 1.0
        research_facts = {
            "input_context_results": cached.get("input_context_results", ""),
            "external_rag_results": cached.get("external_rag_results", ""),
            "objective_facts": objective_facts_text,
            "user_image": user_image_text,
            "character_image": character_image_text,
        }
        logger.info(
            "RAG summary: user=%s global_user=%s cache_hit=%s depth=%s sources=%s input_context=%s external=%s input=%s",
            user_name,
            global_user_id,
            True,
            metadata.get("depth"),
            metadata.get("rag_sources_used", []),
            log_preview(research_facts["input_context_results"], max_length=140),
            log_preview(research_facts["external_rag_results"], max_length=140),
            log_preview(decontexualized_input, max_length=160),
        )
        return {
            "research_facts": research_facts,
            "research_metadata": [metadata],
        }

    # ── Phase 2: Depth classification ──────────────────────────
    classifier = _get_depth_classifier()
    depth_result = await classifier.classify(
        user_input=decontexualized_input,
        user_topic=state.get("channel_topic", ""),
        affinity=affinity_score,
        input_embedding=input_embedding,
    )
    depth = depth_result["depth"]
    metadata["depth"] = depth
    metadata["depth_confidence"] = depth_result["confidence"]
    metadata["depth_reasoning"] = depth_result["reasoning"]
    metadata["trigger_dispatchers"] = list(depth_result["trigger_dispatchers"])

    # ── Phase 2.5: Knowledge-base pre-probe (DEEP only) ───────
    knowledge_base_results = ""
    if depth == DEEP:
        knowledge_base_results = await _probe_knowledge_base(cache, input_embedding)

    # ── Phase 3: Dispatcher graph ──────────────────────────────
    rag_graph = _build_rag_graph(depth, affinity_percent)

    initial_state: RAGState = {
        "timestamp": state["timestamp"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "platform_message_id": state["platform_message_id"],
        "decontexualized_input": decontexualized_input,
        "channel_topic": state["channel_topic"],
        "input_context_to_timestamp": input_context_to_timestamp,
        "user_name": user_name,
        "global_user_id": global_user_id,
        "platform_bot_id": state["platform_bot_id"],
        "character_profile": state["character_profile"],
        "user_profile": user_profile,
        "input_embedding": input_embedding,
        "depth": depth,
        "depth_confidence": depth_result["confidence"],
        "cache_hit": False,
        "trigger_dispatchers": list(depth_result["trigger_dispatchers"]),
        "rag_metadata": metadata,
    }

    result = await rag_graph.ainvoke(initial_state)

    input_context_results = result.get("input_context_results", "")
    external_rag_results = result.get("external_rag_results", "")

    # Confidence scoring — proxy signal until dispatchers emit explicit scores.
    input_context_conf = _result_confidence(input_context_results)
    external_conf = _result_confidence(external_rag_results)
    metadata["confidence_scores"] = {
        "input_context_rag": input_context_conf,
        "external_rag": external_conf,
    }
    sources_used = []
    if input_context_conf > 0.0:
        sources_used.append("input_context_rag")
    if external_conf > 0.0:
        sources_used.append("external_rag")
    metadata["rag_sources_used"] = sources_used
    metadata["response_confidence"] = max([input_context_conf, external_conf] + [0.0])

    # Record the early-exit decisions implied by the confidence signals.
    metadata["early_exit"] = {
        "deep_input_context_sufficient": depth == DEEP and input_context_conf >= INTERNAL_RAG_STRONG_THRESHOLD,
    }

    logger.debug(
        "RAG metadata: depth=%s confidence=%.3f cache_probe=%s trigger_dispatchers=%s response_confidence=%.3f sources=%s",
        depth,
        depth_result["confidence"],
        metadata.get("cache_probe", []),
        metadata.get("trigger_dispatchers", []),
        metadata.get("response_confidence", 0.0),
        metadata.get("rag_sources_used", []),
    )

    research_facts = {
        "input_context_results": input_context_results,
        "external_rag_results": external_rag_results,
        "objective_facts": objective_facts_text,
        "user_image": user_image_text,
        "character_image": character_image_text,
        "knowledge_base_results": knowledge_base_results,
    }

    logger.info(
        "RAG summary: user=%s global_user=%s cache_hit=%s depth=%s depth_conf=%.2f sources=%s kb_hit=%s input_context=%s external=%s input=%s",
        user_name,
        global_user_id,
        False,
        depth,
        depth_result["confidence"],
        sources_used,
        bool(knowledge_base_results),
        log_preview(input_context_results, max_length=140),
        log_preview(external_rag_results, max_length=140),
        log_preview(decontexualized_input, max_length=160),
    )

    # ── Phase 4: Cache storage ─────────────────────────────────
    await _store_results_in_cache(
        cache, input_embedding, research_facts,
    )

    return {
        "research_facts": research_facts,
        "research_metadata": [metadata],
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.mcp_client import mcp_manager
    from kazusa_ai_chatbot.db import get_character_profile


    # Connect to MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    state: GlobalPersonaState = {
        "decontexualized_input": "千纱晚上要记得奖励我哦♥",
        "channel_topic": "闲聊",
        "platform_bot_id": "1485169644888395817",
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "character_profile": await get_character_profile(),
        "user_profile": {"affinity": 950},
    }

    result = await call_rag_subgraph(state)
    print(f"RAG SubGraph: {result}")


    await mcp_manager.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
