"""Stage 1 / RAG subgraph with semantic cache + depth-based dispatch.

The subgraph has five wrapper phases on top of the existing dispatchers:

* **Phase 0 — Input analysis**: embed ``decontexualized_input`` and initialise
  the unified metadata bundle that accumulates through the whole RAG pass.
* **Phase 1 — Cache check**: probe ``RAGCache`` for ``objective_user_facts``,
  ``character_diary`` and ``external_knowledge`` matches. A strong hit short-
  circuits the rest of the pipeline.
* **Phase 2 — Depth classification**: ``InputDepthClassifier`` selects
  ``SHALLOW`` (user_rag only) or ``DEEP`` (all three dispatchers).
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
    SHALLOW,
    InputDepthClassifier,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm, build_affinity_block

import json
import logging
import operator
from typing import TypedDict, Annotated, Any


logger = logging.getLogger(__name__)


# ── Stage-3 tuning constants ───────────────────────────────────────
# The thresholds below live here until Stage 5 moves them into config.py.

CACHE_HIT_THRESHOLD = 0.82            # similarity required to serve from cache
USER_RAG_STRONG_THRESHOLD = 0.65      # user_rag deemed sufficient for SHALLOW
INTERNAL_RAG_STRONG_THRESHOLD = 0.55  # internal_rag deemed sufficient for DEEP
EXTERNAL_AFFINITY_SKIP_PERCENT = 40   # skip external_rag below this affinity %


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
    decontexualized_input: str
    user_topic: str

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

    # Internal RAG dispatcher output
    internal_rag_next_action: str
    internal_rag_task: str
    internal_rag_context: dict
    internal_rag_expected_response: str

    # Internal RAG output
    internal_rag_results: Annotated[list[str], operator.add]

    # User RAG dispatcher output
    user_rag_next_action: str
    user_rag_task: str
    user_rag_context: dict
    user_rag_expected_response: str

    # User RAG output
    user_rag_results: Annotated[list[str], operator.add]
    user_rag_finalized: str


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
    "user_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）",
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
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name=state["character_profile"]["name"],


    system_prompt = SystemMessage(content=_EXTERNAL_RAG_DISPATCHER_PROMPT.format(
        timestamp=timestamp,
        character_name=character_name
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "user_topic": user_topic
    }, ensure_ascii=False))

    response = await _external_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"External RAG agent dispatcher result: {result}")

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    return {
        "external_rag_next_action": next_action,
        "external_rag_dispatcher_reasoning": dispatcher_reasoning,
        "external_rag_task": task,
        "external_rag_context": context,
        "external_rag_expected_response": expected_response
    }


_INTERNAL_RAG_DISPATCHER_PROMPT = """\
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
    "user_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）"
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
_internal_rag_dispatcher_llm = get_llm(temperature=0, top_p=1.0)
async def internal_rag_dispatcher(state: RAGState) -> dict:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name=state["character_profile"]["name"],
    platform_bot_id = state["platform_bot_id"]
    global_user_id = state["global_user_id"]
    user_name = state["user_name"]

    system_prompt = SystemMessage(content=_INTERNAL_RAG_DISPATCHER_PROMPT.format(
        timestamp=timestamp,
        character_name=character_name,
        platform_bot_id=platform_bot_id,
        global_user_id=global_user_id,
        user_name=user_name
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "user_topic": user_topic
    }, ensure_ascii=False))

    response = await _internal_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Internal RAG agent dispatcher result: {result}")

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    return {
        "internal_rag_next_action": next_action,
        "internal_rag_dispatcher_reasoning": dispatcher_reasoning,
        "internal_rag_task": task,
        "internal_rag_context": context,
        "internal_rag_expected_response": expected_response
    }


_USER_FACT_RAG_DISPATCHER_PROMPT = """\
你负责从角色 {character_name} 的记忆库中提取关于 {user_name} 的原始素材。你需要通过多路查询确保覆盖"当前事实对齐"与"历史情感锚点"。
- 当前的系统时间为 {timestamp}

# 身份锚定 (Identity Anchor)
- 当前检索对象固定为：`{user_name}`。
- 只允许检索"该用户本人"的历史事实、行为记录、承诺状态与关系变化。
- 若 `user_input` 没有明确实体，优先围绕该用户最近互动、最近承诺、最近情绪波动进行检索。
- 严禁把任务扩展为"泛人格分析"或"抽象心理画像"（如"掌控欲/意志力/支配型人格"）除非输入中出现可验证证据。
- 严禁把检索主体漂移到其他用户。

# 检索策略 (Search Strategy)：
1. **语义对齐 (Fact Match)**：提取 `user_input` 中与该用户相关的具体实体（如：礼物、承诺、地点、时间），生成针对性查询。
2. **承诺对齐 (Promise Match)**：优先检索与该用户有关的未完成约定、未来承诺、状态变更（active/unfulfilled）。
3. **关系证据 (Evidence Match)**：检索可验证的互动证据（原话、时间戳、行为、好感变化），避免抽象标签化描述。

# 任务生成约束 (Critical)
- `task` 必须包含 `{user_name}`，并明确写出"检索该用户相关记录"。
- `task` 与 `context.entities` 必须优先使用输入中的原词，不要凭空引入新概念。
- 当输入是日常请求或具体事件时，不要升级为"人格审讯式"任务。
- 优先时间范围：最近 90 天；如证据不足再放宽。

# 输入格式 (Input Format)：
{{
    "user_input": "用户当前的发言内容",
    "user_topic": "当前对话的主题摘要",
    "character_mood": "角色的即时情绪",
    "affinity_context": dict,  // "当前{user_name}在{character_name}心中的好感度描述"
}}

# 输出要求：
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "next_action": "memory_retriever_agent",
    "reasoning": "string",
    "task": "基于输入事实与重大情感转折点的复合检索指令",
    "context": {{
        "target_user_name": "{user_name}",
        "target_global_user_id": "{global_user_id}",
        "entities": ["关键实体词"],
        "search_logic": "user_anchored_multi_track",
        "high_intensity_mode": true,
        "time_range": "last_90_days_then_expand"
    }},
    "expected_response": "仅返回与 {user_name} 相关的原始记录清单：包含时间戳、用户名称、原始行为描述、好感度变动；禁止输出无证据的人格推断"
}}
"""
_user_fact_rag_dispatcher_llm = get_llm(temperature=0, top_p=1.0)
async def user_fact_rag_dispatcher(state: RAGState) -> dict:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name = state["character_profile"]["name"],
    user_name = state["user_name"]

    user_affinity_score = state["user_profile"].get("affinity", AFFINITY_DEFAULT)
    affinity_block = build_affinity_block(user_affinity_score)

    system_prompt = SystemMessage(content=_USER_FACT_RAG_DISPATCHER_PROMPT.format(
        character_name=character_name,
        user_name=user_name,
        timestamp=timestamp,
        global_user_id=state["global_user_id"],
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "user_topic": user_topic,
        "character_mood": state["character_profile"]["mood"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
    }, ensure_ascii=False))

    response = await _user_fact_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"User Fact RAG dispatcher result: {result}")

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    # Override identity fields with ground truth from state — never trust the LLM
    # to copy strings verbatim; tokenizer artifacts can corrupt them (e.g. "蚝爹\t油").
    context["target_user_name"] = user_name
    context["target_global_user_id"] = state["global_user_id"]

    return {
        "user_rag_next_action": next_action,
        "user_rag_dispatcher_reasoning": dispatcher_reasoning,
        "user_rag_task": task,
        "user_rag_context": context,
        "user_rag_expected_response": expected_response
    }



_USER_FACT_RAG_FINALIZER_PROMPT = """\
你负责处理 {character_name} 脑内检索回来的原始碎片。你需要模拟人类大脑，根据当前好感度对记忆进行"主观扭曲"，并执行人工时间衰减。
- 当前的系统时间为 {timestamp}
- 对方用户名为 {user_name}

# 核心处理协议：
1. **好感度滤镜 (affinity_context.lebel)**：
   - **正面词汇**：优先高亮用户的善意。将负面记忆处理为"可原谅的失误"或"傲娇的抱怨点"。
   - **负面词汇**：优先高亮用户的冒犯。将正面记忆处理为"虚伪的讨好"或"值得警惕的异常"。
2. **人工时间衰减 (Temporal Decay Processing)**：
   - **近期 (0-7 days)**：保留高保真细节（具体台词、精确动作）。
   - **中期 (8-60 days)**：压缩为具体事件（发生了什么，结果如何）。
   - **远期 (> 60 days)**：完全抽象化为性格印象（他是个什么样的人）。

# 输入格式 (Input Format)：
{{
    "user_rag_results": ["..."],
    "affinity_context": dict,  // "当前{user_name}在{character_name}心中的好感度描述"
}}

# 输出要求：
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "user_rag_finalized": ["..."]  // 保留与 `user_rag_results` 相同结构，但已根据好感度滤镜和时间衰减处理
}}
"""
_user_fact_rag_finalizer_llm = get_llm(temperature=0.2, top_p=0.9)
async def user_fact_rag_finalizer(state: RAGState) -> dict:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name = state["character_profile"]["name"],
    user_name = state["user_name"]

    user_affinity_score = state["user_profile"].get("affinity", AFFINITY_DEFAULT)
    affinity_block = build_affinity_block(user_affinity_score)

    system_prompt = SystemMessage(content=_USER_FACT_RAG_FINALIZER_PROMPT.format(
        character_name=character_name,
        user_name=user_name,
        timestamp=timestamp,
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_rag_results": state["user_rag_results"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
    }, ensure_ascii=False))

    response = await _user_fact_rag_finalizer_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"User fact RAG finalizer result: {result}")

    user_rag_finalized = result.get("user_rag_finalized", "")

    return {
        "user_rag_finalized": user_rag_finalized
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


async def call_memory_retriever_agent_internal_rag(state: RAGState) -> dict:
    context = dict(state["internal_rag_context"])
    # Override identity fields with ground-truth from pipeline state to prevent
    # the dispatcher LLM from injecting platform IDs (e.g. bot ID) as user UUID.
    context["target_user_name"] = state["user_name"]
    context["target_global_user_id"] = state["global_user_id"]

    result = await memory_retriever_agent(
        task=state["internal_rag_task"],
        context=context,
        expected_response=state["internal_rag_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    return {
        "internal_rag_results": [processed_response]
    }


async def call_memory_retriever_agent_user_rag(state: RAGState) -> dict:
    result = await memory_retriever_agent(
        task=state["user_rag_task"],
        context=state["user_rag_context"],
        expected_response=state["user_rag_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    return {
        "user_rag_results": [processed_response]
    }


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
    logger.info("RAG cache HIT on %s (sim=%.3f)", cache_type, sim)
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
    builder.add_node("internal_rag_dispatcher", internal_rag_dispatcher)
    builder.add_node("user_rag_dispatcher", user_fact_rag_dispatcher)
    builder.add_node("call_web_search_agent", call_web_search_agent)
    builder.add_node("call_memory_retriever_agent_internal_rag", call_memory_retriever_agent_internal_rag)
    builder.add_node("call_memory_retriever_agent_user_rag", call_memory_retriever_agent_user_rag)
    builder.add_node("call_user_fact_rag_finalizer", user_fact_rag_finalizer)

    # user_rag always runs — it is the only dispatcher allowed in SHALLOW.
    builder.add_edge(START, "user_rag_dispatcher")

    # internal_rag + external_rag only when DEEP.
    if depth == DEEP:
        builder.add_edge(START, "internal_rag_dispatcher")
        if affinity_percent >= EXTERNAL_AFFINITY_SKIP_PERCENT:
            builder.add_edge(START, "external_rag_dispatcher")

    # Dispatcher → retriever conditional edges
    builder.add_conditional_edges(
        "external_rag_dispatcher",
        lambda state: state["external_rag_next_action"],
        {"web_search_agent": "call_web_search_agent", "end": END},
    )
    builder.add_conditional_edges(
        "internal_rag_dispatcher",
        lambda state: state["internal_rag_next_action"],
        {"memory_retriever_agent": "call_memory_retriever_agent_internal_rag", "end": END},
    )
    builder.add_conditional_edges(
        "user_rag_dispatcher",
        lambda state: state["user_rag_next_action"],
        {
            "memory_retriever_agent": "call_memory_retriever_agent_user_rag",
            "end": "call_user_fact_rag_finalizer",
        },
    )
    builder.add_edge("call_memory_retriever_agent_user_rag", "call_user_fact_rag_finalizer")

    # Fan-in
    builder.add_edge("call_web_search_agent", END)
    builder.add_edge("call_memory_retriever_agent_internal_rag", END)
    builder.add_edge("call_user_fact_rag_finalizer", END)

    return builder.compile()


async def _store_results_in_cache(
    cache: RAGCache,
    embedding: list[float],
    global_user_id: str,
    research_facts: dict,
    metadata: dict,
) -> None:
    """Write the produced RAG results back into the cache.

    One entry per populated branch — user_rag_finalized as
    ``objective_user_facts``, external_rag_results as global
    ``external_knowledge``, so each can be invalidated independently.

    Args:
        cache: Process-wide cache instance.
        embedding: Query embedding of the current input.
        global_user_id: Current user's internal UUID.
        research_facts: Dict returned to the caller — branch payloads.
        metadata: The metadata bundle to store alongside each entry.
    """
    user_rag = research_facts.get("user_rag_finalized")
    if user_rag:
        await cache.store(
            embedding=embedding,
            results={"user_rag_finalized": user_rag},
            cache_type="objective_user_facts",
            global_user_id=global_user_id,
            metadata={"source": "rag_subgraph", "depth": metadata.get("depth")},
        )

    external = research_facts.get("external_rag_results")
    if external:
        await cache.store(
            embedding=embedding,
            results={"external_rag_results": external},
            cache_type="external_knowledge",
            global_user_id="",
            metadata={"source": "rag_subgraph"},
        )


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
            "user_rag_finalized": cached.get("user_rag_finalized", ""),
            "internal_rag_results": cached.get("internal_rag_results", ""),
            "external_rag_results": cached.get("external_rag_results", ""),
        }
        logger.info(
            f"\n{user_name}(@{global_user_id}): {decontexualized_input}\n"
            f"[CACHE HIT] research_facts served from cache"
        )
        return {
            "research_facts": research_facts,
            "research_metadata": [metadata],
        }

    # ── Phase 2: Depth classification ──────────────────────────
    classifier = _get_depth_classifier()
    depth_result = await classifier.classify(
        user_input=decontexualized_input,
        user_topic=state.get("user_topic", ""),
        affinity=affinity_score,
    )
    depth = depth_result["depth"]
    metadata["depth"] = depth
    metadata["depth_confidence"] = depth_result["confidence"]
    metadata["depth_reasoning"] = depth_result["reasoning"]
    metadata["trigger_dispatchers"] = list(depth_result["trigger_dispatchers"])

    # ── Phase 3: Dispatcher graph ──────────────────────────────
    rag_graph = _build_rag_graph(depth, affinity_percent)

    initial_state: RAGState = {
        "timestamp": state["timestamp"],
        "decontexualized_input": decontexualized_input,
        "user_topic": state["user_topic"],
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

    user_rag_finalized = result.get("user_rag_finalized", "")
    internal_rag_results = result.get("internal_rag_results", "")
    external_rag_results = result.get("external_rag_results", "")

    # Confidence scoring — proxy signal until dispatchers emit explicit scores.
    user_conf = _result_confidence(user_rag_finalized)
    internal_conf = _result_confidence(internal_rag_results)
    external_conf = _result_confidence(external_rag_results)
    metadata["confidence_scores"] = {
        "user_rag": user_conf,
        "internal_rag": internal_conf,
        "external_rag": external_conf,
    }
    sources_used = []
    if user_conf > 0.0:
        sources_used.append("user_rag")
    if internal_conf > 0.0:
        sources_used.append("internal_rag")
    if external_conf > 0.0:
        sources_used.append("external_rag")
    metadata["rag_sources_used"] = sources_used
    metadata["response_confidence"] = max([user_conf, internal_conf, external_conf] + [0.0])

    # Record the early-exit decisions implied by the confidence signals.
    # These are descriptive (documenting what the plan's thresholds would gate
    # on), since depth already controlled which dispatchers actually ran.
    metadata["early_exit"] = {
        "shallow_user_rag_sufficient": depth == SHALLOW and user_conf >= USER_RAG_STRONG_THRESHOLD,
        "deep_internal_rag_sufficient": depth == DEEP and internal_conf >= INTERNAL_RAG_STRONG_THRESHOLD,
    }

    research_facts = {
        "user_rag_finalized": user_rag_finalized,
        "internal_rag_results": internal_rag_results,
        "external_rag_results": external_rag_results,
    }

    logger.info(
        f"\n{user_name}(@{global_user_id}): {decontexualized_input}\n"
        f"Depth: {depth} (conf={depth_result['confidence']:.2f})\n"
        f"User RAG finalized: {user_rag_finalized}\n"
        f"Internal RAG results: {internal_rag_results}\n"
        f"External RAG results: {external_rag_results}"
    )

    # ── Phase 4: Cache storage ─────────────────────────────────
    await _store_results_in_cache(
        cache, input_embedding, global_user_id, research_facts, metadata,
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
        "user_topic": "闲聊",
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
