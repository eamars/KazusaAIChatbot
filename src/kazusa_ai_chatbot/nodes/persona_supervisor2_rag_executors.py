"""Retrieval dispatchers, agent wrappers, and tier-gate functions.

All executor nodes and their LLM singletons live here. Moved from
``persona_supervisor2_rag.py`` during Phase 6 decomposition.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.agents.memory_retriever_agent import memory_retriever_agent
from kazusa_ai_chatbot.agents.web_search_agent2 import web_search_agent
from kazusa_ai_chatbot.db import get_user_profile
from kazusa_ai_chatbot.db.conversation import (
    get_conversation_history,
    search_conversation_history,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_schema import RAGState, _build_image_context
from kazusa_ai_chatbot.utils import (
    get_llm,
    log_dict_subset,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)


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
    "channel_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）"
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
    character_name = state["character_profile"]["name"]

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
        log_preview(task),
        log_preview(expected_response),
        log_dict_subset(
            context if isinstance(context, dict) else {},
            ["target_user_input", "target_user_topic", "entities", "referenced_event"],
        ),
        log_preview(dispatcher_reasoning),
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
    character_name = state["character_profile"]["name"]
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
        log_preview(task),
        log_preview(expected_response),
        log_dict_subset(
            context if isinstance(context, dict) else {},
            ["target_user_input", "target_user_topic", "entities", "time_horizon", "referenced_event"],
        ),
        log_preview(dispatcher_reasoning),
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
        "external_rag_results": [processed_response],
        "external_rag_is_empty_result": bool(result.get("is_empty_result", False)),
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
        "input_context_results": [processed_response],
        "input_context_is_empty_result": bool(result.get("is_empty_result", False)),
    }


# ── Phase 1 — Continuation Resolver ───────────────────────────────

_CONTINUATION_RESOLVER_PROMPT = """\
你是一个上下文延续解析器。你的任务是判断用户的输入是否是一个"延续性话语"（continuation / follow-up / reply-only），如果是，则从最近的对话历史中补全被省略的对象或任务。

# 判断标准
- 延续性话语的特征：缺乏独立语义，依赖上文才能理解（如："那后来呢？"、"然后呢？"、"这个呢？"）
- 如果用户输入已经是完整的查询（包含明确主语和宾语），则 `needs_context_resolution` 为 false
- 如果能从 `chat_history_recent` 确定被省略的对象/任务，则补全并输出 `resolved_task`
- 如果无法确定，则保持低置信度，不要猜测

# 输入格式
{{
    "decontextualized_input": "消歧后的用户输入",
    "chat_history_recent": [最近对话记录],
    "channel_topic": "频道话题"
}}

# 输出格式
请务必返回合法的 JSON 字符串：
{{
    "needs_context_resolution": true或false,
    "resolved_task": "补全后的完整查询任务（如果 needs_context_resolution 为 false 则填原始输入）",
    "known_slots": {{"被识别出的实体或对象": "值"}},
    "missing_slots": ["仍然缺失的信息"],
    "confidence": 0.0到1.0,
    "evidence": ["reply_target", "recent_turn", "channel_topic 等使用的线索来源"]
}}
"""
_continuation_resolver_llm = get_llm(temperature=0.1, top_p=0.85)


async def continuation_resolver(state: RAGState) -> dict:
    system_prompt = SystemMessage(content=_CONTINUATION_RESOLVER_PROMPT)
    user_input = {
        "decontextualized_input": state["decontexualized_input"],
        "chat_history_recent": (state.get("chat_history_recent") or [])[-6:],
        "channel_topic": state.get("channel_topic", ""),
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    response = await _continuation_resolver_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    continuation_context = {
        "needs_context_resolution": bool(result.get("needs_context_resolution", False)),
        "resolved_task": str(result.get("resolved_task", state["decontexualized_input"])),
        "known_slots": result.get("known_slots", {}),
        "missing_slots": result.get("missing_slots", []),
        "confidence": float(result.get("confidence", 0.0)),
        "evidence": result.get("evidence", []),
    }

    logger.debug(
        "Continuation resolver: needs_resolution=%s confidence=%.2f resolved_task=%s",
        continuation_context["needs_context_resolution"],
        continuation_context["confidence"],
        log_preview(continuation_context["resolved_task"]),
    )

    return {"continuation_context": continuation_context}


# ── Phase 1 — RAG Planner ────────────────────────────────────────

_RAG_PLANNER_PROMPT = """\
你是检索计划生成器。你的任务是根据用户的输入，决定需要激活哪些检索源来帮助角色生成准确的回应。

# 可用的检索源
- `CURRENT_USER_STABLE`: 当前用户的稳定画像信息（已预加载，通常不需要额外检索）
- `CHANNEL_RECENT_ENTITY`: 同频道近期提到的第三方实体/人物的对话记录
- `THIRD_PARTY_PROFILE`: 已知其他用户的持久画像/印象
- `EXTERNAL_KNOWLEDGE`: 外部知识（新闻、专业知识等）

# 检索模式 (retrieval_mode)
- `NONE`: 不需要任何检索（日常问候、简单情感互动）
- `CURRENT_USER_STABLE`: 只需要当前用户自身信息
- `THIRD_PARTY_PROFILE`: 关于特定已知第三方的持久印象
- `CHANNEL_RECENT_ENTITY`: 关于频道近期提到的人/事
- `EXTERNAL_KNOWLEDGE`: 需要外部知识
- `CASCADED`: 需要组合多个检索源

# 实体识别
从输入中提取所有提及的人物/实体（不含当前用户和角色本身），并标注其类型。

# 输入格式
{{
    "decontextualized_input": "用户输入",
    "resolved_task": "延续解析器补全后的任务（如果没有补全则与 decontextualized_input 相同）",
    "continuation_needs_resolution": true或false,
    "continuation_confidence": 0.0到1.0,
    "user_name": "当前用户名",
    "channel_topic": "频道话题",
    "chat_history_recent_speakers": ["最近对话中出现的发言者名称"],
    "timestamp": "当前时间"
}}

# 输出格式
请务必返回合法的 JSON 字符串：
{{
    "retrieval_mode": "NONE | CURRENT_USER_STABLE | THIRD_PARTY_PROFILE | CHANNEL_RECENT_ENTITY | EXTERNAL_KNOWLEDGE | CASCADED",
    "active_sources": ["需要激活的检索源列表"],
    "task": "检索任务描述",
    "entities": [
        {{
            "surface_form": "实体表面形式",
            "entity_type": "person | group | topic | unknown",
            "resolution_confidence": 0.0到1.0
        }}
    ],
    "subject": {{
        "kind": "current_user | third_party_user | entity | topic | mixed",
        "primary_entity": "主要实体名称（可选）"
    }},
    "time_scope": {{
        "kind": "recent | explicit_range | none",
        "lookback_hours": 72
    }},
    "search_scope": {{
        "same_channel": true,
        "cross_channel": false,
        "current_user_only": false
    }},
    "external_task_hint": "外部搜索提示（可选）",
    "expected_response": "期望的检索结果描述"
}}
"""
_rag_planner_llm = get_llm(temperature=0.1, top_p=0.85)


async def rag_planner(state: RAGState) -> dict:
    continuation_context = state.get("continuation_context") or {}
    resolved_task = continuation_context.get("resolved_task", state["decontexualized_input"])

    recent_speakers = set()
    for msg in (state.get("chat_history_recent") or []):
        name = msg.get("display_name", "")
        if name:
            recent_speakers.add(name)

    system_prompt = SystemMessage(content=_RAG_PLANNER_PROMPT)
    user_input = {
        "decontextualized_input": state["decontexualized_input"],
        "resolved_task": resolved_task,
        "continuation_needs_resolution": continuation_context.get("needs_context_resolution", False),
        "continuation_confidence": continuation_context.get("confidence", 0.0),
        "user_name": state["user_name"],
        "channel_topic": state.get("channel_topic", ""),
        "chat_history_recent_speakers": sorted(recent_speakers),
        "timestamp": state["timestamp"],
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    response = await _rag_planner_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    retrieval_plan = {
        "retrieval_mode": str(result.get("retrieval_mode", "NONE")),
        "active_sources": result.get("active_sources", []),
        "task": str(result.get("task", "")),
        "entities": result.get("entities", []),
        "subject": result.get("subject", {}),
        "time_scope": result.get("time_scope", {"kind": "none", "lookback_hours": 72}),
        "search_scope": result.get("search_scope", {"same_channel": True, "cross_channel": False, "current_user_only": False}),
        "external_task_hint": str(result.get("external_task_hint", "")),
        "expected_response": str(result.get("expected_response", "")),
    }

    logger.debug(
        "RAG planner: mode=%s sources=%s entities=%s subject=%s",
        retrieval_plan["retrieval_mode"],
        retrieval_plan["active_sources"],
        [e.get("surface_form") for e in retrieval_plan["entities"]],
        retrieval_plan.get("subject", {}),
    )

    return {"retrieval_plan": retrieval_plan}


# ── Phase 1 — Entity Grounder ─────────────────────────────────────

_ENTITY_GROUNDER_LLM_PROMPT = """\
你是一个实体消歧器。给定一个表面形式和候选匹配列表，判断最可能的匹配。

# 输入
{{
    "surface_form": "待解析的名称",
    "candidates": [
        {{"display_name": "候选名称", "global_user_id": "UUID", "similarity": "匹配理由"}}
    ]
}}

# 输出
请务必返回合法的 JSON 字符串：
{{
    "resolved_global_user_id": "匹配的 UUID 或空字符串",
    "confidence": 0.0到1.0,
    "reasoning": "判断理由"
}}
"""
_entity_grounder_llm = get_llm(temperature=0.0, top_p=1.0)


async def entity_grounder(state: RAGState) -> dict:
    retrieval_plan = state.get("retrieval_plan") or {}
    entities = retrieval_plan.get("entities", [])
    if not entities:
        return {
            "resolved_entities": [],
            "entity_resolution_notes": "",
        }

    chat_history_recent = state.get("chat_history_recent") or []
    participants: dict[str, str] = {}
    for msg in chat_history_recent:
        display_name = msg.get("display_name", "")
        global_user_id = msg.get("global_user_id", "")
        if display_name and global_user_id:
            participants[display_name.lower()] = global_user_id

    resolved_entities: list[dict] = []
    resolution_notes: list[str] = []

    for entity in entities:
        surface_form = str(entity.get("surface_form", ""))
        entity_type = str(entity.get("entity_type", "unknown"))

        resolved_id = ""
        confidence = 0.0
        method = "unresolved"

        # Step 1: Deterministic match against recent history participants
        lower_surface = surface_form.lower()
        if lower_surface in participants:
            resolved_id = participants[lower_surface]
            confidence = 0.95
            method = "exact_display_name"
        else:
            for name, uid in participants.items():
                if lower_surface in name or name in lower_surface:
                    resolved_id = uid
                    confidence = 0.80
                    method = "partial_display_name"
                    break

        # Step 2: If still unresolved and entity type is person, try LLM fallback
        if not resolved_id and entity_type == "person" and participants:
            candidates = [
                {"display_name": name, "global_user_id": uid, "similarity": "chat participant"}
                for name, uid in participants.items()
            ]
            try:
                system_prompt = SystemMessage(content=_ENTITY_GROUNDER_LLM_PROMPT)
                human_message = HumanMessage(content=json.dumps({
                    "surface_form": surface_form,
                    "candidates": candidates,
                }, ensure_ascii=False))
                response = await _entity_grounder_llm.ainvoke([system_prompt, human_message])
                llm_result = parse_llm_json_output(response.content)
                llm_resolved_id = str(llm_result.get("resolved_global_user_id", ""))
                llm_confidence = float(llm_result.get("confidence", 0.0))
                if llm_resolved_id and llm_confidence >= 0.6:
                    resolved_id = llm_resolved_id
                    confidence = llm_confidence
                    method = "llm_disambiguation"
            except Exception:
                logger.warning("Entity grounder LLM fallback failed for %s", surface_form, exc_info=True)

        resolved_entity = {
            "surface_form": surface_form,
            "entity_type": entity_type,
            "resolved_global_user_id": resolved_id,
            "resolution_confidence": confidence,
            "resolution_method": method,
        }
        resolved_entities.append(resolved_entity)

        if resolved_id:
            resolution_notes.append(f"{surface_form} → resolved (method={method}, confidence={confidence:.2f})")
        else:
            resolution_notes.append(f"{surface_form} → unresolved (type={entity_type})")

    entity_resolution_notes = "; ".join(resolution_notes) if resolution_notes else ""

    logger.debug(
        "Entity grounder: resolved=%d/%d notes=%s",
        sum(1 for e in resolved_entities if e["resolved_global_user_id"]),
        len(resolved_entities),
        log_preview(entity_resolution_notes),
    )

    return {
        "resolved_entities": resolved_entities,
        "entity_resolution_notes": entity_resolution_notes,
    }


# ── Phase 2 — Channel Recent Entity RAG ──────────────────────────

CHANNEL_RECENT_ENTITY_LOOKBACK_HOURS = 72
CHANNEL_RECENT_ENTITY_MAX_RESULTS = 20


async def channel_recent_entity_rag(state: RAGState) -> dict:
    resolved_entities = state.get("resolved_entities") or []
    retrieval_plan = state.get("retrieval_plan") or {}
    active_sources = retrieval_plan.get("active_sources", [])
    ledger = dict(state.get("retrieval_ledger") or {})

    if "CHANNEL_RECENT_ENTITY" not in active_sources:
        return {"channel_recent_entity_results": "", "retrieval_ledger": ledger}

    platform = state["platform"]
    platform_channel_id = state["platform_channel_id"]
    lookback_hours = retrieval_plan.get("time_scope", {}).get("lookback_hours", CHANNEL_RECENT_ENTITY_LOOKBACK_HOURS)
    time_threshold = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()

    all_results: list[str] = []

    for entity in resolved_entities:
        surface_form = entity.get("surface_form", "")
        resolved_id = entity.get("resolved_global_user_id", "")
        ledger_key = f"channel_recent_entity:{surface_form}"

        if ledger_key in ledger:
            continue

        messages: list[dict] = []

        if resolved_id:
            raw = await get_conversation_history(
                platform=platform,
                platform_channel_id=platform_channel_id,
                global_user_id=resolved_id,
                from_timestamp=time_threshold,
                limit=CHANNEL_RECENT_ENTITY_MAX_RESULTS,
            )
            messages.extend(raw)

        if len(messages) < 3:
            keyword_results = await search_conversation_history(
                query=surface_form,
                platform=platform,
                platform_channel_id=platform_channel_id,
                limit=CHANNEL_RECENT_ENTITY_MAX_RESULTS,
                method="keyword",
                from_timestamp=time_threshold,
            )
            seen_ids = {m.get("platform_message_id") for m in messages}
            for _, msg in keyword_results:
                if msg.get("platform_message_id") not in seen_ids:
                    messages.append(msg)

        if messages:
            formatted_lines = []
            for msg in messages[-CHANNEL_RECENT_ENTITY_MAX_RESULTS:]:
                speaker = msg.get("display_name", "unknown")
                content = msg.get("content", "")
                ts = msg.get("timestamp", "")
                formatted_lines.append(f"[{ts}] {speaker}: {content}")

            entity_summary = f"### {surface_form} 的近期频道记录\n" + "\n".join(formatted_lines)
            all_results.append(entity_summary)

        ledger[ledger_key] = {"found": len(messages), "method": "id+keyword" if resolved_id else "keyword"}

    channel_recent_entity_results = "\n\n".join(all_results) if all_results else ""

    logger.debug(
        "Channel recent entity RAG: entities=%d results_len=%d",
        len(resolved_entities),
        len(channel_recent_entity_results),
    )

    return {
        "channel_recent_entity_results": channel_recent_entity_results,
        "retrieval_ledger": ledger,
    }


# ── Phase 2 — Third Party Profile RAG ────────────────────────────


async def third_party_profile_rag(state: RAGState) -> dict:
    resolved_entities = state.get("resolved_entities") or []
    retrieval_plan = state.get("retrieval_plan") or {}
    active_sources = retrieval_plan.get("active_sources", [])
    ledger = dict(state.get("retrieval_ledger") or {})

    if "THIRD_PARTY_PROFILE" not in active_sources:
        return {"third_party_profile_results": "", "retrieval_ledger": ledger}

    current_user_id = state["global_user_id"]
    all_results: list[str] = []

    for entity in resolved_entities:
        resolved_id = entity.get("resolved_global_user_id", "")
        surface_form = entity.get("surface_form", "")

        if not resolved_id:
            continue

        # Prevent fetching the current user's profile as a third party
        if resolved_id == current_user_id:
            continue

        ledger_key = f"third_party_profile:{resolved_id}"
        if ledger_key in ledger:
            continue

        try:
            profile = await get_user_profile(resolved_id)
        except Exception:
            logger.warning("Failed to fetch third-party profile for %s", resolved_id, exc_info=True)
            profile = {}

        if not profile:
            ledger[ledger_key] = {"found": False}
            continue

        profile_image = _build_image_context(profile.get("user_image") or {})

        profile_lines = [f"### {surface_form} 的用户画像"]
        if profile_image.get("milestones"):
            profile_lines.append(f"里程碑: {json.dumps(profile_image['milestones'], ensure_ascii=False)}")
        if profile_image.get("historical_summary"):
            profile_lines.append(f"历史总结: {profile_image['historical_summary']}")
        if profile_image.get("recent_observations"):
            profile_lines.append(f"近期观察: {', '.join(profile_image['recent_observations'])}")

        diary = profile.get("character_diary") or []
        if diary:
            recent_diary = [str(e.get("entry", "")).strip() for e in diary[-5:] if str(e.get("entry", "")).strip()]
            if recent_diary:
                profile_lines.append(f"主观日记: {'; '.join(recent_diary)}")

        all_results.append("\n".join(profile_lines))
        ledger[ledger_key] = {"found": True}

    third_party_profile_results = "\n\n".join(all_results) if all_results else ""

    logger.debug(
        "Third party profile RAG: profiles_found=%d results_len=%d",
        len(all_results),
        len(third_party_profile_results),
    )

    return {
        "third_party_profile_results": third_party_profile_results,
        "retrieval_ledger": ledger,
    }


# ── Phase 1+2 — Tier Gate functions ──────────────────────────────


def _should_run_tier2(state: RAGState) -> str:
    retrieval_plan = state.get("retrieval_plan") or {}
    active_sources = retrieval_plan.get("active_sources", [])
    has_tier2_source = any(s in active_sources for s in ("CHANNEL_RECENT_ENTITY", "THIRD_PARTY_PROFILE"))
    if has_tier2_source:
        return "run"
    return "skip"


def _should_run_tier3(state: RAGState) -> str:
    retrieval_plan = state.get("retrieval_plan") or {}
    active_sources = retrieval_plan.get("active_sources", [])
    if "EXTERNAL_KNOWLEDGE" in active_sources:
        return "run"
    return "skip"

