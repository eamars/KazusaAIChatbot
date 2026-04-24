"""Pre-retrieval resolution layer: continuation resolver, RAG planner, entity grounder.

These three nodes run sequentially before the tiered retrieval dispatchers.
They produce the ``continuation_context``, ``retrieval_plan``, and
``resolved_entities`` that gate and parameterize all downstream retrieval.

Moved from ``persona_supervisor2_rag.py`` during Phase 6 decomposition.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_schema import RAGState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

import json
import logging

logger = logging.getLogger(__name__)


# ── Phase 1 — Continuation Resolver ───────────────────────────────

_CONTINUATION_RESOLVER_PROMPT = """\
你是一个上下文延续解析器。你的任务是判断用户的输入是否是一个"延续性话语"（continuation / follow-up / reply-only），如果是，则从最近的对话历史中补全被省略的对象或任务。

# 判断标准
- 延续性话语的特征：缺乏独立语义，依赖上文才能理解（如："那后来呢？"、"然后呢？"、"这个呢？"）
- 如果用户输入已经是完整的查询（包含明确主语和宾语），则 `needs_context_resolution` 为 false
- 如果能从 `chat_history_recent` 确定被省略的对象/任务，则补全并输出 `resolved_task`
- 如果无法确定，则保持低置信度，不要猜测
- 如果输入中包含 URL、文件名、引用文本、专有名词等字面锚点，必须原样保留这些锚点，不得替换成猜测出的别名或相近实体
- `resolved_task` 必须表达用户想检索/确认的任务，不能改写成助手视角的追问、澄清问题或反问句
- 如果你只能看出"用户在指这个链接/这个对象"，但无法确定具体想问哪一部分，应保留原始锚点并降低置信度，而不是把 `resolved_task` 改写成"是指哪一部分？"这类问题

# URL / 字面锚点示例
- 输入：`这个 https://example.com/page`
  错误输出：`这个链接是哪个页面？`
  错误原因：把用户任务改写成了助手澄清问题
  正确输出示例 1：`用户在提及这条链接：https://example.com/page`
  正确输出示例 2：`用户在确认是否记得这条链接及其相关内容：https://example.com/page`
- 输入：`这个文件 README.md`
  错误输出：`README.md 是什么内容？`
  正确输出：`用户在提及 README.md 这个文件`

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

当前固定扮演的角色名为：{character_name}
当前角色在平台上的账号 ID 为：<@{platform_bot_id}>

# 可用的检索源
- `CURRENT_USER_STABLE`: 当前用户的稳定画像信息（已预加载，通常不需要额外检索）
- `CHARACTER_SELF_KNOWLEDGE`: 角色自身可公开表达的稳定资料（运行时注入，无需额外检索）
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
- 如果输入或 `resolved_task` 含有 URL、引用文本、文件名、页面标题等字面锚点，必须优先保留这些锚点；不要把它们替换成猜测出的相近实体名
- 只有当实体在输入、`resolved_task` 或最近历史里有明确依据时，才允许写入 `entities`
- 如果 `continuation_confidence` 偏低或 `missing_slots` 非空，且任务仍不具备可检索性，应选择 `NONE`，让主回复阶段进行澄清，而不是硬启动检索
- 如果用户问的是角色自己可公开回答的资料（如名字、角色描述、性别、年龄、生日、公开背景），通常应选择 `retrieval_mode=NONE`、`active_sources=[]`，并把 `subject.kind` 标记为 `character_self`。这类信息不属于第三方检索，也不应放入 `entities`
- 不要把角色自己误判成 `third_party_user`；角色名或其常用简称若是在问“你/千纱/你自己”的资料，应归到 `character_self`
- 如果输入中出现 `<@...>` 形式的 mention，必须结合角色平台 ID `<@{platform_bot_id}>` 判断它是不是角色本人；命中角色本人时，应优先视为 `character_self` / 受话对象，而不是第三方实体

# URL / 字面锚点示例
- 若 `resolved_task` 是 `用户在确认是否记得这条链接及其相关内容：https://example.com/page`
  则允许把 URL 原样写入 `task`、`entities.surface_form` 或 `external_task_hint`
- 禁止把它改写成猜测出的近似实体名、近音名或错别字
- 当页面真实标题不确定时，宁可保留 URL，也不要输出一个看似具体但缺乏依据的人名
- 当 `subject.primary_entity` 没有来自用户输入、最近历史或其他明确证据时，应填写原始 URL 锚点或空字符串；不要仅凭 URL 编码内容猜测页面标题

# 输入格式
{{
    "decontextualized_input": "用户输入",
    "resolved_task": "延续解析器补全后的任务（如果没有补全则与 decontextualized_input 相同）",
    "continuation_needs_resolution": true或false,
    "continuation_confidence": 0.0到1.0,
    "continuation_missing_slots": ["仍然缺失的信息"],
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
        "kind": "current_user | character_self | third_party_user | entity | topic | mixed",
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

    system_prompt = SystemMessage(
        content=_RAG_PLANNER_PROMPT.format(
            character_name=state["character_profile"]["name"],
            platform_bot_id=state["platform_bot_id"],
        )
    )
    user_input = {
        "decontextualized_input": state["decontexualized_input"],
        "resolved_task": resolved_task,
        "continuation_needs_resolution": continuation_context.get("needs_context_resolution", False),
        "continuation_confidence": continuation_context.get("confidence", 0.0),
        "continuation_missing_slots": continuation_context.get("missing_slots", []),
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
