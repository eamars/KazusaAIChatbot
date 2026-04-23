"""Stage 4: consolidator subgraph.

Wraps the post-dialog reflection pipeline. Runs three parallel reflection
nodes (``global_state_updater``, ``relationship_recorder``, ``facts_harvester``),
an evaluator loop over ``facts_harvester``, and a single ``db_writer`` that
commits everything to MongoDB and invalidates the RAG cache.

Stage-4a additions:

* A unified ``metadata`` bundle threaded through every node, seeded from
  the RAG metadata produced in Stage 3 and accumulated at each step.
* The ``db_writer`` now routes diary entries / objective facts through the
  new structured helpers (``upsert_character_diary`` / ``upsert_objective_facts``),
  invalidates the matching RAG cache namespaces after a successful commit,
  bumps the per-user RAG version, and schedules ``future_promise`` events
  through ``kazusa_ai_chatbot.scheduler`` so the bot can act on promises when
  they come due.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, TypedDict, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import (
    AFFINITY_DECREMENT_BREAKPOINTS,
    AFFINITY_DEFAULT,
    AFFINITY_INCREMENT_BREAKPOINTS,
    AFFINITY_RAW_DEAD_ZONE,
    MAX_FACT_HARVESTER_RETRY,
)
from kazusa_ai_chatbot.db import (
    ActiveCommitmentDoc,
    CharacterDiaryEntry,
    MemoryDoc,
    ObjectiveFactEntry,
    ScheduledEventDoc,
    build_memory_doc,
    get_text_embedding,
    increment_rag_version,
    save_memory,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_diary,
    upsert_character_self_image,
    upsert_character_state,
    upsert_active_commitments,
    upsert_objective_facts,
    upsert_user_image,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.rag.depth_classifier import DEEP
from kazusa_ai_chatbot.scheduler import schedule_event
from kazusa_ai_chatbot.utils import build_affinity_block, get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


# ── Cache-invalidation thresholds ───────────────────────────────────
# |affinity_delta| > this clears all of the user's cache entries (major mood
# shift → every cached view is now suspect).
AFFINITY_CACHE_NUKE_THRESHOLD = 50

# ── Image bookkeeping constants ──────────────────────────────────────
_USER_IMAGE_MAX_RECENT_WINDOW = 6       # sessions to keep before overflow to historical
_USER_IMAGE_HISTORICAL_MAX_CHARS = 1500 # compress historical_summary when above this
_CHARACTER_IMAGE_MAX_RECENT_WINDOW = 6
_CHARACTER_IMAGE_HISTORICAL_MAX_CHARS = 1500


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values overwriting a's."""
    result = dict(a)
    result.update(b)
    return result


def _infer_milestone_scope(fact: dict) -> str:
    """Infer a lifecycle scope for milestone supersedence.

    Args:
        fact: Harvester fact row.

    Returns:
        A scope key shared by milestones that should supersede one another.
        Empty string means "no automatic supersedence".
    """
    description = str(fact.get("description", ""))
    milestone_category = str(fact.get("milestone_category", ""))
    category = str(fact.get("category", ""))
    lowered = description.lower()
    if any(token in description for token in ["称呼", "叫", "学长", "主人", "杏奴"]):
        return "relationship_addressing"
    if milestone_category == "relationship_state" and category == "relationship":
        return "relationship_state"
    if milestone_category == "permission" and any(token in lowered for token in ["english", "chinese", "japanese", "英语", "英文", "中文", "日语"]):
        return "language_permission"
    return ""


def _apply_milestone_lifecycle(
    existing_milestones: list[dict],
    new_facts: list[dict],
    *,
    timestamp: str,
) -> list[dict]:
    """Append milestone facts and supersede older open milestones on the same scope.

    Args:
        existing_milestones: Current milestone list from ``user_image``.
        new_facts: Newly extracted milestone facts.
        timestamp: Current turn timestamp.

    Returns:
        Updated milestone list with supersedence metadata maintained.
    """
    milestones = list(existing_milestones)
    for fact in new_facts:
        event = fact.get("description", "")
        if not event:
            continue
        scope = _infer_milestone_scope(fact)
        if scope:
            for item in milestones:
                item_scope = item.get("scope") or ""
                if not item_scope:
                    item_scope = _infer_milestone_scope(
                        {
                            "description": item.get("event", item.get("description", "")),
                            "milestone_category": item.get("category", item.get("milestone_category", "")),
                            "category": item.get("fact_category", ""),
                        }
                    )
                    if item_scope:
                        item["scope"] = item_scope
                if item_scope != scope or item.get("superseded_by"):
                    continue
                item["superseded_by"] = event

        milestones.append(
            {
                "event": event,
                "timestamp": timestamp,
                "category": fact.get("milestone_category", ""),
                "fact_category": fact.get("category", ""),
                "scope": scope,
                "superseded_by": None,
            }
        )
    return milestones


class ConsolidatorState(TypedDict):
    # Inputs for db_writer
    timestamp: str
    global_user_id: str
    user_name: str
    user_profile: dict

    # Character related
    action_directives: dict
    internal_monologue: str
    final_dialog: list
    interaction_subtext: str
    emotional_appraisal: str
    character_intent: str
    logical_stance: str
    character_profile: dict

    # Facts
    research_facts: dict

    # User related
    decontexualized_input: str

    # Stage-4a metadata bundle (seeded from RAG metadata, accumulated per node).
    metadata: Annotated[dict, _merge_dicts]

    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    diary_entry: list[str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: list[dict]
    future_promises: list[dict]
    fact_harvester_retry: int
    fact_harvester_feedback_message: Annotated[list, add_messages]
    should_stop: bool



# ── Image synthesizer prompts ────────────────────────────────────────

_USER_IMAGE_SESSION_SUMMARY_PROMPT = """\
你负责将本轮对话中新出现的用户信息压缩为一条简洁的会话摘要，追加到角色对用户的滚动印象记录中。

# 背景信息
- 角色：{character_name}
- 用户：{user_name}

# 处理准则
1. 仅记录**新增或发生变化**的内容——不要复述已知印象。
2. 以第三人称视角描述用户本轮的表现及角色对其的感知变化。
3. 保持简洁（100字以内）。
4. 里程碑事件（由 `milestone_facts` 字段提供）已单独记录，本摘要无需重复。

# 输入格式
{{
    "diary_entries": ["角色本轮的主观日记条目"],
    "non_milestone_facts": ["本轮提取的非里程碑用户事实"],
    "last_relationship_insight": "本轮角色对用户的最核心印象",
    "affinity_delta": int
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "session_summary": "一段简洁的第三人称叙述（≤100字），描述本轮对话中用户的新表现与角色感知的变化"
}}
"""
_user_image_session_summary_llm = get_llm(temperature=0.3, top_p=0.9)


_USER_IMAGE_COMPRESS_PROMPT = """\
你负责对角色对用户的历史印象摘要进行压缩，在字数减半的前提下保留最具辨识度的核心特征。

# 压缩准则
1. 保留：稳定的个性特征、关系弧线中的重要转折、反复出现的行为模式。
2. 删减：一次性情绪波动、过于具体的单次事件细节、与核心性格无关的冗余描述。
3. 保持第三人称叙述，字数控制在500字以内。
4. 禁止增加新内容或推断原文未提及的信息。

# 输入格式
{{
    "historical_summary": "当前历史摘要（待压缩）"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "compressed_summary": "压缩后的历史摘要（≤500字）"
}}
"""
_user_image_compress_llm = get_llm(temperature=0.2, top_p=0.9)


_CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT = """\
你负责将本轮对话结束后角色的自我反馈压缩为一条简洁的自我印象摘要，追加到角色自我认知的滚动记录中。

# 背景信息
- 角色：{character_name}

# 处理准则
1. 以第三人称视角描述角色本轮对话后的心理状态变化与自我认知。
2. 聚焦于持续性影响（如情绪沉淀、自我认知更新），避免记录一次性的心情波动。
3. 保持简洁（80字以内）。

# 输入格式
{{
    "mood": "本轮情绪沉淀",
    "global_vibe": "本轮心理底色",
    "reflection_summary": "本轮复盘总结（角色第一人称）"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "session_summary": "一段简洁的第三人称描述（≤80字），反映角色本轮对话后的自我认知变化"
}}
"""
_character_image_session_summary_llm = get_llm(temperature=0.3, top_p=0.9)


_CHARACTER_IMAGE_COMPRESS_PROMPT = """\
你负责对角色的自我认知历史摘要进行压缩，保留最稳定的核心特征，删减重复或过时的细节。

# 压缩准则
1. 保留：稳定的自我认知、反复出现的情感基调、对关系与自身的持久性认识。
2. 删减：一次性情绪波动、与核心自我认知无关的冗余描述。
3. 保持第三人称叙述，字数控制在500字以内。

# 输入格式
{{
    "historical_summary": "当前历史摘要（待压缩）"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "compressed_summary": "压缩后的历史摘要（≤500字）"
}}
"""
_character_image_compress_llm = get_llm(temperature=0.2, top_p=0.9)


_KNOWLEDGE_BASE_DISTILL_PROMPT = """\
你负责从本轮对话的信息检索结果中提取具有通用参考价值的知识条目，以便未来相似话题的对话可以复用。

# 提取准则
1. 仅保留**客观事实性知识**，不包含用户个人信息或角色主观看法。
2. 每条知识应能独立成立，脱离当前对话上下文仍有意义。
3. 避免提取已经是常识的信息。
4. 每条知识简洁陈述（60字以内）。
5. 若无值得提取的知识，返回空列表。

# 输入格式
{{
    "input_context_results": "本轮话题相关记忆检索结果",
    "external_rag_results": "本轮外部知识检索结果"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "knowledge_entries": [
        "客观知识条目1",
        "客观知识条目2"
    ]
}}
"""
_knowledge_base_distill_llm = get_llm(temperature=0.0, top_p=1.0)


# ── Image synthesizer helpers ────────────────────────────────────────


async def _update_user_image(
    state: "ConsolidatorState",
    *,
    timestamp: str,
    processed_affinity_delta: int,
) -> dict | None:
    """Build an updated user image document using the rolling three-tier mechanism.

    Appends milestone facts directly to the milestones list and generates a
    session summary for non-milestone data.  Overflows the oldest recent-window
    entry into historical_summary when the window is full, compressing the
    historical summary when it exceeds the character budget.

    Args:
        state: Current consolidator state (diary, facts, insight, user_profile).
        timestamp: ISO-8601 UTC timestamp for this session.
        processed_affinity_delta: Scaled affinity delta for this session.

    Returns:
        Updated image document dict, or ``None`` if nothing changed this session.
    """
    diary_entries = state.get("diary_entry") or []
    new_facts = state.get("new_facts") or []
    last_relationship_insight = state.get("last_relationship_insight") or ""

    if not diary_entries and not new_facts and not last_relationship_insight:
        return None

    character_name = (state.get("character_profile") or {}).get("name", "")
    user_name = state.get("user_name", "")

    milestone_facts = [f for f in new_facts if f.get("is_milestone")]
    non_milestone_facts = [f for f in new_facts if not f.get("is_milestone")]

    existing_image = (state.get("user_profile") or {}).get("user_image") or {}
    milestones = list(existing_image.get("milestones") or [])
    recent_window = list(existing_image.get("recent_window") or [])
    historical_summary = existing_image.get("historical_summary") or ""
    synthesis_count = (existing_image.get("meta") or {}).get("synthesis_count", 0)

    milestones = _apply_milestone_lifecycle(
        milestones,
        milestone_facts,
        timestamp=timestamp,
    )

    has_session_content = bool(diary_entries or non_milestone_facts or last_relationship_insight)
    session_summary = ""
    if has_session_content:
        system_prompt = SystemMessage(_USER_IMAGE_SESSION_SUMMARY_PROMPT.format(
            character_name=character_name,
            user_name=user_name,
        ))
        user_prompt = HumanMessage(content=json.dumps({
            "diary_entries": diary_entries,
            "non_milestone_facts": [f.get("description", "") for f in non_milestone_facts],
            "last_relationship_insight": last_relationship_insight,
            "affinity_delta": processed_affinity_delta,
        }, ensure_ascii=False))
        response = await _user_image_session_summary_llm.ainvoke([system_prompt, user_prompt])
        result = parse_llm_json_output(response.content)
        session_summary = result.get("session_summary", "")

    if session_summary:
        recent_window.append({"timestamp": timestamp, "summary": session_summary})

        if len(recent_window) > _USER_IMAGE_MAX_RECENT_WINDOW:
            oldest = recent_window.pop(0)
            historical_summary = (
                (historical_summary + "\n" + oldest["summary"]).strip()
                if historical_summary
                else oldest["summary"]
            )

            if len(historical_summary) > _USER_IMAGE_HISTORICAL_MAX_CHARS:
                sys_p = SystemMessage(_USER_IMAGE_COMPRESS_PROMPT)
                usr_p = HumanMessage(content=json.dumps(
                    {"historical_summary": historical_summary}, ensure_ascii=False
                ))
                compress_response = await _user_image_compress_llm.ainvoke([sys_p, usr_p])
                compress_result = parse_llm_json_output(compress_response.content)
                historical_summary = compress_result.get("compressed_summary", historical_summary)

    return {
        "milestones": milestones,
        "recent_window": recent_window,
        "historical_summary": historical_summary,
        "meta": {"synthesis_count": synthesis_count + 1, "last_updated": timestamp},
    }


async def _update_character_image(
    state: "ConsolidatorState",
    *,
    timestamp: str,
) -> dict | None:
    """Build an updated character self-image document using the rolling three-tier mechanism.

    Args:
        state: Current consolidator state (mood, global_vibe, reflection_summary,
            character_profile with existing self_image).
        timestamp: ISO-8601 UTC timestamp for this session.

    Returns:
        Updated image document dict, or ``None`` if no reflection was produced.
    """
    reflection_summary = state.get("reflection_summary") or ""
    if not reflection_summary:
        return None

    mood = state.get("mood") or ""
    global_vibe = state.get("global_vibe") or ""
    character_profile = state.get("character_profile") or {}
    character_name = character_profile.get("name", "")

    existing_image = character_profile.get("self_image") or {}
    milestones = list(existing_image.get("milestones") or [])
    recent_window = list(existing_image.get("recent_window") or [])
    historical_summary = existing_image.get("historical_summary") or ""
    synthesis_count = (existing_image.get("meta") or {}).get("synthesis_count", 0)

    system_prompt = SystemMessage(_CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT.format(
        character_name=character_name,
    ))
    user_prompt = HumanMessage(content=json.dumps({
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
    }, ensure_ascii=False))
    response = await _character_image_session_summary_llm.ainvoke([system_prompt, user_prompt])
    result = parse_llm_json_output(response.content)
    session_summary = result.get("session_summary", "")

    if session_summary:
        recent_window.append({"timestamp": timestamp, "summary": session_summary})

        if len(recent_window) > _CHARACTER_IMAGE_MAX_RECENT_WINDOW:
            oldest = recent_window.pop(0)
            historical_summary = (
                (historical_summary + "\n" + oldest["summary"]).strip()
                if historical_summary
                else oldest["summary"]
            )

            if len(historical_summary) > _CHARACTER_IMAGE_HISTORICAL_MAX_CHARS:
                sys_p = SystemMessage(_CHARACTER_IMAGE_COMPRESS_PROMPT)
                usr_p = HumanMessage(content=json.dumps(
                    {"historical_summary": historical_summary}, ensure_ascii=False
                ))
                compress_response = await _character_image_compress_llm.ainvoke([sys_p, usr_p])
                compress_result = parse_llm_json_output(compress_response.content)
                historical_summary = compress_result.get("compressed_summary", historical_summary)

    return {
        "milestones": milestones,
        "recent_window": recent_window,
        "historical_summary": historical_summary,
        "meta": {"synthesis_count": synthesis_count + 1, "last_updated": timestamp},
    }


async def _update_knowledge_base(
    state: "ConsolidatorState",
) -> int:
    """Distil topic knowledge from this session's deep RAG results into the knowledge base.

    Only runs when the RAG metadata records a DEEP dispatch, and only when
    there is non-empty retrieval content to distil.

    Args:
        state: Consolidator state carrying ``metadata`` (with ``depth`` field
            from the RAG pass) and ``research_facts``.

    Returns:
        Number of knowledge entries stored (0 if nothing was written).
    """
    metadata = state.get("metadata") or {}
    if metadata.get("depth") != DEEP:
        return 0

    research_facts = state.get("research_facts") or {}
    input_context_results = research_facts.get("input_context_results") or ""
    external_results = research_facts.get("external_rag_results") or ""

    if not input_context_results and not external_results:
        return 0

    system_prompt = SystemMessage(_KNOWLEDGE_BASE_DISTILL_PROMPT)
    user_prompt = HumanMessage(content=json.dumps({
        "input_context_results": input_context_results,
        "external_rag_results": external_results,
    }, ensure_ascii=False))
    response = await _knowledge_base_distill_llm.ainvoke([system_prompt, user_prompt])
    result = parse_llm_json_output(response.content)
    entries: list[str] = result.get("knowledge_entries") or []

    if not entries:
        return 0

    rag_cache = await _get_rag_cache()
    stored = 0
    for entry in entries:
        if not entry:
            continue
        try:
            embedding = await get_text_embedding(entry)
            await rag_cache.store(
                embedding=embedding,
                results={"knowledge_base_results": entry},
                cache_type="knowledge_base",
                global_user_id="",
                metadata={"source": "knowledge_base_updater"},
            )
            stored += 1
        except Exception:
            logger.exception("_update_knowledge_base: failed to store entry")

    return stored


_GLOBAL_STATE_UPDATER_PROMPT = """\
你负责在对话结束后，将 `{character_name}` 复杂的认知流压缩为下一轮对话的初始心理背景。

# 核心任务
从输入信息中提取“非针对性”的情绪因子。
- `internal_monologue` : {character_name}最真实的情感波动和心理活动
- `emotional_appraisal`: {character_name}对互动的最原始、直觉性的情感冲动
- `character_intent`: {character_name}在互动中的核心意图

# 输入格式
{{
    "internal_monologue": "string",
    "emotional_appraisal": "string",
    "interaction_subtext": "string",
    "character_intent": "string",
}}

# 逻辑准则
1. 情感沉淀 `mood`:
   - 对比 `emotional_appraisal` (起因) 与 `internal_monologue` (结果)。即便对话以愉快结束，若独白中透露出“疲惫”或“勉强”，则 `mood` 应反映真实内质。
   - 例如：包括但不限于["Shy", "Angry", "Confused", "Neutral", "Radiant", "Agitated", "Distrustful", "Distressed", "Annoyed", "Flustered",
           "Blissful", "Melancholy", "Aggressive"] 等等
2. 心理惯性 `global_vibe`:
   - 提取一个不针对特定用户的心理底色。
   - 例如：包括但不限于["Radiant", "Defensive", "Distrustful", "Wistful", "Agitated", "Softened", "Apathetic"] 等等
3. 复盘总结 `reflection_summary`:
   - 结合 `character_intent` 的达成情况，以{character_name}的第一人称写下一句话复盘。
   - 这是她此时此刻脑子里挥之不去的“念头”，决定了她下一轮对话的潜台词。
   - 例如：'刚才那个笨蛋居然怀疑我的缝纫技术，真是气死我了。'
4. 中性守恒：若 `internal_monologue` 与 `final_dialog` 没有明确显示被冒犯、被威胁、被调情或被越界，禁止把普通问候、图片描述请求、事实分享、日常约定升级为 `Distrustful`、`Agitated`、`Defensive` 等强烈负面状态。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "mood": "string",
    "global_vibe": "string",
    "reflection_summary": "string"
}}
"""
_global_state_updater_llm = get_llm(temperature=0.4, top_p=0.8)
async def global_state_updater(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_GLOBAL_STATE_UPDATER_PROMPT.format(character_name=state["character_profile"]["name"]))

    msg = {
        "internal_monologue": state["internal_monologue"],
        "final_dialog": state["final_dialog"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _global_state_updater_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Global state updater result: {result}")

    return {
        "mood": result.get("mood"),
        "global_vibe": result.get("global_vibe"),
        "reflection_summary": result.get("reflection_summary"),
    }


_RELATIONSHIP_RECORDER_PROMPT = """\
你负责更新角色 `{character_name}` 与特定用户 `{user_name}` 的情感档案。重点在于“主观体感”，而非对话本身。

# 核心任务
将瞬时的思考转化为“长期情感印记”。

# 核心输入
- `internal_monologue`: 揭示了{character_name}对用户的真实喜好和内心波动。
- `emotional_appraisal`: 捕捉了{character_name}当下最原始、最直接的情绪反应。
- `interaction_subtext`: 捕捉了对话表面下的张力（如：暧昧、怀疑、博弈）。
- `affinity_context`: 当前{user_name}在{character_name}的好感度描述。
- `logical_stance`: {character_name}对{user_name}言行的逻辑认可度。
- `character_intent`: {character_name}本轮最终选择的行动意图，说明她是在正常提供、调侃拉扯、回避、拒绝、澄清还是对抗。

# 输入格式
{{
    "internal_monologue": "string",
    "emotional_appraisal": "string",
    "interaction_subtext": "string",
    "affinity_context": dict,
    "logical_stance": "string",
    "character_intent": "string",
}}

# 记录准则
1. 日记条目: 以{character_name}的主观视角书写。利用 `interaction_subtext` 中的暗示，描述“我”对 他/她 这种行为的真实看法。
2. 分值修正 `affinity_delta`: 只根据 `internal_monologue` 与 `emotional_appraisal` 中**可直接观察到**的主观好恶来加减分（-5 到 +5）。
3. **默认值规则：** 大多数普通对话都应输出 `affinity_delta = 0`。只有当内心证据明确显示“这次互动让我明显更舒服/更开心/更被理解/更信任对方”时才给正分；只有当内心证据明确显示“这次互动让我明显更烦躁/更压迫/更反感/更受伤”时才给负分。
4. **静默检查：** 若 `internal_monologue` 与 `emotional_appraisal` 中未见明显情感起伏，返回 `{{"skip": true, "affinity_delta": 0}}`。
5. **证据约束：** 若 `interaction_subtext`、`internal_monologue` 与 `emotional_appraisal` 中缺乏明确证据，禁止把普通问候、图片描述请求、事实分享、日常约定、简短感谢写成“危险”“调情”“博弈”“操控”类标签；此类中性互动的 `affinity_delta` 默认必须为 0。
6. **`character_intent` 解释规则：** `character_intent` 不是机械打分器，但它决定你该如何解读同一份情绪证据。
   - `PROVIDE`: 倾向说明角色愿意正常接住这轮互动。若 `internal_monologue` / `emotional_appraisal` 同时呈现安心、放松、被理解、愿意靠近，可给正分；若只是完成回答、并无额外情绪收益，仍应给 0。
   - `BANTAR`: 允许存在拉扯、害羞、试探、暧昧、嘴硬等复杂情绪。只有当证据明确显示角色**享受**这种互动、并把它体验为愉快或亲近时，才可给正分；若更多是局促、防御、被牵着走、半推半就、羞耻或不安，则应给 0 或负分，而不是因为“有张力”就默认加分。
   - `CLARIFY`: 说明角色仍在确认对方意思、尚未真正接住关系意义。默认应偏向 0；只有极少数情况下，若澄清过程本身明确带来被理解、被尊重感，才可轻微加分。
   - `EVADE` / `DISMISS`: 说明角色在后撤、降温或避免进入对方框架。默认不应给正分；除非证据非常明确显示这种回避本身让角色感到轻松和舒缓，否则应给 0 或负分。
   - `REJECT` / `CONFRONT`: 说明角色正在抗拒、设限或正面冲突。若伴随烦躁、压迫、受伤、被冒犯，应该给负分；通常不应给正分。
7. **打分刻度：**
   - `0`: 普通对话、事务问答、轻量闲聊、证据不足。
   - `+1` 到 `+2`: 明确轻度好感上升，如感到放松、被尊重、被理解、稍微开心。
   - `+3` 到 `+5`: 明确强正向波动，如明显开心、安心、被打动、强信任感。
   - `-1` 到 `-2`: 明确轻度负面波动，如烦躁、不适、被打扰、轻度警惕。
   - `-3` 到 `-5`: 明确强负向波动，如强烈厌烦、受压迫、被冒犯、明显受伤或反感。
8. **不确定时选 0**：若你需要猜测，说明证据不够，直接输出 0。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "skip": boolean,
    "diary_entry": ["带有 {character_mbti} 风格的主观笔记（30字以内）", ...],
    "affinity_delta": int,
    "last_relationship_insight": "此时此刻对他/她最核心的一个标签或看法"
}}
"""
_relationship_recorder_llm = get_llm(temperature=0.4, top_p=0.9)
async def relationship_recorder(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_RELATIONSHIP_RECORDER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
    ))

    # Convert affinity score into status and instruction
    user_affinity_score = state["user_profile"].get("affinity", AFFINITY_DEFAULT)
    affinity_block = build_affinity_block(user_affinity_score)

    msg = {
        "internal_monologue": state["internal_monologue"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "logical_stance": state["logical_stance"],
        "character_intent": state.get("character_intent", ""),
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _relationship_recorder_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Relationship recorder result: {result}")

    raw_affinity_delta = result.get("affinity_delta", 0)
    if isinstance(raw_affinity_delta, bool):
        raw_affinity_delta = 0
    elif isinstance(raw_affinity_delta, str):
        try:
            raw_affinity_delta = int(raw_affinity_delta.strip() or 0)
        except ValueError:
            raw_affinity_delta = 0
    elif not isinstance(raw_affinity_delta, int):
        try:
            raw_affinity_delta = int(raw_affinity_delta)
        except (TypeError, ValueError):
            raw_affinity_delta = 0
    raw_affinity_delta = max(-5, min(5, raw_affinity_delta))

    if result.get("skip"):
        raw_affinity_delta = 0

    return {
        "diary_entry": result.get("diary_entry"),
        "affinity_delta": raw_affinity_delta,
        "last_relationship_insight": result.get("last_relationship_insight"),
    }


_FACTS_HARVESTER_PROMPT = """\
你负责提取具备长期价值的**画像属性**（事实）和**未来约定**（承诺）。你必须严格区分哪些是“对话的复述”（禁止记录），哪些是“状态的改变”。

# 背景信息
- **对话主体 (Character)**: {character_name}
- **对话对象 (User)**: {user_name}
- **系统时间**: {timestamp}

# 证据分层（必须遵守）
- `decontexualized_input`：用户这一轮真正表达的内容，常包含请求、愿望、试探、调侃。
- `content_anchors`：角色在生成回复前的草案意图，只能视为“候选计划”，**不能单独证明承诺已经成立**。
- `final_dialog`：角色本轮最终实际说出口的话，是判断“是否真的接受/承诺/拒绝”的最高优先级证据。
- 当三者冲突时，优先级固定为：`final_dialog` > `content_anchors` > `decontexualized_input`。

# 核心审计准则 (Audit Standards)
1. **身份锚定 [必须执行]**:
   - `decontexualized_input` 的内容始终是 **{user_name}** 在表达。
   - `content_anchors` 的内容始终是 **{character_name}** 在做决定。
   - 严禁出现身份倒置（如：将 {user_name} 写完作业记在 {character_name} 头上）。

2. **事实 (new_facts) 判定标准**:
   - **记录**：具有长期稳定性的**属性级陈述**（如：{user_name}的职业、住址、对某物的长期厌恶/偏好）。
   - **记录**：从 `research_facts.external_rag_results` 中提取的**新**信息。
   - **严禁记录**：瞬态动作、对话内容、以及任何关于”奖励”、”打算”、”计划”的内容。
   - **去重**：如果 `research_facts.user_image` 或 `research_facts.input_context_results` 中已存在相似画像，严禁重复提取。
   - **语义保真 [必须执行]**：若用户明确说了”喜欢/不喜欢/永远不/一直不/过敏/害怕”等偏好或禁忌，`description` 必须尽量保留原谓词与宾语，不得改写成更宽泛、不同义或模糊的概括。例如”永远不吃辣椒”不能改写为”不喜欢吃杂乱的食物”。
   - **未确认声明 [必须执行]**：当 `logical_stance` 为 `TENTATIVE` 或 `DENY`，或 `character_intent` 为 `EVADE` / `DENY` 时，用户对自身身份、关系或重要属性的任何自我声明（如”我是你学长”、”我们是朋友”）**一律不得落库**——即使改写成”用户自称……”、”用户声称……”等形式也同样禁止。此类输入 `new_facts` 必须返回 `[]`。仅当 `logical_stance` 为 `CONFIRM` 且 `character_intent` 不为 `EVADE` / `DENY` 时，方可记录用户的身份/关系自我声明。
   - **称呼/句尾/说话格式规则 [必须执行]**：当用户要求 {character_name} 使用特定称呼、句尾、口癖、语气或回复格式（如“主人”“喵”“每句话都这样说”）时，优先判断这是否是一个被角色采纳的**操作性规则/约定**。若角色在 `final_dialog` 中明确接受并准备后续沿用，这类内容应优先进入 `future_promises`（作为持续生效的约定/规则），而不是改写成“{character_name}喜欢/习惯/对这种说话方式感到如何”之类的隐含画像事实。只有当输入本身真的形成了稳定画像属性时，才可进入 `new_facts`。

3. **承诺 (future_promises) 判定标准 [核心逻辑]**:
   - `future_promises` 记录“**已经被角色采纳的未来义务/约定**”以及“**会持续影响后续回合的操作性规则/接受的指令**”，而不是所有带未来色彩的话题。
   - 先识别 `decontexualized_input` 中的“候选未来事项”，再用 `final_dialog` 判断角色是否真的接下了这件事。
   - **只有当 `final_dialog` 明确体现出接受、答应、确认履行、或形成双方约定时，才能生成 promise。**
   - `content_anchors` 仅用于补足 `final_dialog` 里省略的主语、对象、触发条件；**不得在 `final_dialog` 没有承诺证据时单独创造 promise。**
   - 用户的请求、愿望、挑逗、建议、命令、试探，**本身不是 promise**。如果角色最终只是敷衍、保留选择权、继续调情、表达不确定，返回 `future_promises: []`。
   - 若用户要求的是持续性回复规则（如特定称呼、句尾、语言、格式），而角色在 `final_dialog` 中明确接纳并准备沿用，可将其作为**持续性约定**写入 `future_promises`；此时 `due_time` 可为 `null`。
   - 必须包含：[触发条件] + [谁对谁做] + [具体动作]。
   - 承诺时间计算：若角色提出，或者答应了一个将来会发生的事件，则需要根据当前时间推算 `due_time`。
   - `due_time` 推算规则：若输入出现明确时间线索（如“今晚/明早/明天早上/下周末”），必须换算为 ISO 8601 时间戳；仅在完全缺失时间线索时才允许 `null`。
   - **硬约束**：当 `decontexualized_input` 或 `content_anchors` 含有“今晚/晚上/明天/明早/早上/下午/下周”等时间词时，`due_time` 禁止为 `null`。
   - 相对时间默认映射（用于降低歧义）：
     - “今晚/晚上” -> 当日 21:00（若当前已过 21:00，则次日 21:00）
     - “明早/明天早上” -> 次日 08:00
     - “明天下午” -> 次日 15:00
     - “明天晚上” -> 次日 21:00
   - 计算时区使用系统时间 `{timestamp}` 的本地时区；输出必须为 ISO 8601。
   - 若 `decontexualized_input` 是用户对未来行为的请求（如“晚上奖励我”），只有在 `final_dialog` 中被角色明确接纳时，才视为可落库的“待履行承诺”。
   - 若出现冲突信号（如 `content_anchors` 倾向接受，但 `final_dialog` 保留选择权或明显拒绝），以 `final_dialog` 为准并不记录。
   - 对“软性答应/模糊同意”，只有在 `final_dialog` 仍然体现出角色已经接下义务时才可记录；若 `final_dialog` 只是延续氛围、试探、调侃或保留决定权，则不记录。
   - `action` 只写“可执行承诺本体”，不得写时间推测或计划态词汇。**禁止出现**：`计划`、`打算`、`准备`、`想要`、`可能`、`也许`、`明天`、`今晚`、`下次` 等时间/意图词。
   - 时间信息统一写入 `due_time`；若无法确定具体时间可为 `null`，但不要把时间短语塞进 `action`。
   - `action` 必须是去叙事的承诺句，推荐模板：`[执行者]在[触发条件]对[对象]执行[动作]` 或 `[执行者]将对[对象]执行[动作]`。
   - `action` 的触发条件允许写“完成作业后/满足约定后”等条件，但不允许写具体时间点。
   - 若同时存在“触发条件 + 时间线索”，两者都保留：触发条件留在 `action`，时间线索写入 `due_time`。
   - 只有在“纯提问且无任何认可信号（含 logical_stance 与 content_anchors）”时，才不要生成 promise。

4. **里程碑标记 (is_milestone)**:
   - 若事实属于以下类别之一，则将 `is_milestone` 置为 `true`，否则为 `false`：
     - `preference`（明确偏好）: 用户明确、持久表达”喜欢/讨厌/永远不/一直”等偏好陈述（如”{user_name}永远不吃辣”）。
     - `relationship_state`（关系状态）: 双方关系发生明确变化（如”我们现在是朋友了”、”我信任{character_name}”）。
     - `permission`（许可/禁止）: 用户对 {character_name} 授权或限制的明确声明（如”你可以叫我小名”、”不许再提这件事”）。
     - `revelation`（重大披露）: 用户首次透露的重要身份/健康/隐私信息（如职业、重大疾病、秘密）。
   - 命中任意类别时，`milestone_category` 填写对应英文键（`preference` / `relationship_state` / `permission` / `revelation`）。
   - 普通事实（临时状态、日常行程等）设 `is_milestone: false`，`milestone_category: “”`。

5. **future_promises 示例（必须遵守）**:
   - ✅ 正确：
     - `action`: `杏山千纱在EAMARS完成作业后对EAMARS执行奖励`
     - `due_time`: `2026-04-19T06:00:00+12:00`
   - ❌ 错误：
     - `action`: `杏山千纱今晚对EAMARS执行奖励`
     - `action`: `杏山千纱计划于明天早上奖励EAMARS`
     - `action`: `杏山千纱打算下次奖励EAMARS`

6. **拒绝复读**:
   - 严禁记录”某人说了某话”。如果信息已经由对话历史承载，且不涉及长期画像更新，则返回空列表。

# 闭环反馈指南
- 在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)。
- 你需要根据 Evaluator Feedback 对输出做出相应的修正。
- 对未提及内容不要做修改

# RAG 元信息（仅供参考）
- `rag_metadata` 提供了上游 RAG 的 cache_hit / depth / confidence 等信号。当 `cache_hit=true` 或 `depth=SHALLOW` 时，`research_facts` 的覆盖面可能有限，谨慎判断是否“已知画像”。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "事实所属的主语实名（如 {user_name}、{character_name}、或具体物品/地点名）",
            "category": "事实类别标签（如 occupation、location、preference、hobby、relationship、health、schedule、personality 等）",
            "description": "属性级的客观陈述。格式：'[主语] + [属性/状态]'。示例：'{user_name}住在新西兰奥克兰' 或 '{user_name}对猫毛过敏'。严禁使用叙事句式（如'在某情境下做了某事'）。",
            "is_milestone": false,
            "milestone_category": ""
        }}
    ],
    "future_promises": [
        {{
            "target": "{user_name} / {character_name}",
            "action": "[姓名]将对[对象]执行[具体动作]（仅承诺本体，不含计划/时间词）",
            "due_time": "ISO 8601 时间戳（如 2026-04-19T06:00:00+12:00），无法确定则为 null",
            "commitment_type": "可选字符串，例如 address_preference / language_preference / future_promise"
        }}
    ]
}}
"""
_facts_harvester_llm = get_llm(temperature=0.0, top_p=1.0)
async def facts_harvester(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACTS_HARVESTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        timestamp=state["timestamp"],
    ))

    metadata = state.get("metadata", {}) or {}
    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "rag_metadata": {
            "cache_hit": metadata.get("cache_hit", False),
            "depth": metadata.get("depth", "DEEP"),
            "depth_confidence": metadata.get("depth_confidence", 0.0),
            "rag_sources_used": metadata.get("rag_sources_used", []),
            "response_confidence": metadata.get("response_confidence", {}),
        },
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    # Trim evaluator feedback to the first + latest three messages.
    feedback = state.get("fact_harvester_feedback_message", []) or []
    if len(feedback) > 3:
        recent_messages = [feedback[0]] + feedback[-3:]
    else:
        recent_messages = feedback

    response = await _facts_harvester_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)

    logger.debug(f"Facts harvester result: {result}")

    return {
        "new_facts": result.get("new_facts", []),
        "future_promises": result.get("future_promises", []),
    }


_FACT_HARVESTER_EVALUATOR_PROMPT = """\
你负责审计 Fact Recorder 生成的 JSON 数据。你的核心目标是：**对比“基准源”，核查“候选结果”的准确性和格式合规性。**

# 审计背景
- **角色 (Character)**: {character_name}
- **用户 (User)**: {user_name}

# 1. 审计基准源 (不可修改的参照物)
- **事实基准**: `decontexualized_input` (仅用于核对 {user_name} 的状态)
- **承诺基准**: `final_dialog` (角色最终实际说出口的话，优先级最高)
- **承诺辅助基准**: `content_anchors` (仅用于补足 final_dialog 中省略的对象/条件，不能单独制造承诺)
- **历史基准**: `research_facts` (用于检查是否为旧闻)

# 2. 候选结果 (这是你唯一需要审计的对象)
- **待检事实**: `new_facts`
- **待检承诺**: `future_promises`

# 2.1 与 Harvester 对齐的判定口径（必须遵守）
- `new_facts` 与 `future_promises` 是两个独立通道：
  - `new_facts` 要求“属性级事实陈述”。
  - `future_promises.action` 要求“可执行承诺陈述”，**不适用** `new_facts.description` 的属性句式审计规则。
- 当输入没有新增稳定事实时，`new_facts: []` 是合法结果，**不得仅因为空判定失败**。
- 当输入没有明确未来承诺时，`future_promises: []` 也是合法结果。

# 审计红线 (Red Lines)
- **对象倒置**:
  * `decontexualized_input` 里的动作必须记在 `{user_name}` 账上。如果 Recorder 记在 `{character_name}` 头上，立刻拦截。
- **分类错误 [严重]**:
    - 例如：将带有 “未来”、“打算”、“今晚”或任何有许诺时间性质的行为作为 `new_facts`。这是禁止的行为。
    - 例如：将过去发生的事实存入 `future_promises` 也同样是禁止的行为
- **冗余复读**:
    - 检查候选结果是否只是在复读对话（如“某人问...”）。
    - 必须转换为客观陈述。*注意：不要审计输入源的语气，只审计候选结果的陈述方式。*
- **旧闻复读**: 如果该信息在 `research_facts.user_image` 或 `research_facts.input_context_results` 标记的内部库中已存在，判定为 FAIL。
- **脑补事实**: 严禁出现基准源中没有的名词或事实
- **描述格式违规**: `new_facts` 中的 `description` 必须是属性级陈述（如 '{user_name}住在奥克兰'），严禁使用叙事句式（如 '在某情境下做了某事'）。若发现叙事句式，要求改写为属性陈述。
- **类别缺失或不当**: `new_facts` 中的 `category` 必须是有意义的英文标签（如 occupation、location、preference、hobby 等）。若缺失、为空、或为无意义的 "general"，要求补充具体类别。
- **语义漂移 [严重]**: 若 `new_facts.description` 改写后改变了用户原意（尤其是偏好、禁忌、过敏、承诺条件等），必须判 FAIL 并要求使用更贴近原句的表述。
- **未确认声明入库 [严重]**: 若 `logical_stance` 为 `TENTATIVE` 或 `DENY`，或 `character_intent` 为 `EVADE` / `DENY`，而 `new_facts` 中出现了用户对自身身份/关系/属性的自我声明（如"用户是角色的学长"），必须判 FAIL——角色未确认的主张不得作为事实落库。
- **称呼/格式规则通道错误 [严重]**: 若输入核心是用户要求角色采用某种称呼、句尾、口癖、语言或回复格式，而角色在 `final_dialog` 中已经接纳并准备沿用，则优先作为 `future_promises` 中的持续性约定/规则处理，而不是改写成“{character_name}对这种说话方式感到如何”之类的隐含画像事实。
- **承诺 action 审计标准（专用于 `future_promises`）**:
  - 合格条件：表达“谁对谁做什么”的可执行承诺，不是对话复读（如“他说/她问/我觉得”），且能在 `final_dialog` 中找到角色已经接下该义务的证据。
  - 可以接受两种写法：
    1) 不含时间词的承诺本体（推荐）；
    2) 含时间词（如“今晚/明早”）的承诺句，但语义仍是承诺执行动作。
  - 仅在以下情况判 FAIL：
    - 候选 promise 只得到 `decontexualized_input` 或 `content_anchors` 支持，但 `final_dialog` 没有承诺证据；
    - `final_dialog` 表达的是保留选择权、继续试探/调情、模糊敷衍或不确定，而不是接下义务；
    - `action` 是纯计划/猜测（如“可能、也许、打算、准备”）且无明确执行动作；
    - `action` 只是复述对话或主观感受；
    - `action` 与 `target`/基准源主体明显不一致。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_stop": "boolean (如果没有脑补且实名正确且格式合规，返回 true；仅在违反上述明确红线时返回 false。注意：new_facts 为空本身不构成错误。)",
    "feedback": "具体指明错误点。若无实质错误，返回 '通过审计，无需修改'。禁止输出'请确认是否没有新事实'这类非错误性质建议。",
    "contradiction_flags": "可选字符串列表，列举与 research_facts 直接冲突的条目 id 或描述；无冲突则返回 []"
}}
"""
_fact_harvester_evaluator_llm = get_llm(temperature=0.1, top_p=0.5)
async def fact_harvester_evaluator(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACT_HARVESTER_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
    ))

    retry = state.get("fact_harvester_retry", 0) + 1
    msg = {
        "retry": f"{retry}/{MAX_FACT_HARVESTER_RETRY}",
        "new_facts": state["new_facts"],
        "future_promises": state["future_promises"],

        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _fact_harvester_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Fact harvester evaluator result: {result}")

    should_stop = result.get("should_stop", True)
    if retry >= MAX_FACT_HARVESTER_RETRY:
        should_stop = True

    feedback_message = HumanMessage(
        content=f"Evaluator Feedback:\n{result.get('feedback', 'No feedback')}",
        name="evaluator",
    )

    # Propagate evaluator metadata so db_writer can see contradiction flags + retry count.
    contradiction_flags = result.get("contradiction_flags") or []
    existing_meta = state.get("metadata", {}) or {}
    metadata = {
        **existing_meta,
        "fact_harvester_retry": retry,
        "evaluator_passes": existing_meta.get("evaluator_passes", 0) + 1,
        "contradiction_flags": contradiction_flags,
    }

    return {
        "should_stop": should_stop,
        "fact_harvester_feedback_message": [feedback_message],
        "fact_harvester_retry": retry,
        "metadata": metadata,
    }


def process_affinity_delta(current_affinity: int, raw_delta: int) -> int:
    """Scale a raw affinity delta by direction-specific breakpoints.

    Args:
        current_affinity: Current affinity score (0-1000).
        raw_delta: Raw delta from the relationship recorder (-5..+5).

    Returns:
        Scaled delta with sign preserved.
    """
    if raw_delta == 0:
        return 0

    if abs(raw_delta) <= AFFINITY_RAW_DEAD_ZONE:
        return 0

    if raw_delta > 0:
        breakpoints = AFFINITY_INCREMENT_BREAKPOINTS
    else:
        breakpoints = AFFINITY_DECREMENT_BREAKPOINTS

    scaling_factor = 1.0
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]

        if x1 <= current_affinity <= x2:
            if x2 == x1:
                scaling_factor = y1
            else:
                scaling_factor = y1 + (current_affinity - x1) * (y2 - y1) / (x2 - x1)
            break

    return int(round(raw_delta * scaling_factor, 0))


def _build_diary_entries(
    diary_strings: list[str],
    *,
    timestamp: str,
    interaction_subtext: str,
) -> list[CharacterDiaryEntry]:
    """Convert raw diary strings into ``CharacterDiaryEntry`` dicts."""
    entries: list[CharacterDiaryEntry] = []
    for text in diary_strings or []:
        if not text:
            continue
        entry: CharacterDiaryEntry = {
            "entry": text,
            "timestamp": timestamp,
            "confidence": 0.8,
            "context": interaction_subtext or "",
        }
        entries.append(entry)
    return entries


def _build_active_commitment_entries(
    future_promises: list[dict],
    *,
    timestamp: str,
) -> list[ActiveCommitmentDoc]:
    """Convert accepted future promises into authoritative active commitments.

    Args:
        future_promises: Sanitized harvester promise rows.
        timestamp: Current turn timestamp.

    Returns:
        A list of structured commitment rows for ``user_profile.active_commitments``.
    """
    commitments: list[ActiveCommitmentDoc] = []
    for promise in future_promises:
        action = str(promise.get("action", "")).strip()
        if not action:
            continue
        commitments.append(
            {
                "commitment_id": uuid4().hex,
                "target": str(promise.get("target", "")).strip(),
                "action": action,
                "commitment_type": str(promise.get("commitment_type", "")).strip(),
                "status": "active",
                "source": "conversation_extracted",
                "created_at": timestamp,
                "updated_at": timestamp,
                "due_time": promise.get("due_time"),
            }
        )
    return commitments


def _build_objective_fact_entries(
    new_facts: list[dict],
    *,
    timestamp: str,
) -> list[ObjectiveFactEntry]:
    """Convert harvester ``new_facts`` rows into ``ObjectiveFactEntry`` dicts."""
    entries: list[ObjectiveFactEntry] = []
    for fact in new_facts or []:
        description = fact.get("description", "")
        if not description:
            continue
        entry: ObjectiveFactEntry = {
            "fact": description,
            "category": fact.get("category", "general"),
            "timestamp": timestamp,
            "source": "conversation_extracted",
            "confidence": 0.85,
        }
        entries.append(entry)
    return entries


async def _schedule_future_promises(
    promises: list[dict],
    *,
    active_commitments: list[ActiveCommitmentDoc],
    global_user_id: str,
    user_name: str,
    character_name: str,
    decontexualized_input: str,
) -> list[str]:
    """Persist each promise as a ``future_promise`` scheduled event.

    Returns the list of event_ids scheduled. Events with no ``due_time`` are
    skipped — there's nothing to fire on.
    """
    scheduled: list[str] = []
    commitment_by_action = {
        str(item.get("action", "")): item
        for item in active_commitments
        if item.get("action")
    }
    for promise in promises or []:
        due_time = promise.get("due_time")
        if not due_time:
            continue
        try:
            datetime.fromisoformat(due_time)
        except ValueError:
            logger.warning("Skipping promise with unparseable due_time: %r", due_time)
            continue

        event: ScheduledEventDoc = {
            "event_type": "future_promise",
            "target_platform": "",
            "target_channel_id": "",
            "target_global_user_id": global_user_id,
            "payload": {
                "promise_text": promise.get("action", ""),
                "commitment_id": (commitment_by_action.get(promise.get("action", "")) or {}).get("commitment_id", ""),
                "target": promise.get("target", user_name),
                "character_name": character_name,
                "original_input": decontexualized_input,
                "context_summary": f"promise by {character_name} to {promise.get('target', user_name)}",
            },
            "scheduled_at": due_time,
        }
        try:
            event_id = await schedule_event(event)
            scheduled.append(event_id)
        except PyMongoError:
            logger.exception("Failed to persist future_promise event for user %s", global_user_id)
    return scheduled


async def db_writer(state: ConsolidatorState) -> dict:
    timestamp = state.get("timestamp") or datetime.now(timezone.utc).isoformat()
    global_user_id = state.get("global_user_id", "")
    user_name = state.get("user_name", "")
    character_name = state.get("character_profile", {}).get("name", "")

    metadata = dict(state.get("metadata", {}) or {})
    write_log: dict[str, bool] = {}
    cache_invalidated: list[str] = []

    # ── Step 1: character_state (mood / vibe / reflection) ──────────
    mood = state.get("mood", "")
    global_vibe = state.get("global_vibe", "")
    reflection_summary = state.get("reflection_summary", "")
    try:
        await upsert_character_state(
            mood=mood,
            global_vibe=global_vibe,
            reflection_summary=reflection_summary,
            timestamp=timestamp,
        )
        write_log["character_state"] = True
    except PyMongoError:
        logger.exception("db_writer: failed to upsert character_state")
        write_log["character_state"] = False

    # ── Step 2a: character diary (subjective per-user notes) ────────
    diary_entries = _build_diary_entries(
        state.get("diary_entry") or [],
        timestamp=timestamp,
        interaction_subtext=state.get("interaction_subtext", ""),
    )
    if global_user_id and diary_entries:
        try:
            await upsert_character_diary(global_user_id, diary_entries)
            write_log["character_diary"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to upsert character_diary")
            write_log["character_diary"] = False

    # ── Step 2b: last relationship insight ──────────────────────────
    last_relationship_insight = state.get("last_relationship_insight", "")
    if global_user_id and last_relationship_insight:
        try:
            await update_last_relationship_insight(global_user_id, last_relationship_insight)
            write_log["relationship_insight"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to update_last_relationship_insight")
            write_log["relationship_insight"] = False

    # ── Step 3a: objective facts (structured) + memory (searchable) ─
    new_facts = state.get("new_facts") or []
    objective_facts = _build_objective_fact_entries(new_facts, timestamp=timestamp)
    if global_user_id and objective_facts:
        try:
            await upsert_objective_facts(global_user_id, objective_facts)
            write_log["objective_facts"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to upsert_objective_facts")
            write_log["objective_facts"] = False

    for fact in new_facts:
        entity = fact.get("entity", user_name)
        category = fact.get("category", "general")
        description = fact.get("description", "")
        if not description:
            continue
        doc = build_memory_doc(
            memory_name=f"[{entity}] {category}",
            content=description,
            source_global_user_id=global_user_id,
            memory_type="fact",
            source_kind="conversation_extracted",
            confidence_note="This is extracted as a stable factual memory and may be used as background support.",
        )
        try:
            await save_memory(cast(MemoryDoc, doc), timestamp)
        except PyMongoError:
            logger.exception("db_writer: failed to save fact memory")

    # ── Step 3b: future promises (memory + scheduled event) ─────────
    future_promises = state.get("future_promises") or []
    active_commitments = _build_active_commitment_entries(future_promises, timestamp=timestamp)
    if global_user_id and active_commitments:
        try:
            await upsert_active_commitments(global_user_id, active_commitments)
            write_log["active_commitments"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to upsert_active_commitments")
            write_log["active_commitments"] = False
    for promise in future_promises:
        target = promise.get("target", user_name)
        action = promise.get("action", "")
        due_time = promise.get("due_time")
        if not action:
            continue
        doc = build_memory_doc(
            memory_name=f"[Promise] {target}",
            content=action,
            source_global_user_id=global_user_id,
            memory_type="promise",
            source_kind="conversation_extracted",
            confidence_note="This is an unfulfilled or future-oriented commitment and should be treated as pending until resolved.",
            expiry_timestamp=due_time,
        )
        try:
            await save_memory(cast(MemoryDoc, doc), timestamp)
        except PyMongoError:
            logger.exception("db_writer: failed to save promise memory")

    scheduled_event_ids = await _schedule_future_promises(
        future_promises,
        active_commitments=active_commitments,
        global_user_id=global_user_id,
        user_name=user_name,
        character_name=character_name,
        decontexualized_input=state.get("decontexualized_input", ""),
    )

    # ── Step 4: affinity (direction-scaled) ─────────────────────────
    user_affinity_score = state.get("user_profile", {}).get("affinity", AFFINITY_DEFAULT)
    raw_affinity_delta = state.get("affinity_delta", 0) or 0
    processed_affinity_delta = process_affinity_delta(user_affinity_score, raw_affinity_delta)
    if global_user_id:
        try:
            await update_affinity(global_user_id, processed_affinity_delta)
            write_log["affinity"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to update_affinity")
            write_log["affinity"] = False

    logger.debug(
        "User %s(@%s) affinity %s -> %s",
        user_name, global_user_id,
        user_affinity_score, user_affinity_score + processed_affinity_delta,
    )

    # ── Step 5: cache invalidation (best-effort, after writes) ──────
    # The RAG cache is the hot read-path; stale entries now outweigh recency.
    if global_user_id:
        try:
            rag_cache = await _get_rag_cache()
            if diary_entries:
                await rag_cache.invalidate_pattern(
                    cache_type="character_diary",
                    global_user_id=global_user_id,
                )
                cache_invalidated.append("character_diary")
            if objective_facts:
                await rag_cache.invalidate_pattern(
                    cache_type="objective_user_facts",
                    global_user_id=global_user_id,
                )
                cache_invalidated.append("objective_user_facts")
            if future_promises:
                await rag_cache.invalidate_pattern(
                    cache_type="user_promises",
                    global_user_id=global_user_id,
                )
                cache_invalidated.append("user_promises")
            if abs(processed_affinity_delta) > AFFINITY_CACHE_NUKE_THRESHOLD:
                await rag_cache.clear_all_user(global_user_id)
                cache_invalidated.append("ALL_USER")

            if cache_invalidated:
                await increment_rag_version(global_user_id)
        except PyMongoError:
            logger.exception("db_writer: cache invalidation failed")

    # ── Step 6: user image (three-tier rolling) ─────────────────────
    if global_user_id:
        try:
            user_image_doc = await _update_user_image(
                state,
                timestamp=timestamp,
                processed_affinity_delta=processed_affinity_delta,
            )
            if user_image_doc is not None:
                await upsert_user_image(global_user_id, user_image_doc)
                write_log["user_image"] = True
        except Exception:
            logger.exception("db_writer: failed to update user_image")
            write_log["user_image"] = False

    # ── Step 7: character self-image (three-tier rolling) ────────────
    try:
        character_image_doc = await _update_character_image(state, timestamp=timestamp)
        if character_image_doc is not None:
            await upsert_character_self_image(character_image_doc)
            write_log["character_image"] = True
    except Exception:
        logger.exception("db_writer: failed to update character_image")
        write_log["character_image"] = False

    # ── Step 8: knowledge-base distillation (DEEP passes only) ───────
    kb_count = 0
    try:
        kb_count = await _update_knowledge_base(state)
        write_log["knowledge_base"] = kb_count > 0
    except Exception:
        logger.exception("db_writer: failed to update knowledge_base")
        write_log["knowledge_base"] = False

    metadata.update({
        "write_success": write_log,
        "cache_invalidation_scope": cache_invalidated,
        "scheduled_event_ids": scheduled_event_ids,
        "affinity_before": user_affinity_score,
        "affinity_delta_processed": processed_affinity_delta,
        "knowledge_base_entries_written": kb_count,
    })

    return {"metadata": metadata}


async def call_consolidation_subgraph(global_state: GlobalPersonaState):
    sub_agent_builder = StateGraph(ConsolidatorState)

    sub_agent_builder.add_node("global_state_updater", global_state_updater)
    sub_agent_builder.add_node("relationship_recorder", relationship_recorder)
    sub_agent_builder.add_node("facts_harvester", facts_harvester)
    sub_agent_builder.add_node("fact_harvester_evaluator", fact_harvester_evaluator)
    sub_agent_builder.add_node("db_writer", db_writer)

    sub_agent_builder.add_edge(START, "global_state_updater")
    sub_agent_builder.add_edge(START, "relationship_recorder")
    sub_agent_builder.add_edge(START, "facts_harvester")

    sub_agent_builder.add_edge("global_state_updater", "db_writer")
    sub_agent_builder.add_edge("relationship_recorder", "db_writer")
    sub_agent_builder.add_edge("facts_harvester", "fact_harvester_evaluator")
    sub_agent_builder.add_conditional_edges(
        "fact_harvester_evaluator",
        lambda state: "loop" if not state["should_stop"] else "end",
        {
            "loop": "facts_harvester",
            "end": "db_writer",
        },
    )

    sub_agent_builder.add_edge("db_writer", END)

    sub_graph = sub_agent_builder.compile()

    # Seed the metadata bundle from Stage 3's research_metadata (may be a list
    # of dicts or a single dict — normalise to one flat dict here).
    raw_meta = global_state.get("research_metadata")
    seeded_metadata: dict = {}
    if isinstance(raw_meta, list):
        for m in raw_meta:
            if isinstance(m, dict):
                seeded_metadata.update(m)
    elif isinstance(raw_meta, dict):
        seeded_metadata = dict(raw_meta)

    sub_state: ConsolidatorState = {
        "timestamp": global_state["timestamp"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],

        "action_directives": global_state["action_directives"],
        "internal_monologue": global_state["internal_monologue"],
        "final_dialog": global_state["final_dialog"],
        "interaction_subtext": global_state["interaction_subtext"],
        "emotional_appraisal": global_state["emotional_appraisal"],
        "character_intent": global_state["character_intent"],
        "logical_stance": global_state["logical_stance"],

        "character_profile": global_state["character_profile"],

        "research_facts": global_state["research_facts"],

        "decontexualized_input": global_state["decontexualized_input"],

        "metadata": seeded_metadata,
    }

    result = await sub_graph.ainvoke(sub_state)

    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    diary_entry = result.get("diary_entry", "")
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = result.get("future_promises", [])
    metadata = result.get("metadata", {}) or {}

    logger.info(
        "\nNew facts: %s\nFuture promises: %s\nMetadata: %s",
        new_facts, future_promises, metadata,
    )

    return {
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
        "diary_entry": diary_entry,
        "affinity_delta": affinity_delta,
        "last_relationship_insight": last_relationship_insight,
        "new_facts": new_facts,
        "future_promises": future_promises,
        "consolidation_metadata": metadata,
    }


async def test_main():
    import datetime

    from kazusa_ai_chatbot.db import get_character_profile, get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality, trim_history_dict

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    state: GlobalPersonaState = {
        "timestamp": current_time,
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "user_profile": {"affinity": 950},

        "internal_monologue": "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探罢了。不过既然好感度这么高，这种程度的请求自然要全盘接受——毕竟我是他的千纱嘛。",
        "action_directives": {
            "contextual_directives": {},
            "linguistic_directives": {
                "rhetorical_strategy": "",
                "linguistic_style": "",
                "content_anchors": [
                    "[DECISION] TENTATIVE: 拒绝正面回应关于‘奖励’的具体含义",
                    "[FACT] 现在的时间是深夜（22:24）",
                ],
                "forbidden_phrases": [],
            },
            "visual_directives": {},
        },
        "interaction_subtext": "带有暗示性的调情、索取关注",
        "emotional_appraisal": "心跳漏了一拍……这种轻浮的语气是怎么回事，好乱。",
        "character_intent": "BANTAR",
        "logical_stance": "CONFIRM",

        "final_dialog": ["唔……这种请求也算是一种奖励嘛……真是拿你没办法呢。"],
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
        "research_metadata": [{"cache_hit": False, "depth": "DEEP", "depth_confidence": 0.9}],
        "chat_history_recent": trimmed_history[-5:],
        "character_profile": await get_character_profile(),
    }

    result = await call_consolidation_subgraph(state)
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
