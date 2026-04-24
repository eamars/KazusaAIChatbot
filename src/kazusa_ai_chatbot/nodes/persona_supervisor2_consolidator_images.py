"""Stage 4 consolidator image synthesis helpers."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
    normalize_diary_entries,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output


# ── Image bookkeeping constants ──────────────────────────────────────
_USER_IMAGE_MAX_RECENT_WINDOW = 6       # sessions to keep before overflow to historical
_USER_IMAGE_HISTORICAL_MAX_CHARS = 1500 # compress historical_summary when above this
_CHARACTER_IMAGE_MAX_RECENT_WINDOW = 6
_CHARACTER_IMAGE_HISTORICAL_MAX_CHARS = 1500

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
    diary_entries = normalize_diary_entries(state.get("diary_entry"))
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
