"""Stage 4 consolidator image synthesis helpers."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output


# ── Image bookkeeping constants ──────────────────────────────────────
_CHARACTER_IMAGE_MAX_RECENT_WINDOW = 6
_CHARACTER_IMAGE_HISTORICAL_MAX_CHARS = 1500


# ── Image synthesizer prompts ────────────────────────────────────────


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
_character_image_session_summary_llm = get_llm(
    temperature=0.3,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


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
_character_image_compress_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)

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

    return_value = {
        "milestones": milestones,
        "recent_window": recent_window,
        "historical_summary": historical_summary,
        "meta": {"synthesis_count": synthesis_count + 1, "last_updated": timestamp},
    }
    return return_value
