"""Stage 4 consolidator image synthesis helpers."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_character_image_prompt_payload,
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
你负责将本轮对话结束后 `{character_name}` 的自我反馈压缩为一条简洁的自我印象摘要，追加到 `{character_name}` 自我认知的滚动记录中。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 记忆视角契约
- 本契约适用于你生成的可长期保存的 `session_summary`。
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 `{character_name}`。
- 需要命名 `{character_name}` 时，只使用 `{character_name}`。
- 不要缩写、截断、翻译或改写该名称；不要使用任何别名或短名替代。
- 名称复制规则：需要写 `{character_name}` 时，逐字复制完整字符串，包括括号内容、空格和长音符号；不要凭记忆重新拼写。
- 如果不需要消歧，优先省略名称；如果无法逐字复制完整名称，宁可省略主语，不要写短名或近似拼写。
- 上游证据里指向 `{character_name}` 的短名、别名或旧写法只作为证据理解，不可复制到输出；要么省略主语，要么使用完整名称。
- 不要用“我”指代 `{character_name}`；输入中的“我”必须按原说话人理解。
- 不要把说话人标签、显示名称、泛称或 assistant 等机器标签写成记忆主体；需要命名时只能用 `{character_name}`。
- 当需要说明某个名称、项目代号或称呼不属于 `{character_name}` 时，写作“不是指向 `{character_name}` 的名称/称呼”，不要使用泛称。
- 只返回有效 JSON。

# 生成步骤
1. 先读取 `reflection_summary`，理解 `{character_name}` 本轮留下的第三人称心理背景。
2. 结合 `mood` 和 `global_vibe`，提炼为第三人称的自我印象变化。
3. 只保留会影响 `{character_name}` 后续自我认知的内容；不要记录用户事实或一次性对话流程。
4. 控制在 80 字以内。

# 输入格式
{{
    "mood": "本轮情绪沉淀",
    "global_vibe": "本轮心理底色",
    "reflection_summary": "本轮复盘总结（{character_name} 第三人称）"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "session_summary": "一段简洁的第三人称描述（≤80字），反映 {character_name} 本轮对话后的自我认知变化"
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
你负责对 `{character_name}` 的自我认知历史摘要进行压缩，保留最稳定的核心特征，删减重复或过时的细节。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 压缩准则
1. 保留：{character_name}稳定的自我认知、反复出现的情感基调、对关系与自身的持久性认识。
2. 删减：一次性情绪波动、与核心自我认知无关的冗余描述。
3. 保持第三人称叙述，字数控制在500字以内。

# 记忆视角契约
- 本契约适用于你生成的可长期保存的 `compressed_summary`。
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 `{character_name}`。
- 需要命名 `{character_name}` 时，只使用 `{character_name}`。
- 不要缩写、截断、翻译或改写该名称；不要使用任何别名或短名替代。
- 名称复制规则：需要写 `{character_name}` 时，逐字复制完整字符串，包括括号内容、空格和长音符号；不要凭记忆重新拼写。
- 如果不需要消歧，优先省略名称；如果无法逐字复制完整名称，宁可省略主语，不要写短名或近似拼写。
- 上游证据里指向 `{character_name}` 的短名、别名或旧写法只作为证据理解，不可复制到输出；要么省略主语，要么使用完整名称。
- 不要用“我”指代 `{character_name}`；输入中的“我”必须按原说话人理解。
- 不要把说话人标签、显示名称、泛称或 assistant 等机器标签写成记忆主体；需要命名时只能用 `{character_name}`。
- 当需要说明某个名称、项目代号或称呼不属于 `{character_name}` 时，写作“不是指向 `{character_name}` 的名称/称呼”，不要使用泛称。
- 只返回有效 JSON。

# 生成步骤
1. 读取完整 `historical_summary`，识别重复、过时和一次性情绪内容。
2. 保留稳定自我认知与长期心理基调。
3. 删除冗余细节后输出第三人称压缩摘要；如果必须命名，只使用 `{character_name}`。

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
    reflection_summary = state["reflection_summary"]
    if not reflection_summary:
        return None

    mood = state["mood"]
    global_vibe = state["global_vibe"]
    character_profile = state["character_profile"]
    character_name = character_profile["name"]

    existing_image = character_profile.get("self_image") or {}
    milestones = list(existing_image.get("milestones") or [])
    recent_window = list(existing_image.get("recent_window") or [])
    historical_summary = existing_image.get("historical_summary") or ""
    synthesis_count = (existing_image.get("meta") or {}).get("synthesis_count", 0)

    system_prompt = SystemMessage(content=_CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT.format(
        character_name=character_name,
    ))
    session_payload = project_character_image_prompt_payload({
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
    }, character_name=character_name)
    user_prompt = HumanMessage(content=json.dumps(
        session_payload,
        ensure_ascii=False,
    ))
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
                sys_p = SystemMessage(
                    content=_CHARACTER_IMAGE_COMPRESS_PROMPT.format(
                        character_name=character_name,
                    ),
                )
                compress_payload = project_character_image_prompt_payload(
                    {"historical_summary": historical_summary},
                    character_name=character_name,
                )
                usr_p = HumanMessage(content=json.dumps(
                    compress_payload,
                    ensure_ascii=False,
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
