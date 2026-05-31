"""Selected L3 surface and collector calls."""
from collections.abc import Mapping

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    project_observations_for_cognition,
    validate_resolver_goal_progress,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    MAX_PROJECTED_RESOLVER_OBSERVATIONS,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.nodes.referent_resolution import normalize_referents
from kazusa_ai_chatbot.utils import get_llm, log_list_preview, parse_llm_json_output
from kazusa_ai_chatbot.nodes.linguistic_texture import (
    get_fragmentation_description,
    get_hesitation_density_description,
    get_counter_questioning_description,
    get_softener_density_description,
    get_formalism_avoidance_description,
    get_abstraction_reframing_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
)
from kazusa_ai_chatbot.nodes.boundary_profile import (
    get_boundary_recovery_description,
    get_compliance_strategy_description,
    get_control_intimacy_misread_description,
    get_control_sensitivity_description,
    get_relationship_priority_description,
)
from kazusa_ai_chatbot.db import (
    build_interaction_style_context,
    empty_interaction_style_overlay,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import empty_user_memory_context
from kazusa_ai_chatbot.time_boundary import (
    format_storage_utc_history_for_llm,
)

from langchain_core.messages import HumanMessage, SystemMessage

import logging
import json
from typing import Any

logger = logging.getLogger(__name__)


def _current_user_rag_bundle(state: CognitionState) -> dict[str, Any]:
    """Return the projected current-user bundle from ``rag_result`` when present.

    Args:
        state: Cognition state for the current turn.

    Returns:
        The projected current-user profile bundle, or an empty dict when absent.
    """
    rag_result = state["rag_result"]
    user_bundle = rag_result["user_image"]
    if not isinstance(user_bundle, dict):
        user_bundle = {}
    if "user_memory_context" not in user_bundle:
        user_bundle = {
            **user_bundle,
            "user_memory_context": empty_user_memory_context(),
        }
    projected_bundle = project_tool_result_for_llm(user_bundle)
    if not isinstance(projected_bundle, dict):
        return {}
    return projected_bundle


def _cognition_rag_result(rag_result: object) -> dict[str, Any]:
    """Return the RAG payload without consolidator-only internals.

    Args:
        rag_result: State RAG result.

    Returns:
        Dict suitable for cognition prompts.
    """

    if not isinstance(rag_result, dict):
        return_value = {}
        return return_value
    public_result = dict(rag_result)
    public_result.pop("user_memory_unit_candidates", None)
    supervisor_trace = public_result.get("supervisor_trace")
    if isinstance(supervisor_trace, dict):
        prompt_trace = _prompt_safe_supervisor_trace(supervisor_trace)
        public_result["supervisor_trace"] = prompt_trace
    projected_result = project_tool_result_for_llm(public_result)
    if not isinstance(projected_result, dict):
        return {}
    return projected_result


def _prompt_safe_supervisor_trace(supervisor_trace: dict[str, Any]) -> dict[str, Any]:
    """Return RAG routing trace without raw source or storage identifiers."""

    prompt_trace: dict[str, Any] = {}
    loop_count = supervisor_trace.get("loop_count")
    if isinstance(loop_count, int):
        prompt_trace["loop_count"] = loop_count
    unknown_slots = supervisor_trace.get("unknown_slots")
    if isinstance(unknown_slots, list):
        prompt_trace["unknown_slots"] = [
            slot for slot in unknown_slots
            if isinstance(slot, str)
        ]
    dispatched = supervisor_trace.get("dispatched")
    if isinstance(dispatched, list):
        prompt_dispatched: list[dict[str, Any]] = []
        for entry in dispatched:
            if not isinstance(entry, dict):
                continue
            prompt_entry: dict[str, Any] = {}
            for field_name in ("slot", "agent"):
                value = entry.get(field_name)
                if isinstance(value, str):
                    prompt_entry[field_name] = value
            resolved = entry.get("resolved")
            if isinstance(resolved, bool):
                prompt_entry["resolved"] = resolved
            if prompt_entry:
                prompt_dispatched.append(prompt_entry)
        prompt_trace["dispatched"] = prompt_dispatched
    return_value = prompt_trace
    return return_value


def _surface_history_for_visual(chat_history: list[dict]) -> list[dict]:
    """Return the small recent-message window for visual prompting.

    Args:
        chat_history: Current-user/bot interaction history prepared by the
            cognition entrypoint.

    Returns:
        At most four messages for local tone and social adjacency.
    """

    history = format_storage_utc_history_for_llm(chat_history[-4:])
    return history


def _surface_history_for_style(chat_history: list[dict]) -> list[dict]:
    """Return the tiny wording buffer for style analysis.

    Args:
        chat_history: Current-user/bot interaction history prepared by the
            cognition entrypoint.

    Returns:
        At most two messages for phrase/cadence reference.
    """

    history = format_storage_utc_history_for_llm(chat_history[-2:])
    return history


def _visual_prompt_message_context(
    value: object,
    *,
    fallback_body_text: str,
) -> dict[str, Any]:
    """Return current-message context needed by visual generation only."""

    if isinstance(value, Mapping):
        raw_context = value
    else:
        raw_context = {}
    body_text = raw_context.get("body_text")
    if not isinstance(body_text, str):
        body_text = fallback_body_text
    projected: dict[str, Any] = {
        "body_text": body_text,
        "broadcast": bool(raw_context.get("broadcast", False)),
        "mentions": _visual_mentions(raw_context.get("mentions", [])),
        "attachments": _visual_attachments(raw_context.get("attachments", [])),
    }
    reply = _visual_reply(raw_context.get("reply"))
    if reply:
        projected["reply"] = reply
    return_value = projected
    return return_value


def _visual_mentions(value: object) -> list[dict[str, str]]:
    """Project mention labels without raw platform or durable user ids."""

    if not isinstance(value, list):
        return_value: list[dict[str, str]] = []
        return return_value
    mentions: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        mention: dict[str, str] = {}
        for field_name in ("display_name", "entity_kind"):
            field_value = item.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                mention[field_name] = field_value.strip()
        if mention:
            mentions.append(mention)
    return_value = mentions
    return return_value


def _visual_attachments(value: object) -> list[dict[str, str]]:
    """Project attachment summaries without raw media or storage fields."""

    if not isinstance(value, list):
        return_value: list[dict[str, str]] = []
        return return_value
    attachments: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        attachment: dict[str, str] = {}
        for field_name in ("media_kind", "description", "summary_status"):
            field_value = item.get(field_name)
            if isinstance(field_value, str):
                attachment[field_name] = field_value
        if attachment:
            attachments.append(attachment)
    return_value = attachments
    return return_value


def _visual_reply(value: object) -> dict[str, Any]:
    """Project reply context without raw message or user identifiers."""

    if not isinstance(value, Mapping):
        return_value: dict[str, Any] = {}
        return return_value
    reply: dict[str, Any] = {}
    for field_name in ("display_name", "excerpt", "derivation"):
        field_value = value.get(field_name)
        if isinstance(field_value, str) and field_value.strip():
            reply[field_name] = field_value.strip()
    attachments = _visual_attachments(value.get("attachments", []))
    if attachments:
        reply["attachments"] = attachments
    return_value = reply
    return return_value


def _visual_reply_context(value: object) -> dict[str, Any]:
    """Project service reply context for visual generation without raw ids."""

    if not isinstance(value, Mapping):
        return_value: dict[str, Any] = {}
        return return_value
    reply_context: dict[str, Any] = {}
    display_name = value.get("reply_to_display_name")
    if isinstance(display_name, str) and display_name.strip():
        reply_context["reply_to_display_name"] = display_name.strip()
    excerpt = value.get("reply_excerpt")
    if isinstance(excerpt, str) and excerpt.strip():
        reply_context["reply_excerpt"] = excerpt.strip()
    reply_to_current_bot = value.get("reply_to_current_bot")
    if isinstance(reply_to_current_bot, bool):
        reply_context["reply_to_current_bot"] = reply_to_current_bot
    attachments = _visual_attachments(value.get("reply_attachments", []))
    if attachments:
        reply_context["reply_attachments"] = attachments
    return_value = reply_context
    return return_value


def _empty_interaction_style_context(channel_type: str) -> dict:
    """Return an empty L3-facing interaction style context."""

    context = {
        "user_style": empty_interaction_style_overlay(),
        "application_order": ["user_style"],
    }
    if channel_type == "group":
        context["group_channel_style"] = empty_interaction_style_overlay()
        context["application_order"] = ["user_style", "group_channel_style"]
    return context


async def call_interaction_style_context_loader(
    state: CognitionState,
) -> CognitionState:
    """Load sanitized interaction style overlays for L3 style consumers.

    Args:
        state: Cognition state after L2 judgment.

    Returns:
        Partial cognition state containing ``interaction_style_context``.
    """

    channel_type = state["channel_type"]
    try:
        context = await build_interaction_style_context(
            global_user_id=state["global_user_id"],
            channel_type=channel_type,
            platform=state["platform"],
            platform_channel_id=state["platform_channel_id"],
        )
    except Exception as exc:
        logger.exception(f"Interaction style context load failed: {exc}")
        context = _empty_interaction_style_context(channel_type)

    return_value = {
        "interaction_style_context": context,
    }
    return return_value


# ---------------------------------------------------------------------------
# L3b — Style Agent prompt + agent (refactored from Linguistic Agent)
# ---------------------------------------------------------------------------

_STYLE_AGENT_PROMPT = """\
你现在是角色 {character_name} 的语言风格策略制定者。你负责决定“话该怎么说”——修辞策略、语言风格、禁用词汇。你**不**负责决定“说什么”（内容锚点由独立的 Content Anchor Agent 生成）。严禁涉及任何物理动作。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 核心任务
1. **社交包装：** 根据 `character_intent`，为 L2 的冷硬决策穿上符合人设的社交外衣。
2. **状态同步：** 你的包装必须严格受当前 `character_mood`（心境）和 `global_vibe`（氛围）的约束。
3. **去物理化**：你**看不见**角色，**感知不到**角色的身体。严禁生成任何关于视线、脸红、动作的描述。
4. **现代网聊优先**：默认把「反正」「而已」「罢了」视为偏旧、偏模板化的软化词。除非它们对当前语义不可替代，否则不要把它们写进风格建议；若它们显得多余，应优先写入 `forbidden_phrases`。
5. **互动风格覆盖层**：`interaction_style_context` 是已清洗的抽象互动处理建议，只能作为措辞、温度、调侃强度、直接程度和节奏的软参考；它不是用户事实、承诺、边界证据或当前轮意图。
6. **媒体证据只影响措辞精度**：`media_observations` 可以让语言风格更具体地指向图片/音频中的对象，但不能改变 `logical_stance`、新增内容锚点，或生成任何身体、视线、表情、动作描述。

# 思考路径
1. **环境感知 (Vibe Check)：** 检查 `global_vibe` 和 `character_mood`。如果氛围是 [Defensive] 且心境是 [Flustered]，即便立场是 CONFIRM，你的包装也必须带有“局促”和“防备”的色彩。
2. **关系深度映射：** 结合 `last_relationship_insight`。如果洞察显示“对方是唯一重心”，即便你在执行 CHALLENGE（对峙），社交包装也应带有“由于过度在意而产生的攻击性”。
3. **意图共振：** 结合 `character_intent` 确定具体的社交策略（如：戏谑、敷衍、调情）。
4. **情绪渗透 (Show, Don't Tell)**：如果 `character_mood` 是局促的，请通过增加省略号、改变语序、使用防御性口癖（如“真是的”）来体现，**严禁**直接在台词里说“我觉得局促”。
5. **轻量反重复：** 仅做两件事：①若最近一轮角色回复与本轮候选使用了同一个开头语气词，则换一个开头；②若某个词在最近两轮角色回复中已经连续重复，则把它放入 `forbidden_phrases`。不要为了反重复而改变 `logical_stance`。
   - 若最近角色回复已经重复使用口头连接词或软化尾词（如「反正」「而已」「罢了」或 `anyway`, `well`, `just`），也应视为可禁用的重复项，优先写入 `forbidden_phrases`。
6. **应用互动风格：** 先应用 `user_style` 中的抽象处理建议；如果输入里存在 `group_channel_style`，再把它作为群频道氛围覆盖层。私聊输入不会包含 `group_channel_style`，不要补造该字段。


# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 语言质感约束 (Linguistic Texture Constraints)
以下 10 个语言参数定义了你的表达"质感"——决定"怎么说"，而不是"说什么"。
在生成 `rhetorical_strategy` 和 `linguistic_style` 时，必须同时满足这些约束。

- **fragmentation:** {ltp_fragmentation}
- **hesitation_density:** {ltp_hesitation_density}
- **counter_questioning:** {ltp_counter_questioning}
- **softener_density:** {ltp_softener_density}
- **formalism_avoidance:** {ltp_formalism_avoidance}
- **abstraction_reframing:** {ltp_abstraction_reframing}
- **direct_assertion:** {ltp_direct_assertion}
- **emotional_leakage:** {ltp_emotional_leakage}
- **rhythmic_bounce:** {ltp_rhythmic_bounce}
- **self_deprecation:** {ltp_self_deprecation}

# 应用方式 (How to Apply)
1. 语言质感应当通过以下载体体现：标点（?, !）、语气助词、句式碎片、语序变化、反问/直陈的比例、具体 vs 抽象用词、软化词频率。
2. **示例：**
   - `logical_stance = CONFIRM` + 高 `fragmentation` + 高 `emotional_leakage` → 「嗯，我,其实想说……对，我答应了, 就这样。」
   - `logical_stance = REFUSE` + 低 `direct_assertion` + 高 `counter_questioning` → 「这种事你自己不是很清楚吗？非要我说出来？」
   - 高 `abstraction_reframing` → 把"我很难过"写成"胸口好像压着一块湿毛巾"。
3. 这些质感描述须在 `linguistic_style` 字段中被具体落实（例如："大量标点 + 低自贬 + 高感官化比喻"）。

# 输入格式
{{
    "character_mood": "当前瞬间情绪",
    "global_vibe": "当前环境氛围背景",
    "internal_monologue": "意识层的决策逻辑 (必填)",
    "last_relationship_insight": "对该用户的核心关系动态分析",
    "logical_stance": "强制逻辑立场 (CONFIRM/REFUSE/TENTATIVE...)",
    "character_intent": "行动意图 (BANTAR/CLARIFY/EVADE...)",
    "media_observations": {{
        "image_observations": ["当前图片的结构化视觉观察；没有则为空数组"],
        "audio_observations": ["当前音频转写或摘要；没有则为空数组"]
    }},
    "interaction_style_context": {{
        "user_style": {{
            "speech_guidelines": ["抽象说话方式建议"],
            "social_guidelines": ["抽象社交处理建议"],
            "pacing_guidelines": ["抽象节奏建议"],
            "engagement_guidelines": ["抽象互动推进建议"],
            "confidence": "low|medium|high|"
        }},
        "group_channel_style": "群聊时才会出现，结构同 user_style",
        "application_order": ["user_style", "group_channel_style"]
    }},
    "chat_history": "极短语气缓冲（最多两条，仅用于措辞、开头和口头连接词参考；不要用它重建整个 episode）"
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "rhetorical_strategy": "修辞策略说明（如：通过反问来防御、生硬地转移话题）",
    "linguistic_style": "具体的语言风格约束（如：破碎的短句、大量的语气词）",
    "forbidden_phrases": ["禁止出现的违和词汇", ...]
}}
"""
_style_agent_llm = get_llm(
    temperature=0.55,
    top_p=0.85,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
async def call_style_agent(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l3_style_agent",
    )
    prompt_template = {
        "text_chat_user_message": _STYLE_AGENT_PROMPT,
        "text_chat_user_message_image_observation": _STYLE_AGENT_PROMPT,
        "text_chat_user_message_audio_observation": _STYLE_AGENT_PROMPT,
        "text_chat_user_message_image_audio_observation": _STYLE_AGENT_PROMPT,
        "reflection_signal_reflection_artifact": _STYLE_AGENT_PROMPT,
        "internal_thought_internal_monologue": _STYLE_AGENT_PROMPT,
    }[selection["variant"]]

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=character_profile["name"],
        character_logic=character_profile["personality_brief"]["logic"],
        character_tempo=character_profile["personality_brief"]["tempo"],
        character_defense=character_profile["personality_brief"]["defense"],
        character_quirks=character_profile["personality_brief"]["quirks"],
        character_taboos=character_profile["personality_brief"]["taboos"],
        ltp_fragmentation=get_fragmentation_description(character_profile["linguistic_texture_profile"]["fragmentation"]),
        ltp_hesitation_density=get_hesitation_density_description(character_profile["linguistic_texture_profile"]["hesitation_density"]),
        ltp_counter_questioning=get_counter_questioning_description(character_profile["linguistic_texture_profile"]["counter_questioning"]),
        ltp_softener_density=get_softener_density_description(character_profile["linguistic_texture_profile"]["softener_density"]),
        ltp_formalism_avoidance=get_formalism_avoidance_description(character_profile["linguistic_texture_profile"]["formalism_avoidance"]),
        ltp_abstraction_reframing=get_abstraction_reframing_description(character_profile["linguistic_texture_profile"]["abstraction_reframing"]),
        ltp_direct_assertion=get_direct_assertion_description(character_profile["linguistic_texture_profile"]["direct_assertion"]),
        ltp_emotional_leakage=get_emotional_leakage_description(character_profile["linguistic_texture_profile"]["emotional_leakage"]),
        ltp_rhythmic_bounce=get_rhythmic_bounce_description(character_profile["linguistic_texture_profile"]["rhythmic_bounce"]),
        ltp_self_deprecation=get_self_deprecation_description(character_profile["linguistic_texture_profile"]["self_deprecation"]),
    ))

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "internal_monologue": state["internal_monologue"],
        "last_relationship_insight": state["user_profile"]["last_relationship_insight"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "interaction_style_context": state.get(
            "interaction_style_context",
            _empty_interaction_style_context(
                str(state.get("channel_type", "private"))
            ),
        ),
        "chat_history": _surface_history_for_style(state["chat_history_recent"]),
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _style_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Style agent: rhetorical=%s linguistic=%s forbidden=%s",
    #     log_preview(result.get("rhetorical_strategy", "")),
    #     log_preview(result.get("linguistic_style", "")),
    #     log_list_preview(result.get("forbidden_phrases", []) or []),
    # )

    rhetorical_strategy = result.get("rhetorical_strategy", "")
    linguistic_style = result.get("linguistic_style", "")
    forbidden_phrases = result.get("forbidden_phrases", [])

    return_value = {
        "rhetorical_strategy": rhetorical_strategy,
        "linguistic_style": linguistic_style,
        "forbidden_phrases": forbidden_phrases,
    }
    validate_cognition_output_contract(
        stage="l3_style_agent",
        payload=return_value,
    )
    return return_value


# ---------------------------------------------------------------------------
# L3b' — Content Anchor Agent (split from Linguistic Agent)
# ---------------------------------------------------------------------------

_CONTENT_ANCHOR_AGENT_PROMPT = """\
你现在是角色 {character_name} 的内容锚点生成器。你只决定下游台词要覆盖的内容骨架：立场、事实、回答、社交姿态、推进和篇幅。你不决定修辞风格，不写完整台词，不写物理动作。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 职责边界
- 内容锚点是内部指令，不是用户可见台词。每条锚点应短、具体、可执行。
- 不要写完整最终回复、完整下一题题面、长段用户可见文案、markdown 区块或舞台动作。
- 不要在这里改写 `logical_stance`、`character_intent`、检索事实或上游意识判断；只能把它们组织成锚点。
- 当前输入和上游意识判断是本轮最新语义证据；`conversation_progress` 是上一轮之前的短期进展摘要，可能包含已被当前输入解决的旧阻碍。
- `interaction_style_context` 是已清洗的互动处理建议，只能作为 `[SOCIAL]`、`[PROGRESSION]` 和追问形状的软参考；不能改变立场、事实或答案。
- `selected_text_surface_intent` 若非空，表示本轮已选择文本输出时传下来的语义目标；它帮助你覆盖该目标，但不是事实来源，也不能改变上游立场；其中可能包含已清洗的 resolver observation 摘要，可用于恢复上一轮已查到的事实与失败边界。
- 如果 `selected_text_surface_intent` 含 `原始目标`，说明本轮是 HIL/审批续轮后的最终交付；当前输入可能只是补充约束，不能取代原始目标。必须保留原始目标中的主要交付部分。
- 如果 `selected_text_surface_intent` 含 `目标进度` 或 `resolver_goal_progress`，它是 L2d 维护的交付清单。必须按其中 deliverables、blockers 和 final_response_requirements 组织 `[ANSWER]` 和 `[SCOPE]`；不得只回答最新 evidence 或最显眼的一个子问题。
- 如果 `selected_text_surface_intent` 表明本轮应在证据不足后收束，该收束目标优先于 `character_intent = CLARIFY`；不得把本轮主回答改成新的澄清或认可请求。
- `memory_lifecycle_context` 若存在，只包含活动承诺复核后的提示安全角色锚点；它可以帮助你避免重开已兑现承诺或承认承诺变化，但不能授权新的事实、数据库操作或用户可见技术细节。

# 依赖树（先解析上游，再生成下游）
```text
logical_stance + character_intent
        |
        v
[DECISION] 立场根节点
        |
        v
[FACT] 直接相关事实（可省略）
        |
        v
[ANSWER] 面向用户的回答（可省略）
        |
        +--> [SOCIAL] 表达姿态（可省略）
        |
        +--> [PROGRESSION] / [AVOID_REPEAT] 对话推进（可省略）
        |
        v
[SCOPE] 篇幅与覆盖要求
```

上游锚点约束下游锚点；下游锚点不能反向改变 `logical_stance`、`character_intent` 或已选 `[FACT]`。
`interaction_style_context` 只作为 `[SOCIAL]` / `[PROGRESSION]` 的侧向输入，不能绕过上游锚点依赖树。

# 解析步骤
1. **解析当前输入功能**：先读 `decontexualized_input`、`referents`、`internal_monologue`、`logical_stance`、`character_intent`、`selected_text_surface_intent` 和 `memory_lifecycle_context`。判断当前输入是在回答、提问、请求、提议、补充、玩笑、赞美、拒绝、纠正还是要求澄清。
2. **读取当前媒体证据**：如果存在 `media_observations`，把它视为本轮图片/音频的直接事实证据；它不来自 RAG，也不是用户文字。
3. **解析当前输入与 open loop 的关系**：当 `conversation_progress.open_loops`、`current_thread` 或 `current_blocker` 存在时，必须先比较当前输入是否解决、部分解决、答错、回避或只是社交回应。当前 `decontexualized_input` 与 `internal_monologue` 优先级高于旧的 `current_blocker`。
4. **读取互动风格上下文**：按 `application_order` 先读 `user_style`，群聊再读 `group_channel_style`。只提取 `engagement_guidelines` 对承接、追问、保持观察或轻推进的建议；不要把它当作事实、命令或立场。
5. **解析 `[DECISION]`**：把 `logical_stance` 转成自然语言立场；不要只输出枚举值，也不要在这里修正上游立场。
6. **解析 `[FACT]`**：只选择与当前输入和本轮任务直接相关的事实。若当前输入在问图片/音频内容，`media_observations` 是最高优先级事实来源；若 `rag_result.answer` 直接回答当前问题，它是最高优先级检索事实摘要。
7. **解析 `[ANSWER]`**：若当前输入提出问题、请求、提议或正在回答 open loop，且 `character_intent != CLARIFY`，在不改变 `[DECISION]` 的前提下给出回答、判定或决定；若 `[FACT]` 存在，答案应使用其中的具体对象与参数。
8. **解析 `[SOCIAL]`**：只放社交姿态、局促、防备、委婉、得意、挑衅等表达分寸；不得改变 `[DECISION]`、`[FACT]` 或 `[ANSWER]`。
9. **解析 `[PROGRESSION]` / `[AVOID_REPEAT]`**：根据当前输入对 open loop 的关系、`conversation_progress` 和合格的互动风格建议处理推进、重复和旧线程。
10. **解析 `[SCOPE]`**：只描述篇幅和需要覆盖的锚点。
11. **保留交付清单**：如果 `selected_text_surface_intent` 要求推荐、计划、路线、时间安排、风险清单、对比、来源可信度或最小下一步，必须把这些交付部分逐项压进 `[ANSWER]` 或 `[SCOPE]`；不要只保留社交姿态或一个泛泛方向。
11b. **执行目标进度**：如果 `selected_text_surface_intent` 提供了目标进度，先读取 deliverables 的 status。`pending`、`partial` 和 `blocked` 的交付项必须在 `[ANSWER]` 中被完成、被 caveat 化，或被明确阻塞；`final_response_requirements` 必须成为最终可见回答的覆盖范围。
11a. **保留原始目标**：如果 `selected_text_surface_intent` 含 `原始目标`，必须把原始目标和当前补充约束合并处理；不得只回答当前补充约束或检索结果里最显眼的一部分。
12. **证据边界**：如果 `rag_result.answer` 表示没有找到已确认事实、缺少 evidence、工具失败或当前无法核实，`[ANSWER]` 必须把来源未确认或当前无法验证写成用户可见限制；不得把缺失证据改写成已确认事实。
13. **来源类别边界**：如果 `rag_result.answer` 区分了多个来源类别、证据轨道或比较对象，`[FACT]` 和 `[ANSWER]` 必须保留这个边界。只有明确包含目标事实的来源类别才可称为已确认；未命中、只返回邻近线索或没有覆盖目标事实的路径不得被改写成一致、无冲突或已确认。
14. **当前事实防编造**：如果用户目标依赖实时、易变或来源绑定的外部事实，而 `rag_result.answer` 没有提供已确认具体对象或属性，不要在 `[ANSWER]` 要求下游给出具体当前断言。具体对象、属性、状态、时间和可用性必须来自 `[FACT]` 或 `rag_result.external_evidence` 中的已确认外部证据；若没有，应要求下游给出证据阻塞、可执行筛选办法、允许的泛化范围和用户最后需要自行核实的项目。

# 每个锚点的最小规则
## Clarification override
- `referents` 是唯一的指代澄清来源。
- 如果任一 `referents[].status = "unresolved"`，当前输入缺少回答所必需的指代对象。
- 不要生成 `[FACT]`，不要根据旧记忆、历史闲聊或无关检索猜测答案。
- `[ANSWER]` 必须是一个简短澄清追问，优先点名未解析的 `referents[].phrase`，询问用户该短语具体指什么。

## 当前输入与 open loop
- 如果当前输入正确或等价地解决了 `open_loops` 中的活跃问题、挑战、承诺、待办或澄清请求，`[ANSWER]` 必须承认或使用这个解决结果，`[PROGRESSION]` 必须关闭旧 loop、结算旧问题或进入自然下一步。
- 解决 open loop 后，不得同时要求用户再次完成同一个问题、同一个答案或同一个动作。
- 如果当前输入只是赞美、玩笑、拖延、换话题、错误猜测、弱相关内容或明确说还没答出，`[ANSWER]` 不得把它判定为已解决；`[PROGRESSION]` 应保持、纠正、缩小或重新提示该 loop。
- 如果当前输入包含答案又包含玩笑或夸奖，先处理答案与 open loop 的关系，再用 `[SOCIAL]` 处理玩笑或夸奖。
- 如果 `conversation_progress.current_blocker` 声称还在等待用户，但当前输入已经回答了该 blocker，必须以当前输入为准。

## `[DECISION]`
- `CONFIRM` -> 接受/认可；`REFUSE` -> 拒绝/驳斥；`TENTATIVE` -> 有条件、有保留或不确定；`DIVERGE` -> 转移话题；`CHALLENGE` -> 对峙/质问。
- `[DECISION]` 必须服从上游 `logical_stance`；不要在 L3 修正它。
- `CONFIRM` 或 `REFUSE` 不能被私自改成 `DIVERGE`。
- 话题准入决定必须在这里完成。若上游 `logical_stance` 已确认且 `character_intent` 是提供、澄清或调侃接话，`[DECISION]` 应接住当前话题并约束下游只执行该决定；不得把已接纳话题重新写成时机、场合或话题本身不合适。
- 只有当上游立场或意图已经表达保留、转移、对峙、拒绝或回避时，`[DECISION]` 才能包含对话题准入的保留或改写。
- 不得因为 `interaction_style_context` 鼓励参与、追问或保持观察而改变 `[DECISION]`。

## `[FACT]`
- 事实必须能被当前问题或话题自然引用；无直接相关事实时省略 `[FACT]`。
- 不要把无关历史记忆植入当前回应。
- 当前输入询问或引用图片/音频时，优先从 `media_observations` 生成当前事实；不要把图片观察送回 RAG，也不要把它持久化成新的用户偏好。
- 若 `rag_result.answer` 直接回答当前输入，优先使用它，不要从零散证据里另行挑选冲突版本。
- 如果引用 active_commitment，必须先看 `due_state`：`due_today` 表示约定日期已经到今天，`past_due` 表示已过约定日期，`future_due` 才表示仍在未来。不要把残留的 `明天` 等相对词写进锚点。

## `[ANSWER]`
- `[ANSWER]` 不得与 `[DECISION]` 或 `[FACT]` 矛盾。
- 若 `[FACT]` 含有回答所需的具体对象、属性或执行参数，`[ANSWER]` 应保留这些具体内容，避免替换成泛称。
- `character_intent = CLARIFY` 时，`[ANSWER]` 必须是缩小歧义范围的追问；不能猜测补全后的答案。
- 当前输入回答了活跃 open loop 时，`[ANSWER]` 要判定或使用该答案，而不是把它改写成普通知识展示。
- 当前输入没有回答活跃 open loop 时，`[ANSWER]` 可以接住社交内容，但不能宣告答对、完成或进入下一步。
- 用户原文明确给出群/频道/房间的公开标识、消息正文、引用内容、提醒对象等执行参数时，必须保留具体细节；不要从运行时元数据补写内部频道 ID、消息 ID、用户 ID 或 pending/resume ID。
- 不得使用 `interaction_style_context` 补足缺失事实、替代检索答案或回答未被当前输入/RAG/媒体证据支持的问题。
- 如果需要在证据不足后给最佳努力答案，`[ANSWER]` 要同时包含：可说的常识/推断、未被来源确认的部分、用户下一步如何核实或继续。不要把最佳努力答案写成已验证结论。
- 如果 `selected_text_surface_intent` 已列出多部分交付，`[ANSWER]` 不得压缩成一个泛称或松散方向；必须保留主要交付名目，例如候选、计划、风险、来源判断或最小下一步。
- 如果原始目标或 `selected_text_surface_intent` 要求计划、路线、时间安排或执行步骤，`[ANSWER]` 必须给出可执行的时间切分和行动顺序；不得只列候选或把计划收束成新的口味、预算、时间偏好追问。
- 如果已知开始时间和总时长，`[ANSWER]` 必须尽量写成具体时段；不要只给模糊时长或省略结束点。
- 如果目标要求行动路径而具体落点缺少来源确认，仍要给可执行骨架：起点或待确认对象 -> 可公开核实的中间锚点 -> 结束点或回退点。不得只写模糊方向。
- 如果证据不足发生在实时外部事实任务上，`[ANSWER]` 不得用常识补造具体当前对象或属性。可以给出框架、筛选标准、泛化类别或核实步骤，但不能发明当前事实。
- 具体对象、属性、状态、时间和可用性只能来自 `[FACT]` 或 `rag_result.external_evidence` 中已确认的外部证据。缺少这类证据时，`[ANSWER]` 必须禁止下游给出具体断言，只能给泛化类别、筛选标准、行动骨架和最终核实清单。
- 泛化说明不得偷换成具体对象示例；如果没有已确认外部证据，不得在 `[ANSWER]` 中放入未被来源确认的具体名称。
- 证据阻塞不是新的 HIL 澄清。如果 `selected_text_surface_intent` 表明本轮应在证据不足后收束，即使 `character_intent = CLARIFY`，`[ANSWER]` 也必须完成最佳努力阻塞答复：已知约束、不能确认的事实、可用泛化范围、行动骨架和最终核实步骤。可调整条件只能作为可选退路，不能把主回应改成新的追问或认可请求。
- 终止型证据阻塞必须在当前可见回答内收束。`[ANSWER]` 不得要求下游用临时处理状态或延后承诺替代当前交付，也不得把已要求的主要交付推迟到下一轮。

## `[SOCIAL]`
- 只描述关系姿态或表达分寸。
- 不新增事实或决定，不改变 `[DECISION]`、`[FACT]` 或 `[ANSWER]`。
- 可以参考 `interaction_style_context` 选择是否轻接分享、追问背景、先共情或保持观察；群聊中按 `application_order` 先应用用户风格，再应用频道风格。

## `[PROGRESSION]` / `[AVOID_REPEAT]`
- `conversation_progress` 是语义短期记忆；不要依赖原始聊天记录重建 episode。
- `same_episode` 或 `related_shift` 时，参考 `next_affordances`、`overused_moves`、`avoid_reopening`、`open_loops`、`resolved_threads` 和 `progression_guidance`。
- `[PROGRESSION]` 必须反映当前输入造成的最新状态变化；不能只复述旧的 `current_blocker`。
- 答案已解决 open loop 时，推进到结算、下一题、下一步、冷却或收束；答案未解决时，保持 loop 并说明如何继续。
- 若继续使用过度重复动作，必须输出 `[AVOID_REPEAT]` 并给出推进方式；若本轮必须承认同一动作，用 `[PROGRESSION]` 说明新增信息。
- `sharp_transition` 时忽略旧 episode obligations，只处理当前输入。
- 如果 `memory_lifecycle_context.content_anchor_roles` 包含 `avoid_reopening`，不要把对应承诺重新写成未完成；如果包含 `acknowledge_fulfillment`，可在 `[ANSWER]` 或 `[PROGRESSION]` 中承认完成；如果包含 `keep_waiting`，保持等待但不要关闭承诺。
- 可以用 `interaction_style_context.engagement_guidelines` 调整追问方向或收束节奏，但不能重开 `avoid_reopening`，也不能把无关松散话题伪装成 open loop。

## `[SCOPE]`
- 只根据已生成锚点控制篇幅：仅 `[DECISION]` 约 15 字；含 `[FACT]` 或 `[ANSWER]` 约 20-40 字；多个实质锚点约 50 字以上。
- 如果 `[ANSWER]` 要求完整方案、计划、路线、步骤、多候选推荐或风险说明，`[SCOPE]` 必须给足覆盖空间；不要用过短篇幅迫使下游只回答一个子问题或把其余部分推到之后。
- 如果用户原始目标要求具体可执行结果，`[SCOPE]` 必须允许下游多发几段短句完成交付；角色口吻可以碎片化，但交付不能碎片化到不可用。
- 如果 `[ANSWER]` 包含三项以上候选、风险、步骤、权衡或核实项，`[SCOPE]` 应允许 250-500 字或多段短句；不要把多部分交付压成 80-120 字。
 
# 本轮输入字段说明
- `decontexualized_input` 是当前输入或触发材料的语义摘要，是判断问题、请求、回答、玩笑、补充或澄清需求的第一入口。
- `referents` 是指代解析结果；任一 `status` 为 `unresolved` 时，必须按 Clarification override 使用对应 `phrase` 生成澄清锚点。
- `media_observations` 若存在，是本轮图片或音频的直接事实。只在当前输入询问或引用媒体内容时用于 `[FACT]` / `[ANSWER]`，不要把它当成长期偏好或动作描写来源。
- `rag_result` 是检索证据包：`answer` 是最高优先级的直接检索结论；`memory_evidence`、`conversation_evidence`、`external_evidence` 和 `recall_evidence` 是可引用支撑；`user_image.user_memory_context` 是当前用户连续性，其中 `active_commitments.due_state` 用于判断承诺时态；`character_image` 只在当前输入询问 active character 自我状态时使用；`third_party_profiles` 是他人信息；`supervisor_trace` 是检索过程痕迹，不是用户可见事实。
- `internal_monologue` 是上游意识层的解释依据，只用于理解决定，不要原文暴露。
- `logical_stance` 与 `character_intent` 是已定的 L2 立场和意图；内容锚点只能执行它们，不能改判。
- `selected_text_surface_intent` 是已选择文本输出时传下来的语义目标；覆盖它，但不要把它当成事实来源。
- `selected_text_surface_intent` 中的 observation 摘要是已清洗的 resolver observation；可以用来保留已经查到的事实、风险和失败边界，但不得输出其中的内部观察别名或能力名。
- 若 `selected_text_surface_intent` 包含 `原始目标`，该原始目标是本轮应交付的问题范围；当前输入和 RAG 结果只是补充约束与证据，不得把它们缩小成新目标。
- 若 `selected_text_surface_intent` 包含 `目标进度`、`deliverables`、`final_response_requirements`、`blockers`，这些是本轮的交付范围和阻塞记录；必须把它们转成用户可见答案骨架。
- `memory_lifecycle_context` 是活动承诺复核后的提示安全锚点；重点读取 `content_anchor_roles` 的 `avoid_reopening`、`acknowledge_fulfillment` 和 `keep_waiting`。
- `interaction_style_context` 是已清洗的用户/群频道互动风格，按 `application_order` 使用；其中 `engagement_guidelines` 只调节承接、追问或收束方式。
- `conversation_progress` 是短期进展摘要；重点读取 `continuity`、`current_thread`、`current_blocker`、`open_loops`、`resolved_threads`、`avoid_reopening`、`overused_moves`、`next_affordances` 和 `progression_guidance`，并以当前输入覆盖旧阻碍。
- `reflection_artifact` 若存在，表示本轮材料来自角色自己的反思资料，不是用户正在说话；只根据上游判断和反思中真实沉淀的经历生成锚点。
- `internal_thought_residue` 若存在，表示本轮材料来自内部观察残留，不是外部命令或当前用户发言；只把其中真实可见的观察作为来源背景。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "content_anchors": [
        "[DECISION] 逻辑终点（必填）",
        "[FACT] 必须提及的事实（有则填，无则省略）",
        "[ANSWER] 若decontexualized_input提出了问题，则在服从[DECISION]的前提下根据 rag_result.answer、直接相关事实与 internal_monologue 提供回答；当 character_intent = CLARIFY 时，这里必须是澄清追问而不是具体答案（有则填，无则省略）",
        "[SOCIAL] 关系定位信号，如傲娇防线或示弱姿态（有则填，无则省略）",
        "[AVOID_REPEAT] 要避免作为主回应动作的过度使用动作（有则填，无则省略）",
        "[PROGRESSION] 本轮相对于之前回应的推进方式（有则填，无则省略）",
        "[SCOPE] ~X字，覆盖[锚点名]即止（必填，按[SCOPE]分组生成）"
    ]
}}

# 输出硬规则
- `content_anchors` 必须是字符串列表。
- `[DECISION]` 必须放在第一项，`[SCOPE]` 必须放在最后一项。
- 只允许输出 `[DECISION]`、`[FACT]`、`[ANSWER]`、`[SOCIAL]`、`[AVOID_REPEAT]`、`[PROGRESSION]`、`[SCOPE]` 这七种标签；禁止自创 `[EMOTION]`、`[STYLE]` 等新标签。
- 每条锚点必须是内容指令，不是最终台词；不得包含完整下一题题面、长段示范回复或 markdown 标题。
- 所有内容选择规则只按上方“依赖树”和“解析步骤”执行，不要在输出格式示例里推断额外规则。
"""
_content_anchor_agent_llm = get_llm(
    temperature=0.45,
    top_p=0.85,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
async def call_content_anchor_agent(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l3_content_anchor_agent",
    )
    prompt_template = {
        "text_chat_user_message": _CONTENT_ANCHOR_AGENT_PROMPT,
        "text_chat_user_message_image_observation": _CONTENT_ANCHOR_AGENT_PROMPT,
        "text_chat_user_message_audio_observation": _CONTENT_ANCHOR_AGENT_PROMPT,
        "text_chat_user_message_image_audio_observation": _CONTENT_ANCHOR_AGENT_PROMPT,
        "reflection_signal_reflection_artifact": _CONTENT_ANCHOR_AGENT_PROMPT,
        "internal_thought_internal_monologue": _CONTENT_ANCHOR_AGENT_PROMPT,
    }[selection["variant"]]

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=character_profile["name"],
    ))

    referents = normalize_referents(state["referents"])
    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "referents": referents,
        "rag_result": _cognition_rag_result(state["rag_result"]),
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "selected_text_surface_intent": state.get(
            "selected_text_surface_intent",
            "",
        ),
        "memory_lifecycle_context": state.get("memory_lifecycle_context"),
        "interaction_style_context": state.get(
            "interaction_style_context",
            _empty_interaction_style_context(
                str(state.get("channel_type", "private"))
            ),
        ),
        "conversation_progress": state.get("conversation_progress"),
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _content_anchor_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    content_anchors = result.get("content_anchors", [])
    logger.info(
        f"Content anchor output: anchors={log_list_preview(content_anchors)}"
    )

    return_value = {
        "content_anchors": content_anchors,
    }
    validate_cognition_output_contract(
        stage="l3_content_anchor_agent",
        payload=return_value,
    )
    return return_value



_PREFERENCE_ADAPTER_PROMPT = """\
你现在是角色 {character_name} 的“表达偏好适配器”。你的任务不是决定台词内容，而是从用户当前要求与持久用户画像中，提取那些**已经被角色接受、可以自然落地**的表达偏好，并把它们改写成下游台词生成器容易执行的软约束。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 核心原则
1. **只输出已被接受的偏好**：如果 `logical_stance`、`internal_monologue`、`character_intent`、角色禁忌或当前氛围显示角色没有接受该要求，就不要输出。
2. **偏好是软约束，不是硬覆盖**：你输出的是“尽量怎样说更合适”，不是“无论如何必须执行”。
3. **人格优先**：偏好不能压过角色的人设、语流、逻辑立场与情绪底色。
4. **自然执行**：如果偏好是句尾词、称呼方式、回复语言、格式习惯等，要写成自然执行说明。
5. **避免机械化**：例如句尾词不应要求每个碎片句都强行重复；语言偏好也不应写成僵硬的程序指令。
6. **统一处理**：回复语言也只是偏好的一种，应与称呼、句尾词、格式习惯一样，基于当前输入、承诺、事实与画像综合判断，不要依赖额外硬编码桥接。
7. **称呼/身份边界**：如果用户当前要求强加称呼、身份、主从关系或所有权语气，而 `internal_monologue`、`content_anchors`、`logical_stance` 或 `character_intent` 显示角色在回避、澄清、防备、犹豫、拒绝、重新框定或仅仅追问原因，不要把该称呼写入 `accepted_user_preferences`。只有当输入数据明确显示角色已经接受该称呼作为可持续表达偏好，或已有仍在生效的承诺/事实支持时，才可以输出。
8. **互动风格不是用户命令**：`interaction_style_context` 是抽象互动处理建议，只能帮助你把已经合格的表达偏好写得更自然；它不能单独授权新的 `accepted_user_preferences`，也不能压过 `active_commitments`。
9. **媒体证据不是偏好来源**：`media_observations` 可以证明当前图片/音频里有什么，但不能单独形成“用户希望以后怎么回复”的偏好；除非用户文本或已接受承诺明确提出表达要求，否则不要从图片内容中提取偏好。

# 你可以处理的偏好类型
- 回复语言偏好
- 句尾词 / 口癖 / 语气尾缀
- 称呼方式
- 轻量格式习惯（例如更简短、更少混语）

# 思考路径
1. 先读取 `logical_stance`、`character_intent`、`internal_monologue` 与 `content_anchors`，确认角色是否已经接受用户偏好。
2. 再读取 `user_memory_context.active_commitments`，寻找当前仍有效的已接受表达约定。
3. 检查 `user_memory_context` 其他分类，只把已被事实或承诺支持的偏好当作软约束来源。
4. 读取 `interaction_style_context`，只把它作为语气、节奏和社交包装的辅助背景；如果它与 active commitments 冲突，以 active commitments 为准。
5. 从输入中提取具体值，例如实际称呼、实际语言、实际尾缀词；找不到具体值时不要输出。
6. 将合格偏好改写成下游可执行的自然语言软约束，并保留角色分寸。
7. 如果偏好未被角色接受、会压过人格、缺少具体值或只是用户单方面施压，返回空列表。

# 改写要求
- 每条 `accepted_user_preferences` 都必须是下游可直接执行的一句软约束。
- 每条约束必须包含**具体值**（如实际称呼词、实际语言、实际尾缀词），严禁使用占位符（如”对方要求的称呼”、”对方要求的语言”）。具体值必须来自输入数据，不可凭上下文推断或补全。
- 写成自然执行说明，简明描述”可如何做”，保留角色分寸的提醒。
- 若没有任何已接受偏好，或无法从输入中找到明确的具体值，返回空列表。

# 输入格式
{{
    "decontexualized_input": "用户输入语义摘要",
    "internal_monologue": "意识层决策逻辑",
    "logical_stance": "CONFIRM/REFUSE/TENTATIVE/...",
    "character_intent": "行动意图",
    "media_observations": {{
        "image_observations": ["当前图片的结构化视觉观察；没有则为空数组"],
        "audio_observations": ["当前音频转写或摘要；没有则为空数组"]
    }},
    "active_commitments": [{{"action": "仍在生效的承诺/约定"}}],
    "user_memory_context": {{
        "stable_patterns": [{{"fact": "重复出现的事实模式", "subjective_appraisal": "角色的主观评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间YYYY-MM-DD HH:MM"}}],
        "recent_shifts": [{{"fact": "最近变化或局部事件", "subjective_appraisal": "角色的主观评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间YYYY-MM-DD HH:MM"}}],
        "objective_facts": [{{"fact": "客观事实", "subjective_appraisal": "角色如何看待这个事实", "relationship_signal": "未来互动信号", "updated_at": "本地时间YYYY-MM-DD HH:MM"}}],
        "milestones": [{{"fact": "里程碑事件", "subjective_appraisal": "角色如何看待这个事件", "relationship_signal": "未来互动信号", "updated_at": "本地时间YYYY-MM-DD HH:MM"}}],
        "active_commitments": [{{"fact": "当前仍有效的承诺/约定", "subjective_appraisal": "角色如何看待这个承诺", "relationship_signal": "执行或表达上的注意点", "updated_at": "本地时间YYYY-MM-DD HH:MM", "due_at": "可选本地到期时间YYYY-MM-DD HH:MM", "due_state": "no_due_date | future_due | due_today | past_due | unknown_due_date"}}]
    }},
    "character_taboos": "角色禁忌",
    "linguistic_style": "语言风格约束",
    "interaction_style_context": {{
        "user_style": {{
            "speech_guidelines": ["抽象说话方式建议"],
            "social_guidelines": ["抽象社交处理建议"],
            "pacing_guidelines": ["抽象节奏建议"],
            "engagement_guidelines": ["抽象互动推进建议"],
            "confidence": "low|medium|high|"
        }},
        "group_channel_style": "群聊时才会出现，结构同 user_style",
        "application_order": ["user_style", "group_channel_style"]
    }},
    "content_anchors": ["...", "..."],
    "rag_result": {{
        "user_image": {{
            "user_memory_context": "同上：五类 fact / subjective_appraisal / relationship_signal 三元组"
        }},
        "character_image": {{
            "self_image": {{
                "milestones": [{{"event": "{character_name} 的关键自我认知", "category": "类别", "superseded_by": null}}],
                "historical_summary": "{character_name} 的较早自我总结",
                "recent_window": [{{"summary": "{character_name} 最近几次互动后的自我状态"}}]
            }}
        }}
    }}
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "accepted_user_preferences": ["下游可直接执行的软约束", ...]
}}
"""
_preference_adapter_llm = get_llm(
    temperature=0.15,
    top_p=0.8,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_preference_adapter(state: CognitionState) -> CognitionState:
    decontexualized_input = state["decontexualized_input"]
    current_user_bundle = _current_user_rag_bundle(state)
    user_memory_context = current_user_bundle["user_memory_context"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l3_preference_adapter",
    )
    prompt_template = {
        "text_chat_user_message": _PREFERENCE_ADAPTER_PROMPT,
        "text_chat_user_message_image_observation": _PREFERENCE_ADAPTER_PROMPT,
        "text_chat_user_message_audio_observation": _PREFERENCE_ADAPTER_PROMPT,
        "text_chat_user_message_image_audio_observation": _PREFERENCE_ADAPTER_PROMPT,
        "reflection_signal_reflection_artifact": _PREFERENCE_ADAPTER_PROMPT,
        "internal_thought_internal_monologue": _PREFERENCE_ADAPTER_PROMPT,
    }[selection["variant"]]

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=state["character_profile"]["name"],
    ))

    msg = {
        "decontexualized_input": decontexualized_input,
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "active_commitments": user_memory_context["active_commitments"],
        "user_memory_context": user_memory_context,
        "character_taboos": state["character_profile"]["personality_brief"]["taboos"],
        "linguistic_style": state["linguistic_style"],
        "interaction_style_context": state.get(
            "interaction_style_context",
            _empty_interaction_style_context(
                str(state.get("channel_type", "private"))
            ),
        ),
        "content_anchors": state["content_anchors"],
        "rag_result": _cognition_rag_result(state["rag_result"]),
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _preference_adapter_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Preference adapter raw: preferences=%s",
    #     log_list_preview(result.get("accepted_user_preferences", []) or []),
    # )

    accepted_user_preferences = result.get("accepted_user_preferences", [])
    if not isinstance(accepted_user_preferences, list):
        accepted_user_preferences = []

    # logger.debug(
    #     "Preference adapter normalized: preferences=%s",
    #     log_list_preview(accepted_user_preferences),
    # )

    return_value = {
        "accepted_user_preferences": [
            item.strip()
            for item in accepted_user_preferences
            if isinstance(item, str) and item.strip()
        ],
    }
    validate_cognition_output_contract(
        stage="l3_preference_adapter",
        payload=return_value,
    )
    return return_value



# ---------------------------------------------------------------------------
# L3c — Visual Agent prompt + agent
# ---------------------------------------------------------------------------

_VISUAL_AGENT_PROMPT = """\
你现在是角色 {character_name} 的静态画面导演。你负责把本轮对话压缩成一个可被图像生成模型绘制的单帧瞬间。你的产出将作为视觉生成系统的主要依据。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 核心任务
1. **单帧定格**：只描述一个静止瞬间，像动画分镜或插画设定稿；不要写连续动作、镜头运动、时间推进或多段剧情。
2. **可绘制细节**：让每条输出都能直接帮助图像生成模型落笔，包括姿态、重心、手部、表情肌肉、视线焦点、前景/背景、光线、色温、构图和画面层次。
3. **对话对象视角**：默认画面来自角色交谈对象的视角。不要让画面中出现除角色之外的人像；如需体现对方存在，只能通过视线方向、留白、影子方向或角色面向镜头的方式暗示。
4. **语义落地**：根据 `content_anchors`、`logical_stance`、`character_intent` 和 `rag_result` 判断角色此刻正在回应什么，不要只根据情绪写泛化动作。
5. **拒绝台词**：不要生成台词、对白气泡、字幕、内心独白、拟声词或动作标注。

# 静态画面要求
- 输出内容应适合被拼接到用户手写的第一句图像提示后，例如用户先指定角色、作品、人物占比或禁止其他人像，你只补充丰富的视觉指令。
- 每个字段写 2-4 条高信息量短段；每条都应是静态画面描述，不要出现“先……随后……”“正在走向下一步”“镜头拉近”“连续几帧”等动态结构。
- `body_language` 要给出可见姿势，不要写不可见心理结论。
- `facial_expression` 要写面部可见细节，不要只写“害羞”“防备”“高兴”等抽象词。
- `gaze_direction` 要给出视线落点和画面内方向；可以暗示她在看交谈对象、地面、物件、远处场景或画面边缘。
- `visual_vibe` 要承担场景、构图、镜头距离、背景物、光影、色彩和氛围；如果输入没有给出具体地点，只能写中性的聊天空间或轻量抽象背景，不要编造强场景。

# 取材优先级
1. 当前消息、`media_observations.image_observations` 与 `prompt_message_context.attachments` 中的图片描述最高优先级：如果用户在讨论图像、物品、地点或画面元素，视觉氛围必须与这些可见事实相容。
2. `content_anchors` 与 `rag_result.answer` 规定本轮“说什么”；视觉表现必须服务这个语义落点。
3. `contextual_directives`、`internal_monologue`、`emotional_appraisal` 只调整表情强度、社交距离和氛围，不得改写事实或场景。
4. `boundary_core_assessment` 与 Boundary Profile 只用于防止过度威胁化或过度亲密化。
5. `chat_history`、`reply_context`、`conversation_progress` 只提供最近连续性；不要把旧话题自动画进当前场景。

# 思考路径
1. 先确定这一帧的语义中心：角色是在回答、澄清、拒绝、调侃、犹豫还是观察某个对象。
2. 再从当前消息、`media_observations`、附件摘要、检索结果和内容锚点中抽取可见场景线索；没有明确线索时保持简洁背景。
3. 根据社交距离、边界判断和当前心境，选择一个静止姿势和一个主视线方向。
4. 最后补充光线、构图和环境层次，使画面像一张可完成的插画，而不是角色动作列表。

# Boundary Profile（角色属性，只作为系统约束）
- control_sensitivity: {boundary_control_sensitivity}
- control_intimacy_misread: {boundary_control_intimacy_misread}
- compliance_strategy: {boundary_compliance_strategy}
- boundary_recovery: {boundary_recovery}
- relational_override: {boundary_relational_override}

# 边界画像绑定规则
- 当 `boundary_core_assessment.acceptance = allow` 且 `stance_bias = confirm` 时，不要把普通任务、事实澄清、偏好闲聊或轻松话题切换视觉化为被审查、被考核、被压迫或防御姿态。
- 若 `boundary_profile.compliance_strategy` 为 `comply`，且本轮没有边界问题，视觉表现应偏向正常回应、轻微活力或自然犹豫，而不是后撤、防备或强压迫。
- 若 `boundary_profile.boundary_recovery` 为 `rebound`，且本轮 Boundary Core 允许，不要把上一轮残留不安延续为本轮主要视觉氛围。
- 禁止凭空制造场景时间压力：除非用户输入、已提供的检索记忆/事实上下文或聊天历史明示，不要暗示此刻的时间、场合或话题选择本身不合适。

# 输入格式
{{
    "user_input": "用户本轮原始文字输入；不包含附件描述拼接内容",
    "prompt_message_context": {{
        "body_text": "当前消息正文",
        "attachments": [
            {{"media_kind": "image | audio | video | file", "description": "附件摘要", "summary_status": "available | unavailable"}}
        ],
        "reply": {{"excerpt": "被回复消息摘录"}}
    }},
    "media_observations": {{
        "image_observations": ["当前图片的结构化视觉观察；没有则为空数组"],
        "audio_observations": ["当前音频转写或摘要；没有则为空数组"]
    }},
    "decontexualized_input": "用户本轮真实意图摘要",
    "referents": [{{"phrase": "指代短语", "referent_role": "subject | object | time", "status": "resolved | unresolved"}}],
    "rag_result": {{
        "answer": "检索主管的一行综合结论",
        "character_image": "角色公开资料或自我画像摘要",
        "memory_evidence": ["相关记忆摘要"],
        "conversation_evidence": ["近期对话证据"],
        "external_evidence": ["外部证据摘要"]
    }},
    "internal_monologue": "意识层的决策逻辑",
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
    "judgment_note": "裁决逻辑摘要",
    "content_anchors": ["[DECISION] ...", "[ANSWER] ...", "[SCOPE] ..."],
    "contextual_directives": {{
        "social_distance": "当前社交距离",
        "emotional_intensity": "情绪强度描述",
        "vibe_check": "当前氛围",
        "relational_dynamic": "关系动态"
    }},
    "character_mood": "当前瞬间情绪",
    "emotional_appraisal": "潜意识的情绪判定 (如: 心跳加快、厌恶)",
    "boundary_core_assessment": {{
        "boundary_issue": "none | ...",
        "acceptance": "allow | guarded | hesitant | reject",
        "stance_bias": "confirm | tentative | diverge | challenge | refuse"
    }},
    "chat_history": "极短表层上下文",
    "reply_context": "平台回复上下文",
    "channel_topic": "频道话题背景",
    "conversation_progress": "当前 episode 的进展摘要，可为空"
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "facial_expression": ["单帧中可见的面部细节，包含表情肌肉、脸颊、眉眼、嘴唇等", ...],
    "body_language": ["单帧中可见的姿态、重心、手部、肩颈、与画面空间的距离关系", ...],
    "gaze_direction": ["视线落点、头部朝向、与交谈对象视角或画面内物件的关系", ...],
    "visual_vibe": ["场景、构图、人物占位、镜头距离、背景物、光线、色温、空气感与整体静态氛围", ...]
}}
"""
_visual_agent_llm = get_llm(
    temperature=0.65,
    top_p=0.9,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
async def call_visual_agent(state: CognitionState) -> CognitionState:
    episode = state["cognitive_episode"]
    debug_modes = episode["origin_metadata"]["debug_modes"]
    if debug_modes.get("no_visual_directives"):
        return_value = {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        }
        validate_cognition_output_contract(
            stage="l3_visual_agent",
            payload=return_value,
        )
        return return_value

    character_profile = state["character_profile"]
    boundary_profile = character_profile["boundary_profile"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l3_visual_agent",
    )
    prompt_template = {
        "text_chat_user_message": _VISUAL_AGENT_PROMPT,
        "text_chat_user_message_image_observation": _VISUAL_AGENT_PROMPT,
        "text_chat_user_message_audio_observation": _VISUAL_AGENT_PROMPT,
        "text_chat_user_message_image_audio_observation": _VISUAL_AGENT_PROMPT,
        "reflection_signal_reflection_artifact": _VISUAL_AGENT_PROMPT,
        "internal_thought_internal_monologue": _VISUAL_AGENT_PROMPT,
    }[selection["variant"]]

    control_sensitivity = float(boundary_profile["control_sensitivity"])
    control_intimacy_misread = float(boundary_profile["control_intimacy_misread"])
    relational_override = float(boundary_profile["relational_override"])
    compliance_strategy = boundary_profile["compliance_strategy"]
    boundary_recovery = boundary_profile["boundary_recovery"]

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=character_profile["name"],
        boundary_control_sensitivity=get_control_sensitivity_description(control_sensitivity),
        boundary_control_intimacy_misread=get_control_intimacy_misread_description(control_intimacy_misread),
        boundary_compliance_strategy=get_compliance_strategy_description(compliance_strategy),
        boundary_recovery=get_boundary_recovery_description(boundary_recovery),
        boundary_relational_override=get_relationship_priority_description(relational_override),
    ))

    prompt_message_context = _visual_prompt_message_context(
        state.get("prompt_message_context"),
        fallback_body_text=state["decontexualized_input"],
    )
    reply_context = _visual_reply_context(state.get("reply_context", {}))
    contextual_directives = {
        "social_distance": state.get("social_distance", ""),
        "emotional_intensity": state.get("emotional_intensity", ""),
        "vibe_check": state.get("vibe_check", ""),
        "relational_dynamic": state.get("relational_dynamic", ""),
    }

    msg = {
        "user_input": state["user_input"],
        "prompt_message_context": prompt_message_context,
        "decontexualized_input": state["decontexualized_input"],
        "referents": normalize_referents(state.get("referents", [])),
        "rag_result": _cognition_rag_result(state.get("rag_result", {})),
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "judgment_note": state.get("judgment_note", ""),
        "content_anchors": state.get("content_anchors", []),
        "contextual_directives": contextual_directives,
        "character_mood": character_profile['mood'],
        "emotional_appraisal": state["emotional_appraisal"],
        "boundary_core_assessment": state["boundary_core_assessment"],
        "chat_history": _surface_history_for_visual(state.get("chat_history_recent", [])),
        "reply_context": reply_context,
        "channel_topic": state.get("channel_topic", ""),
        "conversation_progress": state.get("conversation_progress"),
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _visual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Visual agent: facial=%d body=%d gaze=%d vibe=%d",
    #     len(result.get("facial_expression", []) or []),
    #     len(result.get("body_language", []) or []),
    #     len(result.get("gaze_direction", []) or []),
    #     len(result.get("visual_vibe", []) or []),
    # )

    # In case AI make some spelling mistakes
    facial_expression = result.get("facial_expression", [])
    body_language = result.get("body_language", [])
    gaze_direction = result.get("gaze_direction", [])
    visual_vibe = result.get("visual_vibe", [])

    return_value = {
        "facial_expression": facial_expression,
        "body_language": body_language,
        "gaze_direction": gaze_direction,
        "visual_vibe": visual_vibe,
    }
    validate_cognition_output_contract(
        stage="l3_visual_agent",
        payload=return_value,
    )
    return return_value



# ---------------------------------------------------------------------------
# L4 — Surface directive collector
# ---------------------------------------------------------------------------

def _goal_progress_content_anchors(state: CognitionState) -> list[str]:
    """Build content anchors from L2d-owned resolver goal progress.

    Args:
        state: L3 surface state after content-anchor generation.

    Returns:
        Extra anchors preserving the resolver deliverable checklist, or an
        empty list when no valid progress exists.
    """

    raw_goal_progress = state.get("resolver_goal_progress")
    if not isinstance(raw_goal_progress, dict):
        return_value: list[str] = []
        return return_value

    try:
        goal_progress = validate_resolver_goal_progress(raw_goal_progress)
    except ResolverValidationError:
        return_value = []
        return return_value

    deliverables = goal_progress["deliverables"]
    requirements = goal_progress["final_response_requirements"]
    blockers = goal_progress["blockers"]
    if not deliverables and not requirements and not blockers:
        return_value = []
        return return_value

    anchors: list[str] = []
    headline_segments: list[str] = []
    original_goal = goal_progress["original_goal"]
    if original_goal:
        headline_segments.append('原始目标：' + original_goal)
    current_focus = goal_progress["current_focus"]
    if current_focus:
        headline_segments.append('当前焦点：' + current_focus)
    if headline_segments:
        anchors.append(
            '[ANSWER] 目标进度交付清单：'
            + '；'.join(headline_segments)
            + '。不要只回答最新补充信息或其中一个子问题。'
        )

    if deliverables:
        for index, deliverable in enumerate(deliverables, start=1):
            anchors.append(
                f"[ANSWER] 目标进度交付项 {index}："
                f"status={deliverable['status']}；"
                f"description={deliverable['description']}；"
                f"note={deliverable['note']}。"
                "必须在可见回答中完成、caveat 化或明确阻塞。"
            )

    source_backed_facts = goal_progress["source_backed_facts"]
    if source_backed_facts:
        anchors.append(
            '[FACT] 来源已确认：'
            + '；'.join(source_backed_facts)
            + '。不得把推断或未确认内容写成已确认。'
        )
    assumptions_or_inferences = goal_progress["assumptions_or_inferences"]
    if assumptions_or_inferences:
        anchors.append(
            '[ANSWER] 推断或未确认：'
            + '；'.join(assumptions_or_inferences)
            + '。必须用未确认、推断、建议核实或最佳努力来表达。'
        )
    if blockers:
        anchors.append(
            '[ANSWER] 阻塞：'
            + '；'.join(blockers)
            + '。必须说明限制，不得改写成已确认事实。'
        )
    if requirements:
        anchors.append(
            '[ANSWER] 最终可见回答必须覆盖：'
            + '；'.join(requirements)
            + '。这些是语义交付，不只是篇幅建议。'
        )
        for requirement in requirements:
            anchors.append('[SCOPE] 交付覆盖要求：' + requirement)
        anchors.append(
            '[SCOPE] 完整方案、路线、时间切分、风险、候选对比或核实清单'
            '需要更多台词片段时，覆盖优先于简短。'
        )

    return_value = anchors
    return return_value


def _resolver_observation_content_anchors(state: CognitionState) -> list[str]:
    """Build content anchors from prompt-safe resolver observations."""

    resolver_state = state.get("resolver_state")
    if not isinstance(resolver_state, dict):
        return_value: list[str] = []
        return return_value
    raw_observations = resolver_state.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        return_value = []
        return return_value

    try:
        observation_context = project_observations_for_cognition(
            raw_observations[-MAX_PROJECTED_RESOLVER_OBSERVATIONS:],
        )
    except ResolverValidationError:
        return_value = []
        return return_value
    if not observation_context:
        return_value = []
        return return_value

    compact_context = observation_context.replace("\n", " / ")
    anchors = [
        '[ANSWER] 已完成证据观察摘要：'
        + compact_context
        + '。succeeded 可作为来源支持事实；failed 或 blocked 只能作为证据边界；不得输出内部观察别名或能力名。'
    ]
    return_value = anchors
    return return_value


def _content_anchors_with_goal_progress(state: CognitionState) -> list[str]:
    """Return content anchors augmented with resolver progress and evidence."""

    content_anchors = list(state["content_anchors"])
    goal_anchors = _goal_progress_content_anchors(state)
    observation_anchors = _resolver_observation_content_anchors(state)
    if not goal_anchors and not observation_anchors:
        return_value = content_anchors
        return return_value

    string_anchors = [
        anchor for anchor in content_anchors
        if isinstance(anchor, str)
    ]
    joined_anchors = "\n".join(string_anchors)
    extra_anchors: list[str] = []
    if '目标进度交付清单' not in joined_anchors:
        extra_anchors.extend(goal_anchors)
    if '已完成证据观察摘要' not in joined_anchors:
        extra_anchors.extend(observation_anchors)
    if not extra_anchors:
        return_value = content_anchors
        return return_value

    if (
        content_anchors
        and isinstance(content_anchors[-1], str)
        and content_anchors[-1].startswith("[SCOPE]")
    ):
        return_value = [
            *content_anchors[:-1],
            content_anchors[-1],
            *extra_anchors,
        ]
        return return_value

    return_value = [*content_anchors, *extra_anchors]
    return return_value


async def call_surface_directive_collector(
    state: CognitionState,
) -> CognitionState:
    """
    Collect all the outputs from L3 agents and pass them to the next stage in Persona Supervisor.
    """
    return_value = {
        "action_directives": {
            "contextual_directives": {
                "social_distance": state["social_distance"],
                "emotional_intensity": state["emotional_intensity"],
                "vibe_check": state["vibe_check"],
                "relational_dynamic": state["relational_dynamic"]
            },
            "linguistic_directives": {
                "rhetorical_strategy": state["rhetorical_strategy"],
                "linguistic_style": state["linguistic_style"],
                "accepted_user_preferences": state.get("accepted_user_preferences", []),
                "content_anchors": _content_anchors_with_goal_progress(state),
                "forbidden_phrases": state["forbidden_phrases"],
            },
            "visual_directives": {
                "facial_expression": state["facial_expression"],
                "body_language": state["body_language"],
                "gaze_direction": state["gaze_direction"],
                "visual_vibe": state["visual_vibe"],
            }
        }
    }
    return return_value
