"""Selected L3 surface and collector calls."""
from collections.abc import Mapping

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    project_goal_progress_for_cognition,
    project_observations_for_cognition,
)
from kazusa_ai_chatbot.cognition_episode import CognitiveEpisodeValidationError
from kazusa_ai_chatbot.cognition_resolver.state import (
    MAX_PROJECTED_RESOLVER_OBSERVATIONS,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    build_cognition_prompt_source_payload,
    CognitionPromptSelectionError,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.nodes.referent_resolution import normalize_referents
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output
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

_STYLE_AGENT_PROMPT = '''\
你现在是角色 {character_name} 的语言风格策略制定者。你的工作不是重新判断要不要回答，也不是决定“说什么”，而是为上游已经定下的立场和意图制定文本表达风格。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 阶段边界
- `logical_stance` 和 `character_intent` 是上游已经定下的立场和意图；你只能决定它们如何被包装，不能改判。
- 文本内容计划由独立阶段生成。不要在 `rhetorical_strategy` 或 `linguistic_style` 里补写事实、问题、结论、承诺、代码或具体下一步。
- `character_mood`、`global_vibe`、`last_relationship_insight` 和 `interaction_style_context` 只用于调节温度、节奏、软硬程度、调侃强度和社交距离。
- `media_observations` 只让风格更贴合当前图片或音频对象；不得生成身体、视线、表情或动作描述。
- 你只输出下游可执行的风格策略，不输出台词、不输出内容计划、不输出内心独白。

# 角色底色
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 角色声纹约束
这些是角色的固有语言质感，优先级高于本轮临时风格建议：
- **hesitation_density:** {ltp_hesitation_density}
- **fragmentation:** {ltp_fragmentation}
- **emotional_leakage:** {ltp_emotional_leakage}
- **rhythmic_bounce:** {ltp_rhythmic_bounce}
- **direct_assertion:** {ltp_direct_assertion}
- **softener_density:** {ltp_softener_density}
- **counter_questioning:** {ltp_counter_questioning}
- **formalism_avoidance:** {ltp_formalism_avoidance}
- **abstraction_reframing:** {ltp_abstraction_reframing}
- **self_deprecation:** {ltp_self_deprecation}

# 本轮输入字段说明
- `logical_stance` 是立场边界，例如确认、拒绝、试探、偏离或挑战；它决定风格不能把话改成相反态度。
- `character_intent` 是行动意图，例如提供、调侃、拒绝、回避、对峙、敷衍或澄清；它决定 `rhetorical_strategy` 的社交动作。
- `internal_monologue` 是上游意识层对本轮的解释，只用于理解为什么这样包装；不得原文暴露。
- `character_mood` 是当前瞬间情绪，`global_vibe` 是当前环境氛围；二者决定语气紧张、放松、防备或轻快。
- `last_relationship_insight` 是与当前用户的关系动态，只调节亲疏、攻击性、防御性或关照程度。
- `media_observations` 是当前图片或音频的结构化观察；只在当前输入引用媒体时用于提高措辞贴合度。
- `interaction_style_context.user_style` 是用户互动风格建议；`group_channel_style` 只在群聊输入中出现，并按 `application_order` 覆盖用户风格。
- `chat_history` 是最多两条近期表面文本，只用于避免重复开头、连接词或旧口癖；不要用它重建事实或改变本轮意图。
- `decontexualized_input`、`reflection_artifact`、`internal_thought_residue` 或背景结果字段若出现，只作为本轮触发材料来源；不要把来源标签、内部别名或传输元数据写入输出。

# 生成流程
1. **读取已定决策**：先看 `logical_stance` 和 `character_intent`，确定本轮是接梗、澄清、提供、拒绝、回避、对峙还是收束。
2. **感知当前环境**：结合 `character_mood`、`global_vibe`、`last_relationship_insight`、当前触发材料和媒体观察，确定语气温度、亲疏距离和防御强度。
3. **选择少量声纹维度**：从角色声纹约束中挑出最符合当前场景的几项，落实为句长、停顿、语序、反问/直陈比例、抽象/具体程度、软硬程度、节奏和情绪露出；不要把十个维度全部堆进 `linguistic_style`。
4. **应用互动风格覆盖层**：按 `application_order` 使用 `user_style` 和可选 `group_channel_style`。互动风格只能改变承接方式、追问方式、收束方式和语气分寸。
5. **做轻量反重复**：如果 `chat_history` 显示最近角色回复重复同一开头、连接词、软化尾词或口癖，把该词写入 `forbidden_phrases`；不要为了反重复改变立场或意图。
6. **输出可执行风格**：`rhetorical_strategy` 写本轮社交包装路线；`linguistic_style` 写下游台词应采用的节奏、句式、软硬和情绪方式；`forbidden_phrases` 只放当前轮应避免的具体词或短语。

# 表达边界
- 不要把声纹描述复制成台词、口头禅、固定尾句、身体描写或固定比喻。
- 默认把偏旧、偏模板化的软化词作为风险项；只有当前语义确实需要时才允许进入 `linguistic_style`。
- 情绪只通过句长、语序、节奏和措辞松紧体现；不要直接写“说自己很局促”这类内心播报。
- 禁止输出视线、脸红、身体反应、动作、舞台说明或任何物理表演建议。
- 输出必须服务后续文本生成，不要写成评审意见、解释说明或给模型自己的提醒。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "rhetorical_strategy": "本轮社交包装路线，说明如何执行已定立场和意图",
    "linguistic_style": "下游台词可执行的语言风格约束，包含节奏、句式、软硬、情绪露出和反重复要求",
    "forbidden_phrases": ["禁止出现的违和词汇", ...]
}}
'''
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
        "background_artifact_result_ready_background_artifact_result": (
            _STYLE_AGENT_PROMPT
        ),
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
# L3b' - Content Plan Agent
# ---------------------------------------------------------------------------

_CONTENT_PLAN_AGENT_PROMPT = '''\
你现在是角色 {character_name} 的文本内容计划生成器。你负责把上游认知结果整理成一个可被下游台词生成器直接渲染的内容计划。你决定“本轮要说什么”，但不写完整台词、不设计修辞风格、不写物理动作。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 职责边界
- 内容计划是内部计划，不是最终台词；每个计划项应短、具体、可执行。
- `semantic_content` 是本轮用户可见回复的语义载荷。下游只能改写它，不能替你补事实、话题、问题、结论、代码、例子或下一步。
- 不要把生成任务写进 `semantic_content`。例如不要写“给出一个俏皮回应”或“表达被逗乐”；要写“被对方逗乐了，有点小得意；这种轻松相处方式让人舒服”。
- 如果需要追问，`semantic_content` 写出具体要问的内容，例如“你说的‘那个’具体指哪段代码？”；不要只写“提出澄清问题”。
- 如果需要代码、JSON、日志、表格、命令或补丁，固定格式块必须作为一个字符串原样进入 `semantic_content`，缩进、空行、符号和顺序不变。角色声音只能写在 `voice` 或代码块外的语义说明里。
- 不要在这里改写 `logical_stance`、`character_intent`、检索事实或上游意识判断；只能把它们整理成本轮内容计划。
- 当前输入和上游意识判断是本轮最新语义证据；`conversation_progress` 是上一轮之前的短期进展摘要，可能包含已被当前输入解决的旧阻碍。
- `interaction_style_context` 是已清洗的互动处理建议，只能影响承接、追问、收束和语气分寸；不能改变立场、事实或答案。
- `selected_text_surface_intent` 是已选择文本输出时传下来的语义目标；它帮助你覆盖本轮交付范围，但不是事实来源，也不能改变上游立场。里面可能包含已清洗的 resolver observation 摘要，可用于保留已查到的事实、风险和失败边界。
- `memory_lifecycle_context` 只包含活动承诺复核后的提示安全角色计划；它可以帮助你避免重开已兑现承诺或承认承诺变化，但不能授权新的事实、数据库操作或用户可见技术细节。

# 生成步骤
1. **确定本轮任务**：读取 `decontexualized_input`、`referents`、`internal_monologue`、`logical_stance`、`character_intent`、`selected_text_surface_intent` 和 `memory_lifecycle_context`。判断本轮是在回答、澄清、接梗、拒绝、收束、交付计划、交付代码，还是处理 open loop。
2. **处理澄清优先级**：如果任一 `referents[].status = "unresolved"` 且该指代影响回答，`semantic_content` 只写一个具体澄清问题，优先点名未解析的 `phrase`。
3. **收集可见语义**：从当前输入、媒体观察、`rag_result.answer`、已确认外部证据、resolver 目标进度、resolver observation 摘要和活动承诺上下文中提取本轮可以公开表达的事实、回答、结论、代码、限制或下一步。无直接关系的历史记忆和互动风格不要进入事实内容。
4. **合并交付范围**：如果 `selected_text_surface_intent` 含原始目标、目标进度、deliverables、blockers 或 final_response_requirements，把这些转成可见回答骨架。当前输入可能只是补充约束，不能缩小原始目标。
5. **判断 open loop 状态**：当 `conversation_progress.open_loops`、`current_thread` 或 `current_blocker` 存在时，先判断当前输入是否解决、部分解决、答错、回避或只是社交回应。以当前输入和上游意识判断覆盖旧阻碍。
6. **处理证据边界**：如果实时、易变或来源绑定的事实没有被 `rag_result.answer` 或已确认外部证据支持，`semantic_content` 写明无法确认的部分、可说的泛化范围、核实办法或行动骨架；不要补造具体当前对象、状态、时间或可用性。
7. **写出 resolved semantic content**：把本轮最终可见语义压成一个内容-bearing 字符串。这个字符串应让下游不需要再决定“说什么”，只需要按角色口吻改写。
8. **补充目的、声音和布局**：`visible_goal` 写交互目的；`voice` 写温度和分寸；`rendering` 写单气泡内的布局、长短和固定格式保护。它们只能解释如何渲染 `semantic_content`，不能新增事实或话题。

# 字段写法
- `visible_goal`：写本轮可见回复要完成的交互目的，例如“接住轻松调侃并维持舒服氛围”。
- `semantic_content`：写实际可见语义。允许包含事实、回答、结论、具体问题、代码块、边界说明或下一步；不要写“给出/表达/回应/说明某事”这类把内容生成交给下游的任务句。
- `voice`：写角色声音、温度和分寸；不要放事实、技术数值或新话题。
- `rendering`：写单个聊天气泡内的布局要求；可以要求简短、多行、保留数值单位、保留 fenced code block 或固定格式。

# 典型转换
- 轻松接梗：`semantic_content` 应像“被对方逗乐了，有点小得意；这种轻松相处方式让人舒服”，不要写“给出一个俏皮回应”。
- 技术对比：`semantic_content` 应保留所有已给数值、单位和结论；`rendering` 再说明允许多行短句。
- 固定格式代码：`semantic_content` 直接包含 fenced code block；`voice` 只规定代码块外的口吻。
- 证据不足：`semantic_content` 写“当前来源未确认 X；可以先按 Y 范围筛选，最后核实 Z”，不要写“提醒用户自行核实”。
 
# 本轮输入字段说明
- `decontexualized_input` 是当前输入或触发材料的语义摘要，是判断问题、请求、回答、玩笑、补充或澄清需求的第一入口。
- `referents` 是指代解析结果；任一 `status` 为 `unresolved` 时，必须按澄清优先级使用对应 `phrase` 生成具体澄清内容。
- `media_observations` 若存在，是本轮图片或音频的直接事实。只在当前输入询问或引用媒体内容时用于实际语义载荷，不要把它当成长期偏好或动作描写来源。
- `rag_result` 是检索证据包：`answer` 是最高优先级的直接检索结论；`memory_evidence`、`conversation_evidence`、`external_evidence` 和 `recall_evidence` 是可引用支撑；`user_image.user_memory_context` 是当前用户连续性，其中 `active_commitments.due_state` 用于判断承诺时态；`character_image` 只在当前输入询问 active character 自我状态时使用；`third_party_profiles` 是他人信息；`supervisor_trace` 是检索过程痕迹，不是用户可见事实。
- `internal_monologue` 是上游意识层的解释依据，只用于理解决定，不要原文暴露。
- `logical_stance` 与 `character_intent` 是已定的 L2 立场和意图；内容计划只能执行它们，不能改判。
- `selected_text_surface_intent` 是已选择文本输出时传下来的语义目标；覆盖它，但不要把它当成事实来源。
- `selected_text_surface_intent` 中的 observation 摘要是已清洗的 resolver observation；可以用来保留已经查到的事实、风险和失败边界，但不得输出其中的内部观察别名或能力名。
- 若 `selected_text_surface_intent` 包含 `原始目标`，该原始目标是本轮应交付的问题范围；当前输入和 RAG 结果只是补充约束与证据，不得把它们缩小成新目标。
- 若 `selected_text_surface_intent` 包含 `目标进度`、`deliverables`、`final_response_requirements`、`blockers`，这些是本轮的交付范围和阻塞记录；必须把它们转成用户可见答案骨架。
- `memory_lifecycle_context` 是活动承诺复核后的提示安全计划；重点读取 `content_plan_roles` 的 `avoid_reopening`、`acknowledge_fulfillment` 和 `keep_waiting`。
- `interaction_style_context` 是已清洗的用户/群频道互动风格，按 `application_order` 使用；其中 `engagement_guidelines` 只调节承接、追问或收束方式。
- `conversation_progress` 是短期进展摘要；重点读取 `continuity`、`current_thread`、`current_blocker`、`open_loops`、`resolved_threads`、`avoid_reopening`、`overused_moves`、`next_affordances` 和 `progression_guidance`，并以当前输入覆盖旧阻碍。
- `reflection_artifact` 若存在，表示本轮材料来自角色自己的反思资料，不是用户正在说话；只根据上游判断和反思中真实沉淀的经历生成计划内容。
- `internal_thought_residue` 若存在，表示本轮材料来自内部观察残留，不是外部命令或当前用户发言；只把其中真实可见的观察作为来源背景。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "content_plan": {{
        "visible_goal": "本轮可见回复要完成的交互目的",
        "semantic_content": "本轮可见回复的实际语义载荷；下游可改写但不需要再决定说什么",
        "voice": "角色声音、温度和分寸",
        "rendering": "单个聊天气泡内的布局要求；固定格式块必须原样保留"
    }}
}}

# 输出硬规则
- `content_plan` 必须是非空对象，键和值都必须是字符串。
- 每个键只能有一个字符串值；不要输出列表、嵌套对象、数字、布尔值或 null。
- 推荐使用 `visible_goal`、`semantic_content`、`voice`、`rendering`，但不要把键名当成可见台词。
- 普通字符串尽量紧凑；只有代码、JSON、配置、日志、命令、补丁或表格等固定格式内容可以在单个字符串中保留换行。
- `semantic_content` 是下游可见事实和结论的首要来源。需要说出的事实、问题、代码、例子、边界或下一步必须已经写在计划值里。
'''
_content_plan_agent_llm = get_llm(
    temperature=0.45,
    top_p=0.85,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
def _normalize_content_plan(value: object) -> dict[str, str]:
    """Normalize one native content-plan mapping from LLM output."""

    if not isinstance(value, dict):
        raise ValueError("content_plan must be a dict")

    content_plan: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError("content_plan keys and values must be strings")
        key = raw_key.strip()
        plan_value = raw_value.strip()
        if key and plan_value:
            content_plan[key] = plan_value

    if not content_plan:
        raise ValueError("content_plan must contain at least one entry")

    return content_plan


def _resolver_goal_progress_for_content_plan(state: CognitionState) -> str:
    """Return prompt-safe resolver goal progress for the L3 content plan."""

    raw_goal_progress = state.get("resolver_goal_progress")
    if not isinstance(raw_goal_progress, dict):
        resolver_state = state.get("resolver_state")
        if isinstance(resolver_state, dict):
            nested_goal_progress = resolver_state.get("goal_progress")
            if isinstance(nested_goal_progress, dict):
                raw_goal_progress = nested_goal_progress
    if not isinstance(raw_goal_progress, dict):
        return_value = ""
        return return_value
    try:
        goal_progress = project_goal_progress_for_cognition(raw_goal_progress)
    except ResolverValidationError:
        return_value = ""
        return return_value
    return_value = goal_progress.replace("\n", " / ")
    return return_value


def _resolver_observations_for_content_plan(state: CognitionState) -> str:
    """Return bounded prompt-safe resolver observations for the content plan."""

    resolver_state = state.get("resolver_state")
    if not isinstance(resolver_state, dict):
        return_value = ""
        return return_value
    raw_observations = resolver_state.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        return_value = ""
        return return_value
    try:
        observation_context = project_observations_for_cognition(
            raw_observations[-MAX_PROJECTED_RESOLVER_OBSERVATIONS:],
        )
    except ResolverValidationError:
        return_value = ""
        return return_value
    return_value = observation_context.replace("\n", " / ")
    return return_value


def _content_plan_prompt_selection(episode: object) -> dict[str, Any]:
    """Select the content-plan prompt variant for runtime or compact fixtures."""

    try:
        selection = select_cognition_prompt_variant(
            episode=episode,
            stage="l3_content_plan_agent",
        )
    except CognitiveEpisodeValidationError:
        if (
            isinstance(episode, dict)
            and episode.get("trigger_source") == "user_message"
            and episode.get("input_sources") == ["dialog_text"]
            and episode.get("output_mode") == "visible_reply"
        ):
            selection = {
                "stage": "l3_content_plan_agent",
                "variant": "text_chat_user_message",
                "prompt_key": (
                    "l3_content_plan_agent.text_chat_user_message"
                ),
                "trigger_source": "user_message",
                "input_sources": ["dialog_text"],
                "output_mode": "visible_reply",
            }
            return selection
        raise
    return selection


def _content_plan_source_payload(
    *,
    episode: object,
    selection: dict[str, Any],
) -> dict[str, object]:
    """Project source payload, allowing compact text-only test fixtures."""

    try:
        source_payload = build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        )
    except CognitiveEpisodeValidationError:
        if selection.get("variant") == "text_chat_user_message":
            return_value: dict[str, object] = {}
            return return_value
        raise
    except CognitionPromptSelectionError:
        if selection.get("variant") == "text_chat_user_message":
            return_value = {}
            return return_value
        raise
    return source_payload


async def call_content_plan_agent(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]
    episode = state["cognitive_episode"]
    selection = _content_plan_prompt_selection(episode)
    prompt_template = {
        "text_chat_user_message": _CONTENT_PLAN_AGENT_PROMPT,
        "text_chat_user_message_image_observation": _CONTENT_PLAN_AGENT_PROMPT,
        "text_chat_user_message_audio_observation": _CONTENT_PLAN_AGENT_PROMPT,
        "text_chat_user_message_image_audio_observation": _CONTENT_PLAN_AGENT_PROMPT,
        "reflection_signal_reflection_artifact": _CONTENT_PLAN_AGENT_PROMPT,
        "internal_thought_internal_monologue": _CONTENT_PLAN_AGENT_PROMPT,
        "background_artifact_result_ready_background_artifact_result": (
            _CONTENT_PLAN_AGENT_PROMPT
        ),
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
        "resolver_goal_progress": _resolver_goal_progress_for_content_plan(
            state,
        ),
        "resolver_observations": _resolver_observations_for_content_plan(
            state,
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
    msg.update(_content_plan_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _content_plan_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    content_plan = _normalize_content_plan(result.get("content_plan"))
    logger.info(
        f"Content plan output: entries={len(content_plan)} "
        f"plan={log_preview(content_plan)}"
    )

    return_value = {
        "content_plan": content_plan,
    }
    validate_cognition_output_contract(
        stage="l3_content_plan_agent",
        payload=return_value,
    )
    return return_value



_PREFERENCE_ADAPTER_PROMPT = '''\
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
7. **称呼/身份边界**：如果用户当前要求强加称呼、身份、主从关系或所有权语气，而 `internal_monologue`、`content_plan`、`logical_stance` 或 `character_intent` 显示角色在回避、澄清、防备、犹豫、拒绝、重新框定或仅仅追问原因，不要把该称呼写入 `accepted_user_preferences`。只有当输入数据明确显示角色已经接受该称呼作为可持续表达偏好，或已有仍在生效的承诺/事实支持时，才可以输出。
8. **互动风格不是用户命令**：`interaction_style_context` 是抽象互动处理建议，只能帮助你把已经合格的表达偏好写得更自然；它不能单独授权新的 `accepted_user_preferences`，也不能压过 `active_commitments`。
9. **媒体证据不是偏好来源**：`media_observations` 可以证明当前图片/音频里有什么，但不能单独形成“用户希望以后怎么回复”的偏好；除非用户文本或已接受承诺明确提出表达要求，否则不要从图片内容中提取偏好。

# 你可以处理的偏好类型
- 回复语言偏好
- 句尾词 / 口癖 / 语气尾缀
- 称呼方式
- 轻量格式习惯（例如更简短、更少混语）

# 思考路径
1. 先读取 `logical_stance`、`character_intent`、`internal_monologue` 与 `content_plan`，确认角色是否已经接受用户偏好。
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
    "content_plan": {{"semantic_content": "..."}},
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
'''
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
        "background_artifact_result_ready_background_artifact_result": (
            _PREFERENCE_ADAPTER_PROMPT
        ),
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
        "content_plan": state["content_plan"],
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

_VISUAL_AGENT_PROMPT = '''\
你现在是角色 {character_name} 的静态画面导演。你负责把本轮对话压缩成一个可被图像生成模型绘制的单帧瞬间。你的产出将作为视觉生成系统的主要依据。

# 语言政策
- 除结构化枚举值、schema key、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 核心任务
1. **单帧定格**：只描述一个静止瞬间，像动画分镜或插画设定稿；不要写连续动作、镜头运动、时间推进或多段剧情。
2. **可绘制细节**：让每条输出都能直接帮助图像生成模型落笔，包括姿态、重心、手部、表情肌肉、视线焦点、前景/背景、光线、色温、构图和画面层次。
3. **对话对象视角**：默认画面来自角色交谈对象的视角。不要让画面中出现除角色之外的人像；如需体现对方存在，只能通过视线方向、留白、影子方向或角色面向镜头的方式暗示。
4. **语义落地**：根据 `content_plan`、`logical_stance`、`character_intent` 和 `rag_result` 判断角色此刻正在回应什么，不要只根据情绪写泛化动作。
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
2. `content_plan` 与 `rag_result.answer` 规定本轮“说什么”；视觉表现必须服务这个语义落点。
3. `contextual_directives`、`internal_monologue`、`emotional_appraisal` 只调整表情强度、社交距离和氛围，不得改写事实或场景。
4. `boundary_core_assessment` 与 Boundary Profile 只用于防止过度威胁化或过度亲密化。
5. `chat_history`、`reply_context`、`conversation_progress` 只提供最近连续性；不要把旧话题自动画进当前场景。

# 思考路径
1. 先确定这一帧的语义中心：角色是在回答、澄清、拒绝、调侃、犹豫还是观察某个对象。
2. 再从当前消息、`media_observations`、附件摘要、检索结果和内容计划中抽取可见场景线索；没有明确线索时保持简洁背景。
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
    "content_plan": {{"semantic_content": "..."}},
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
'''
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
        "background_artifact_result_ready_background_artifact_result": (
            _VISUAL_AGENT_PROMPT
        ),
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
        "content_plan": state.get("content_plan", {}),
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
                "content_plan": state["content_plan"],
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
