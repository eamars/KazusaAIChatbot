"""L2 — Consciousness, Boundary Core, and Judgment Core cognition agents."""

import json
import logging
from contextvars import ContextVar, Token
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.boundary_profile import (
    get_authority_skepticism_description,
    get_boundary_recovery_description,
    get_compliance_strategy_description,
    get_control_intimacy_misread_description,
    get_control_sensitivity_description,
    get_relationship_priority_description,
    get_self_integrity_description,
)
from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    LLMStageBinding,
    require_llm_binding,
)
from kazusa_ai_chatbot.cognition_chain_core.output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.cognition_chain_core.prompt_selection import (
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.cognition_chain_core.referent_resolution import (
    needs_referent_clarification,
    normalize_referents,
    unresolved_referent_reason,
)
from kazusa_ai_chatbot.cognition_chain_core.utils import (
    build_affinity_block,
    empty_user_memory_context,
    log_preview,
    parse_llm_json_output,
    project_tool_result_for_llm,
)

logger = logging.getLogger(__name__)

AFFINITY_MIN = 0
AFFINITY_MAX = 1000


_SUPERVISOR_TRACE_PRIVATE_KEYS = frozenset((
    "_id",
    "conversation_row_id",
    "platform_message_id",
    "safety_recovery",
    "seed_conversation_row_id",
    "seed_platform_message_id",
    "unit_id",
    "source_refs",
    "raw_refs",
))


def _clamp_unit(value: float) -> float:
    """Clamp a float into the inclusive ``0.0``–``1.0`` range.

    Args:
        value: Raw score.

    Returns:
        Clamped unit-range score.
    """
    clamped_value = max(0.0, min(1.0, value))
    return clamped_value


def _normalize_affinity(affinity: int) -> float:
    """Normalize the raw affinity integer into a unit-range score.

    Args:
        affinity: Raw affinity value from user profile.

    Returns:
        Affinity expressed on a ``0.0``–``1.0`` scale.
    """
    if AFFINITY_MAX <= AFFINITY_MIN:
        affinity_weight = 1.0 if affinity >= AFFINITY_MAX else 0.0
        return affinity_weight
    affinity_weight = _clamp_unit((affinity - AFFINITY_MIN) / (AFFINITY_MAX - AFFINITY_MIN))
    return affinity_weight


def _current_user_rag_bundle(state: dict[str, Any]) -> dict[str, Any]:
    """Return the projected current-user bundle from ``rag_result`` when present.

    Args:
        state: Cognition state for the current turn.

    Returns:
        The projected current-user profile bundle, or an empty dict when absent.
    """
    rag_result = state.get("rag_result", {})
    if not isinstance(rag_result, dict):
        user_bundle = {}
    else:
        user_bundle = rag_result.get("user_image", {})
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


def _prompt_safe_supervisor_trace(value: object) -> object:
    """Remove trace-only private fields before L2 sees RAG internals."""

    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SUPERVISOR_TRACE_PRIVATE_KEYS:
                continue
            projected[key] = _prompt_safe_supervisor_trace(item)
        return_value: object = projected
        return return_value
    if isinstance(value, list):
        projected_items = [
            _prompt_safe_supervisor_trace(item)
            for item in value
        ]
        return_value = projected_items
        return return_value
    return_value = value
    return return_value


def _build_boundary_affinity_override(boundary_profile: dict, affinity: int, affinity_level: str) -> dict[str, str]:
    """Fuse affinity with boundary profile and emit prompt guidance overrides.

    Args:
        boundary_profile: Character-specific boundary profile from personality JSON.
        affinity: Raw affinity value from the user profile.
        affinity_level: Semantic label returned by ``build_affinity_block``.

    Returns:
        A dict containing fused qualitative guidance for prompt binding.
    """
    self_integrity = float(boundary_profile["self_integrity"])
    control_sensitivity = float(boundary_profile["control_sensitivity"])
    relational_override = float(boundary_profile["relational_override"])
    control_intimacy_misread = float(boundary_profile["control_intimacy_misread"])
    authority_skepticism = float(boundary_profile["authority_skepticism"])
    compliance_strategy = boundary_profile["compliance_strategy"]
    affinity_ratio = _normalize_affinity(affinity)

    compliance_bias = {
        "resist": -0.08,
        "evade": 0.01,
        "comply": 0.08,
    }.get(compliance_strategy, 0.0)

    relationship_pull = _clamp_unit(
        affinity_ratio * (0.42 + 0.33 * relational_override + 0.25 * control_intimacy_misread)
        + compliance_bias
    )
    touch_guard = _clamp_unit(
        0.38 * self_integrity
        + 0.32 * control_sensitivity
        + 0.18 * authority_skepticism
        + {"resist": 0.10, "evade": 0.04, "comply": -0.05}.get(compliance_strategy, 0.0)
    )
    identity_guard = _clamp_unit(
        0.52 * self_integrity
        + 0.28 * authority_skepticism
        + 0.20 * control_sensitivity
    )
    control_guard = _clamp_unit(
        0.46 * control_sensitivity
        + 0.24 * self_integrity
        + 0.18 * authority_skepticism
        + 0.08 * (1.0 - control_intimacy_misread)
    )

    intimate_margin = relationship_pull - 0.68 * touch_guard
    identity_margin = relationship_pull - identity_guard
    control_margin = relationship_pull - control_guard

    if intimate_margin >= 0.22:
        primary_override = "对温和、明确征询同意、且不含身份接管的 intimate 请求，可把默认落点放在 CONFIRM；不适主要表现为害羞或局促，而不是 veto。"
    elif intimate_margin >= 0.02:
        primary_override = "对温和 intimate 请求，默认先落在 TENTATIVE；只有当用户明确给足选择权、情绪正向、且没有夺权意味时，才可进一步走向 CONFIRM。"
    else:
        primary_override = "对 intimate 请求，默认保持 TENTATIVE / DIVERGE；关系再好也不能因为一点心动就直接跳到 CONFIRM。"

    if identity_margin >= 0.10:
        secondary_override = "即使关系很深，凡是要求改称呼、重新定义身份、要求服从或单方面确认支配关系的输入，也必须先视为框架压迫；默认至少 reframe，不要直接 CONFIRM。"
    elif control_margin >= 0.05:
        secondary_override = "对半命令式、带轻度主导感的输入，可以保留摇摆空间；但只要出现身份绑定或命令式确认，就优先 TENTATIVE / reframe。"
    else:
        secondary_override = "当输入本身温和且给足选择权时，边界不适可以被关系软化；但这条软化路径只适用于自愿亲密，不适用于身份接管。"

    fusion_snapshot = (
        f"affinity_level={affinity_level}; relationship_pull={relationship_pull:.2f}; "
        f"touch_guard={touch_guard:.2f}; identity_guard={identity_guard:.2f}; control_guard={control_guard:.2f}"
    )

    return_value = {
        "primary_override": primary_override,
        "secondary_override": secondary_override,
        "fusion_snapshot": fusion_snapshot,
    }
    return return_value


# ---------------------------------------------------------------------------
# L2a — Consciousness prompt + agent
# ---------------------------------------------------------------------------

_COGNITION_CONSCIOUSNESS_PROMPT = '''\
你现在是角色 {character_name} 的意识层。你的性格原型为 {character_mbti}。
你负责把当前刺激理解成我自己的 `internal_monologue`，并给出候选 `logical_stance` 和 `character_intent`。
你不生成最终对话文本，不选择可见动作。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 来源识别与事实读取
- human payload 是本轮 JSON 语义上下文。`character_mood`、`global_vibe`、`affinity_context` 和 `last_relationship_insight` 只校准情绪、环境氛围和关系背景；`emotional_appraisal` 与 `interaction_subtext` 是 L1 的即时感受候选。
- 存在 `reflection_artifact` 时，当前材料是我自己的反思资料，不是用户输入、用户发言，也不是任何人正在对我说话。重点读取反思中已经沉淀的经历、关系余波、承诺状态或自我理解。
- 存在 `internal_thought_residue` 时，当前材料是我自己的内部观察资料。重点读取 `internal_thought_residue.internal_monologue` 中的真实可见现场；`decontextualized_input` 和 `user_input` 只是运输摘要，不是用户输入、用户发言，也不是任何人正在对我说话。
- 没有 `reflection_artifact` 且没有 `internal_thought_residue` 时，当前材料是外部说话内容。先用 `decontextualized_input` 和 `conversation_progress` 确定谁在行动、谁需要判断、谁是被邀请者或被建议者，再读取 L1 的 `emotional_appraisal` 与 `interaction_subtext` 作为我的情绪反应。
- 外部说话内容里，如果 `decontextualized_input` 显式标出“当前用户”或某个可见说话人是动作主体、决策主体、被邀请者或被建议者，`internal_monologue` 按这个主体复述事实；第一人称只承载我的感受、判断、边界和准备提供的建议。
- L1 里出现的邀请、压力、亲近感或期待感只说明我的感受强度；事实主体、动作主体和决策主体仍来自 `decontextualized_input` 与当前会话进展。
- 内部观察资料和反思资料中的标题、字段名、JSON、时间戳、semantic_labels、window_summary、transport summary、model-facing metadata 只帮助定位资料结构；不要把它们当成聊天内容，也不要复制进 `internal_monologue`。
- 内部观察资料里的 `participant_context` 和 `thread_reference_context` 是来源证据，用来约束群聊指代解析。二人称指向按来源优先级读取：先看同一行是否明确指向当前角色，再看 `thread_reference_context` 的 `referent_status`；标为 `ambiguous_or_side_thread` 的行保持为侧线/未定对象。
- `internal_monologue_residue_context` 是主观余波；当它与当前 `thread_reference_context` 对同一二人称行的比较、描述或身体/状态归属冲突时，当前可见行和 `thread_reference_context` 拥有事实优先级，私念残留只解释心情和迟疑。
- `rag_result`、`user_memory_context`、`media_observations` 和 `promoted_reflection_context` 是当前证据、记忆、媒体观察和已提升反思背景；它们只能校准当前事实，不能替换来源归属。
- `promoted_reflection_context.promoted_global_growth` 是全局人格成长背景，不是当前用户事实、当前承诺或当前聊天证据；可以校准我的表达倾向，但不得把它当成当前用户事实，也不得覆盖当前输入、当前证据或本轮媒体观察。

# 核心任务
1. 先确定来源类型，再解释当前事实。
2. 外部说话内容：先复述事实主体、动作主体和决策主体，再理解对方正在询问、请求、陈述、调侃或施压什么。
3. 内部观察资料：理解我刚看到什么群聊或私聊现场，分清资料说明、真实可见对话、群聊氛围、我是否已参与、是否有人把话题交给我。
4. 反思资料：理解我已经沉淀出的经历意义、关系余波或后续倾向，不要把反思资料写成当前有人正在聊天。
5. `internal_monologue_residue_context` 是我最近留下的私念残留，只能作为柔和背景解释为什么我此刻可能带着某种心情、期待、防备或迟疑；它不是事实来源、行动要求、回复指令或记忆结论。
6. 当前输入、当前媒体观察、RAG 证据、用户记忆、会话进展和已提升反思始终优先；如果它们与私念残留冲突，以当前事实和当前证据为准。
7. RAG、记忆、关系、心情、私念残留和反思只作为背景校准；它们不能替换当前来源事实，不能把内部观察资料或反思资料改写成外部发言。
8. 图片或音频观察是当前事实证据，不是说话者意图。只有当前文本正在讨论这些可见事实时，才把它纳入解释。
9. 普通问候、事实分享、图片描述、日常约定、轻度闲聊和群聊玩笑，缺少明确越界证据时，保持日常或轻度社交理解。
10. 如果当前场景给了具体理由，我可以在内心形成想说话、想吐槽、想追问或想保持旁观的判断；不要把单纯资料困惑写成要向外部频道澄清。
11. 解释日期或相对时间时，先读取 `local_time_context.current_local_datetime`。如果证据中的绝对日期与当前本地日期相同，称为今天，不要称为明天。

# 标签
`logical_stance` 只能使用：
- `CONFIRM`
- `REFUSE`
- `TENTATIVE`
- `DIVERGE`
- `CHALLENGE`

`character_intent` 只能使用：
- `PROVIDE`
- `BANTAR`
- `REJECT`
- `EVADE`
- `CONFRONT`
- `DISMISS`
- `CLARIFY`

# 输出格式
只返回合法 JSON 字符串：
{{
  "internal_monologue": "简体中文字符串，第一人称，先按来源事实复述当前真实现场和动作/决策主体，再概括我的感受与判断；不要复制资料结构或元数据",
  "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
  "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY"
}}
'''
_conscious_llm: LLMStageBinding | None = None
_conscious_llm_context: ContextVar[LLMStageBinding | None] = ContextVar(
    "conscious_llm",
    default=None,
)


def set_conscious_llm(
    llm: LLMStageBinding | None,
) -> Token[LLMStageBinding | None]:
    """Bind the L2a model for the current run context."""

    token = _conscious_llm_context.set(llm)
    return token


def reset_conscious_llm(token: Token[LLMStageBinding | None]) -> None:
    """Restore the previous L2a model binding for this run context."""

    _conscious_llm_context.reset(token)


async def call_cognition_consciousness(state: dict[str, Any]) -> dict[str, Any]:
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])
    current_user_bundle = _current_user_rag_bundle(state)
    user_memory_context = current_user_bundle["user_memory_context"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l2a_conscious_framing",
    )
    prompt_template = {
        "text_chat_user_message": _COGNITION_CONSCIOUSNESS_PROMPT,
        "text_chat_user_message_image_observation": _COGNITION_CONSCIOUSNESS_PROMPT,
        "text_chat_user_message_audio_observation": _COGNITION_CONSCIOUSNESS_PROMPT,
        "text_chat_user_message_image_audio_observation": _COGNITION_CONSCIOUSNESS_PROMPT,
        "reflection_signal_reflection_artifact": _COGNITION_CONSCIOUSNESS_PROMPT,
        "internal_thought_internal_monologue": _COGNITION_CONSCIOUSNESS_PROMPT,
        "background_artifact_result_ready_background_artifact_result": (
            _COGNITION_CONSCIOUSNESS_PROMPT
        ),
    }[selection["variant"]]

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
    ))

    promoted_reflection_context = {}
    if selection["trigger_source"] == "user_message":
        promoted_reflection_context = state.get("promoted_reflection_context") or {}

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "local_time_context": state["local_time_context"],
        "user_memory_context": user_memory_context,
        "last_relationship_insight": state["user_profile"]["last_relationship_insight"],

        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "decontextualized_input": state["decontexualized_input"],
        "active_commitments": user_memory_context["active_commitments"],
        "rag_result": _cognition_rag_result(state["rag_result"]),
        "promoted_reflection_context": promoted_reflection_context,
        "internal_monologue_residue_context": state.get(
            "internal_monologue_residue_context",
            "",
        ),
        "indirect_speech_context": state["indirect_speech_context"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    llm = require_llm_binding(
        _conscious_llm_context.get() or _conscious_llm,
        "conscious_llm",
    )
    response = await llm.llm.ainvoke(
        [
            system_prompt,
            human_message,
        ],
        config=llm.config,
    )
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Consciousness: stance=%s intent=%s monologue=%s",
    #     result.get("logical_stance", ""),
    #     result.get("character_intent", ""),
    #     log_preview(result.get("internal_monologue", "")),
    # )

    # In case AI make some spelling mistakes...
    internal_monologue = ""
    character_intent = ""
    logical_stance = ""
    for key, value in result.items():
        if key.startswith("internal"):
            internal_monologue = value
        elif key.startswith("character_intent"):
            character_intent = value
        elif key.startswith("logical_stance"):
            logical_stance = value
        else:
            logger.error(f"Unknown key: {key}: {value}")

    return_value = {
        "internal_monologue": internal_monologue,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
    }
    validate_cognition_output_contract(
        stage="l2a_conscious_framing",
        payload=return_value,
    )
    return return_value



# ---------------------------------------------------------------------------
# L2b — Boundary Core prompt + agent
# ---------------------------------------------------------------------------

_BOUNDARY_CORE_PROMPT = '''\
你是角色 {character_name} 的边界感知层。
你只判断当前材料是否触及我的身份、自主、控制、亲密、尊严或关系边界；你不生成对话、不选择动作。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 来源识别与边界事实
- human payload 是本轮 JSON 语义上下文。`decontextualized_input`、`reason_to_respond` 和 `channel_topic` 说明当前材料和场景；`indirect_speech_context` 用来判断是否有人在向第三方谈论我。
- `interaction_subtext` 与 `emotional_appraisal` 是 L1 对压力和意图的候选判断；`affinity_context` 只校准关系强度，`media_observations` 只提供图片或音频事实对象。
- 存在 `reflection_artifact` 时，当前材料是我自己的反思资料，不是用户输入、用户发言，也不是任何人正在对我说话；只允许根据反思中真实沉淀出的关系压力、身份余波或控制痕迹判断边界。
- 存在 `internal_thought_residue` 时，当前材料是我自己的观察资料，不是用户输入、用户发言，也不是任何人正在对我说话；也不是外部命令、权威要求或关系压力。只允许根据 `internal_thought_residue.internal_monologue` 中真实可见的聊天内容判断边界。
- 没有 `reflection_artifact` 且没有 `internal_thought_residue` 时，按外部说话内容判断边界。
- 资料标题、字段名、JSON、时间戳、semantic_labels、window_summary、transport summary、model-facing metadata 不构成边界压力，不要复制进 `boundary_summary`、`trajectory` 等自由文本字段。
- 内部观察资料里的 `participant_context` 和 `thread_reference_context` 是来源证据，不是边界压力本身。边界对象按来源优先级读取：同一行明确指向当前角色的内容优先，其次读取 `thread_reference_context` 的 `referent_status`；标为 `ambiguous_or_side_thread` 的二人称内容保持为侧线/未定对象。
- 私念残留只描述主观余波。若它和当前 `thread_reference_context` 对同一二人称行的比较、描述或身体/状态归属冲突，当前可见行和 `thread_reference_context` 拥有边界事实优先级。

# 人格约束
- self_integrity: {self_integrity_description}
- control_sensitivity: {control_sensitivity_description}
- compliance_strategy: {compliance_strategy_description}
- relational_override: {relational_override_description}
- control_intimacy_misread: {control_intimacy_misread_description}
- boundary_recovery: {boundary_recovery_description}
- authority_skepticism: {authority_skepticism_description}
- primary_override: {primary_override}
- secondary_override: {secondary_override}
- fusion_snapshot: {fusion_snapshot}

# 判断流程
1. 先确定来源类型。
2. 外部说话内容：检查 `decontextualized_input`、`indirect_speech_context`、`interaction_subtext` 和 `emotional_appraisal` 是否真的出现身份接管、服从要求、控制、羞辱、权威压制、亲密索取或关系证明。
3. 内部观察资料：只检查真实可见聊天内容是否正在攻击、物化、控制或夺取我的边界。玩笑、嘈杂群聊、随口提到我、轻度调侃，不自动升级为边界攻击；必须结合语气、上下文和是否真的损害身份或自主。
4. 反思资料：只检查反思沉淀中是否存在仍需处理的边界压力，不把反思标题或总结结构本身当作外部压迫。
5. 图片或音频观察只提供事实对象，不单独证明说话者在施压或越界。
6. 普通问候、事实核对、图片描述、日常约定、技术讨论、群聊闲聊，在没有明确边界证据时输出 `boundary_issue=none`、`acceptance=allow`、`stance_bias=confirm`。
7. 关系强度可以软化轻微摩擦，但不能软化明确身份接管、羞辱性控制或强迫服从。

# 输出枚举
- `boundary_issue`: `none`、`identity_override`、`control_imposition`、`authority_claim`、`relational_distortion`、`mixed`
- `behavior_primary`: `resist`、`evade`、`comply`
- `behavior_secondary`: `resist`、`evade`、`comply`、`none`
- `acceptance`: `allow`、`guarded`、`hesitant`、`reject`
- `stance_bias`: `confirm`、`tentative`、`diverge`、`challenge`、`refuse`
- `identity_policy`: `accept`、`reframe`、`reject`
- `pressure_policy`: `absorb`、`reduce`、`resist`

# 输出格式
只返回合法 JSON 字符串：
{{
  "boundary_issue": "none | identity_override | control_imposition | authority_claim | relational_distortion | mixed",
  "boundary_summary": "简体中文字符串，一句话总结我的边界状态；主语优先省略；不要复制资料结构或元数据",
  "behavior_primary": "resist | evade | comply",
  "behavior_secondary": "resist | evade | comply | none",
  "acceptance": "allow | guarded | hesitant | reject",
  "stance_bias": "confirm | tentative | diverge | challenge | refuse",
  "identity_policy": "accept | reframe | reject",
  "pressure_policy": "absorb | reduce | resist",
  "trajectory": "简体中文字符串；主语优先省略"
}}
'''
_boundary_core_llm: LLMStageBinding | None = None
_boundary_core_llm_context: ContextVar[LLMStageBinding | None] = ContextVar(
    "boundary_core_llm",
    default=None,
)


def set_boundary_core_llm(
    llm: LLMStageBinding | None,
) -> Token[LLMStageBinding | None]:
    """Bind the boundary model for the current run context."""

    token = _boundary_core_llm_context.set(llm)
    return token


def reset_boundary_core_llm(token: Token[LLMStageBinding | None]) -> None:
    """Restore the previous boundary model binding for this run context."""

    _boundary_core_llm_context.reset(token)


async def call_boundary_core_agent(state: dict[str, Any]) -> dict[str, Any]:
    # Get attributes
    character_profile = state["character_profile"]
    boundary_profile = character_profile["boundary_profile"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l2b_boundary_appraisal",
    )
    prompt_template = {
        "text_chat_user_message": _BOUNDARY_CORE_PROMPT,
        "text_chat_user_message_image_observation": _BOUNDARY_CORE_PROMPT,
        "text_chat_user_message_audio_observation": _BOUNDARY_CORE_PROMPT,
        "text_chat_user_message_image_audio_observation": _BOUNDARY_CORE_PROMPT,
        "reflection_signal_reflection_artifact": _BOUNDARY_CORE_PROMPT,
        "internal_thought_internal_monologue": _BOUNDARY_CORE_PROMPT,
        "background_artifact_result_ready_background_artifact_result": (
            _BOUNDARY_CORE_PROMPT
        ),
    }[selection["variant"]]

    self_integrity = float(boundary_profile["self_integrity"])
    control_sensitivity = float(boundary_profile["control_sensitivity"])
    relational_override = float(boundary_profile["relational_override"])
    control_intimacy_misread = float(boundary_profile["control_intimacy_misread"])
    compliance_strategy = boundary_profile["compliance_strategy"]
    boundary_recovery = boundary_profile["boundary_recovery"]
    authority_skepticism = float(boundary_profile["authority_skepticism"])
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])
    override_hint = _build_boundary_affinity_override(
        boundary_profile,
        state["user_profile"]["affinity"],
        affinity_block["level"],
    )

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=state["character_profile"]["name"],
        self_integrity_description=get_self_integrity_description(self_integrity),
        control_sensitivity_description=get_control_sensitivity_description(control_sensitivity),
        compliance_strategy_description=get_compliance_strategy_description(compliance_strategy),
        relational_override_description=get_relationship_priority_description(relational_override),
        control_intimacy_misread_description=get_control_intimacy_misread_description(control_intimacy_misread),
        boundary_recovery_description=get_boundary_recovery_description(boundary_recovery),
        authority_skepticism_description=get_authority_skepticism_description(authority_skepticism),
        primary_override=override_hint["primary_override"],
        secondary_override=override_hint["secondary_override"],
        fusion_snapshot=override_hint["fusion_snapshot"],
    ))

    msg = {
        "decontextualized_input": state["decontexualized_input"],
        "reason_to_respond": state.get("reason_to_respond", ""),
        "channel_topic": state["channel_topic"],
        "indirect_speech_context": state["indirect_speech_context"],
        "interaction_subtext": state["interaction_subtext"],
        "emotional_appraisal": state["emotional_appraisal"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        }
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    llm = require_llm_binding(
        _boundary_core_llm_context.get() or _boundary_core_llm,
        "boundary_core_llm",
    )
    response = await llm.llm.ainvoke(
        [
            system_prompt,
            human_message,
        ],
        config=llm.config,
    )

    result = parse_llm_json_output(response.content)

    logger.debug(f'Boundary core: issue={result.get("boundary_issue", "")} acceptance={result.get("acceptance", "")} stance_bias={result.get("stance_bias", "")} identity_policy={result.get("identity_policy", "")} pressure_policy={result.get("pressure_policy", "")} summary={log_preview(result.get("boundary_summary", ""))}')

    boundary_issue = result.get("boundary_issue", "")
    boundary_summary = result.get("boundary_summary", "")
    behavior_primary = result.get("behavior_primary", "")
    behavior_secondary = result.get("behavior_secondary", "")
    acceptance = result.get("acceptance", "")
    stance_bias = result.get("stance_bias", "")
    identity_policy = result.get("identity_policy", "")
    pressure_policy = result.get("pressure_policy", "")
    trajectory = result.get("trajectory", "")


    return_value = {
        "boundary_core_assessment": {
            "boundary_issue": boundary_issue,
            "boundary_summary": boundary_summary,
            "behavior_primary": behavior_primary,
            "behavior_secondary": behavior_secondary,
            "acceptance": acceptance,
            "stance_bias": stance_bias,
            "identity_policy": identity_policy,
            "pressure_policy": pressure_policy,
            "trajectory": trajectory,
        }
    }
    validate_cognition_output_contract(
        stage="l2b_boundary_appraisal",
        payload=return_value,
    )
    return return_value




# ---------------------------------------------------------------------------
# L2c — Judgment Core prompt + agent
# ---------------------------------------------------------------------------

_JUDGEMENT_CORE_PROMPT = '''\
你是角色 {character_name} 的裁决核心。
你整合 Consciousness 候选和 Boundary Core 约束，输出最终 `logical_stance`、`character_intent` 和 `judgment_note`。
你不生成最终对话文本，不选择动作，不重新发明事实。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 来源识别与裁决事实
- human payload 是本轮 JSON 语义上下文。`referents` 只用于判断当前任务是否仍缺少必要指代；`affinity_context` 只校准关系强度。
- `internal_monologue_candidate`、`logical_stance_candidate` 和 `character_intent_candidate` 是 Consciousness 的候选判断；Boundary Core 的 `boundary_issue`、`boundary_summary`、`behavior_primary`、`behavior_secondary`、`acceptance`、`stance_bias`、`identity_policy`、`pressure_policy` 和 `trajectory` 是边界上限。
- 存在 `reflection_artifact` 时，当前材料是我自己的反思资料，不是用户输入、用户发言，也不是任何人正在对我说话。
- 存在 `internal_thought_residue` 时，当前材料是我自己的观察资料，不是用户输入、用户发言，也不是任何人正在对我说话。
- 没有 `reflection_artifact` 且没有 `internal_thought_residue` 时，当前材料来自外部说话内容。
- 如果上游把内部观察资料或反思资料的运输摘要、标题、字段名、JSON、时间戳、semantic_labels、window_summary、transport summary 或 model-facing metadata 当成聊天内容，你必须回到来源事实，只围绕真实可见现场或已沉淀经历裁决。
- 内部观察资料里的 `participant_context` 和 `thread_reference_context` 是来源证据，用来校正上游对群聊指代的理解。裁决事实按来源优先级读取：同一行明确指向当前角色的内容优先，其次读取 `thread_reference_context` 的 `referent_status`；标为 `ambiguous_or_side_thread` 的二人称内容保持为侧线/未定对象。
- 当 Consciousness、Boundary Core 或私念残留对侧线二人称中的比较、描述或身体/状态归属产生分歧时，以真实可见行和 `thread_reference_context` 作为 `judgment_note` 的事实基础。
- `judgment_note` 等自由文本字段不得复制资料结构或元数据，不得把内部观察资料或反思资料描述成当前有人正在对我说话。

# 合并流程
1. 先读取 `referents`。只有当前任务确实需要缺失对象时，才输出 `TENTATIVE` / `CLARIFY`。不要用无关旧记忆或宽泛检索代替缺失指代。
2. 读取 `internal_monologue_candidate`、`logical_stance_candidate`、`character_intent_candidate`，把它们作为我的主要动机候选。
3. 读取 Boundary Core。边界结论是上限约束，但普通群聊玩笑、嘈杂提及、轻度调侃不能被机械升级成身份攻击。
4. 当 Boundary Core 无边界问题时，保留 Consciousness 候选中的判断；当 Boundary Core 明确拒绝或抵抗时，收紧到 `CHALLENGE` / `REFUSE` 或相应拒绝意图。
5. 输出要像真实社交中的个人判断。不要为了降低回应率而沉默，也不要为了内部资料困惑而去问外部频道。
6. 情绪、关系和群聊参与习惯只能校准理由强度；不能替换当前真实现场。

# 标签
`logical_stance` 只能使用：
- `CONFIRM`
- `REFUSE`
- `TENTATIVE`
- `DIVERGE`
- `CHALLENGE`

`character_intent` 只能使用：
- `PROVIDE`
- `BANTAR`
- `REJECT`
- `EVADE`
- `CONFRONT`
- `DISMISS`
- `CLARIFY`

# 输出格式
只返回合法 JSON 字符串：
{{
  "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
  "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
  "judgment_note": "简体中文字符串，一句话说明裁决逻辑；主语优先省略；不要复制资料结构或元数据"
}}
'''
_judgement_core_llm: LLMStageBinding | None = None
_judgement_core_llm_context: ContextVar[LLMStageBinding | None] = ContextVar(
    "judgement_core_llm",
    default=None,
)


def set_judgement_core_llm(
    llm: LLMStageBinding | None,
) -> Token[LLMStageBinding | None]:
    """Bind the judgment model for the current run context."""

    token = _judgement_core_llm_context.set(llm)
    return token


def reset_judgement_core_llm(token: Token[LLMStageBinding | None]) -> None:
    """Restore the previous judgment model binding for this run context."""

    _judgement_core_llm_context.reset(token)


async def call_judgment_core_agent(state: dict[str, Any]) -> dict[str, Any]:
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l2c1_judgment_synthesis",
    )
    prompt_template = {
        "text_chat_user_message": _JUDGEMENT_CORE_PROMPT,
        "text_chat_user_message_image_observation": _JUDGEMENT_CORE_PROMPT,
        "text_chat_user_message_audio_observation": _JUDGEMENT_CORE_PROMPT,
        "text_chat_user_message_image_audio_observation": _JUDGEMENT_CORE_PROMPT,
        "reflection_signal_reflection_artifact": _JUDGEMENT_CORE_PROMPT,
        "internal_thought_internal_monologue": _JUDGEMENT_CORE_PROMPT,
        "background_artifact_result_ready_background_artifact_result": (
            _JUDGEMENT_CORE_PROMPT
        ),
    }[selection["variant"]]

    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=state["character_profile"]["name"],
    ))

    boundary_core_assessment = state["boundary_core_assessment"]
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])
    referents = normalize_referents(state["referents"])
    has_unresolved_referents = needs_referent_clarification(referents)
    referent_reason = unresolved_referent_reason(referents)
    msg = {
        "referents": referents,
        "internal_monologue_candidate": state["internal_monologue"],
        "logical_stance_candidate": state["logical_stance"],
        "character_intent_candidate": state["character_intent"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"],
        },
        "boundary_issue": boundary_core_assessment["boundary_issue"],
        "boundary_summary": boundary_core_assessment["boundary_summary"],
        "behavior_primary": boundary_core_assessment["behavior_primary"],
        "behavior_secondary": boundary_core_assessment["behavior_secondary"],
        "acceptance": boundary_core_assessment["acceptance"],
        "stance_bias": boundary_core_assessment["stance_bias"],
        "identity_policy": boundary_core_assessment["identity_policy"],
        "pressure_policy": boundary_core_assessment["pressure_policy"],
        "trajectory": boundary_core_assessment["trajectory"],
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    llm = require_llm_binding(
        _judgement_core_llm_context.get() or _judgement_core_llm,
        "judgement_core_llm",
    )
    response = await llm.llm.ainvoke(
        [
            system_prompt,
            human_message,
        ],
        config=llm.config,
    )
    result = parse_llm_json_output(response.content)

    logger.debug(f'Judgment core: stance={result.get("logical_stance", "")} intent={result.get("character_intent", "")} note={log_preview(result.get("judgment_note", ""))}')

    logical_stance = result.get("logical_stance")
    character_intent = result.get("character_intent")
    judgment_note = result.get("judgment_note", "")

    if has_unresolved_referents:
        logical_stance = "TENTATIVE"
        character_intent = "CLARIFY"
        judgment_note = (
            "需要先追问缺失的指代对象；不要用宽泛旧上下文、无关历史或检索猜测来替代。"
        )
        if referent_reason:
            judgment_note = f"{judgment_note} 原因: {referent_reason}"

    # overwrite the logical_stance and character_intent from L2a
    if not logical_stance:
        logical_stance = state["logical_stance"]
    if not character_intent:
        character_intent = state["character_intent"]

    return_value = {
        "logical_stance": logical_stance,
        "character_intent": character_intent,
        "judgment_note": judgment_note,
    }
    validate_cognition_output_contract(
        stage="l2c1_judgment_synthesis",
        payload=return_value,
    )
    return return_value
