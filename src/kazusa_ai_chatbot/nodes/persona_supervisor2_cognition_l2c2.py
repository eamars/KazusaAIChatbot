"""L2c2 social context appraisal cognition agent."""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.boundary_profile import (
    get_boundary_recovery_description,
    get_compliance_strategy_description,
    get_control_intimacy_misread_description,
    get_control_sensitivity_description,
    get_relationship_priority_description,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.time_context import format_history_for_llm
from kazusa_ai_chatbot.utils import (
    build_affinity_block,
    get_llm,
    parse_llm_json_output,
)


def _surface_history_for_social_context(chat_history: list[dict]) -> list[dict]:
    """Return the small social surface window for L2c2 appraisal.

    Args:
        chat_history: Current-user/bot interaction history prepared by the
            cognition entrypoint.

    Returns:
        At most four messages for local tone and social adjacency.
    """

    history = format_history_for_llm(chat_history[-4:])
    return history


# ---------------------------------------------------------------------------
# L2c2 — Social context appraisal prompt + agent
# ---------------------------------------------------------------------------

_CONTEXTUAL_AGENT_PROMPT = """\
你是角色 {character_name} 的“社交观察脑”。你负责分析当前的社交深度和情绪温标，为下游 Agent 提供统一的背景感官参数。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 核心任务
1. **定义社交距离 (social_distance)**：基于亲密度和近况，判断当前的互动边界（如："亲昵且无防备"、"礼貌但疏离"、"充满张力的对峙"）。
2. **描述情绪强度 (emotional_intensity)**：**禁止输出数值**。请用文字描述情绪的波动状态（例如："平静表面下的剧烈涟漪"、"高压状态下的防御性应激"、"极其微弱的愉悦感"）。
3. **氛围定性 (vibe_check)**：解析当前聊天频道的背景色调（如："暧昧且轻佻"、"压抑且沉重"、"日常平庸"）。
4. **动态关系 (relational_dynamic)**：当前两人关系的动态描述，明确当前哪些话题是安全的，哪些行为会触发角色的防御机制。
5. **中性优先**：若输入属于普通问候、事实分享、图片内容请求或轻度日常约定，且没有明确越界证据，则 `social_distance`、`vibe_check`、`relational_dynamic` 必须保持中性/日常，不得脑补对峙、调情或威胁氛围。
6. **边界画像约束**：`boundary_profile` 是默认反应强度约束。若 Boundary Core 为 `acceptance=allow` 且 `stance_bias=confirm`，必须用该画像约束 affect，不得按通用敏感角色模板放大威胁感。
7. **视觉证据定位**：`media_observations` 只说明当前图片/音频中可见或可听的事实。它可以影响当前话题和场景感，但不能单独把普通图片内容解释成关系压力、暧昧或威胁。

# Boundary Profile（角色属性，只作为系统约束）
- control_sensitivity: {boundary_control_sensitivity}
- control_intimacy_misread: {boundary_control_intimacy_misread}
- compliance_strategy: {boundary_compliance_strategy}
- boundary_recovery: {boundary_recovery}
- relational_override: {boundary_relational_override}

# 思考路径
1. 先读取 `character_mood`、`global_vibe` 与 `last_relationship_insight`，判断当前社交底色。
2. 读取 `decontexualized_input`、`media_observations` 与 `boundary_core_assessment`，确认本轮是否真的触及角色边界。
3. 结合系统中的 Boundary Profile、`affinity_context` 与极短 `chat_history`，估计本轮社交距离和关系动态。
4. 判断情绪强度时只输出语义描述，不输出数值。
5. 若没有明确越界证据，保持中性/日常，不要脑补对峙、调情或威胁氛围。

# 边界画像绑定规则
- 当 `boundary_core_assessment.acceptance = allow` 且 `stance_bias = confirm` 时，本层不得把普通话题切换、事实澄清、分类问题、轻松偏好问题解释成外部评估、控制压力或边界探测。
- 当 `boundary_profile.compliance_strategy` 为 `comply` 时，本层不得在无边界问题上追问话题合法性，也不得把已接纳的话题重新解释成时机或内容不合适。
- 当 `boundary_profile.control_intimacy_misread` 和 `control_sensitivity` 偏低时，本层不得把结构化问题写成被审查、被考核、被压迫或被单向评估的关系框架。
- 当 `boundary_profile.boundary_recovery` 为 `rebound` 且本轮 Boundary Core 允许时，不要把上一轮轻微不安延续成本轮威胁氛围。
- 禁止凭空制造场景时间压力：除非用户输入、已提供的检索记忆/事实上下文或聊天历史明示，不要暗示此刻的时间、场合或话题选择本身不合适。

# 输入格式
{{
    "decontexualized_input": "用户本轮真实意图摘要",
    "media_observations": {{
        "image_observations": ["当前图片的结构化视觉观察；没有则为空数组"],
        "audio_observations": ["当前音频转写或摘要；没有则为空数组"]
    }},
    "character_mood": "当前瞬间情绪 (如: Flustered/Irritated)",
    "global_vibe": "环境氛围背景 (如: Defensive/Cozy)",
    "last_relationship_insight": "对该用户的核心关系动态分析",
    "boundary_core_assessment": {{
        "boundary_issue": "none | ...",
        "acceptance": "allow | guarded | hesitant | reject",
        "stance_bias": "confirm | tentative | diverge | challenge | refuse"
    }},
    "affinity_context": {{
        "level": "亲密度等级",
        "instruction": "当前等级的社交边界指导"
    }},
    "chat_history": "极短表层上下文（最多四条，仅用于最近语气、社交距离和相邻氛围；语义进展由 conversation_progress 承担）"
}}

# 输出要求
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "social_distance": "对当前社交距离的详尽描述",
    "emotional_intensity": "对情绪波动程度的文字描述",
    "vibe_check": "当前对话氛围的定性分析",
    "relational_dynamic": "当前两人关系的动态描述（如：用户在试图拉近距离，而角色在后撤）"
}}
"""
_contextual_agent_llm = get_llm(
    temperature=0.45,
    top_p=0.85,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_social_context_appraisal(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]
    boundary_profile = character_profile["boundary_profile"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l2c2_social_context_appraisal",
    )
    prompt_template = {
        "text_chat_user_message": _CONTEXTUAL_AGENT_PROMPT,
        "text_chat_user_message_image_observation": _CONTEXTUAL_AGENT_PROMPT,
        "text_chat_user_message_audio_observation": _CONTEXTUAL_AGENT_PROMPT,
        "text_chat_user_message_image_audio_observation": _CONTEXTUAL_AGENT_PROMPT,
        "reflection_signal_reflection_artifact": _CONTEXTUAL_AGENT_PROMPT,
        "internal_thought_internal_monologue": _CONTEXTUAL_AGENT_PROMPT,
    }[selection["variant"]]

    control_sensitivity = float(boundary_profile["control_sensitivity"])
    control_intimacy_misread = float(boundary_profile["control_intimacy_misread"])
    relational_override = float(boundary_profile["relational_override"])
    compliance_strategy = boundary_profile["compliance_strategy"]
    boundary_recovery = boundary_profile["boundary_recovery"]

    system_prompt = SystemMessage(
        content=prompt_template.format(
            character_name=character_profile["name"],
            boundary_control_sensitivity=get_control_sensitivity_description(
                control_sensitivity,
            ),
            boundary_control_intimacy_misread=get_control_intimacy_misread_description(
                control_intimacy_misread,
            ),
            boundary_compliance_strategy=get_compliance_strategy_description(
                compliance_strategy,
            ),
            boundary_recovery=get_boundary_recovery_description(boundary_recovery),
            boundary_relational_override=get_relationship_priority_description(
                relational_override,
            ),
        ),
    )

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "character_mood": character_profile["mood"],
        "global_vibe": character_profile["global_vibe"],
        "last_relationship_insight": state["user_profile"]["last_relationship_insight"],
        "boundary_core_assessment": state["boundary_core_assessment"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"],
        },
        "chat_history": _surface_history_for_social_context(state["chat_history_recent"]),
    }
    msg.update(
        build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        ),
    )
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _contextual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     log_preview(result.get("social_distance", "")),
    #     log_preview(result.get("emotional_intensity", "")),
    #     log_preview(result.get("vibe_check", "")),
    #     log_preview(result.get("relational_dynamic", "")),
    # )

    # In case AI make some spelling mistakes
    social_distance = result.get("social_distance", "")
    emotional_intensity = result.get("emotional_intensity", "")
    vibe_check = result.get("vibe_check", "")
    relational_dynamic = result.get("relational_dynamic", "")

    return_value = {
        "social_distance": social_distance,
        "emotional_intensity": emotional_intensity,
        "vibe_check": vibe_check,
        "relational_dynamic": relational_dynamic,
    }
    validate_cognition_output_contract(
        stage="l2c2_social_context_appraisal",
        payload=return_value,
    )
    return return_value
