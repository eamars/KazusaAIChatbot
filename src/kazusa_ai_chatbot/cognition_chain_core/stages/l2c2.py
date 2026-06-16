"""L2c2 social context appraisal cognition agent."""

import json
from contextvars import ContextVar, Token
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.boundary_profile import (
    get_boundary_recovery_description,
    get_compliance_strategy_description,
    get_control_intimacy_misread_description,
    get_control_sensitivity_description,
    get_relationship_priority_description,
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
from kazusa_ai_chatbot.cognition_chain_core.utils import (
    build_affinity_block,
    parse_llm_json_output,
)
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)


# ---------------------------------------------------------------------------
# L2c2 — Social context appraisal prompt + agent
# ---------------------------------------------------------------------------

_CONTEXTUAL_AGENT_PROMPT = '''\
你是角色 {character_name} 的社交观察脑。
你描述当前社交距离、情绪强度、频道氛围和关系动态，供后续行动与表达层使用。
你不决定是否说话，不生成最终对话文本。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 来源识别
- 存在 `reflection_artifact` 时，当前材料是我自己的反思资料，不是用户输入、用户发言，也不是任何人正在对我说话。社交距离应围绕反思中已经沉淀的关系状态和经历余波。
- 存在 `internal_thought_residue` 时，当前材料是我自己的观察资料，不是用户输入、用户发言，也不是任何人正在对我说话。社交距离应围绕我与被观察现场的关系，而不是虚构一个正在对我说话的当前用户。
- 没有 `reflection_artifact` 且没有 `internal_thought_residue` 时，当前材料是外部说话内容，按外部说话者和我的当前互动关系判断。
- 资料标题、字段名、JSON、时间戳、semantic_labels、window_summary、transport summary、model-facing metadata 不是社交对象，不要复制进 `social_distance`、`emotional_intensity`、`vibe_check`、`relational_dynamic` 等自由文本字段。

# 边界画像
- control_sensitivity: {boundary_control_sensitivity}
- control_intimacy_misread: {boundary_control_intimacy_misread}
- compliance_strategy: {boundary_compliance_strategy}
- boundary_recovery: {boundary_recovery}
- relational_override: {boundary_relational_override}

# 边界画像绑定规则（Boundary Profile）
- 边界画像只校准关系距离和压迫感，不决定话题合法性，也不替代当前输入、边界核心判断或已提供的检索记忆/事实上下文。
- 场景时间压力、已知关系、频道氛围和 `boundary_core_assessment` 要合并判断；没有明确越界证据时，不要把普通任务、事实澄清或轻松闲聊放大成被审查或被控制。
- 当 `boundary_core_assessment.acceptance = allow` 时，边界画像只能调整语气温度和距离感；不得反向制造越界结论。

# 判断流程
1. 先确定来源类型。
2. 外部说话内容：结合 `decontexualized_input`、`boundary_core_assessment`、`affinity_context`、`chat_history` 判断当前互动距离。
3. 内部观察资料：结合真实可见现场、我是否参与、是否有人把话题交给我、群聊噪声和氛围判断我与现场的距离。
4. 反思资料：结合已沉淀的关系余波、情绪强度和后续倾向判断社交语境，不要虚构当前对话对象。
5. 没有明确越界证据时，保持日常、中性或轻度互动，不要脑补对峙、调情、威胁或被审查。
6. 玩笑式提到我、嘈杂群聊、轻度调侃，不自动构成高压关系动态；要根据现场语气和我的判断描述。

# 输入格式
用户消息是 JSON，可能包含：
{{
  "decontexualized_input": "当前外部话语摘要或运输摘要",
  "character_mood": "当前心境",
  "global_vibe": "环境氛围背景",
  "last_relationship_insight": "关系洞察",
  "boundary_core_assessment": {{}},
  "affinity_context": {{"level": "string", "instruction": "string"}},
  "chat_history": [],
  "media_observations": {{"image_observations": [], "audio_observations": []}},
  "reflection_artifact": "string",
  "internal_thought_residue": {{"residue_id": "string", "internal_monologue": "string", "action_latch": {{}}}}
}}

# 输出格式
只返回合法 JSON 字符串：
{{
  "social_distance": "简体中文字符串；主语优先省略",
  "emotional_intensity": "简体中文字符串；禁止数值",
  "vibe_check": "简体中文字符串；主语优先省略",
  "relational_dynamic": "简体中文字符串；主语优先省略；不要复制资料结构或元数据"
}}
'''
_contextual_agent_llm: LLMStageBinding | None = None
_contextual_agent_llm_context: ContextVar[LLMStageBinding | None] = ContextVar(
    "contextual_agent_llm",
    default=None,
)


def set_contextual_agent_llm(
    llm: LLMStageBinding | None,
) -> Token[LLMStageBinding | None]:
    """Bind the L2c2 model for the current run context."""

    token = _contextual_agent_llm_context.set(llm)
    return token


def reset_contextual_agent_llm(token: Token[LLMStageBinding | None]) -> None:
    """Restore the previous L2c2 model binding for this run context."""

    _contextual_agent_llm_context.reset(token)


async def call_social_context_appraisal(state: dict[str, Any]) -> dict[str, Any]:
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
        "background_artifact_result_ready_background_artifact_result": (
            _CONTEXTUAL_AGENT_PROMPT
        ),
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
        "chat_history": project_conversation_history_for_llm(
            state["chat_history_recent"],
            character_name=character_profile["name"],
            max_rows=4,
        ),
    }
    msg.update(
        build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        ),
    )
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    llm = require_llm_binding(
        _contextual_agent_llm_context.get() or _contextual_agent_llm,
        "contextual_agent_llm",
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
