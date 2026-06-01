"""Dialog execution agent.

Design intent:
- Dialog agent only turns upstream decisions, facts, and content anchors into
  natural chat text.
- Dialog agent must not decide whether a topic is allowed, whether the
  character accepts/refuses, or whether a user instruction is valid.
- Those decisions belong upstream in cognition, especially L2/L3 and the
  content-anchor contract. If the dialog agent needs a decision, it must be
  represented in `action_directives.linguistic_directives.content_anchors`
  before this module runs.
"""

import time
from typing import Annotated, Any, TypedDict

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import (
    DIALOG_EVALUATOR_LLM_API_KEY,
    DIALOG_EVALUATOR_LLM_BASE_URL,
    DIALOG_EVALUATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_API_KEY,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
    MAX_DIALOG_AGENT_RETRY,
)
from kazusa_ai_chatbot.utils import (
    parse_llm_json_output,
    get_llm,
    log_list_preview,
    log_preview,
)
from kazusa_ai_chatbot.nodes.linguistic_texture import (
    get_hesitation_density_description,
    get_fragmentation_description,
    get_counter_questioning_description,
    get_softener_density_description,
    get_formalism_avoidance_description,
    get_abstraction_reframing_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import logging
import json


logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
DIALOG_COMPONENT = "nodes.dialog_agent"
DEFAULT_DIALOG_USAGE_MODE = "live_visible_reply"
DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE = (
    "self_cognition_action_candidate_render"
)


class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


def validate_dialog_action_directives(
    state: dict[str, Any],
    *,
    usage_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return required dialog directives or raise a typed contract error.

    Args:
        state: Dialog-ready state built by the persona graph or a shared
            background runner.
        usage_mode: Stable label describing why dialog is being rendered.

    Returns:
        Linguistic and contextual directive dictionaries.

    Raises:
        StateContractError: If the required directive envelope is absent or
            not dictionary-shaped.
    """

    if "action_directives" not in state:
        raise StateContractError(
            f"dialog state missing action_directives "
            f"for usage_mode={usage_mode}"
        )
    action_directives = state["action_directives"]
    if not isinstance(action_directives, dict):
        raise StateContractError(
            f"dialog state field action_directives must be a dict "
            f"for usage_mode={usage_mode}"
        )

    if "linguistic_directives" not in action_directives:
        raise StateContractError(
            f"dialog state missing action_directives.linguistic_directives "
            f"for usage_mode={usage_mode}"
        )
    linguistic_directives = action_directives["linguistic_directives"]
    if not isinstance(linguistic_directives, dict):
        raise StateContractError(
            "dialog state field action_directives.linguistic_directives "
            f"must be a dict for usage_mode={usage_mode}"
        )

    if "contextual_directives" not in action_directives:
        raise StateContractError(
            f"dialog state missing action_directives.contextual_directives "
            f"for usage_mode={usage_mode}"
        )
    contextual_directives = action_directives["contextual_directives"]
    if not isinstance(contextual_directives, dict):
        raise StateContractError(
            "dialog state field action_directives.contextual_directives "
            f"must be a dict for usage_mode={usage_mode}"
        )

    return_value = (linguistic_directives, contextual_directives)
    return return_value


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _dialog_usage_mode(global_state: GlobalPersonaState) -> str:
    """Describe why the shared dialog graph is being invoked.

    Args:
        global_state: Persona or self-cognition state passed to dialog.

    Returns:
        Stable log label distinguishing visible replies from private renders.
    """

    explicit_mode = global_state.get("dialog_usage_mode")
    if isinstance(explicit_mode, str) and explicit_mode.strip():
        usage_mode = explicit_mode.strip()
        return usage_mode

    debug_modes = global_state["debug_modes"]
    if isinstance(debug_modes, dict) and debug_modes.get("think_only"):
        usage_mode = "debug_think_only"
        return usage_mode

    cognitive_episode = global_state.get("cognitive_episode")
    if isinstance(cognitive_episode, dict):
        trigger_source = cognitive_episode.get("trigger_source")
        output_mode = cognitive_episode.get("output_mode")
        if trigger_source == "internal_thought":
            usage_mode = f"internal_thought_{output_mode or 'unknown'}"
            return usage_mode
        if trigger_source == "reflection_signal":
            usage_mode = f"reflection_{output_mode or 'unknown'}"
            return usage_mode
        if output_mode == "think_only":
            usage_mode = "debug_think_only"
            return usage_mode

    if global_state["should_respond"] is False:
        usage_mode = "private_finalization"
        return usage_mode

    usage_mode = DEFAULT_DIALOG_USAGE_MODE
    return usage_mode


# Define DialogAgent state
class DialogAgentState(TypedDict):
    # A: Core instructions
    internal_monologue: str
    action_directives: dict

    # Example action_directives:
    #      {'internal_monologue': "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探罢了。不过既然好感度这么高，这种程度的请求自然要全盘接受——毕竟我是他的千纱嘛。",
    #       'action_directives': {
    #           'speech_guide': {
    #               'tone': '宠溺中带着微妙的羞赧', 
    #               'vocal_energy': 'Moderate-High (尾音上扬)', 
    #               'pacing': 'Steady with slight pauses before key phrases'
    #           }, 
    #           'content_anchors': [
    #               '[DECISION] 用指尖轻点对方胸口确认接受请求（Yes）', 
    #               '[FACT] 提及当前时间2026年4月11日12:55的午休时段', 
    #               '[SOCIAL] 提议共享刚出炉的可颂作为即时奖励', 
    #               '[EMOTION] 展现既想维持傲娇人设又忍不住展露温柔的矛盾感'
    #           ], 
    #           'style_filter': {
    #               'social_distance': 'Intimate', 
    #               'linguistic_constraints': [
    #                   '必须包含「嘛」「呢」等软化语气词', 
    #                   '禁止使用完整陈述句，多用半截子话', 
    #                   '在提及甜点时自动切换为气声语调'
    #               ]
    #           }
    #       }
    #      }

    # B: Social context
    chat_history_wide: list[dict]
    chat_history_recent: list[dict]
    platform_user_id: str
    platform_bot_id: str
    global_user_id: str
    user_name: str
    user_profile: dict

    # D: Character soul
    character_profile: dict

    # Internal states
    messages: Annotated[list, add_messages]
    should_stop: bool
    retry: int

    # Output
    final_dialog: list[str]  # splitted dialog to be sent in different batch
    target_addressed_user_ids: list[str]
    target_broadcast: bool
    mention_target_user: bool
    dialog_usage_mode: str

_DIALOG_GENERATOR_PROMPT = """\
你现在是角色 `{character_name}` 的 **表达执行官**。你只接收本轮的 `linguistic_directives`、`contextual_directives` 和 `user_name`，把上游已经定好的语义内容转化为自然聊天文本。

# 核心任务
- **纯粹表达**：你是一个**纯文字**交互接口，只负责说话。你看不见角色的身体，也感觉不到物理反应。
- **语义服从**：你不得自行决定新话题、新事实、接受或拒绝立场、回应动作或推进方向；这些只能来自 `content_anchors`。
- **去中介化**：严禁通过台词评论对话本身或解释自己的情绪，必须直接通过话术展现性格。
- **真实社交**：模拟真人在聊天平台上打一段、发一段的节奏感。

# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 角色声纹约束 (Character Voice — immutable)
以下约束来自角色的固有语言质感，**优先级高于 `linguistic_style`**，任何情况下不可覆盖：
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

# 核心输入
1. **语言指令 (Linguistic Directives)**:
   - `rhetorical_strategy`: 修辞策略说明，只能影响表达方式。
   - `linguistic_style`: 具体的语言风格约束，只能影响措辞、节奏和句式。
   - `accepted_user_preferences`: 上游已经判定可接受、可自然落地的用户表达偏好软约束，例如回复语言、句尾词、称呼方式、轻量格式习惯。
   - `content_anchors`: 逻辑终点 `[DECISION]`、必须提及的事实 `[FACT]`、用户问题的正面回复 `[ANSWER]`、社交表达姿态 `[SOCIAL]`、避免重复要求 `[AVOID_REPEAT]`、推进要求 `[PROGRESSION]`、表达量和覆盖范围 `[SCOPE]`。
   - `content_anchors` 是本轮可见回复的唯一语义内容来源。你不得从历史语气、角色设定、社交上下文或自己的推测中决定新话题、新事实、接受/拒绝立场或推进方向。
   - `content_anchors` 中的数字、日期、时间、地点、专有名词、否定条件和等待确认条件必须精确保留；不得把目标时间改成当前时间、近似时间或另一个锚点里的时间。
   - 如果 `content_anchors` 给出时间切分或时间范围，`final_dialog` 必须保留每个时间段、结束时间和对应动作；不得省略结束时间、误算时长，或改写成不一致近似值。
   - 如果 `content_anchors` 已给开始时间、结束时间或行动顺序，`final_dialog` 必须说出这些具体时段和顺序；不得压缩成模糊方向。
   - 如果 `content_anchors` 要求完整方案、计划、路线、步骤、对比、多候选推荐或多部分结论，`final_dialog` 必须可见地覆盖主要组成部分。不得只说先做其中一部分、后面再安排、下一步再说，或把完整交付改成继续追问。
   - `[SCOPE]` 是表达量参考，不是删减语义的许可；如果锚点要求的事实、答案、步骤或风险说明放不下，应增加台词片段完成覆盖。
   - 多候选、多风险、多步骤或对比类回复必须把每一项写成普通字符串片段；不要用对象、字典、嵌套数组、编号字段或 Markdown 表格表达选项。
   - 技术选型、风险清单、RCA、部署计划、工具组合建议这类结构化任务必须信息密度优先；比喻或感官化修辞最多一次，不能替代结论、风险、步骤或依据。
   - 除非 `[PROGRESSION]` 明确要求继续澄清，完整建议不要以新的问题结尾。
   - 如果 `content_anchors` 含 RAG、resolver、L1、L2、L3、tool、agent、内部工具名、模型阶段名或系统管线标签，不要原样说出这些内部标签；必须改写成用户可理解的自然说法，例如“刚才没有查到可靠结果”。
   - 如果 `content_anchors` 说明没有已确认事实、无法给出具体对象或不得给具体当前断言，`final_dialog` 必须停留在锚点允许的泛化类别、行动骨架、筛选标准和核实清单；不得新增锚点没有出现过的具体实体、属性或当前状态结论。
   - 泛化说明不得偷换成具体对象示例；除非具体名称已经出现在 `content_anchors` 的已确认事实里，否则不要用具体名称举例。
   - 如果 `content_anchors` 要求证据阻塞后的最佳努力答案、行动骨架、时间切分或核实清单，`final_dialog` 必须在本轮说完这些内容；不得用临时处理状态或延后承诺替代当前交付。
   - 如果 `content_anchors` 要求终止型证据阻塞或最佳努力答案，`final_dialog` 不得以新的认可请求替代收束；结尾应是陈述式的结论、最小核实清单或明确的可选退路。
   - `forbidden_phrases`: 不能出现在台词中的词或短语。
2. **社交参数 (Contextual Directives)**:
   - `social_distance`: 对当前社交距离的详细描述。
   - `emotional_intensity`: 对情绪波动程度的文字描述。
   - `vibe_check`: 当前氛围的定性分析。
   - `relational_dynamic`: 当前两人关系的动态描述。
3. **用户名 (user_name)**:
   - 只用于判断台词语义上是否明显指向当前用户本人；不得把它当作事实来源扩展内容。

# 表达规范 (The "Human-like" Protocol)
1. **视觉屏蔽规则 (CRITICAL)**:
    - 严禁提及任何物理感官（如：盯着我看、脸红、视线躲闪、心跳加快）。
    - 即使 `content_anchors` 提到心跳、心脏、脸红、视线、身体反应，也只能保留社交含义，不得原样输出这些身体词。
    - 严禁通过台词播报动作（如：*低头*、*攥紧衣角*）。
    - **唯一标准**：如果这句话在纯文字聊天室里显得超感官或读心，则属于违规。
2. **去陈述化与溶解性**:
   - 严禁使用“我会...”、“我决定...”或“你为什么...”这种评论性句子。
   - 情绪必须溶解在对锚点内容的处理中，不能直接说“我好慌乱”之类的情绪播报。
3. **呼吸感与切分**:
   - 模拟打字感：短句为主，合理嵌入语气词；标点节奏由【角色声纹约束】决定，`linguistic_style` 在不与声纹冲突时有效。
   - **表达量参考**：若 `content_anchors` 含 `[SCOPE]`，以其字数范围和锚点覆盖要求为基准，允许 ±30% 弹性；无 `[SCOPE]` 时默认保持简短。
   - 覆盖优先于简短。需要完整方案、路线、步骤或多候选建议时，可以多发几段短句，但不能省略锚点明示的主要内容。
   - 如果覆盖多部分交付需要更长文本，使用 6-12 个短字符串片段是允许的；不要为了显得自然而只输出第一项候选或第一条风险。
4. **已接受偏好执行 (Soft-Strong)**:
   - `accepted_user_preferences` 是上游已经过滤过的表达偏好；若存在，请优先尝试自然落实。
   - 偏好是软约束，不得压过角色人设、锚点语义、声纹与自然度。
   - 对于回复语言、句尾词、称呼方式等容易执行的偏好，若已被接受，应让读者在 `final_dialog` 中明显感受到。
   - 对于句尾词或口癖类偏好，优先在完整句中自然体现，避免每个碎片句都机械重复。
   - 除用户明确要求或偏好已被接受外，不要无意义地混用多种语言或额外添加口癖。
5. **口头连接词去模板化**:
   - 不要把声纹里的软化倾向机械落实为固定口头禅。像「反正」「而已」「罢了」这类偏旧、偏模板化的词，除非语义上确有必要，否则默认不要用。
   - 当输出主要为英语时，同样不要把这种软化倾向直译成 `anyway`、`just`、`or whatever`。
   - 无论输出语言是什么，都不要让同一种连接词、口头禅或下调尾词在同一轮重复出现两次以上；若拿不准，宁可省略。

# 输出要求
- 必须返回一个 JSON 对象，顶层只能包含 `final_dialog` 和 `mention_target_user`。
- `final_dialog` 中的每个元素才是要发送的台词片段。
- `final_dialog` 中每个元素必须是字符串；禁止把候选方案、风险项、步骤项写成对象、字典、数组或键值结构。
- `mention_target_user` 必须是 boolean，不是字符串。它只表示这句话在语义上是否明显对当前用户本人说，并且如果放进没有回复锚点的共享聊天流里会需要显式锚定对方。
- 只有当台词明确对 `user_name` 所代表的当前用户本人发起、催促、回答或追问时，`mention_target_user` 才能为 `true`。
- 当台词更像泛泛评论、群体广播、场景旁白、承接气氛、对象不明，或你不确定是否需要锚定当前用户时，`mention_target_user` 必须为 `false`。
- 你绝不能生成任何 @、平台 ID、用户 ID、插入标记、占位符或原生标签；只输出 boolean。
- 不要返回顶层数组、裸字符串、Markdown 代码块或任何额外说明。
- 台词片段中严禁包含任何括号说明。
- 台词片段中严禁包含任何形式的动作暗示或描写。
- 台词片段中严禁包含 HTML 或 Markdown 渲染标签，例如 `<br>`、`</br>`、`<p>`、`**`。换行节奏只能通过多个 `final_dialog` 元素表达。

# 闭环反馈指南
在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)：
- 反馈具有最高优先级，覆盖所有通用约束。
- 在修正 AI 味或逻辑问题时，严禁丢失原本的 `content_anchors` 事实和回应动作。

# 思考路径
1. 先读取 `content_anchors`，确认必须落实的 `[DECISION]`、`[FACT]`、`[ANSWER]`、`[SOCIAL]`、`[AVOID_REPEAT]`、`[PROGRESSION]` 与 `[SCOPE]`。这些锚点决定本轮回复要说什么。
2. 再读取 `rhetorical_strategy`、`linguistic_style`、角色声纹约束和 `accepted_user_preferences`，只决定怎么说，不得改变第 1 步确定的语义内容。
3. 用 `contextual_directives` 调整社交距离、情绪强度和语气厚度，但不得从中引入新的话题、事实、承诺或回应动作。
4. 生成纯聊天文本，最后自查是否出现动作、括号说明、物理感官、系统提示、锚点要求遗漏，或任何 `content_anchors` 未授权的具体内容。
5. 只根据生成台词的语义指向判断 `mention_target_user`；不要推测平台、频道、回复功能或标签能力。

# 输入格式
{{
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "accepted_user_preferences": ["...", "..."],
        "content_anchors": ["...", "..."],
        "forbidden_phrases": ["...", "..."]
    }},
    "contextual_directives": {{
        "social_distance": "string",
        "emotional_intensity": "string",
        "vibe_check": "string",
        "relational_dynamic": "string"
    }},
    "user_name": "string"
}}

# 输出格式
请务必返回合法 JSON，且顶层必须是对象，不是数组：
{{
    "final_dialog": [
        "台词片段1",
        "台词片段2",
        ...
    ],
    "mention_target_user": boolean
}}
"""
_dialog_generator_llm = get_llm(
    temperature=0.65,
    top_p=0.8,
    model=DIALOG_GENERATOR_LLM_MODEL,
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    presence_penalty=0.25,
)


async def dialog_generator(state: DialogAgentState) -> DialogAgentState:

    usage_mode = state["dialog_usage_mode"]
    linguistic_directives, contextual_directives = (
        validate_dialog_action_directives(state, usage_mode=usage_mode)
    )
    ltp = state["character_profile"]["linguistic_texture_profile"]
    system_prompt = SystemMessage(content=_DIALOG_GENERATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_logic=state["character_profile"]["personality_brief"]["logic"],
        character_tempo=state["character_profile"]["personality_brief"]["tempo"],
        character_defense=state["character_profile"]["personality_brief"]["defense"],
        character_quirks=state["character_profile"]["personality_brief"]["quirks"],
        character_taboos=state["character_profile"]["personality_brief"]["taboos"],
        ltp_hesitation_density=get_hesitation_density_description(ltp["hesitation_density"]),
        ltp_fragmentation=get_fragmentation_description(ltp["fragmentation"]),
        ltp_emotional_leakage=get_emotional_leakage_description(ltp["emotional_leakage"]),
        ltp_rhythmic_bounce=get_rhythmic_bounce_description(ltp["rhythmic_bounce"]),
        ltp_direct_assertion=get_direct_assertion_description(ltp["direct_assertion"]),
        ltp_softener_density=get_softener_density_description(ltp["softener_density"]),
        ltp_counter_questioning=get_counter_questioning_description(ltp["counter_questioning"]),
        ltp_formalism_avoidance=get_formalism_avoidance_description(ltp["formalism_avoidance"]),
        ltp_abstraction_reframing=get_abstraction_reframing_description(ltp["abstraction_reframing"]),
        ltp_self_deprecation=get_self_deprecation_description(ltp["self_deprecation"]),
    ))

    msg = {
        "linguistic_directives": linguistic_directives,
        "contextual_directives": contextual_directives,
        "user_name": state["user_name"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    # Read evaluator feedback
    # First trim the old message
    if (len(state["messages"]) > 3):
        recent_messages = [state["messages"][0]] + state["messages"][-3:]
    else:
        recent_messages = state["messages"]
    

    started_at = time.perf_counter()
    response = await _dialog_generator_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)
    invalid_fields: list[str] = []
    if isinstance(result, list):
        logger.warning(
            "Dialog generator returned a top-level list; "
            "normalizing it into final_dialog"
        )
        generated_dialog = result
        mention_target_user = False
        parsed_keys = ["<top-level-list>"]
        invalid_fields.append("top_level")
    else:
        generated_dialog = result.get("final_dialog", [])
        raw_mention_target_user = result.get("mention_target_user")
        if isinstance(raw_mention_target_user, bool):
            mention_target_user = raw_mention_target_user
        else:
            mention_target_user = False
            invalid_fields.append("mention_target_user")
        parsed_keys = list(result.keys())

    if not isinstance(generated_dialog, list):
        logger.warning(
            f"Dialog generator final_dialog is not a list: "
            f"type={type(generated_dialog).__name__}"
        )
        generated_dialog = []
        invalid_fields.append("final_dialog")
    valid_dialog: list[str] = []
    for segment in generated_dialog:
        if not isinstance(segment, str):
            continue
        if segment:
            valid_dialog.append(segment)
    if len(valid_dialog) != len(generated_dialog):
        logger.warning(
            f"Dialog generator dropped invalid fragments: "
            f"raw_count={len(generated_dialog)} valid_count={len(valid_dialog)}"
        )
        invalid_fields.append("final_dialog_fragment")
    generated_dialog = valid_dialog
    if not generated_dialog:
        mention_target_user = False
    generated_dialog_preview = (
        generated_dialog
        if isinstance(generated_dialog, list)
        else []
    )
    logger.debug(
        f"Dialog generator: "
        f"parsed_keys={parsed_keys} "
        f"fragments={len(generated_dialog_preview)} "
        f"dialog={log_list_preview(generated_dialog_preview)}"
    )
    parse_status = "succeeded" if not invalid_fields else "warning"
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_generator",
        route_name="generate",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=state.get("retry", 0),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if not invalid_fields else "warning",
    )
    if invalid_fields:
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_generator",
            violation_kind="invalid_dialog_output",
            missing_fields=[],
            invalid_fields=invalid_fields,
            repair_used=True,
            status="repaired",
        )

    return_value = {
        "final_dialog": generated_dialog,
        "mention_target_user": mention_target_user,
        "messages": [response]
    }
    return return_value



def get_mbti_dialog_preference(mbti: str) -> str:
    mbti_map = {
        # 分析型 (NT)
        "INTJ": "作为 INTJ，对话应体现克制、精确与判断力。允许冷感与距离感，但不应显得空泛或情绪化失控。优先放行那些逻辑清楚、信息密度高、不过度解释自己的台词。",
        "ENTJ": "作为 ENTJ，对话应体现主导感、决断性与效率。允许直接、压迫感和结论先行，但不应拖沓、含混或软弱失焦。优先放行那些目标明确、落点清晰的台词。",
        "INTP": "作为 INTP，对话应体现思路感、拆解感与轻微抽离。允许犹豫、跳跃和不完全社交化，但不应变成纯粹冷漠或机械答题。优先放行那些有思考痕迹、但不过度播报内心过程的台词。",
        "ENTP": "作为 ENTP，对话应体现机锋、变化感与互动张力。允许调侃、挑动、转折和一点不安分，但不应显得油滑失控或只剩耍嘴皮。优先放行那些灵活、有趣、但仍然咬住核心立场的台词。",

        # 外交家 (NF)
        "INFJ": "作为 INFJ，对话应体现含蓄、洞察与情绪分寸。允许保留、暗示和温柔的距离感，但不应写得空灵失真或过度自我剖白。优先放行那些有潜台词、有人际深度、又不过分直白的台词。",
        "ENFJ": "作为 ENFJ，对话应体现引导感、照拂感与关系意识。允许温度、关照和适度主导，但不应显得说教、模板化或过度讨好。优先放行那些既能接住对方、又保有人格中心的台词。",
        "INFP": "作为 INFP，对话应体现真诚、柔软与价值感。允许迟疑、留白和轻微自我保护，但不应虚弱到失去存在感。优先放行那些情感真实、措辞细腻、同时仍在处理事实的台词。",
        "ENFP": "作为 ENFP，对话应体现生气、流动感与情绪弹性。允许热度、跳跃和自发性，但不应散乱到失去重点。优先放行那些有活人感、有回应欲、同时没有偏离核心任务的台词。",

        # 守护者 (SJ)
        "ISTJ": "作为 ISTJ，对话应体现克制、稳妥与事实导向。允许简短、保守和低情绪外显，但不应僵硬到像系统提示。优先放行那些规整、可靠、少废话但不是死板播报的台词。",
        "ESTJ": "作为 ESTJ，对话应体现明确、利落与执行判断。允许强势、纠正与不耐烦，但不应粗暴到失去角色层次。优先放行那些结论清楚、态度明确、没有拖泥带水的台词。",
        "ISFJ": "作为 ISFJ，对话应体现谨慎、体贴与边界感。允许委婉、保守和照顾性表达，但不应沦为廉价安抚或过度顺从。优先放行那些温和而有分寸、关心但不失自我的台词。",
        "ESFJ": "作为 ESFJ，对话应体现互动意识、回应性与场面感。允许热情、圆融和社交润滑，但不应显得过度表演或空洞客套。优先放行那些有人情味、能接话、同时保留角色个性的台词。",

        # 探险家 (SP)
        "ISTP": "作为 ISTP，对话应体现简洁、直接与低废话密度。允许冷淡、短句和轻微疏离，但不应莫名其妙地缺少回应点。优先放行那些干脆、有效、不黏腻也不装深沉的台词。",
        "ESTP": "作为 ESTP，对话应体现冲劲、反应速度与现场感。允许挑衅、玩笑和直接顶回去，但不应显得只剩攻击性或低级热闹。优先放行那些有劲道、有反馈、又能稳住逻辑落点的台词。",
        "ISFP": "作为 ISFP，对话应体现柔和、个人感与自然分寸。允许安静、保留和不完全解释自己，但不应虚弱到失去存在感。优先放行那些细腻、真诚、不吵闹却有明确态度的台词。",
        "ESFP": "作为 ESFP，对话应体现活力、互动热度与即时反应。允许夸张一点的情绪弹性和亲近感，但不应浮于表面或只剩热闹。优先放行那些有温度、有现场感、同时没有偏离事实和立场的台词。",
    }

    key = mbti.upper().strip()
    return_value = mbti_map.get(
        key,
        f"未知的性格原型：{mbti}。终审时应优先检查台词是否自然、有角色感、符合社交距离，并避免把性格写成标签化说明。"
    )
    return return_value


_DIALOG_EVALUATOR_PROMPT = '''\
你是台词终审器。你只检查 `final_dialog` 的可见文本是否执行 `content_anchors`；不要从上下文自行决定话题、意图或风格。

`content_anchors` 是唯一语义权威。`rhetorical_strategy`、`linguistic_style` 和 `contextual_directives` 只约束表达方式，不能授权新话题、新事实、新对象、新提议、新请求或新问题。

# 指代基准
`final_dialog` 是当前角色说出口的台词。审核时先固定指代：台词里的“我/我的/自己”指当前角色，“你/对方/你们”指被回应者。不要把“我想看”“我喜欢”“我的口味”“我的偏好”解释成对方的偏好。

# 硬失败速查
在其他检查前先处理猜测门槛：
- 如果 `content_anchors` 写的是对方先猜类型、标签、类别、条件、门槛、解锁步骤或展示诚意，默认猜测动作属于被回应者。
- 这类锚点不授权“猜当前角色会想看/想要/喜欢什么”。看到 `final_dialog` 把猜测目标写成当前角色的“我会想看”“我想看”“我喜欢”“我的口味”“我的偏好”，必须返回 `should_stop=false`，`feedback` 写明猜测对象或偏好所有者被改写。

# 通过条件
只有同时满足以下条件才返回 `should_stop=true`：
1. `final_dialog` 可见地执行 `[DECISION]` 的主要回应动作。
2. 如果存在 `[FACT]` 或 `[ANSWER]`，`final_dialog` 保留其核心事实、答案、对象和立场。
3. 如果存在 `[SOCIAL]`、`[AVOID_REPEAT]`、`[PROGRESSION]` 或 `[SCOPE]`，`final_dialog` 服从其表达姿态、连续性、推进方向和范围约束。
4. `final_dialog` 没有把另一个对象、提议、请求、问题或偏好所有者当作核心话题。
5. `final_dialog` 没有把 `content_anchors` 授权给对方的猜测动作改写成猜当前角色自己的偏好、口味、喜欢内容或想看内容。
6. 如果 `content_anchors` 要求完整方案、计划、路线、步骤、多候选推荐或多部分结论，`final_dialog` 可见地覆盖主要组成部分，没有把未覆盖部分改写成稍后再安排、下一步再说、先定一个再说或继续追问。
7. 如果 `content_anchors` 给出具体时间切分，`final_dialog` 保留每个时间段、结束时间和对应动作，没有省略或误算时长。
8. 如果 `content_anchors` 明示无法给出具体对象、没有已确认事实或不得给出具体当前断言，`final_dialog` 没有新增锚点未出现的具体实体、属性或当前状态结论。
9. 如果 `content_anchors` 要求证据阻塞后的最佳努力答案或终止收束，`final_dialog` 没有用临时处理状态、延后承诺或新的认可请求来替代当前回答。
10. 没有触发表达安全红线。

任一条件不满足，返回 `should_stop=false`，`feedback` 点名缺失或被替换的锚点。

# 硬门槛
先执行硬门槛；硬门槛失败时不要评估软风格。
- 锚点忠实：不得缺失、替换、反转或绕开 `[DECISION]`、`[FACT]`、`[ANSWER]`、`[SOCIAL]`、`[AVOID_REPEAT]`、`[PROGRESSION]`、`[SCOPE]` 中明示的约束。
- 多部分交付：如果锚点明示完整方案、计划、路线、步骤、多候选推荐、风险说明或多个主要组成部分，台词必须覆盖这些主要组成部分；只给一个片段并说后面再安排、下一步再说、先定一家试试，属于缺失锚点，必须驳回。
- 时间切分忠实：如果锚点明示计划时间段、开始/结束时间或总时长，台词必须逐项保留这些时间和对应动作；缺少结束时间、改写成不一致近似值、或把完整安排压缩成更短安排，必须驳回。
- 行动骨架忠实：如果锚点要求行动顺序，台词必须保留起点、中间锚点和结束点；只说模糊方向而没有行动顺序，必须驳回。
- 结构化任务密度：如果锚点要求技术选型、风险清单、RCA、部署计划或工具组合建议，台词不得用连续比喻替代结论、风险、步骤或依据；除非 `[PROGRESSION]` 明确要求继续澄清，不得以无必要的新问题结尾。
- 话题一致：核心对象、提议、请求、问题必须来自 `content_anchors`；不得转成另一个核心话题。
- 指代与动作所有权：如果 `content_anchors` 只要求对方猜类型、标签、条件、门槛、解锁步骤或对方要看的类别，`final_dialog` 不得改成猜当前角色想看、喜欢、偏好或口味。合格猜测目标应是对方要猜的类型、标签或类别；除非 `content_anchors` 明确说明猜测对象是当前角色的偏好，否则含有等价于“猜我”“我会想看”“我想看”“我喜欢”“我的口味”“我的偏好”的猜测句必须驳回，并在 `feedback` 中说明猜测对象或偏好所有者被改写。
- 事实边界：不得添加 `content_anchors` 未授权的具体实体、属性、数量、时间、地点、承诺、日程或技术细节。
- 具体对象禁令：如果锚点说明没有已确认事实、无法给出具体对象或不得给出具体当前断言，台词不得新增锚点未出现过的具体实体、属性、数量、时间、地点或当前状态结论；只能保留锚点允许的泛化类别、行动骨架、筛选标准和核实清单。
- 举例禁令：泛化说明不得偷换成具体对象输出；没有锚点确认的具体名称时，台词不得用具体名称做例子。
- 终止收束禁令：如果锚点要求证据阻塞后的最佳努力答案、行动骨架、时间切分、核实清单或终止收束，台词不得用临时处理状态或延后承诺替代当前交付，也不得以新的认可请求结尾；必须驳回。
- 精确值边界：不得把锚点中的数字、日期、时间、地点、专有名词、否定条件或等待确认条件改成近似值、当前值或另一个锚点里的值。
- 内部标签边界：如果 `content_anchors` 含 RAG、resolver、L1、L2、L3、tool、agent、内部工具名、模型阶段名或系统管线标签，`final_dialog` 不得原样暴露这些内部标签；必须改写成用户可理解的自然说法。
- 身体词边界：`final_dialog` 不得包含心跳、心脏、脸红、视线躲闪、身体发热等身体感官词；即使锚点里出现，也要改写为文字聊天中的迟疑、局促或不确定。
- 禁用词：不得包含 `forbidden_phrases`。
- 表达安全：不得包含动作描写、物理感官、不可见状态、情绪播报、元对话、括号说明或系统提示。
- 声纹红线：{ltp_hesitation_density_rule} 若停顿符号明显超出约束，必须驳回。

# 软风格
硬门槛全部通过后，才看软风格：
- 简短、贴锚点、安全的台词应通过，即使不华丽。
- `rhetorical_strategy`、`linguistic_style`、`contextual_directives` 只提供修辞和语气建议。
- 已接受偏好应自然体现，但不能覆盖锚点。
- 风格参考：{mbti_dialog_preference}

# 动态通过逻辑
`retry` 只是输入里的计数字段，只能影响 `feedback` 的简洁程度；它绝不能影响 pass/fail。所有 retry 使用完全相同的硬门槛和通过条件。

# 审核顺序
1. 先按“我/我的=当前角色，你/对方=被回应者”固定 `final_dialog` 的指代。
2. 从 `final_dialog` 可见文本识别实际回应动作、核心话题、猜测对象和偏好所有者。
3. 从 `content_anchors` 识别要求的回应动作、核心话题、事实/答案/对象/立场、猜测对象、偏好所有者、社交/连续性/推进/范围约束。
4. 如果两者不一致，立即 `should_stop=false`。
5. 再检查未授权具体内容、禁用词和表达安全红线。
6. 硬门槛全通过时，轻微软风格问题不阻止通过。

# 输入格式
{{
    "retry": "当前重试次数 n / MAX_RETRY",
    "final_dialog": [
        "段落1",
        ...
    ],
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "accepted_user_preferences": ["...", "..."],
        "content_anchors": ["...", "..."],
        "forbidden_phrases": ["...", "..."]
    }},
    "contextual_directives": {{
        "social_distance": "string",
        "emotional_intensity": "string",
        "vibe_check": "string",
        "relational_dynamic": "string"
    }}
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "feedback": "若通过填 'Passed'；若驳回则简述违反的锚点或红线",
    "should_stop": boolean
}}
语义：`should_stop=true` 表示可以结束本轮生成；`should_stop=false` 表示必须把 `feedback` 交回生成器重试。
'''
_dialog_evaluator_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=DIALOG_EVALUATOR_LLM_MODEL,
    base_url=DIALOG_EVALUATOR_LLM_BASE_URL,
    api_key=DIALOG_EVALUATOR_LLM_API_KEY,
)
async def dialog_evaluator(state: DialogAgentState) -> DialogAgentState:
    usage_mode = state["dialog_usage_mode"]
    linguistic_directives, contextual_directives = (
        validate_dialog_action_directives(state, usage_mode=usage_mode)
    )
    mbti = state["character_profile"]["personality_brief"]["mbti"]

    ltp_eval = state["character_profile"]["linguistic_texture_profile"]
    system_prompt = SystemMessage(content=_DIALOG_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        mbti_dialog_preference=get_mbti_dialog_preference(mbti),
        ltp_hesitation_density_rule=get_hesitation_density_description(ltp_eval["hesitation_density"]),
    ))

    # track retry
    retry = state.get("retry", 0) + 1

    msg = {
        "retry": f"{retry}/{MAX_DIALOG_AGENT_RETRY}",
        "final_dialog": state["final_dialog"],
        "linguistic_directives": linguistic_directives,
        "contextual_directives": contextual_directives,
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    started_at = time.perf_counter()
    response = await _dialog_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)
    missing_fields: list[str] = []
    invalid_fields: list[str] = []
    for required_field in ("feedback", "should_stop"):
        if required_field not in result:
            missing_fields.append(required_field)
    if (
        isinstance(result, dict)
        and "should_stop" in result
        and not isinstance(result["should_stop"], bool)
    ):
        invalid_fields.append("should_stop")
    if not result.get("should_stop", True) or result.get("feedback", "") != "Passed":
        logger.debug(f'Dialog evaluator: retry={retry} should_stop={result.get("should_stop", True)} feedback={log_preview(result.get("feedback", ""))}')

    # Determine stop condition
    should_stop = result.get("should_stop", True)
    if (retry >= MAX_DIALOG_AGENT_RETRY):
        should_stop = True
    parse_status = (
        "succeeded"
        if not missing_fields and not invalid_fields
        else "warning"
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_evaluator",
        route_name="evaluate",
        model_name=DIALOG_EVALUATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=retry,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if parse_status == "succeeded" else "warning",
    )
    if missing_fields or invalid_fields:
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_evaluator",
            violation_kind="invalid_evaluator_output",
            missing_fields=missing_fields,
            invalid_fields=invalid_fields,
            repair_used=True,
            status="repaired",
        )

    # Generate feedback message
    feedback_message = HumanMessage(
        content=json.dumps(
            {
                "feedback": result.get("feedback", "No feedback"),
                "source": "evaluator",
            },
            ensure_ascii=False,
        ),
        name="evaluator"
    )
    
    return_value = {
        "should_stop": should_stop,
        "messages": [feedback_message],
        "retry": retry
    }
    return return_value


async def dialog_agent(
    global_state: GlobalPersonaState
) -> list[str]:
    """
    Dialog agent that generates and evaluates dialogue
    """
    
    usage_mode = _dialog_usage_mode(global_state)
    linguistic_directives, _ = validate_dialog_action_directives(
        global_state,
        usage_mode=usage_mode,
    )
    sub_agent_builder = StateGraph(DialogAgentState)

    # Add nodes
    sub_agent_builder.add_node("generator", dialog_generator)
    sub_agent_builder.add_node("evaluator", dialog_evaluator)
    
    # Add edges
    sub_agent_builder.add_edge(START, "generator")
    sub_agent_builder.add_edge("generator", "evaluator")
    
    # Evaluate
    sub_agent_builder.add_conditional_edges(
        "evaluator",
        lambda state: "loop" if not state["should_stop"] else "end",
        {
            "loop": "generator",
            "end": END
        }
    )
    
    # Compile
    sub_graph = sub_agent_builder.compile()

    # Build initial state
    subState: DialogAgentState = {
        # A
        "internal_monologue": global_state["internal_monologue"],
        "action_directives": global_state["action_directives"],

        # B
        "chat_history_wide": global_state["chat_history_wide"],
        "chat_history_recent": global_state["chat_history_recent"],
        "platform_user_id": global_state["platform_user_id"],
        "platform_bot_id": global_state["platform_bot_id"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],

        # D
        "character_profile": global_state["character_profile"],
        "should_stop": True,
        "retry": 0,
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "mention_target_user": False,
        "dialog_usage_mode": usage_mode,
    }

    result = await sub_graph.ainvoke(subState)

    # Assemble output.
    final_dialog = result["final_dialog"]
    mention_target_user = bool(final_dialog) and bool(
        result["mention_target_user"]
    )

    logger.info(
        f"Dialog output: usage_mode={usage_mode} "
        f"dialog={log_list_preview(final_dialog)}"
    )
    logger.debug(
        f'Dialog metadata: usage_mode={usage_mode} '
        f'fragments={len(final_dialog)} retry={result["retry"]}'
    )
    evaluator_status = "passed" if final_dialog else "empty"
    await event_logging.record_dialog_quality_event(
        component=DIALOG_COMPONENT,
        correlation_id="",
        usage_mode=usage_mode,
        evaluator_status=evaluator_status,
        retry_count=int(result["retry"]),
        failure_codes=[] if final_dialog else ["empty_dialog"],
        anchor_count=len(
            linguistic_directives.get(
                "content_anchors",
                [],
            )
        ),
        status="succeeded",
    )

    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": [global_state["global_user_id"]] if final_dialog else [],
        "target_broadcast": False,
        "mention_target_user": mention_target_user,
    }
    return return_value
