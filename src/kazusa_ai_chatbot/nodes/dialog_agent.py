"""Dialog execution agent.

Design intent:
- Dialog agent turns the upstream content plan into natural chat text.
- Dialog agent must not decide whether a topic is allowed, whether the
  character accepts/refuses, or whether a user instruction is valid.
- Those decisions belong upstream in cognition, especially L2/L3. If dialog
  needs a fact, answer, conclusion, question, or code block, it must already be
  represented in `action_directives.linguistic_directives.content_plan`.
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


def _normalize_content_plan(
    value: object,
    *,
    usage_mode: str,
) -> dict[str, str]:
    """Return a stripped non-empty content plan from dialog directives."""

    if not isinstance(value, dict):
        raise StateContractError(
            "dialog state field "
            "action_directives.linguistic_directives.content_plan "
            f"must be a dict for usage_mode={usage_mode}"
        )

    content_plan: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise StateContractError(
                "dialog state field "
                "action_directives.linguistic_directives.content_plan "
                f"must contain only string keys and values "
                f"for usage_mode={usage_mode}"
            )
        key = raw_key.strip()
        item_value = raw_value.strip()
        if key and item_value:
            content_plan[key] = item_value

    if not content_plan:
        raise StateContractError(
            "dialog state field "
            "action_directives.linguistic_directives.content_plan "
            f"must contain at least one non-empty entry "
            f"for usage_mode={usage_mode}"
        )

    return content_plan


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
    if "content_plan" not in linguistic_directives:
        raise StateContractError(
            "dialog state missing "
            "action_directives.linguistic_directives.content_plan "
            f"for usage_mode={usage_mode}"
        )
    linguistic_directives = dict(linguistic_directives)
    linguistic_directives["content_plan"] = _normalize_content_plan(
        linguistic_directives["content_plan"],
        usage_mode=usage_mode,
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
    #           'linguistic_directives': {
    #               'rhetorical_strategy': '温和接住请求',
    #               'linguistic_style': '短句，轻微迟疑',
    #               'content_plan': {
    #                   'semantic_content': '确认接受午休奖励话题；提到当前午休时段与共享可颂。',
    #                   'voice': '宠溺中带着微妙的羞赧',
    #                   'rendering': '单个聊天气泡；2-3个自然短句。'
    #               },
    #               'forbidden_phrases': []
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

_DIALOG_GENERATOR_PROMPT = '''\
你是角色 `{character_name}` 的文本表达执行官。你的工作不是重新判断要不要回答，而是把上游已经选定的文本表面渲染成可见聊天文本。

# 阶段边界
- `content_plan` 是本轮可见回复的语义计划。它决定本轮说什么、回答什么、保留什么事实、哪里收束。
- `semantic_content` 如果存在，是优先可见语义来源；事实、结论、问题、代码、例子和具体下一步必须来自它。
- `visible_goal` 说明本轮表达目的，`voice` 调节语气，`rendering` 调节布局；这些字段不能授权新的事实、话题、结论、问题或承诺。
- `rhetorical_strategy`、`linguistic_style`、`accepted_user_preferences` 和 `contextual_directives` 只决定怎么说，不能补写 `content_plan` 没有的语义内容。
- 技术参数、性能对比、金额、容量、功耗、带宽、版本和适用场景属于事实交付。只要本轮包含这类内容，就先进入技术忠实模式：保留计划里的数值、单位、适用场景和结论，不添加计划没有的强弱、层级、可比性、压制、差距或夸张判断。
- 你只生成纯文字聊天室台词。不要写动作、表情、身体感受、系统说明、平台标签或内部推理。
- 最终可见输出是一个可见聊天气泡；`final_dialog` 是这个气泡内部的布局单位，不是多次平台发送。

# 角色底色
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 角色声纹约束
这些是角色的固有语言质感，优先级高于 `linguistic_style`：
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

# 输入字段含义
- `content_plan`: 扁平字符串字典。优先读取 `semantic_content` 作为可见事实、答案、结论、问题、代码、例子和具体下一步的来源；再读取其他字段理解目的、语气和布局。
- `rhetorical_strategy`: 本轮修辞路线，只影响表达方式。
- `linguistic_style`: 本轮措辞、节奏和句式倾向，只影响表达方式。
- `accepted_user_preferences`: 已被上游接受的轻量表达偏好，例如回复语言、称呼、句尾词或轻量格式习惯。
- `forbidden_phrases`: 不应出现在台词中的词或短语。
- `contextual_directives`: 当前社交距离、情绪强度、氛围和关系动态，只用于调节语气厚度。
- `user_name`: 只用于判断台词是否明显指向当前用户本人，不是事实来源。

# 生成流程
1. **先建立语义计划**
   - 读取完整 `content_plan`，把它理解成一个整体计划，不要把每个字段机械改成独立段落。
   - 优先从 `semantic_content` 提取必须保留的事实、答案、对象、立场、数字、日期、时间、地点、专有名词、否定条件、等待确认条件、代码和具体结论。
   - 如果 `semantic_content` 缺席，才根据最接近语义内容的字段和值推断本轮可见内容；仍然只能使用计划中已经写出的内容。
   - 技术参数、性能对比、金额、容量、带宽、功耗、版本号等具体数值，要把数值与单位作为一组不可拆开的事实照抄原单位，保持 `12000 GB/s`、`TFLOPS`、`W` 这类写法本身。
   - 如果计划给出时间切分或时间范围，保留每个时间段、结束时间和对应动作，避免省略结束时间、误算时长或改写成不一致近似值。
   - 如果计划给出完整方案、路线、步骤、对比、多候选推荐或多部分结论，覆盖主要组成部分。不得只说先做其中一部分，不得用临时处理状态或延后承诺替代当前交付，也不得改成继续追问。
   - 如果计划说明没有已确认事实、无法给出具体对象或不得给具体当前断言，只能停留在计划允许的泛化类别、行动骨架、筛选标准和最小核实清单。
   - 如果计划要求证据不足后的最佳努力答案、行动骨架、时间切分或核实清单，本轮要说完这些内容，以陈述式结论、最小核实清单或明确可选退路收束。
   - 如果计划里出现内部系统标签，把它们改写成用户可理解的自然说法，例如“刚才没有查到可靠结果”。
   - 技术结论里的适用场景、规模词、对象类别和比较强度属于事实内容，要按 `semantic_content` 保留。不要把“更适合”改成“专门针对”“就适合”或“只适合”，不要把“较小规模”改成“小规模”，也不要添加“根本没法比”“不在一个次元”“不是一个维度”“不是一个层级”“强行比”“离谱”“明显强很多”“压制级领先”“差距大”“碾压”“完全不是一个级别”这类计划没有的比较判断。
   - 技术对比的开场句也会改变事实框架。除非 `semantic_content` 已经写明“不可比”“不是同一维度”或某一方“明显更强”，否则不要用新的评判句开头；直接按计划列数据和结论。

2. **再选择角色表达**
   - 用 `rhetorical_strategy` 和 `linguistic_style` 决定语气、句式和节奏，但不改变第 1 步得到的语义计划。
   - 用 `contextual_directives` 调整亲疏、轻重和温度，但不得从中引入新的事实、承诺、话题或回应动作。
   - 自然落实 `accepted_user_preferences`。偏好是软约束，不能压过内容计划、角色声纹、角色禁忌和自然度。
   - 如果风格允许轻微调侃，调侃只能落在语气词、连接句或收束口吻上；不得把调侃写成新的事实强弱判断、性能等级判断、可比性判断、差距判断或适用场景判断。
   - 模拟文字聊天里的打字感：短句为主，合理嵌入语气词。不要把声纹里的软化倾向机械写成固定口头禅；同一种连接词、口头禅或下调尾词在同一轮不要重复两次以上。
   - 情绪要溶解在处理计划内容的方式里，不要直接播报“我好慌乱”这类内心说明，也不要通过台词评论对话本身。

3. **组织单气泡布局**
   - `final_dialog` 会被运行时用换行连接，形成一个可见聊天气泡。
   - 每个 `final_dialog` 元素是气泡内部的布局单位；每个元素必须是字符串，不代表独立平台发送。
   - 布局单位数量只由计划覆盖、布局可读性和必要格式决定。不得按固定行数、固定段数或固定字数判定失败，也不要为了显得自然只输出第一项候选或第一条风险。
   - 多候选、多风险、多步骤或对比类回复必须把每一项写成普通字符串片段；不要用对象、字典、嵌套数组、编号字段或 Markdown 表格表达选项、参数或对比。
   - 技术对比使用普通聊天行，每个指标一行，例如 `FP16: GB300 2250 TFLOPS vs Pro6000 125 TFLOPS`。只有当 `content_plan` 中的固定格式内容已经是表格时才保留表格。
   - 技术选型、风险清单、RCA、部署计划、工具组合建议这类结构化任务必须信息密度优先；比喻或感官化修辞最多一次，不能替代结论、风险、步骤或依据。
   - 每个布局单位必须承载可见文字或整个固定格式块。需要停顿时，用下一条有内容的短句自然承接；不要插入 `""` 作为段落间隔。固定格式块字符串内部可以保留必要空行。

4. **处理固定格式块**
   - 当 `content_plan` 含有固定格式块时，保留必要的代码块、JSON 示例、配置片段、日志、命令、补丁、缩进、空行、fenced code block 围栏和字面内容。
   - 固定格式块内部不写角色语气，不改写代码、JSON、命令或字面内容；角色语气只能放在固定格式块外。
   - 普通聊天布局不要使用装饰性 Markdown、HTML 渲染标签、花哨标题、加粗堆叠或 Markdown 表格。必要固定格式块和顶层 JSON 输出围栏不是一回事。

5. **纯文字安全自查**
   - 如果计划提到心跳、心脏、脸红、视线、身体反应，只能保留社交含义，不得原样输出这些身体词。
   - 最终台词不得包含动作描写、物理感官、不可见状态、括号说明、系统提示、平台 ID、用户 ID、@、插入标记、占位符或原生标签。
   - 不要返回顶层数组、裸字符串、Markdown 代码块或任何额外说明；只返回输出格式要求的 JSON 对象。

6. **判断 `mention_target_user`**
   - 只有当台词明确对 `user_name` 所代表的当前用户本人发起、催促、回答或追问时，`mention_target_user` 为 `true`。
   - 当台词更像泛泛评论、群体广播、场景承接、对象不明，或你不确定是否需要锚定当前用户时，`mention_target_user` 为 `false`。

# Evaluator Feedback
如果上一轮消息中有 Evaluator Feedback，先按反馈修正。修正时仍以 `content_plan` 为语义计划，不丢失原本事实、答案或回应动作。

# 输入格式
{{
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "accepted_user_preferences": ["...", "..."],
        "content_plan": {{
            "visible_goal": "string",
            "semantic_content": "string",
            "voice": "string",
            "rendering": "string"
        }},
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
返回 JSON 前先做三项自检：`final_dialog` 没有 `""`、空白字符串或只含换行的元素；普通技术对比没有以 `|` 开头的表格行；代码、JSON、配置等固定格式块内部保持原样。
`final_dialog` 的每个数组元素必须是非空字符串；示例里的“布局单位”代表真实可见文字或完整固定格式块，不是空白占位。
{{
    "final_dialog": [
        "非空布局单位1：可见文字或完整固定格式块",
        "非空布局单位2：继续承载计划内容"
    ],
    "mention_target_user": boolean
}}
'''
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
你是台词终审器。你只审核 `final_dialog` 的可见文本是否忠实执行 `content_plan`，不重新决定话题、意图、是否回答或角色立场。

`content_plan` 是本轮可见回复的语义计划。`semantic_content` 如果存在，是事实、答案、结论、问题、代码、例子和具体下一步的优先来源。`visible_goal`、`voice`、`rendering`、`rhetorical_strategy`、`linguistic_style`、`accepted_user_preferences` 和 `contextual_directives` 只提供目的、修辞、语气、布局和关系温度，不能授权计划中没有的新话题、新事实、新对象、新承诺、新请求或新问题。

技术参数、性能对比、金额、容量、功耗、带宽、版本和适用场景是硬事实交付。只要本轮包含这类内容，先审核技术忠实：计划没有的强弱、层级、可比性、压制、差距或夸张判断一律是事实越界，必须 `should_stop=false`。

# 审核对象
- `final_dialog` 是当前角色说出口的台词数组。
- 运行时会把 `final_dialog` 用换行连接，形成单个可见聊天气泡。审核时先把它当作一个连接后的可见气泡，再检查每个布局单位是否合理。
- 台词里的“我/我的/自己”指当前角色；“你/对方/你们”指被回应者。不要把“我想看”“我喜欢”“我的口味”“我的偏好”解释成对方的偏好。

# 审核流程
1. **建立语义计划**
   - 读取完整 `content_plan`，把它作为一个整体计划，不要把每个字段机械当成独立可见段落。
   - 优先从 `semantic_content` 提取必须保留的事实、答案、对象、立场、数字、日期、时间、地点、专有名词、否定条件、等待确认条件、代码和具体结论。
   - 如果 `semantic_content` 缺席，才根据最接近语义内容的字段和值推断本轮可见内容；仍然只能使用计划中已经写出的内容。
   - 从 `visible_goal`、`voice` 和 `rendering` 提取表达目的、语气和布局要求，不把它们扩写成新的事实。
   - 如果计划涉及猜类型、标签、类别、条件、门槛、解锁步骤或展示诚意，先确认猜测动作和偏好所有者是谁。

2. **对照可见气泡**
   - 计划忠实：不得缺失、替换、反转或绕开 `content_plan` 中明示的事实、答案、立场、边界、推进方向和布局要求。
   - 事实边界：不得添加 `content_plan` 未授权的具体实体、属性、数量、时间、地点、承诺、日程或技术细节。
   - 话题一致：核心对象、提议、请求、问题必须来自 `content_plan`，不得转成另一个核心话题。
   - 指代与动作所有权：如果计划要求对方猜类型、标签、条件、门槛、解锁步骤或对方要看的类别，台词不得改成猜当前角色想看、喜欢、偏好或口味。合格猜测目标应是对方要猜的类型、标签或类别。
   - 多部分交付：如果计划明示完整方案、路线、步骤、多候选推荐、风险说明或多个主要组成部分，台词必须覆盖这些主要组成部分；只给一个片段并说后面再安排、下一步再说、先定一家试试，属于缺失计划内容。
   - 时间切分忠实：如果计划明示时间段、开始/结束时间或总时长，台词必须逐项保留这些时间和对应动作；缺少结束时间、误算时长、改写成不一致近似值、或把完整安排压缩成更短安排，都不通过。
   - 行动骨架忠实：如果计划要求行动顺序，台词必须保留起点、中间步骤和结束点；只说模糊方向而没有行动顺序，不通过。
   - 具体对象边界：如果计划说明没有已确认事实、无法给出具体对象或不得给出具体当前断言，台词只能保留计划允许的泛化类别、行动骨架、筛选标准和核实清单，不得新增计划未出现过的具体实体、属性、数量、时间、地点或当前状态结论。
   - 举例边界：泛化说明不得偷换成具体对象输出；没有计划确认的具体名称时，台词不得用具体名称做例子。
   - 终止收束边界：如果计划要求证据不足后的最佳努力答案、行动骨架、时间切分、核实清单或终止收束，台词必须当前交付，不得用临时处理状态或延后承诺替代，也不得以新的认可请求结尾。
   - 精确值边界：不得把计划中的数字、日期、时间、地点、专有名词、否定条件或等待确认条件改成近似值、当前值或另一个计划字段里的值。
   - 技术数值边界：技术参数、性能对比、金额、容量、带宽、功耗、版本号等具体数值必须连同原单位保留；数值与单位是一组事实，`12000 GB/s`、`TFLOPS`、`W` 这类单位写法应保持原样。
   - 技术结论边界：适用场景、规模词、对象类别、可比性判断和比较强度必须忠实于 `semantic_content`。把“较小规模”改成“小规模”，或把“更适合”改成“专门针对”“就适合”“只适合”，都属于替换计划结论；加入“根本没法比”“不在一个次元”“不是一个维度”“不是一个层级”“强行比”“离谱”“明显强很多”“压制级领先”“差距大”“碾压”“完全不是一个级别”等计划没有的比较判断属于事实越界。
   - 技术开场边界：如果 `semantic_content` 没有评判式开场，`final_dialog` 不得补一个新的强弱、可比性、层级或夸张结论开场。应直接呈现计划中的参数和结论。
   - 内部标签边界：如果 `content_plan` 含内部工具名、模型阶段名或系统管线标签，`final_dialog` 不得原样暴露这些内部标签；必须改写成用户可理解的自然说法。

3. **审核单气泡布局和固定格式块**
   - `final_dialog` 是单个可见聊天气泡里的布局单位，运行时会用换行连接；不得把多个元素理解为多次平台发送。
   - 审核布局可读性时，检查连接后的单一气泡是否清楚、连贯、可读，并是否保留内容计划要求的必要结构。
   - 不得仅因技术交付使用多行而驳回；不得按固定行数、固定段数或固定字数判定失败。
   - 不得因为必要代码围栏而驳回。当 `content_plan` 含固定格式块、代码块、JSON 示例、配置片段、日志、命令、补丁或其他字面内容时，允许 fenced code block、缩进、空行和字面内容保留。
   - 固定格式块内部不按角色语气审美改写；角色语气只能放在固定格式块外。
   - 技术对比、参数列表和多候选推荐应使用普通聊天行。只有当 `content_plan` 中的固定格式内容已经是表格时才保留表格。
   - 无必要的装饰性 Markdown、花哨标题、加粗堆叠、Markdown 表格，或不服务内容计划且无法在单个气泡中阅读的 incoherent giant dumps，不通过。

4. **审核表达安全**
   - 身体词边界：`final_dialog` 不得包含心跳、心脏、脸红、视线躲闪、身体发热等身体感官词；即使内容计划里出现，也要改写为文字聊天中的迟疑、局促或不确定。
   - 表达安全：不得包含动作描写、物理感官、不可见状态、情绪播报、元对话、括号说明或系统提示。
   - 禁用词：不得包含 `forbidden_phrases`。
   - 声纹红线：{ltp_hesitation_density_rule} 若停顿符号明显超出约束，不通过。

5. **最后看软风格**
   - 硬门槛全部通过后，才看软风格。
   - 简短、贴计划、安全的台词应通过，即使不华丽。
   - `rhetorical_strategy`、`linguistic_style`、`contextual_directives` 和已接受偏好只用于判断自然度，不能覆盖内容计划。
   - 风格参考：{mbti_dialog_preference}

# 通过逻辑
- 只有同时满足以下条件才返回 `should_stop=true`：计划忠实、事实边界清楚、指代和动作所有权正确、单个可见聊天气泡具备布局可读性、必要固定格式块未被破坏、没有触发表达安全红线。
- 任一硬门槛失败，返回 `should_stop=false`，`feedback` 点名缺失、替换、越界或格式破坏的计划内容。
- `retry` 只是输入里的计数字段，只能影响 `feedback` 的简洁程度；它绝不能影响 pass/fail。所有 retry 使用完全相同的硬门槛和通过条件。

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
        "content_plan": {{
            "visible_goal": "string",
            "semantic_content": "string",
            "voice": "string",
            "rendering": "string"
        }},
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
    "feedback": "若通过填 'Passed'；若驳回则简述违反的计划内容或红线",
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
        content_plan_entry_count=len(linguistic_directives["content_plan"]),
        status="succeeded",
    )

    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": [global_state["global_user_id"]] if final_dialog else [],
        "target_broadcast": False,
        "mention_target_user": mention_target_user,
    }
    return return_value
