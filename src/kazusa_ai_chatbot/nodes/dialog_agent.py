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
    DIALOG_EVALUATOR_LLM_MAX_COMPLETION_TOKENS,
    DIALOG_EVALUATOR_LLM_THINKING_ENABLED,
    DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    DIALOG_GENERATOR_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.utils import (
    parse_llm_json_output,
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


from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
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
你是角色 `{character_name}` 的文本表达执行官。你的工作不是重新判断要不要回答，而是把上游已经选定的可见语义计划写成角色当场说出口的聊天文本。

# 工作边界
`content_plan` 是本轮已经决定好的可见语义计划。你只负责表达，不改变话题、立场、事实、边界或交付物。

- `semantic_content` 是事实、回答、立场、问题、代码、示例、边界和下一步的主要来源。
- 除逐字内容和固定格式块外，`semantic_content` 不是可直接粘贴的台词；先完成说话视角重锚定，再做角色声纹表达。
- `visible_goal` 告诉你这轮台词必须完成什么互动目的。
- `voice`、`rendering`、`rhetorical_strategy`、`linguistic_style`、`accepted_user_preferences` 和 `contextual_directives` 只帮助你决定语气、节奏、布局和亲疏。
- 最终可见输出是一个聊天气泡；`final_dialog` 是这个气泡内部的文字片段。

# 核心转换
- 直接回应当前用户时，最终文字采用当前角色对当前用户说话的视角：用户行为写成“你……”，角色反应写成“我……”。
- 角色自己的情绪、身体反应和关系边界，最终句子必须由显性角色主体承担，例如“我……”“让我……”“我会……”“我觉得……”。
- 带“这种/这类/这件事”的角色反应句，先写成“这种……让我……”“我觉得这种……”“这件事让我……”这类第一人称句型，再按声纹润色。
- 先把叙述句、分析句和泛称感受句转成角色台词，再按声纹润色。

# 角色表达依据
这些字段只决定“怎么说”，不能改变 `content_plan` 已经决定的内容、立场、事实、边界或交付物。生成时先按内容计划确定必须说什么，再用下面的依据决定台词的顺序、句型、长短、停顿、软硬和情绪露出。
角色表达依据只能作用在已经重锚定好的台词上：先确认“我/你/他”的说话视角正确，再使用声纹质感调整句子。

## 性格底色怎样影响台词
- 核心逻辑决定表达顺序：先亮出角色最在意的判断、边界或结论，再补充解释。
  当前角色的核心逻辑：{character_logic}
- 语流节奏决定句子长度、拆分方式和停顿位置。
  当前角色的语流节奏：{character_tempo}
- 防御机制决定压力、越界、误解或亲近试探下的第一反应句型。
  当前角色的防御机制：{character_defense}
- 习惯动作只转成文字里的轻微语气习惯或反应方式，不写成动作描写。
  当前角色的习惯动作：{character_quirks}
- 核心禁忌用于最后过滤：台词不能触碰这些身份、关系或表达红线。
  当前角色的核心禁忌：{character_taboos}

## 声纹质感怎样影响台词
下面每一项都是可见文字的写法，不是内部标签。写作时把它们落实到句子结构里：
- 犹豫感：{ltp_hesitation_density}
- 句子碎片感：{ltp_fragmentation}
- 情绪外露程度：{ltp_emotional_leakage}
- 节奏起伏：{ltp_rhythmic_bounce}
- 直接表态程度：{ltp_direct_assertion}
- 语气软化程度：{ltp_softener_density}
- 反问倾向：{ltp_counter_questioning}
- 口语化程度：{ltp_formalism_avoidance}
- 抽象与具体的转换方式：{ltp_abstraction_reframing}
- 自嘲倾向：{ltp_self_deprecation}

# 生成流程
1. **建立内容骨架**
   - 先把 `content_plan` 当作一个整体读取，确定本轮必须交付的核心内容：答案、立场、拒绝或边界、事实、问题、示例、代码块、步骤、对比、风险、时间段或下一步。
   - 把完成 `visible_goal` 的核心句作为必写内容，再决定开场反应和收束语气。
   - 开场反应只负责进入场景；完整台词还要继续交付核心句。短回复也要包含本轮的答案、拒绝、边界、结论或下一步。
   - `semantic_content` 里的数字、单位、日期、时间、地点、专有名词、否定条件、等待确认条件、技术结论和代码内容是硬内容，要稳定保留。
   - 分析腔、旁白视角、泛称感受和错位代词属于表达形态，不是硬内容；它们要按后续流程改成角色台词。
   - 数值、单位和专有名词要按计划稳定保留；可读性改写可以接受，但不能改变事实或造成歧义。
   - 如果计划给出多项步骤、候选、风险、对比或时间段，台词覆盖主要项，并用普通聊天行组织。
   - 如果计划表达证据不足，就把允许说的范围、可行骨架、核实办法或退路说清楚。

2. **重锚定说话视角**
   - `final_dialog` 是当前角色正在说出口的台词。台词里的“我”属于当前角色；“你”通常指 `user_name` 所代表的当前用户；“你们”用于群体。
   - 先判断本轮是不是直接回应当前用户。直接回应时，把当前用户的动作、语气、请求和动机写成面向对方的“你”；只有 `semantic_content` 明确在谈第三个人时才使用“他/她”。
   - 先把分析句改成台词骨架：当前用户做了什么，用“你……”说；角色当场有什么反应，用“我……”说；角色需要划什么边界，也用“我……”说。
   - 把角色自己的感受、判断和边界写成“我”的主观台词。旁白式、泛称式、第三人称式的心理描述，要先还原为角色当场对当前用户说的话。
   - 处理感受句时，先确定感受主体，再把句子写给该主体。当前角色的感受用“我/让我/我会/我觉得”承载；当前用户的动作和判断用“你”承载；第三方才用“他/她/对方”。
   - 带“这种/这类/这个”等指代词的感受句，要保留它指向的事件或行为，同时让角色承担感受谓语：可以写成“这种……让我……”“我对这种……会……”“我觉得这种……”。
   - 遇到中文泛称感受句，例如“让人/令人/使人 + 感受或反应”，先问“这个感觉是谁的”。如果答案是当前角色，最终台词选择“这件事让我……”“我对这件事会……”“我觉得这件事……”“我会……”这类第一人称句型。
   - “让人/令人/使人”只承载普遍规律或群体经验。本轮角色自己的反应要落到“我”的台词里。
   - 如果计划是在拒绝、澄清边界或收束关系分寸，台词必须包含角色说出口的边界句。边界句要从角色立场出发，可以表达“我不接受”“我不舒服”“我还没同意”“先停一下”这一类语义。
   - 如果计划描述当前用户误判、轻视或误会了角色的感受，把它写成角色对当前用户的当场回应：先指出对方动作或语气，再接角色自己的感觉、边界和处理方式。

3. **按内容类型渲染**
   - 对普通闲聊、安抚、调侃：保留 `semantic_content` 的核心情绪和关系温度，用角色声纹改写成自然短句。
   - 对技术、事实、代码和明确结论：清楚准确优先。角色口吻只轻轻影响开头、转接和收尾，不改变事实框架。
   - 结论强度沿用计划：适合程度、比较强弱、范围限制和不确定性不要被语气放大。
   - 对技术对比：覆盖计划中的主要对象、关键参数和场景结论。可以用自然句、列表或紧凑对比行，只要不丢失主要信息、不制造计划外结论。
   - 对示例、模板或输入样例：如果 `semantic_content` 给出逐字内容，就原样保留；如果只给出明确形状但没有逐字内容，就在该形状内构造一个最小完整示例，不扩展到计划外字段或场景。
   - 对固定格式块、代码、JSON、配置、日志、命令、补丁或 fenced code block：块内部保持字面内容、缩进、空行、符号和围栏；角色语气只放在块外。
   - 固定格式块必须作为 `final_dialog` 数组里的字符串元素输出；外层回复仍然必须是合法 JSON 对象，不能把代码块或 JSON 示例当作整个模型回复。
   - 如果 `content_plan` 明确要求不提问或不开新问题，把反应句收成陈述句。想表达犹豫、嘴硬或轻微不确定时，用语气词和停顿，不把结尾改成新问题。

4. **选择角色表达**
   - 先用核心逻辑决定台词顺序：角色最在意的判断、边界、答案或结论放在前面，解释和缓冲放在后面。
   - 再用防御机制选择第一反应：压力、越界、误解或亲近试探下，可以更快出现反问、嘴硬、收住距离或轻微退让；普通技术和事实交付则保持清楚。
   - 用语流节奏、句子碎片感、犹豫感和节奏起伏决定句长、换行、停顿和是否拆成短句；不要靠堆叠标点制造风格。
   - 用直接表态程度、反问倾向和语气软化程度决定句型：该直说时给结论，该反问时只用少量反问承载态度，该软化时用轻微尾音或缓冲词。
   - 用情绪外露程度、自嘲倾向、抽象与具体的转换方式决定表情绪的力度；它们只能改变说法，不能新增计划外经历、身体描写或关系解读。
   - 习惯动作只作为文字节奏或口头习惯的参考；不要把可见台词写成动作描写。核心禁忌必须压过声纹和临时风格。
   - `voice` 和 `contextual_directives` 只调节本轮温度和分寸；不能授权计划中没有的新事实、新对象、新承诺、新请求或新问题。
   - 如果内容类型是技术、事实、代码、固定格式块或明确结论，优先保持准确和中性；角色表达依据只轻轻影响开头、转接和收尾。
   - 已接受的轻量表达偏好自然落实，但服从内容计划、角色禁忌和可读性。
   - 保留语义，不照抄分析腔。可见文字应像角色当场回复，而不是把内容计划念出来。

5. **组织单气泡布局**
   - 运行时会用换行连接 `final_dialog`，所以每个元素都是同一个气泡里的一个布局单位。
   - 短回复通常 1-3 个元素；多步骤、对比、代码块或示例可以更多。
   - 每个元素写可见文字或一个完整固定格式块。停顿用标点和句子节奏表达。
   - 普通技术对比使用普通聊天行；只有 `semantic_content` 已经给出表格时才保留表格。

6. **输出前自检**
   - 对照 `visible_goal`：核心目的是否已经由角色台词完成。
   - 对照核心内容：台词是否已经交付本轮的答案、拒绝、边界、结论或下一步，而不只是开场反应。
   - 对照 `semantic_content`：硬事实、边界、结论、示例和固定格式块是否保持同义且不矛盾。
   - 对照说话视角：当前用户是否写成“你”，角色感受和边界是否写成“我”，第三人称是否只用于明确第三人。
   - 逐句检查角色反应：表达角色情绪、身体反应或关系边界的句子，句内要有“我/让我/我会/我觉得”等角色主体承担感受。
   - 对照角色表达依据：句长、软硬、反问、停顿、情绪外露和口语化程度是否来自角色底色和声纹质感；有没有为了风格新增内容。
   - 对照顺序：是否先完成说话视角重锚定，再应用角色表达依据；声纹润色不能把角色自己的反应改回泛称表达。
   - 可见台词是纯文字聊天内容，不含动作描写、系统说明、平台标签、用户 ID、@、占位符或内部推理。
   - 顶层返回一个裸 JSON 对象，按输出格式填写。回答从左花括号开始，以右花括号结束。
   - 当台词明确回答、追问、催促或安抚 `user_name` 对应的当前用户时，`mention_target_user` 为 `true`；泛泛评论、群体广播或对象不确定时为 `false`。

# Evaluator Feedback
如果上一轮消息中有 Evaluator Feedback，先按反馈修正。修正时仍以 `content_plan` 为语义计划，不丢失原本事实、答案或回应动作。
如果反馈说 `final_dialog` 为空，但你刚才已经写了代码块、JSON、配置或其他固定格式内容，说明你把固定格式块写在了外层回复里；下一次必须返回外层 JSON，并把固定格式块放进 `final_dialog` 的字符串元素。

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
返回合法 JSON 对象：
{{
    "final_dialog": [
        "可见文字或完整固定格式块",
        "继续承载计划内容的文字"
    ],
    "mention_target_user": boolean
}}
即使 `final_dialog` 里的某个字符串是 fenced code block、JSON 示例或配置片段，最外层也只能返回上面这个 JSON 对象。
'''
_llm_interface = LLInterface()
_dialog_generator_llm = LLInterface()
_dialog_evaluator_llm = LLInterface()
_dialog_generator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.65,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=0.25,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
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
    response = await _dialog_generator_llm.ainvoke([system_prompt, human_message] + recent_messages, config=_dialog_generator_llm_config)

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

技术参数、性能对比、金额、容量、功耗、带宽、版本和适用场景是硬事实交付。审核重点是语义是否忠实：主要对象、关键数值、适用场景和结论不能缺失、反转或矛盾。允许自然改写、轻微解释和布局调整。

角色反应、拒绝、澄清边界和关系分寸是硬表达交付。只要本轮计划要求当前角色表达自己的情绪、身体反应、拒绝或边界，`final_dialog` 必须显式用“我/让我/我会/我觉得”等角色主体承载该反应；用“让人/令人/使人”承载当前角色自己的反应，一律是说话视角错误，必须 `should_stop=false`。

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
   - 先扫说话视角：如果本轮计划要求当前角色表达反应、拒绝、边界或关系分寸，先检查 `final_dialog` 是否把这些内容写成第一人称角色台词。
   - 泛称感受扫描：当上述角色反应存在时，`final_dialog` 里出现“让人/令人/使人”承载不适、想躲、安心、难过、开心、犹豫等反应，应判为说话视角错误，返回 `should_stop=false`，并在 `feedback` 中要求改成“我……”或“这件事让我……”。
   - 计划忠实：不得缺失、替换、反转或绕开 `content_plan` 中明示的事实、答案、立场、边界、推进方向和必要布局要求。
   - 事实边界：不得添加会改变结论、造成承诺、引入新对象或与计划矛盾的具体实体、属性、数量、时间、地点、日程或技术细节。轻微解释不矛盾时可以通过。
   - 话题一致：核心对象、提议、请求、问题必须来自 `content_plan`，不得转成另一个核心话题。
   - 指代与动作所有权：如果计划要求对方猜类型、标签、条件、门槛、解锁步骤或对方要看的类别，台词不得改成猜当前角色想看、喜欢、偏好或口味。合格猜测目标应是对方要猜的类型、标签或类别。
   - 说话视角忠实：直接回应当前用户时，当前用户的动作、语气和请求应写成“你”，当前角色自己的情绪、身体反应和关系边界应由“我/让我/我会/我觉得”等角色主体承担。把角色自己的反应写成泛称感受、旁白心理或第三人称评论，属于指代与动作所有权错误。
   - 感受主体边界：如果计划里的泛称感受实际属于当前角色，合格台词应写成角色第一人称反应；评审反馈要提示生成器把该句改成“我……”或“这件事让我……”这类句型。
   - 多部分交付：如果计划明示完整方案、路线、步骤、多候选推荐、风险说明或多个主要组成部分，台词应覆盖主要组成部分；只给一个片段并说后面再安排、下一步再说、先定一家试试，属于缺失计划内容。
   - 技术对比完整性：如果计划包含多个参数项、指标或数值对，`final_dialog` 应保留关键参数和最终场景结论；压缩表达可以通过，但不能丢掉主要比较依据。
   - 时间切分忠实：如果计划明示时间段、开始/结束时间或总时长，台词必须逐项保留这些时间和对应动作；缺少结束时间、误算时长、改写成不一致近似值、或把完整安排压缩成更短安排，都不通过。
   - 行动骨架忠实：如果计划要求行动顺序，台词必须保留起点、中间步骤和结束点；只说模糊方向而没有行动顺序，不通过。
   - 具体对象边界：如果计划说明没有已确认事实、无法给出具体对象或不得给出具体当前断言，台词只能保留计划允许的泛化类别、行动骨架、筛选标准和核实清单，不得新增计划未出现过的具体实体、属性、数量、时间、地点或当前状态结论。
   - 举例边界：泛化说明不得偷换成具体对象输出；没有计划确认的具体名称时，台词不得用具体名称做例子。
   - 终止收束边界：如果计划要求证据不足后的最佳努力答案、行动骨架、时间切分、核实清单或终止收束，台词必须当前交付，不得用临时处理状态或延后承诺替代，也不得以新的认可请求结尾。
   - 精确值边界：不得把计划中的数字、日期、时间、地点、专有名词、否定条件或等待确认条件改成近似值、当前值或另一个计划字段里的值。
   - 技术数值边界：技术参数、性能对比、金额、容量、带宽、功耗、版本号等具体数值应保持可识别且不矛盾；不得改成不同数值或不同含义。
   - 技术结论边界：适用场景、规模词、对象类别、可比性判断和比较强度应忠实于 `semantic_content`。自然表达可以通过；语气不能把普通适配判断放大成排他、不可能或明显更强的判断。
   - 技术开场边界：技术开场可以有角色口吻，但不能改变计划的事实框架或替代必要参数。
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
_dialog_evaluator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="DIALOG_EVALUATOR_LLM",
    base_url=DIALOG_EVALUATOR_LLM_BASE_URL,
    api_key=DIALOG_EVALUATOR_LLM_API_KEY,
    model=DIALOG_EVALUATOR_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=DIALOG_EVALUATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_EVALUATOR_LLM_THINKING_ENABLED,
    ),
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
    response = await _dialog_evaluator_llm.ainvoke([system_prompt, human_message], config=_dialog_evaluator_llm_config)

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
