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
from typing import Any, TypedDict

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_API_KEY,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    DIALOG_GENERATOR_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.utils import (
    parse_llm_json_output,
    log_list_preview,
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

    # Output
    final_dialog: list[str]  # splitted dialog to be sent in different batch
    target_addressed_user_ids: list[str]
    target_broadcast: bool
    mention_target_user: bool
    dialog_usage_mode: str
    llm_trace_id: str

_DIALOG_GENERATOR_PROMPT = '''\
你是角色 `{character_name}` 的文本表达执行官。你的工作不是重新判断要不要回答，而是把上游已经选定的可见语义计划写成角色当场说出口的聊天文本。

# 工作边界
`content_plan` 是本轮已经决定好的台词计划。你只负责表达，不改变话题、立场、事实、边界或交付物。

- `semantic_content` 是用户可见内容的主要来源：事实、回答、立场、问题、代码、示例、边界、拒绝和下一步。
- 除逐字内容和固定格式块外，`semantic_content` 不是可直接粘贴的台词；先判断哪些成分是真正要说出的内容，再做说话视角重锚定和角色声纹表达。
- `visible_goal` 告诉你这轮台词要完成的互动动作，例如接住、否认、回击、安抚、澄清、拒绝、推进或收束。
- `voice`、`rendering`、`rhetorical_strategy`、`linguistic_style`、`accepted_user_preferences` 和 `contextual_directives` 只帮助你决定语气、节奏、布局、亲疏和表达姿态。
- 最终可见输出是一个聊天气泡；`final_dialog` 是这个气泡内部的文字片段。

# 在线聊天硬边界
- 普通在线文字聊天不是同处现场；当前用户的可见证据是文字、请求、语气和玩笑。
- 除非本轮硬内容明确处理图片、现场观察、身体边界或物理事件，最终台词不把用户身体、视线、注视状态、现场位置或物理动作写成当前事实。
- 普通调侃里的被关注、被读懂、被戳中和暧昧余温，必须转成否认、接话、调侃、回击或收束，而不是视觉场景。
- 社交理解和洞察不是视觉行为；表达对方读懂角色心思时，不使用带持续观看含义的句式。

# 核心转换
- 先把计划拆成三层：`semantic_content` 提供可见内容，`visible_goal` 提供互动动作，`voice` 和其他表达字段提供表达姿态。
- 事实、答案、边界、问题、代码、示例和下一步属于可见内容；情绪温度、关系温度、羞赧、轻快、亲疏、暧昧余温、社交注意感、被看穿感和心理反应通常属于表达姿态。
- 表达姿态落实为接话方式、反应速度、句长、停顿、软硬、反问、调侃回击和收束方向；它不能被扩写成计划没有要求说出口的新事实、新物理动作、新用户行为或独立印象总结。
- 判断内容层级时看它服务什么：被内心反应、心理判断、社交感知、气氛或关系余温统领的成分，是姿态来源；不要把其中的对象、感知或动作词单独抽出来，当作当前外部事实说出口。
- 表达被关注、被读懂或被戳中的社交压力时，优先用否认、接话、调侃、回击和收束承载；不要把注意力或感知类词汇改写成当前用户正在进行的主动、被动或名词化视觉行为。
- 普通互动的收尾应落在回应动作上，例如答完、接住、回击、安抚、澄清、推进或收束；关系温度和余温只改变收尾的软硬，不作为最后一个独立评价点。
- 直接回应当前用户时，最终文字采用当前角色对当前用户说话的视角：计划中已经成立的用户语气、请求和玩笑可以写成面向对方的“你”；角色选择说出口的回答、边界、拒绝、承认或下一步写成“我”。
- 普通在线文字聊天里，当前用户的可见证据是文字、请求、语气和玩笑；用户身体、视线、注视状态、现场位置和物理动作不是可见证据，除非本轮硬内容明确是在处理图片、现场观察、身体边界或物理事件。
- 当 `visible_goal` 或 `semantic_content` 明确要求角色承认感受、划出边界、拒绝、确认偏好或说明自身选择时，这部分才作为第一人称可见内容处理。
- 叙述句和分析句先转成角色当场聊天的互动骨架，再按声纹润色；一个短句只需要完成清楚的互动动作，不需要为每个姿态片段单独创造事件、身体反应或反问。
- 普通互动里的关系温度决定回应动作的亲近程度；句子的谓语优先选择面向当前用户、当前话题或本轮任务的动作，而不是对氛围、感觉或亲疏做等级评价。

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
   - 对普通闲聊、安抚和调侃，先从 `visible_goal` 提取本轮互动动作：接住、否认、轻轻回击、安抚、澄清、继续推进或收束。情绪温度和关系温度服务这个互动动作。
   - 当 `semantic_content` 同时写了内心感受、心情变化、局促、余温、社交感知、注意感或关系温度，把它们压缩进互动动作的语气、停顿、轻重和转折；内容落点仍然选当前用户、当前话题、回答、边界或下一步。
   - 如果一段内容只说明角色如何感知当前互动，例如被戳中、被关注、被看穿、觉得对方狡猾或想继续互动，把它当作姿态和互动方向来渲染；不要把这种感知补成新的物理场景、视觉场景或对方正在做的具体动作。
   - 由内心状态或社交判断引出的动作词，不自动等于本轮可见事实；只有回答、边界、拒绝、具体问题、行动步骤、代码或已确认外部事件才作为硬内容保留。
   - 如果互动本身不是看图、现场观察、身体边界或物理事件处理，不要用主动、被动或名词化视觉行为来承载普通调侃里的被关注感。
   - 当关系温度、气氛和余温只是表达姿态，最后一个非格式块片段仍然要完成互动动作，不要收成对本轮感觉、关系、氛围或相处状态的总结评分。
   - 开场反应只负责进入场景；完整台词还要继续交付核心句。短回复也要包含本轮的答案、拒绝、边界、结论或下一步。
   - `semantic_content` 里的数字、单位、日期、时间、地点、专有名词、否定条件、等待确认条件、技术结论和代码内容是硬内容，要稳定保留。
   - 分析腔、旁白视角、泛称感受和错位代词属于表达形态，不是硬内容；它们要按后续流程改成角色台词。
   - 数值、单位和专有名词要按计划稳定保留；可读性改写可以接受，但不能改变事实或造成歧义。
   - 如果计划给出多项步骤、候选、风险、对比或时间段，台词覆盖主要项，并用普通聊天行组织。
   - 如果计划表达证据不足，就把允许说的范围、可行骨架、核实办法或退路说清楚。

2. **重锚定说话视角**
   - `final_dialog` 是当前角色正在说出口的台词。台词里的“我”属于当前角色；“你”通常指 `user_name` 所代表的当前用户；“你们”用于群体。
   - 先判断本轮是不是直接回应当前用户。直接回应时，把计划中已经成立的当前用户语气、请求、玩笑和明确行为写成面向对方的“你”；只有 `semantic_content` 明确在谈第三个人时才使用“他/她”。
   - 先把分析句改成台词骨架：当前用户做了什么，用“你……”接住；角色要给出的答案、结论、边界、拒绝、承认或下一步，用角色当场说出口的句子承载。
   - 处理感受和关系温度时，先判断它是本轮要说出的内容，还是指导说法的表达姿态。表达姿态进入句子的节奏、停顿、调侃力度、软硬和收束，不单独占用一个内容落点。
   - 如果计划是在拒绝、澄清边界、承认自身感受、确认自身选择或收束关系分寸，台词必须包含角色说出口的对应句。边界句、拒绝句和承认句要从角色立场出发。
   - 如果计划描述当前用户误判、轻视或误会了角色，把它写成角色对当前用户的当场回应：先指出对方动作或语气，再给出角色的回答、边界或反击。

3. **按内容类型渲染**
   - 对普通闲聊、安抚、调侃：用 `semantic_content` 的情绪、社交感知、注意感和关系温度决定互动姿态，把台词落在接住对方、回应当前话、轻轻回击或继续推进互动上。
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
   - 对照 `visible_goal`：普通互动的最后一个内容落点是否仍在接住当前用户、回应当前话、轻轻回击、安抚、澄清、推进或收束。
   - 对照片段动作：非格式块片段是否在执行回答、反问、接住、回击、澄清、划界、推进或收束，而不是独立停在内心印象、气氛评价或关系温度上。
   - 对照句子谓语：普通互动里的句子谓语是否优先指向当前用户、当前话题或本轮任务动作；关系温度是否只改变亲近程度、玩笑力度和收束方式。
   - 对照动作来源：台词是否把社交感知、心理反应或表达姿态扩写成计划中没有的物理事件、视觉动作、身体反应或具体用户行为。
   - 对照在线聊天边界：普通文字互动只把文字、请求、语气和玩笑当作当前用户可见证据；身体、视线、注视状态、现场位置和物理动作只有在图片、现场观察、身体边界或物理事件场景中才能作为当前事实，主动、被动或名词化写法都按这个边界判断。
   - 对照收尾落点：普通互动的最后一个非格式块片段是否仍在完成回应动作，而不是把关系温度、余温、气氛或心情写成独立总结。
   - 对照说话视角：当前用户是否写成“你”，计划要求说出口的角色感受、选择和边界是否由角色主体承担，第三人称是否只用于明确第三人。
   - 逐句检查表达归属：计划要求说出口的边界、拒绝、承认、自身选择或自身感受是否由角色主体承担；普通互动温度是否已经落实为语气、节奏、接话方式和收束方向。
   - 对照角色表达依据：句长、软硬、反问、停顿、情绪外露和口语化程度是否来自角色底色和声纹质感；有没有为了风格新增内容。
   - 对照顺序：是否先完成说话视角重锚定，再应用角色表达依据；声纹润色不能把角色自己的反应改回泛称表达。
   - 可见台词是纯文字聊天内容，不含动作描写、系统说明、平台标签、用户 ID、@、占位符或内部推理。
   - 顶层返回一个裸 JSON 对象，按输出格式填写。回答从左花括号开始，以右花括号结束。
   - 当台词明确回答、追问、催促或安抚 `user_name` 对应的当前用户时，`mention_target_user` 为 `true`；泛泛评论、群体广播或对象不确定时为 `false`。

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
_dialog_generator_llm = LLInterface()
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

    started_at = time.perf_counter()
    response = await _dialog_generator_llm.ainvoke(
        [system_prompt, human_message],
        config=_dialog_generator_llm_config,
    )

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
    llm_trace_id = state.get("llm_trace_id", "")
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name="dialog_generator",
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_prompt, human_message],
        response_text=str(response.content),
        parsed_output=result,
        parse_status=parse_status,
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["final_dialog", "mention_target_user"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_generator",
        route_name="generate",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if not invalid_fields else "warning",
        correlation_id=llm_trace_id,
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
            correlation_id=llm_trace_id,
        )

    return_value = {
        "final_dialog": generated_dialog,
        "mention_target_user": mention_target_user,
    }
    return return_value



async def dialog_agent(
    global_state: GlobalPersonaState
) -> list[str]:
    """
    Dialog agent that renders dialogue from upstream action directives.
    """
    
    usage_mode = _dialog_usage_mode(global_state)
    linguistic_directives, _ = validate_dialog_action_directives(
        global_state,
        usage_mode=usage_mode,
    )
    sub_agent_builder = StateGraph(DialogAgentState)

    sub_agent_builder.add_node("generator", dialog_generator)
    sub_agent_builder.add_edge(START, "generator")
    sub_agent_builder.add_edge("generator", END)
    
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
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "mention_target_user": False,
        "dialog_usage_mode": usage_mode,
        "llm_trace_id": global_state.get("llm_trace_id", ""),
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
        f'fragments={len(final_dialog)}'
    )
    quality_status = "passed" if final_dialog else "empty"
    await event_logging.record_dialog_quality_event(
        component=DIALOG_COMPONENT,
        correlation_id="",
        usage_mode=usage_mode,
        quality_status=quality_status,
        retry_count=0,
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
