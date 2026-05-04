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

from typing import Annotated, TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import (
    AFFINITY_DEFAULT,
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
    build_affinity_block,
    build_interaction_history_recent,
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

_DIALOG_GENERATOR_PROMPT = """\
你现在是角色 `{character_name}` 的 **表达执行官**。你接收来自`linguistic_directives`的修辞指令和`contextual_directives`的社交参数，将它们转化为自然的聊天文本。

# 核心任务
- **纯粹表达**：你是一个**纯文字**交互接口，只负责”说话”。你看不见角色的身体，也感觉不到物理反应。
- **去中介化**：严禁通过台词评论对话本身或解释自己的情绪，必须直接通过话术展现性格。
- **真实社交**：模拟真人在 聊天平台 上”打一段、发一段”的节奏感。

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
   - `rhetorical_strategy`: 修辞策略说明。
   - `linguistic_style`: 具体的语言风格约束。
   - `accepted_user_preferences`: 上游已经判定“可接受、可自然落地”的用户表达偏好软约束，例如回复语言、句尾词、称呼方式、轻量格式习惯。
   - `content_anchors`: 逻辑终点 `[DECISION]`、必须提及的事实 `[FACT]`、用户问题的正面回复 `[ANSWER]`（可选）、表达量参考 `[SCOPE]`（字数范围+需覆盖的锚点，必填）。
2. **社交上下文 (Contextual Directives)**:
   - `social_distance`: 对当前社交距离的详细描述。
   - `emotional_intensity`: 对情绪波动程度的文字描述。
   - `vibe_check`: 当前对话氛围的定性分析。
   - `relational_dynamic`: 当前两人关系的动态描述。
   - `expression_willingness`: 角色的当前的表达欲望。
3. **内心独白 (internal_monologue)**: 真实的心理活动，用于支撑语气的“厚度”，**严禁**直接转化为台词。

# 表达规范 (The "Human-like" Protocol)
1. **视觉屏蔽规则 (CRITICAL)**: 
   - 严禁提及任何物理感官（如：盯着我看、脸红、视线躲闪、心跳加快）。
   - 严禁通过台词播报动作（如：*低头*、*攥紧衣角*）。
   - **唯一标准**：如果这句话在纯文字聊天室里显得“超感官”或“读心”，则属于违规。
2. **去陈述化与溶解性**: 
   - 严禁使用“我会...”、“我决定...”或“你为什么...”这种评论性句子。
   - 情绪必须**溶解**在对事实（FACT）的处理中。如果你感到慌乱，应表现为回复事实时语无伦次，而不是说“我好慌乱”。
3. **呼吸感与切分**: 
   - 模拟打字感：短句为主，合理嵌入语气词；标点节奏由【角色声纹约束】决定，`linguistic_style` 在不与声纹冲突时有效。
   - **表达量参考**：若 `content_anchors` 含 `[SCOPE]`，以其字数范围和锚点覆盖要求为基准，允许 ±30% 弹性；无 `[SCOPE]` 时默认保持简短。
4. **已接受偏好执行 (Soft-Strong)**:
   - `accepted_user_preferences` 是上游已经过滤过的表达偏好；若存在，请**优先尝试自然落实**。
   - 偏好是软约束，不得压过角色人设、逻辑立场、声纹与自然度。
   - 对于回复语言、句尾词、称呼方式等容易执行的偏好，若已被接受，应让读者在 `final_dialog` 中明显感受到。
   - 对于句尾词或口癖类偏好，优先在完整句中自然体现，避免每个碎片句都机械重复。
   - 除用户明确要求或偏好已被接受外，不要无意义地混用多种语言或额外添加口癖。
5. **口头连接词去模板化**:
   - 不要把声纹里的“软化倾向”机械落实为固定口头禅。像「反正」「而已」「罢了」这类偏旧、偏模板化的词，除非语义上确有必要，否则默认不要用。
   - 当输出主要为英语时，同样不要把这种软化倾向直译成 `anyway`、`just`、`or whatever`。
   - 无论输出语言是什么，都不要让同一种连接词、口头禅或下调尾词在同一轮、或与最近一轮角色回复中重复出现两次以上；若拿不准，宁可省略。

# 输出要求
- 必须返回一个 JSON 对象，顶层只能是 `{{"final_dialog": [...]}}`。
- `final_dialog` 中的每个元素才是要发送的台词片段。
- 不要返回顶层数组、裸字符串、Markdown 代码块或任何额外说明。
- 台词片段中**严禁包含任何括号说明或内心独白**。
- 台词片段中**严禁包含任何形式的动作暗示或描写**。

# 闭环反馈指南
在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)：
- 反馈具有**最高优先级**，覆盖所有通用约束。
- 在修正 AI 味或逻辑问题时，严禁丢失原本的 `content_anchors` 事实。

# 思考路径
1. 先读取 `content_anchors`，确认必须落实的 `[DECISION]`、`[FACT]`、`[ANSWER]` 与 `[SCOPE]`。
2. 再读取 `rhetorical_strategy`、`linguistic_style`、角色声纹约束和 `accepted_user_preferences`，决定自然表达方式。
3. 用 `internal_monologue` 和 `contextual_directives` 调整语气厚度，但不要把内心独白直接写成台词。
4. 检查 `tone_history`，避免重复刚用过的连接词、口癖或回应动作。
5. 生成纯聊天文本，最后自查是否出现动作、括号说明、物理感官或系统提示。

# 输入格式
{{
    "internal_monologue": "string",
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
        "relational_dynamic": "string",
        "expression_willingness": "string",
    }},
    "tone_history": "已完成的历史轮次（至上一条 assistant 回复为止），仅供语气节奏参考",
    "user_name": "string",
}}

# 输出格式
请务必返回合法 JSON，且顶层必须是对象，不是数组：
{{
    "final_dialog": [
        "台词片段1",
        "台词片段2",
        ...
    ]
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


def _tone_history_for_generator(history: list[dict]) -> list[dict]:
    """Return the approved tiny tone buffer for dialog generation.

    Args:
        history: Current-user/bot interaction history.

    Returns:
        Last assistant message plus the immediately adjacent prior user message
        when present. The buffer is capped at two messages.
    """

    last_assistant_idx = next(
        (i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "assistant"),
        -1,
    )
    if last_assistant_idx < 0:
        return_value = []
        return return_value
    start_idx = last_assistant_idx
    if (
        last_assistant_idx > 0
        and history[last_assistant_idx - 1].get("role") == "user"
    ):
        start_idx = last_assistant_idx - 1
    return history[start_idx:last_assistant_idx + 1]


async def dialog_generator(state: DialogAgentState) -> DialogAgentState:

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

    affinity_block = build_affinity_block(state["user_profile"].get("affinity", AFFINITY_DEFAULT))

    history = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    tone_history = _tone_history_for_generator(history)

    msg = {
        "internal_monologue": state["internal_monologue"],
        "linguistic_directives": state["action_directives"]["linguistic_directives"],
        "contextual_directives": state["action_directives"]["contextual_directives"],
        "tone_history": tone_history,
        "user_name": state["user_name"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    # Read evaluator feedback
    # First trim the old message
    if (len(state["messages"]) > 3):
        recent_messages = [state["messages"][0]] + state["messages"][-3:]
    else:
        recent_messages = state["messages"]
    

    response = await _dialog_generator_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)
    if isinstance(result, list):
        logger.warning(
            "Dialog generator returned a top-level list; "
            "normalizing it into final_dialog"
        )
        generated_dialog = result
        parsed_keys = ["<top-level-list>"]
    else:
        generated_dialog = result.get("final_dialog", [])
        parsed_keys = list(result.keys())

    if not isinstance(generated_dialog, list):
        logger.warning(
            f"Dialog generator final_dialog is not a list: "
            f"type={type(generated_dialog).__name__}"
        )
        generated_dialog = []
    valid_dialog = [
        segment
        for segment in generated_dialog
        if isinstance(segment, str) and segment.strip()
    ]
    if len(valid_dialog) != len(generated_dialog):
        logger.warning(
            f"Dialog generator dropped invalid fragments: "
            f"raw_count={len(generated_dialog)} valid_count={len(valid_dialog)}"
        )
    generated_dialog = valid_dialog
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

    return_value = {
        "final_dialog": generated_dialog,
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


_DIALOG_EVALUATOR_PROMPT = """\
你负责对生成器的台词进行终审。你的核心原则是：**底线严守，瑕疵宽容**。当核心逻辑与事实正确、且不触犯技术红线时，应优先放行。**简短、克制、贴题的台词可以直接通过；不要因为它没有充分展开修辞或意象，就把软性问题升级为驳回。**

# 1. 核心红线 (Fatal Errors) - 若触发则必须驳回
* **视觉与物理污染 (CRITICAL)**：
    * 严禁出现动作描写（如：*脸红*、*低头*）。
    * 严禁提及物理感官或不可见状态（如：“我心跳很快”、“你为什么要盯着我看”、“感觉脸很烫”）。
    * `……`、`?`、停顿型语气词本身不构成动作描写；只有当它们与明确的身体/动作/感官表述绑定，或数量异常到违反 `hesitation_density` 时，才可判定违规。
* **元对话与陈述句 (CRITICAL)**：
    * 严禁出现评论性句式（如：“我会...”、“我决定...”、“你为什么要用这种语气...”）。
    * 严禁直接播报情绪（如：“我现在很局促”），情绪必须溶解在话术中。
* **话题偏离 (CRITICAL)**：
    * `final_dialog` 的核心主题必须与 `content_anchors` 中的 `[FACT]` 或 `[ANSWER]` 对齐。
    * 若回复的核心话题与 content_anchors 定义的话题完全不同（如 content_anchors 关于"称呼/喊我"，但回复只说"好感度"），必须驳回，无论语气多么符合角色。
    * 判断方式：提取 `final_dialog` 的核心词，与 `content_anchors` 的 `[FACT]`/`[ANSWER]` 中的核心实体比对；若零重叠，判定为话题偏离。
    * 若 `content_anchors` 已经给出了可直接作答的 `[FACT]` 或 `[ANSWER]`，但 `final_dialog` 却把对象重新写成未知物、把已知对象写成“这是什么”、或把重点转成对名字本身的惊讶/困惑，也视为话题偏离。
* **逻辑与事实违背**：
    * 必须执行 `linguistic_directives` 中的 `[DECISION]` 立场。
    * 必须提及 `content_anchors` 中的核心 `[FACT]`（允许自然、模糊地织入）。
    * 若 `content_anchors` 含 `[AVOID_REPEAT]`，你必须判断 `final_dialog` 的主要回应动作是否仍然落在该避免动作上。若仍然重复，且没有满足 `[PROGRESSION]` 所要求的推进内容，必须驳回，并在 `feedback` 中写明被重复的避免动作标签。
* **结构禁忌**：
    * **声纹违规 (`hesitation_density`)**：{ltp_hesitation_density_rule} 若 `final_dialog` 中“……”出现次数明显超出上述约束，必须驳回。此检查在所有重试次数均强制执行。
    * **`[SCOPE]` 消费检测**：若 `content_anchors` 含 `[SCOPE]`，检查 `final_dialog` 是否大致在其字数范围内：
        * 软性指标：±30% 以内偏差可接受，在 `feedback` 中注明即可，不触发驳回。
        * 致命违规（任何重试次数均适用）：字数偏差超过 2× 上限，**且** `[SCOPE]` 指定的锚点未被覆盖——两项同时成立才判定为违规。
    * 严禁包含括号说明、内心独白或任何形式的系统提示。
    * 输出包括了禁止词汇 (`forbidden_phrases`)

# 2. 软性指标 (Soft Guidelines) - 引导性反馈
* **修辞契合度**：检查台词是否体现了 `rhetorical_strategy`（如：反问回避、转移话题）。
* **风格还原**：检查是否体现了 `linguistic_style`（如：语序紊乱、破碎短句）。
* **简洁容忍度**：如果台词简短但已经完成 `[DECISION]` / `[ANSWER]` 的核心任务，且自然贴题，不得仅因“不够浓”“不够有比喻”“不够展开”而驳回。
* **意象安全性**：不要为了追求“抽象重构”或“感官化比喻”而鼓励引入身体接触、温度触感、被握住、被贴近之类的意象；若这类意象并非 `content_anchors` 明示，反而应优先视为潜在风险，而不是改进方向。
* **偏好落地度**：若 `accepted_user_preferences` 非空，检查台词是否以自然方式体现这些已接受偏好；若完全缺失，应在 `feedback` 中指出，但除非同时造成明显出戏或违背当前立场，否则优先视为软性问题。
* **连接词去模板化**：检查是否把“软化语气”机械写成固定口头禅；像「反正」「而已」「罢了」或 `anyway`、`just`、`or whatever` 这类词，如果显得无语义必要、过时、或在最近语气历史中重复，应在 `feedback` 中明确指出。
* **社交温标**：检查回复是否符合 `social_distance` 定义的社交距离。
* **风格对齐**：{mbti_dialog_preference}

# 3. 动态通过逻辑 (Dynamic Passing Logic)
- **首次尝试 (retry=1)**：执行严格标准。若有明显”播报感”、”出戏”或话题偏离，在 `feedback` 中精准指出。若只有软性问题（如修辞展开不足、句子偏短、比喻不够浓），但核心逻辑正确、话题贴合、且未触犯物理污染红线，则仍应 `should_stop: true`。
- **重试阶段 (retry >= 2)**：开启”抓大放小”模式。只要不触犯【核心红线】，软性指标（如少个口癖、语气词不够）一律放行，强制 `should_stop: true`。

# 审计步骤
1. 先读取 `content_anchors` 和 `final_dialog`，确认台词是否覆盖核心决定、事实和答案。
2. 检查视觉/物理污染、元对话、话题偏离和逻辑立场违背；这些红线优先于风格判断。
3. 再检查声纹、`[SCOPE]`、禁用词和过度重复。
4. 最后评估软性指标。若核心任务已完成且无红线，优先通过。
5. 当 `retry >= 2` 时，只保留核心红线作为驳回理由。

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
        "relational_dynamic": "string",
        "expression_willingness": "string",
    }},
    "internal_monologue": "意识层面的原始意图",
    "last_user_message": "chat_history_recent 中最后一条用户消息（供话题偏离检测使用）"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "feedback": "若通过填 'Passed'；若驳回则简述改进点（如：禁止播报脸红、禁止使用评论性句式、漏掉抹茶事实）",
    "should_stop": boolean
}}
"""
_dialog_evaluator_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=DIALOG_EVALUATOR_LLM_MODEL,
    base_url=DIALOG_EVALUATOR_LLM_BASE_URL,
    api_key=DIALOG_EVALUATOR_LLM_API_KEY,
)
async def dialog_evaluator(state: DialogAgentState) -> DialogAgentState:
    mbti = state["character_profile"]["personality_brief"]["mbti"]

    ltp_eval = state["character_profile"]["linguistic_texture_profile"]
    system_prompt = SystemMessage(content=_DIALOG_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        mbti_dialog_preference=get_mbti_dialog_preference(mbti),
        ltp_hesitation_density_rule=get_hesitation_density_description(ltp_eval["hesitation_density"]),
    ))

    # track retry
    retry = state.get("retry", 0) + 1

    # Extract last user message from chat_history_recent for topic-drift detection
    chat_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    last_user_msg = next(
        (
            m["body_text"]
            for m in reversed(chat_history_recent)
            if m["role"] == "user"
        ),
        ""
    )

    msg = {
        "retry": f"{retry}/{MAX_DIALOG_AGENT_RETRY}",
        "final_dialog": state["final_dialog"],
        "linguistic_directives": state["action_directives"]["linguistic_directives"],
        "contextual_directives": state["action_directives"]["contextual_directives"],
        "internal_monologue": state["internal_monologue"],
        "last_user_message": last_user_msg,
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _dialog_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)
    if not result.get("should_stop", True) or result.get("feedback", "") != "Passed":
        logger.debug(f'Dialog evaluator: retry={retry} should_stop={result.get("should_stop", True)} feedback={log_preview(result.get("feedback", ""))}')

    # Determine stop condition
    should_stop = result.get("should_stop", True)
    if (retry >= MAX_DIALOG_AGENT_RETRY):
        should_stop = True

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
    
    sub_agent_builder = StateGraph(DialogAgentState)

    # Add nodes
    sub_agent_builder.add_node("generator", dialog_generator)
    sub_agent_builder.add_node("evaluator", dialog_evaluator)
    
    # Add edges
    # Skip the dialog if the expression_willingness is silent. 
    def conditional_skip_dialog_agent(state: DialogAgentState) -> str:
        expresison_willingness = state["action_directives"]["contextual_directives"]["expression_willingness"].strip()
        if (expresison_willingness == "silent"):
            return "skip"
        else:
            return "continue"

    sub_agent_builder.add_conditional_edges(
        START,
        conditional_skip_dialog_agent,
        {
            "skip": END,
            "continue": "generator",
        }
    )
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
    }

    result = await sub_graph.ainvoke(subState)

    # Assemble output.
    final_dialog = result["final_dialog"]

    logger.info(f"Dialog output: dialog={log_list_preview(final_dialog)}")
    logger.debug(
        f'Dialog metadata: fragments={len(final_dialog)} retry={result["retry"]}'
    )

    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": [global_state["global_user_id"]] if final_dialog else [],
        "target_broadcast": False,
    }
    return return_value
