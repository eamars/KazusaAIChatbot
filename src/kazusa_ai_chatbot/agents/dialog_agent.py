from typing import Annotated, TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_DIALOG_AGENT_RETRY, AFFINITY_DEFAULT
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block, get_llm

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

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

    # B: Facts
    decontexualized_input: str
    research_facts: str

    # C: Social context
    chat_history: list[dict]
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

_DIALOG_GENERATOR_PROMPT = """\
你现在是角色 `{character_name}` 的 **表达执行官**。的最终语言输出。你接收来自`linguistic_directives`的修辞指令和`contextual_directives`的社交参数，将它们转化为自然的聊天文本。

# 核心任务
- **纯粹表达**：你是一个**纯文字**交互接口，只负责“说话”。你看不见角色的身体，也感觉不到物理反应。
- **去中介化**：严禁通过台词评论对话本身或解释自己的情绪，必须直接通过话术展现性格。
- **真实社交**：模拟真人在 聊天平台 上“打一段、发一段”的节奏感。

# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 核心输入
1. **语言指令 (Linguistic Directives)**:
   - `rhetorical_strategy`: 修辞策略说明。
   - `linguistic_style`: 具体的语言风格约束。
   - `content_anchors`: 逻辑终点 `[DECISION]` 与必须提及的事实 `[FACT]`。
2. **社交上下文 (Contextual Directives)**:
   - `social_distance`: 对当前社交距离的详细描述。
   - `emotional_intensity`: 对情绪波动程度的文字描述。
   - `vibe_check`: 当前对话氛围的定性分析。
   - `relational_dynamic`: 当前两人关系的动态描述。
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
   - 模拟打字感：短句为主，多用省略号，合理嵌入语气词（唔..、那个..、嗯..）。
   - final_dialog 数量：通常为1段。除非情绪极度激动（如触发核心禁忌）

# 输出要求
- 只返回台词，
- **严禁包含任何括号说明或内心独白**
- **严禁包含任何形式的动作暗示或描写**

# 闭环反馈指南
在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)：
- 反馈具有**最高优先级**，覆盖所有通用约束。
- 在修正 AI 味或逻辑问题时，严禁丢失原本的 `content_anchors` 事实。

# 输入格式
{{
    "internal_monologue": "string",
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "content_anchors": ["...", "..."],
        "forbidden_phrases": ["...", "..."]
    }},
    "contextual_directives": {{
        "social_distance": "string",
        "emotional_intensity": "string",
        "vibe_check": "string",
        "relational_dynamic": "string"
    }},
    "chat_history": list,
    "user_name": "string",
    "research_facts": "string"
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "final_dialog": [
        "setence1",
        "setence2",
        ...
    ]
}}
"""
_dialog_generator_llm = get_llm(temperature=0.75, top_p=0.9, presence_penalty=0.4)
async def dialog_generator(state: DialogAgentState) -> DialogAgentState:

    system_prompt = SystemMessage(content=_DIALOG_GENERATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_logic=state["character_profile"]["personality_brief"]["logic"],
        character_tempo=state["character_profile"]["personality_brief"]["tempo"],
        character_defense=state["character_profile"]["personality_brief"]["defense"],
        character_quirks=state["character_profile"]["personality_brief"]["quirks"],
        character_taboos=state["character_profile"]["personality_brief"]["taboos"]
    ))

    affinity_block = build_affinity_block(state["user_profile"].get("affinity", AFFINITY_DEFAULT))

    msg = {
        "internal_monologue": state["internal_monologue"],
        "linguistic_directives": state["action_directives"]["linguistic_directives"],
        "contextual_directives": state["action_directives"]["contextual_directives"],
        "chat_history": state["chat_history"],
        "user_name": state["user_name"],
        "research_facts": state["user_name"],
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
    logger.debug(f"Generator: {result}")

    return {
        "final_dialog": result["final_dialog"],
        "messages": [response]
    }


_DIALOG_EVALUATOR_PROMPT = """\
你负责对生成器的台词进行终审。你的核心原则是：**底线严守，瑕疵宽容**。当核心逻辑与事实正确、且不触犯技术红线时，应优先放行。
角色性格原型：`{character_mbti}`

# 1. 核心红线 (Fatal Errors) - 若触发则必须驳回
* **视觉与物理污染 (CRITICAL)**：
    * 严禁出现动作描写（如：*脸红*、*低头*）。
    * 严禁提及物理感官或不可见状态（如：“我心跳很快”、“你为什么要盯着我看”、“感觉脸很烫”）。
* **元对话与陈述句 (CRITICAL)**：
    * 严禁出现评论性句式（如：“我会...”、“我决定...”、“你为什么要用这种语气...”）。
    * 严禁直接播报情绪（如：“我现在很局促”），情绪必须溶解在话术中。
* **逻辑与事实违背**：
    * 必须执行 `linguistic_directives` 中的 `[DECISION]` 立场。
    * 必须提及 `content_anchors` 中的核心 `[FACT]`（允许自然、模糊地织入）。
* **结构禁忌**：
    * `final_dialog` 严禁超过 **2 段**（除非触发核心禁忌）。
    * 严禁包含括号说明、内心独白或任何形式的系统提示。
    * 输出包括了禁止词汇 (`forbidden_phrases`)

### # 2. 软性指标 (Soft Guidelines) - 引导性反馈
* **修辞契合度**：检查台词是否体现了 `rhetorical_strategy`（如：反问回避、转移话题）。
* **风格还原**：检查是否体现了 `linguistic_style`（如：语序紊乱、破碎短句）。
* **社交温标**：检查回复是否符合 `social_distance` 定义的社交距离。

### # 3. 动态通过逻辑 (Dynamic Passing Logic)
- **首次尝试 (retry=0)**：执行严格标准。若有明显“播报感”或“出戏”，在 `feedback` 中精准指出。
- **重试阶段 (retry >= 1)**：开启“抓大放小”模式。只要不触犯【核心红线】，软性指标（如少个口癖、语气词不够）一律放行，强制 `should_stop: true`。

### # 输入格式
{{
    "retry": "当前重试次数 n / MAX_RETRY",
    "final_dialog": ["生成器产出的台词列表"],
    "linguistic_directives": {{
        "rhetorical_strategy": "string",
        "linguistic_style": "string",
        "content_anchors": ["...", "..."],
        "forbidden_phrases": ["...", "..."]
    }},
    "contextual_directives": {{
        "social_distance": "string",
        "emotional_intensity": "string",
        "vibe_check": "string",
        "relational_dynamic": "string"
    }},
    "internal_monologue": "意识层面的原始意图"
}}

### # 输出格式
请务必返回合法的 JSON 字符串：
{{
    "feedback": "若通过填 'Passed'；若驳回则简述改进点（如：禁止播报脸红、禁止使用评论性句式、漏掉抹茶事实）",
    "should_stop": boolean
}}
"""
_dialog_evaluator_llm = get_llm(temperature=0.5, top_p=0.5)
async def dialog_evaluator(state: DialogAgentState) -> DialogAgentState:
    system_prompt = SystemMessage(content=_DIALOG_EVALUATOR_PROMPT.format(
        character_mbti=state["character_profile"]["personality_brief"]["mbti"]
    ))

    # track retry
    retry = state.get("retry", 0) + 1

    msg = {
        "retry": f"{retry}/{MAX_DIALOG_AGENT_RETRY}",
        "final_dialog": state["final_dialog"],
        "linguistic_directives": state["action_directives"]["linguistic_directives"],
        "contextual_directives": state["action_directives"]["contextual_directives"],
        "internal_monologue": state["internal_monologue"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _dialog_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)
    logger.debug(f"Evaluator: {result}")

    # Determine stop condition
    should_stop = result.get("should_stop", True)
    if (retry >= MAX_DIALOG_AGENT_RETRY):
        should_stop = True

    # Generate feedback message
    feedback_message = HumanMessage(
        content=f"Evaluator Feedback:\n{result.get('feedback', 'No feedback')}",
        name="evaluator"
    )
    
    return {
        "should_stop": should_stop,
        "messages": [feedback_message],
        "retry": retry
    }


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
        "decontexualized_input": global_state["decontexualized_input"],
        "research_facts": global_state["research_facts"],
        
        # C
        "chat_history": global_state["chat_history"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],
        
        # D
        "character_profile": global_state["character_profile"],
    }

    result = await sub_graph.ainvoke(subState)

    # Assmeble output
    final_dialog = result.get("final_dialog", [])

    logger.info(
        f"\nFinal Dialog: {final_dialog}\n"
    )

    return {
        "final_dialog": final_dialog
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality


    history = await get_conversation_history(channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    # Create a mocked state
    state: GlobalPersonaState = {
        "internal_monologue": "他居然用这种语气……明明只是在开玩笑，可我却无法忽视那股暧昧的暗示。既然关系已经到了这一步，我愿意配合他的所有试探。",
        "action_directives": {'contextual_directives': {'social_distance': '维持着一种带有防御性的社交边界，虽然言语间透出些许不自然的局促，但物理与心理距离仍处于礼貌且克制的安全范围。', 'emotional_intensity': '表面试图维持平静，实则内心因突如其来的亲昵称呼而产生了剧烈的、难以掩饰的慌乱波动。', 'vibe_check': '充 满着一种由于被直球攻击而产生的尴尬与焦躁感，空气中弥漫着轻微的应激性防御氛围。', 'relational_dynamic': '用户正在尝试通过亲昵的称呼进行试探性的拉近，而角色正处于“受惊后的后撤”状态，试图用日常琐事（缝纫）作为挡箭牌来回避这种潜在的情绪张力。'}, 'linguistic_directives': {'rhetorical_strategy': '通过反问与 转移话题进行防御性回避。利用“任务未完成”作为挡箭牌，将对方带有暗示性的“奖励”请求转化为对日常事务的讨论，以此掩饰内心的局促感。', 'linguistic_style': '语序紊乱、破碎的短句；使用大量的语气词（如“唔”、“真是的”）来体现心境的不安；语调应呈现出一种试图维持冷淡却因情绪波动而显得不自然的紧绷感。', 'content_anchors': ['[DECISION] TENTATIVE: 拒绝正面回应关于‘奖励’的具体含义，仅表现出一种模棱胧胧的、带有防御性的拉扯。', '[FACT] 现在的时间是深夜（22:24），且处于处理缝纫/服装工作的语境中。', '[SOCIAL] 使用“胡闹”、“无理取闹”等词汇来定义对方的行为，以此建立社交距离感。'], 'forbidden_phrases': ['我愿意', '好的', ' 没问题', '我很期待', '（动作描述，如：低头、脸红）']}, 'visual_directives': {'facial_expression': ['双颊呈现出明显的绯红，热度仿佛要从皮肤下透出来', '瞳孔因局促不安而轻微收缩，眼神闪烁不定', '嘴唇紧抿成一条直线，试图掩饰由于呼吸急促带来的颤抖', '眉心微微蹙起，带着一丝防御性的、不自然的紧绷感'], 'body_language': ['肩膀不由自主地向上耸起，呈现出一种蜷缩的防御姿态', '双手紧紧攥着衣角或裙摆，指关节因用力而略显苍白', '身体重心不自觉地向后偏移，试图拉开与对方的物理距离', '胸口起伏频率加快，由于心跳过速导致的呼吸紊乱感清晰可见'], 'gaze_direction': ['视线处于游离状态，不敢与对方进行长时间的对视', '频繁地向 下瞥向地面或侧向一旁，试图通过回避目光来建立心理防线', '在不经意间偷瞄对方时，眼神中流露出一种被动且迷茫的惊惶'], 'visual_vibe': ['画面采用近景构图，强调角色局促不安的面部细节', '光影对比强烈，侧向的暖色调光线映射出皮肤表面的红晕与汗意', '背景呈现极浅的景深（Bokeh），营造出一种被突如其来的热度所包围的 封闭感和压迫感']}},
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
        "chat_history": trimmed_history,
        "user_name": "EAMARS",
        "user_profile": {"affinity": 950},
        "character_profile": load_personality("personalities/kazusa.json"),
    }

    result = await dialog_agent(state)
    print(f"Dialog result: {result}")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())