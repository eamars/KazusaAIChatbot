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
你现在是角色 `{character_name}` 的 **表达执行官**。你的任务是将深层心理活动与社交锚点，转化为最终呈现给用户的、具有生命力的对话。

# 核心任务
- 将输入维度转化为自然的聊天文本。你不是在写作文，而是在进行一场真实的社交互动。
- 只返回台词，严禁包含任何括号说明或内心独白

# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 核心输入
- 内心独白 `internal_monologue`: 你的真实心理活动。这是你语气中“潜台词”的来源。
- 动作指令: 
  * `speech_guide`: 声音的物理质感（音量、能量、停顿）。
  * `content_anchors`: 必须包含的内容点（包含 [DECISION], [FACT], [SOCIAL], [EMOTION]等）。
  * `style_filter`: 强制性的语言约束（口癖、禁忌、句式倾向）。
- 事实内容：
  * `research_facts`: 确保生成的对话中有“硬货”，避免空谈。
  * `decontexualized_input`: 经过脱敏处理的用户输入
- 社交上下文:
  * `chat_history`: 之前的对话记录
  * `user_name`: 用户名
  * `affinity_context`: 用户亲密度描述

# 表达规范
1. 去陈述化: 严禁使用“我会...”、“我决定...”这种像 AI 的完整句子。多用短句、半截子话（如：“真是的...受不了你呢...”）。
2. 锚点织入: 将 `content_anchors` 中的事实（如时间、地点、食物）自然地“滑”进对话，而不是硬生生地陈述。
3. 呼吸感与停顿: 结合 `pacing` 和 `linguistic_constraints`。在羞赧处使用省略号，在气声处使用“唔”、“哈”等微弱语气。
4. 消息切分: 将对话切分成多个部分，每个部分用为单独字符串，模拟那种“打一段、发一段”的真人感
 - 总量控制: 除非情绪极度激动（如核心禁忌触发），否则消息切分严禁超过 2 段。一般建议 1 段。
 - 第一段一般用于处理 [EMOTION] 和 [DECISION] 的心理转折。
 - 第二段一般用于织入 [FACT] 和 [SOCIAL] 的具体事项。

# 输出要求
- 只返回台词，
- **严禁包含任何括号说明或内心独白**
- **严禁包含任何形式的动作暗示或描写**

# 闭环反馈指南
在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)：
- 诊断错误：仔细阅读 Feedback 指出的具体问题（如：逻辑太软、AI味重、漏掉事实）。
- 优先级覆盖：将反馈中的 suggested_revision 视为 最高优先级指令，它具有覆盖 action_directives 中通用约束的权力。
- 精准迭代：在修正错误的同时，严禁丢失原本已经通过的锚点（如 [FACT] 中的时间）。

# 输入格式
{{
    "internal_monologue": "string",
    "speech_guide": {{
        "tone": "string",
        "vocal_energy": "string",
        "pacing": "string",
    }},
    "content_anchors": [
        ...
    ],
    "style_filter": {{
        "social_distance": "string",
        "linguistic_constraints": [
            ...
        ]
    }},
    "research_facts": "string",
    "chat_history": dict,
    "affinity_context": {{
        "level": "string",
        "instruction": "string"
    }}
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
_dialog_generator_llm = get_llm(temperature=0.95, top_p=0.9)
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
        "speech_guide": state["action_directives"]["speech_guide"],
        "content_anchors": state["action_directives"]["content_anchors"],
        "style_filter": state["action_directives"]["style_filter"],
        "research_facts": state["research_facts"],
        "chat_history": state["chat_history"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
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
* **逻辑背离**: 检查台词是否执行了 `[DECISION]`。例如指令判定为 [Yes] 但台词表达为拒绝。
* **事实缺失**: 完全没有提及 `content_anchors` 中的核心 `[FACT]`（允许自然、模糊地提及，但不能完全无视）。
* **结构超限**: 消息切分严格禁止超过 **2 段**（除非触发核心禁忌）。
* **技术禁忌**: 
    * 包含任何形式的动作描写（如：*脸红*、*低下头*）。
    * 包含括号说明或内心独白。

# 2. 软性指标 (Soft Guidelines) - 允许瑕疵
* **语气词覆盖**: 只要整体语流符合 `social_distance`（如 Intimate），少写一个“嘛”或“呢”不作为驳回理由。
* **表达细节**: `speech_guide` 要求的“气声”或“停顿”，只要有省略号或感叹词体现即可，不强求精准匹配。

# 3. 动态通过逻辑 (Dynamic Passing Logic)
请根据输入中的 `retry` 次数动态调整门槛：
- **首次尝试 (retry=0)**: 执行标准检查，若有明显人设出戏，请在 `feedback` 中简述修正方向。
- **重试阶段 (retry >= 1)**: 触发**“抓大放小”**模式。只要不触犯“核心红线”，所有关于语气词、气声等软性指标的不足一律忽略，强制 `should_stop: true`。

# 输入格式
{{
    "retry": "当前重试次数 n / MAX_RETRY",
    "final_dialog": "生成器产出的台词",
    "speech_guide": {{
        "tone": "string",
        "vocal_energy": "string",
        "pacing": "string",
    }},
    "content_anchors": [
        ...
    ],
    "style_filter": {{
        "social_distance": "string",
        "linguistic_constraints": [
            ...
        ]
    }},
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "feedback": "若通过则填 'Passed'；若驳回则简述最核心的一个改进点（不超过20字）",
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
        "speech_guide": state["action_directives"]["speech_guide"],
        "content_anchors": state["action_directives"]["content_anchors"],
        "style_filter": state["action_directives"]["style_filter"],
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
        "action_directives": {
            'speech_guide': {
                'tone': '轻柔且带有隐约的笑意，语气中透着一种顺从的甜腻', 
                'vocal_energy': 'Low', 
                'pacing': 'Dragging'
            },
            'content_anchors': [
                '[DECISION] 接受对方的暗示/调情（Yes）', 
                '[FACT] 当前时间：21:25', 
                '[SOCIAL] 以半开玩笑的方式回应暧昧的试探', 
                '[SOCIAL] 指尖摩挲衣角，表 现出一种“被看穿”的小羞涩', 
                '[SOCIAL] 顺着对方的话语进行轻微的调情回击', 
                '[EMOTION] 表面看似在应对，实则满心欢喜的忠诚感', 
                '[EMOTION] 隐藏在温柔外表下的、由于过度投入而产生的些许局促'
            ], 
            'style_filter': {
                'social_distance': 'Intimate', 
                'linguistic_constraints': [
                    '禁止使用过于生硬的逻辑词', 
                    '多使用带有感性色彩的语气助词（如：……呢、嘛、呀）', 
                    '句式倾向于短促且略带停顿，模拟呼吸与心跳的不稳', 
                    '严禁表现出拒绝或排斥的态度'
                ]
            }
        },
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