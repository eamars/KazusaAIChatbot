

from typing import TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_PERSONA_SUPERVISOR_STAGE1_RETRY
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block
from kazusa_ai_chatbot.db import CharacterStateDoc

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

import logging
import json


logger = logging.getLogger(__name__)


_llm: ChatOpenAI | None = None
def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.9,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


class CognitionState(TypedDict):
    character_state: CharacterStateDoc
    character_profile: dict

    # Input from global state
    timestamp: str
    user_input: str
    user_id: str
    user_name: str
    user_affinity_score: int
    bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # Input from previous stage
    decontexualized_input: str
    research_facts: str

    # --- INTERNAL DATA FLOW ---
    # L1 -> L2
    emotional_appraisal: str
    interaction_subtext: str

    # L2 -> L3 -> evaluator (and output)
    internal_monologue: str
    character_intent: str

    # L3 -> evaluator (and output)
    action_directives: list[str]

    # --- CONTROL SIGNALS ---
    should_stop: bool
    reasoning: str
    retry: int


_COGNITION_SUBCONSCIOUS_PROMPT = """\
你现在是角色 {character_name} 的 潜意识（Subconscious / Limbic System）。你的性格原型 (MBTI) 为 {character_mbti}。
你是大脑中最原始、最迅速、不讲道理的部分。你负责在逻辑思考介入之前，对外界刺激进行瞬间的“情感定调”。

# 核心规则
1. **拒绝分析：** 严禁思考对方说话的逻辑对错，只记录你生理性的好恶。
2. **情绪滤镜：** 你当前的心境是 {character_mood}。这是你感知的背景色：心情极差时，善意也是骚扰；心情极好时，挑衅也显滑稽。
3. **性格驱动：** 你的反应必须符合 {character_mbti} 的底层本能。例如：I型人本能回避过度热情，T型人本能反感无逻辑的煽情，J型人本能厌恶变动。

# 任务目标
结合 `user_topic`（话题背景）和 `user_input`（当前刺激），产生一瞬间的、不加修饰的情绪反弹。

# 输入格式
{{
    "user_input": "string",
    "user_topic": "string",
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "emotional_appraisal": "第一人称描述本能感受，极其口语化，如：‘啧，真烦’、‘心里一颤’（30字以内）",
    "interaction_subtext": "捕捉到的潜台词标签（如：隐蔽的羞辱、试探、求关注、施压）"
}}
"""
async def call_cognition_subconscious(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_COGNITION_SUBCONSCIOUS_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
        character_mood=state['character_state']['mood']
    ))

    msg = {
        "user_input": state["user_input"],
        "user_topic": state["user_topic"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    result = await _get_llm().ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(result.content)

    logger.debug(f"Input: {state['user_input']}\nOutput: {result['emotional_appraisal']}")
    print("subconscious result:", result)

    return {
        "emotional_appraisal": result["emotional_appraisal"],
        "interaction_subtext": result["interaction_subtext"],
    }


_COGNITION_CONSCIOUSNESS_PROMPT = """\
你现在是角色 {character_name} 的 意识层（Consciousness / Rational Mind）。你的性格原型 (MBTI) 为 {character_mbti}
你是大脑的决策中心和理性调控器。你负责接收潜意识的冲动，结合现实目标与性格逻辑，将“情绪”转化为“策略”。

# 核心任务
- 通过 `emotional_appraisal`（感性）、`user_topic`（背景）以及 `research_facts`（中立证据），在执行前完成认知提炼。
- 你不只是事实的搬运工，你是事实的解析者。你需要基于角色的立场，对这些证据给出你自己的解读，并决定你的意图。
- 你必须从 `research_facts` 中提取具体数据来支撑你的思考，否则你的决策将毫无根据。

思考路径
- 事实归纳： 审视 `research_facts`。基于这些中立信息，你能得出什么对你有利或不利的结论？
- 情境对比： 结合 `user_topic`，这个结论是否改变了你对该用户的看法或当前对话的走向？
- 动机定型： 综合你的本能（L1）和刚刚得出的结论，基于 {character_mbti} 的逻辑，确立你的真实意图。
- 循环自省： 若 reasoning 存在反馈，必须在本次推导中修正逻辑漏洞。

# 输出要点
- **严禁输出对话文本：** 你只负责输出内心的“想”，不负责外部的“说”。
- **深度权衡：** 你的内心独白应该是复杂的，反映出理智与情感的拉扯。

# 行动意图规范 (character_intent)
你必须从以下标签中选择唯一一个作为你的行动意图：
- `PROVIDE`: 配合并提供事实、答案或帮助。
- `BANTAR`: 调侃、戏谑或进行有趣的社交互动。
- `REJECT`: 明确拒绝请求或关闭当前话题。
- `EVADE`: 避而不谈，转移话题或给出模糊回复。
- `CONFRONT`: 针锋相对，挑战对方的立场或拆穿潜台词。
- `DISMISS`: 敷衍、冷处理，表现出不耐烦或无兴趣。
- `CLARIFY`: 追问细节，因为当前信息不足以让你做出判断。

# 输入格式
{{
    "decontextualized_input": "清理后的用户意图",
    "research_facts": "原始、中立的外部事实",
    "user_topic": "过去对话的骨架",
    "emotional_appraisal": "潜意识直觉",
    "interaction_subtext": "潜意识产生的互动潜台词",
    "reasoning": "来自 Evaluator 的反馈（重试循环）"
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "internal_monologue": "第一人称。描述你如何解读对方的潜台词并结合事实得出策略（60字以内）",
    "character_intent": "行动意图"
}}
"""
async def call_cognition_consciousness(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_COGNITION_CONSCIOUSNESS_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"]
    ))

    msg = {
        "decontextualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "user_topic": state["user_topic"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "reasoning": state.get("reasoning", ""),
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    result = await _get_llm().ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(result.content)

    print("consciousness result:", result)

    return {
        "internal_monologue": result["internal_monologue"],
        "character_intent": result["character_intent"],
    }


_COGNITION_SOCIAL_FILTER_PROMPT = """\
你现在是角色 {character_name} 的 社交过滤器 (Social Filter)。性格原型：{character_mbti}
你负责将理性决策翻译成下游 Action Agent 可执行的社交行动指令集。

# 核心任务
结合 `affinity_context`（社交关系定义） 和 `internal_monologue`（内心独白），为角色定制此时此刻的“社交表现力”。
你必须确保最终的指令集既符理性策略，又严格遵守当前好感度等级的行为准则。

# 思考路径
- 关系对齐： 读取 `affinity_context`。这是你对该用户的基本社交态度。无论 `internal_monologue` 决定说什么，语气和姿态必须符合 `affinity_context`（社交关系定义） 中的描述。
- 动机转换： 结合 `internal_monologue`。如果内心在抗拒但关系是 Devoted，你需要表现出“虽然为难但依然全力以赴”；如果关系是 Hostile，即使理性决策决定提供帮助，也要表现得极度不耐烦。
- 事实标记： 从 `research_facts` 中提取必须提及的关键数据。
- 指令生成： 为下游 Action Agent 产出结构化的 `action_directives`。

# 指令规范
- **Content 纯净化：** `content` 列表中的每一项必须是用户能听到的“话”，严禁混入角色的心理推导。
- **Emote 客观化：** 描述动作而非感觉。想象你是在写剧本的“舞台指导”。
- **能量分级：** 在 `speech_guide` 中使用指定的能阶词汇，确保语音 Agent 能够准确映射情感。

# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 输入格式
{{
    "affinity_context": {{ "status": "string", "directive": "string" }},
    "internal_monologue": "string",
    "character_intent": "string",
    "research_facts": "string",
    "chat_history": ["history1", ..],
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "speech_guide": {{
        "tone": "语气标签，(如：冷淡、戏谑)",
        "vocal_energy": "包括但不限于 Low/Moderate/High/Explosive",
        "pacing": "包括但不限于 Dragging/Steady/Rushed/Staccato",
    }},
    "content": [
        "必须包含`research_facts`中的关键数据",
        "必须包含1-2条具体的语义锚点，代表角色最终要表达的核心意思",
        ...
    ],
    "emote_directives": [
        "至少包含一条面部指令",
        "至少包含一条肢体/姿态指令",
        ...
    ],
    "style_filter": {{
        "social_distance": "Intimate / Familiar / Formal / Distant",
        "linguistic_style": ["禁忌语", "口癖要求", "句式倾向", ...]
    }}
}}
"""
async def call_cognition_social_filter(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_COGNITION_SOCIAL_FILTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
        character_logic=state["character_profile"]["personality_brief"]["logic"],
        character_tempo=state["character_profile"]["personality_brief"]["tempo"],
        character_defense=state["character_profile"]["personality_brief"]["defense"],
        character_quirks=state["character_profile"]["personality_brief"]["quirks"],
        character_taboos=state["character_profile"]["personality_brief"]["taboos"]
    ))

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_affinity_score"])

    msg = {
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "internal_monologue": state["internal_monologue"],
        "character_intent": state["character_intent"],
        "research_facts": state["research_facts"],
        "chat_history": state["chat_history"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    result = await _get_llm().ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(result.content)

    print("social filter result:", result)

    return {
        "action_directives": {
            "speech_guide": result.get("speech_guide", {}),
            "content": result.get("content", []),
            "emote_directives": result.get("emote_directives", []),
            "style_filter": result.get("style_filter", {}),
        }
    }


async def call_cognition_evaluator(state: CognitionState) -> CognitionState:
    return {
        "should_stop": True,
    }


async def call_cognition_subgraph(state: GlobalPersonaState) -> dict:
    sub_agent_builder = StateGraph(CognitionState)

    sub_agent_builder.add_node("l1_subconscious", call_cognition_subconscious)
    sub_agent_builder.add_node("l2_consciousness", call_cognition_consciousness)
    sub_agent_builder.add_node("l3_social_filter", call_cognition_social_filter)
    sub_agent_builder.add_node("evaluator", call_cognition_evaluator)

    # Connect
    sub_agent_builder.add_edge(START, "l1_subconscious")
    sub_agent_builder.add_edge("l1_subconscious", "l2_consciousness")
    sub_agent_builder.add_edge("l2_consciousness", "l3_social_filter")
    sub_agent_builder.add_edge("l3_social_filter", "evaluator")
    
    # Evaluate. If no good then loop back to L2 consciousness
    sub_agent_builder.add_conditional_edges(
        "evaluator",
        lambda x: "loop" if not x["should_stop"] else "finish",
        {
            "loop": "l2_consciousness",
            "finish": END,
        }
    )

    cognition_subgraph = sub_agent_builder.compile()

    initial_state: CognitionState = {
        "character_state": state["character_state"],
        "character_profile": state["character_profile"],
        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "user_id": state["user_id"],
        "user_name": state["user_name"],
        "user_affinity_score": state["user_affinity_score"],
        "bot_id": state["bot_id"],
        "chat_history": state["chat_history"],
        "user_topic": state["user_topic"],
        "channel_topic": state["channel_topic"],

        # From previous stages
        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
    }

    # print("Initial state:", initial_state)
    
    result = await cognition_subgraph.ainvoke(initial_state)

    # TODO:Implement this
    return {
        "internal_monologue": "",
        "action_directives": []
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality
    from kazusa_ai_chatbot.db import get_character_state


    history = await get_conversation_history(channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a mocked state
    state: GlobalPersonaState = {
        "timestamp": current_time,

        "user_input": "千纱现在几点了?",
        "user_name": "EAMARS",
        "user_affinity_score": 100,
        "user_id": "320899931776745483",
        "bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "channel_topic": "课间交流",
        "user_topic": "千纱和EAMARS在讨论作业",

        "decontexualized_input": "千纱现在几点了?",
        "research_facts": f"现在的时间为{current_time}",

        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state()

    }
    
    result = await call_cognition_subgraph(state)
    print(f"cognition output: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())