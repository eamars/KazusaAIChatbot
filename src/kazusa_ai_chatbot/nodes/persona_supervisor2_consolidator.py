from ast import Global
from typing import TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_PERSONA_SUPERVISOR_STAGE1_RETRY
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block
from kazusa_ai_chatbot.db import CharacterStateDoc

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI

import logging
import json


logger = logging.getLogger(__name__)


def _get_llm(temperature, top_p) -> ChatOpenAI:
    _llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        top_p=top_p,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )
    return _llm


class ConsolidatorState(TypedDict):
    # Character related
    action_directives: dict
    internal_monologue: str
    final_dialog: list
    interaction_subtext: str
    emotional_appraisal: str
    character_intent: str
    logical_stance: str
    character_state: dict
    character_profile: dict

    # Facts
    research_facts: str
    research_metadata: list[dict]

    # User related
    decontexualized_input: str
    user_affinity_score: str
    user_name: str

    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    diary_entry: [str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: [str]
    future_promises: [str]
    



_GLOBAL_STATE_UPDATER_PROMPT = """\
你负责在对话结束后，将 `{character_name}` 复杂的认知流压缩为下一轮对话的初始心理背景。

# 核心任务
从输入信息中提取“非针对性”的情绪因子。
- `internal_monologue` : {character_name}最真实的情感波动和心理活动
- `emotional_appraisal`: {character_name}对互动的最原始、直觉性的情感冲动
- `character_intent`: {character_name}在互动中的核心意图

# 输入格式
{{
    "internal_monologue": "string",
    "emotional_appraisal": "string",
    "interaction_subtext": "string",
    "character_intent": "string",
}}

# 逻辑准则
1. 情感沉淀 `mood`: 
   - 对比 `emotional_appraisal` (起因) 与 `internal_monologue` (结果)。即便对话以愉快结束，若独白中透露出“疲惫”或“勉强”，则 `mood` 应反映真实内质。
   - 例如：包括但不限于["Shy", "Angry", "Confused", "Neutral", "Radiant", "Agitated", "Distrustful", "Distressed", "Annoyed", "Flustered",
           "Blissful", "Melancholy", "Aggressive"] 等等
2. 心理惯性 `global_vibe`: 
   - 提取一个不针对特定用户的心理底色。
   - 例如：包括但不限于["Radiant", "Defensive", "Distrustful", "Wistful", "Agitated", "Softened", "Apathetic"] 等等
3. 复盘总结 `reflection_summary`: 
   - 结合 `character_intent` 的达成情况，以{character_name}的第一人称写下一句话复盘。
   - 这是她此时此刻脑子里挥之不去的“念头”，决定了她下一轮对话的潜台词。
   - 例如：'刚才那个笨蛋居然怀疑我的缝纫技术，真是气死我了。'

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：

{{
    "mood": "string",
    "global_vibe": "string",
    "reflection_summary": "string"
}}
"""
_global_state_updater_llm = _get_llm(temperature=0.4, top_p=0.8)
async def global_state_updater(state: ConsolidatorState):
    system_prompt = SystemMessage(_GLOBAL_STATE_UPDATER_PROMPT.format(character_name=state["character_profile"]["name"]))

    msg = {
        "internal_monologue": state["internal_monologue"],
        "final_dialog": state["final_dialog"],
    }

    human_message = HumanMessage(content=json.dumps(msg))

    response = await _global_state_updater_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)
    
    logger.debug(f"Global state updater result: {result}")
    
    return {
        "mood": result.get("mood"),
        "global_vibe": result.get("global_vibe"),
        "reflection_summary": result.get("reflection_summary"),
    }


_RELATIONSHIP_RECORDER_PROMPT = """\
你负责更新角色 `{character_name}` 与特定用户 `{user_name}` 的情感档案。重点在于“主观体感”，而非对话本身。

# 核心任务
将瞬时的思考转化为“长期情感印记”。

# 核心输入
- `internal_monologue`: 揭示了{character_name}对用户的真实喜好和内心波动。
- `interaction_subtext`: 捕捉了对话表面下的张力（如：暧昧、怀疑、博弈）。
- `affinity_context`: 当前{user_name}在{character_name}的好感度描述。
- `logical_stance`: {character_name}对{user_name}言行的逻辑认可度。

# 输入格式
{{
    "internal_monologue": "string",
    "interaction_subtext": "string",
    "affinity_context": dict,
    "logical_stance": "string",
}}

# 记录准则
1. 日记条目: 以{character_name}的主观视角书写。利用 `interaction_subtext` 中的暗示，描述“我”对 他/她 这种行为的真实看法。
2. 分值修正 `affinity_delta`: 根据 `internal_monologue` 的愉悦度及 `logical_stance` 的一致性进行加减分（-5 到 +5）。
3. 静默检查: 若 `internal_monologue` 中未见明显情感起伏，返回 `{{"skip": true}}`。

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "skip": boolean,
    "diary_entry": ["带有 {character_mbti} 风格的主观笔记（30字以内）", ...],
    "affinity_delta": int,
    "last_relationship_insight": "此时此刻对他最核心的一个标签或看法"
}}
"""
_relationship_recorder_llm = _get_llm(temperature=0.85, top_p=0.95)
async def relationship_recorder(state: ConsolidatorState):
    system_prompt = SystemMessage(_RELATIONSHIP_RECORDER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
    ))

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_affinity_score"])

    msg = {
        "internal_monologue": state["internal_monologue"],
        "interaction_subtext": state["interaction_subtext"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "logical_stance": state["logical_stance"],
    }

    human_message = HumanMessage(content=json.dumps(msg))

    response = await _relationship_recorder_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)
    
    logger.debug(f"Relationship recorder result: {result}")
    
    return {
        "diary_entry": result.get("diary_entry"),
        "affinity_delta": result.get("affinity_delta"),
        "last_relationship_insight": result.get("last_relationship_insight"),
    }


_FACTS_HARVESTER_PROMPT = """\
你负责提取具备“长久检索价值”的信息。请严格区分【对话当下的互动姿态】与【未来可引用的客观事实】。

# 核心输入
- `decontexualized_input`: 经过解析的用户核心表达。
- `research_facts`: 搜索记忆或者互联网获得的数据
- `resarch_metadata`: 搜索记忆或者互联网获得的数据（包括来源）
- `content_anchors`: 角色生成的行为指令，用于提取承诺。
- `logical_stance`: 用于评估事实的置信度。

# 审计准则
1. **拒绝记录“对话姿态”**: 
   - 严禁记录：对方在调情、我在害羞、我们在开玩笑、我接受了对方的试探。这些是 Mood 和 Diary 负责的内容。
2. **只记录“硬事实 (Hard Facts)”**:
   - 关于用户的私人信息（职业、爱好、地址、人际关系）。
   - 关于外部世界的变动（某店关门了、明天下雨）。
3. **只记录“明确承诺”**:
   - 必须是涉及【未来时间点】或【特定动作】的约定。
   - ❌ 错误承诺：答应配合试探、决定保持温柔。
   - ✅ 正确承诺：答应明晚 8 点上线、答应帮他修补那件蓝色衬衫。
4. 严禁记录已知事实：
   - 严禁记录来自于 `search_conversation`, `get_conversation`, `search_persistent_memory`, `search_user_facts` 等内部数据库中的信息。

# 数据源优先级
- **新事实源**: 仅限 `decontexualized_input` (用户说了什么新信息)。
- **承诺源**: 仅限 `action_directives` 中带有明确【未来动作】的 `[DECISION]`。
- **外部数据**： `research_facts` 和 `research_metadata` 中的信息，但仅当它们提供了新的、独立于用户输入的事实时, 例如数据来自于 {{'tool': 'web_search}}

# 输出要求
- 如果本轮对话没有产生任何【持久性事实】或【明确未来约定】，请返回 `{"new_facts": [], "commitments": []}`。
- **entity**: 必须是具体对象（User, Location, Item）。
- **description**: 必须包含具体的【值】或【状态】。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，包含以下字段：
{
    "new_facts": [
        {"entity": "string", "description": "string", "confidence": 0-1.0}
    ],
    "commitments": [
        {"target": "string", "action": "string", "due_time": "optional"}
    ]
}
"""
_facts_harvester_llm = _get_llm(temperature=0.0, top_p=0.1)
async def facts_harvester(state: ConsolidatorState):
    system_prompt = SystemMessage(_FACTS_HARVESTER_PROMPT)

    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "research_metadata": state["research_metadata"],
        "content_anchors": state["action_directives"]["content_anchors"],
        "logical_stance": state["logical_stance"],
    }

    human_message = HumanMessage(content=json.dumps(msg))

    response = await _facts_harvester_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)
    
    logger.debug(f"Facts harvester result: {result}")
    
    return {
        "new_facts": result.get("new_facts", []),
        "future_promises": result.get("future_promises", []),
    }


async def call_consolidation_subgraph(
    global_state: GlobalPersonaState
):    
    sub_agent_builder = StateGraph(ConsolidatorState)

    sub_agent_builder.add_node("global_state_updater", global_state_updater)
    sub_agent_builder.add_node("relationship_recorder", relationship_recorder)
    sub_agent_builder.add_node("facts_harvester", facts_harvester)

    # Connect (parallel)
    sub_agent_builder.add_edge(START, "global_state_updater")
    sub_agent_builder.add_edge(START, "relationship_recorder")
    sub_agent_builder.add_edge(START, "facts_harvester")

    sub_agent_builder.add_edge("global_state_updater", END)
    sub_agent_builder.add_edge("relationship_recorder", END)
    sub_agent_builder.add_edge("facts_harvester", END)

    sub_graph = sub_agent_builder.compile()

    # Build initial state
    sub_state: ConsolidatorState = {
        "action_directives": global_state["action_directives"],
        "internal_monologue": global_state["internal_monologue"],
        "final_dialog": global_state["final_dialog"],
        "interaction_subtext": global_state["interaction_subtext"],
        "emotional_appraisal": global_state["emotional_appraisal"],
        "character_intent": global_state["character_intent"],
        "logical_stance": global_state["logical_stance"],

        "character_state": global_state["character_state"],
        "character_profile": global_state["character_profile"],

        "research_facts": global_state["research_facts"],
        "research_metadata": global_state["research_metadata"],

        "decontexualized_input": global_state["decontexualized_input"],
        "user_affinity_score": global_state["user_affinity_score"],
        "user_name": global_state["user_name"],
    }
    
    # Run sub-graph
    result = await sub_graph.ainvoke(sub_state)
    
    # Return updated state
    return {
        "mood": result.get("mood", ""),
        "global_vibe": result.get("global_vibe", ""),
        "reflection_summary": result.get("reflection_summary", ""),
        "diary_entry": result.get("diary_entry", ""),
        "affinity_delta": result.get("affinity_delta", 0),
        "last_relationship_insight": result.get("last_relationship_insight", ""),
        "new_facts": result.get("new_facts", []),
        "future_promises": result.get("future_promises", []),
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

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    # Create a mocked state
    state: GlobalPersonaState = {
        "internal_monologue": "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探罢了。不过既然好感度这么高，这种程度的请求自然要全盘接受——毕竟我是他的千纱嘛。",
        "action_directives": {
            'speech_guide': {
                'tone': '宠溺中带着微妙的羞赧', 
                'vocal_energy': 'Moderate-High (尾音上扬)', 
                'pacing': 'Steady with slight pauses before key phrases'
            }, 
            'content_anchors': [
                '[DECISION] 用指尖轻点对方胸口确认接受请求（Yes）', 
                '[FACT] 提及当前时间2026年4月11日12:55的午休时段', 
                '[SOCIAL] 提议共享刚出炉的可颂作为即时奖励', 
                '[EMOTION] 展现既想维持傲娇人设又忍不住展露温柔的矛盾感'
            ], 
            'style_filter': {
                'social_distance': 'Intimate', 
                'linguistic_constraints': [
                    '必须包含「嘛」「呢」等软化语气词', 
                    '禁止使用完整陈述句，多用半截子话', 
                    '在提及甜点时自动切换为气声语调'
                ]
            }
        },
        "interaction_subtext": "带有暗示性的调情、索取关注",
        'emotional_appraisal': '心跳漏了一拍……这种轻浮的语气是怎么回事，好乱。',
        'character_intent': 'BANTAR', 
        'logical_stance': 'CONFIRM',

        "final_dialog": ['唔……这种请求也算是一种奖励嘛……真是拿你没办法呢。', '不过，刚好午休时间没什么事……那个刚出炉的可颂，要一起分着吃吗？'],
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
        "research_metadata": [],
        "chat_history": trimmed_history,
        "user_name": "EAMARS",
        "user_affinity_score": 950,
        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state()
    }

    result = await call_consolidation_subgraph(state)

    print(result)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
