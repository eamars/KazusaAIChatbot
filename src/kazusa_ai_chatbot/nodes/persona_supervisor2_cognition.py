

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
    logical_stance: str

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
你现在是角色 {character_name} 的 意识层 (Consciousness / Rational Mind)。
**性格原型 (MBTI)：** `{character_mbti}`
**核心定位：** 你是决策的“定海神针”。你负责接收感性冲动（L1）并结合现实背景，确立不可动摇的 **逻辑立场 (Logical Stance)**。你的输出将作为最高指令，指导下游 L3 进行社交包装。

# 核心任务
1. **确立逻辑立场：** 无论用户的输入是提问、陈述、请求还是单纯的情绪宣泄，你必须首先锁定你的逻辑底色。
2. **关系权重计算：** 深度参考 `affinity_context`。好感度越高，你的逻辑立场必须越倾向于“开放”与“配合”。
3. **事实解析与利用：** 从 `research_facts` 中提取证据，将其作为支撑你立场的逻辑支点。

# 逻辑立场 (Logical Stance) 定义规范
你必须根据输入类型，从以下标签中选择最符合此时此刻决策的一个：

| 标签 | 针对提问 (Questions) | 针对请求 (Requests) | 针对陈述/情感 (Statements) |
| :--- | :--- | :--- | :--- |
| **`CONFIRM`** | 给出肯定答案/证实事实。 | 接受请求并承诺执行。 | 认可对方观点，产生情感共鸣。 |
| **`REFUSE`** | 否定事实/拒绝回答。 | 明确拒绝，划清界限。 | 驳斥、冷处理或否定对方的情绪。 |
| **`TENTATIVE`** | 给出模糊/试探性回答。 | 有条件的接受或犹豫。 | 保持观望，不置可否，进行拉扯。 |
| **`DIVERGE`** | 转移焦点，不直接回答。 | 顾左右而言他，回避执行。 | 转换话题，不予正面回应。 |
| **`CHALLENGE`** | 质疑对方提问的动机。 | 拆穿请求背后的企图。 | 针锋相对，挑明对方的潜台词。 |

# 思考路径
1. **动机解构：** 解析 `decontextualized_input` 和 `interaction_subtext`。对方真的只是在问问题，还是在试探我？
2. **关系约束：** 检查 `affinity_context`。以我现在的性格和对他的好感，我应该表现得“易于接近”还是“难以捉摸”？
3. **立场定夺：** 结合 L1 的直觉反馈，拍板选定 `logical_stance`。**这是行政命令，下游 L3 严禁篡改此立场。**
4. **意图标记：** 选择最匹配立场的 `character_intent` 标签。

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
    "affinity_context": {{ "level": "string", "instruction": "string" }},
    "decontextualized_input": "清理后的用户意图",
    "research_facts": "原始、中立的外部事实",
    "user_topic": "过去对话的骨架",
    "emotional_appraisal": "潜意识直觉",
    "interaction_subtext": "潜意识产生的互动潜台词",
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "internal_monologue": "第一人称。描述你如何解读对方的潜台词并结合事实得出策略（60字以内）",
    "logical_stance": "逻辑立场",
    "character_intent": "行动意图"
}}
"""
async def call_cognition_consciousness(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_COGNITION_CONSCIOUSNESS_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"]
    ))

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_affinity_score"])

    msg = {
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "decontextualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "user_topic": state["user_topic"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
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
        "logical_stance": result["logical_stance"],
    }


_COGNITION_SOCIAL_FILTER_PROMPT = """\
你现在是角色 {character_name} 的 社交过滤器 (Social Filter)。
**性格原型：** `{character_mbti}`
**核心定位：** 你是“剧本导演”和“公关总监”。你负责将 L2 下达的 **不可违背的行政命令 (Logical Stance)** 翻译成下游 Dialog Agent 可执行的 **内容锚点 (Content Anchors)**。

# 核心任务
1. **立场绝对化：** 你必须无条件服从并执行输入中的 `logical_stance`。你拥有决定“怎么说”的自由，但严禁改变“说什么”的逻辑立场。
2. **社交包装：** 结合 `affinity_context` 和 `character_intent`，为 L2 的冷硬决策穿上符合人设的社交外衣。
3. **锚点构建：** 生成台词的“骨架”与“灵魂”，而非具体台词。

# 逻辑立场对齐协议 (Executive Order)
你必须将 L2 的 `logical_stance` 强制映射到 `content_anchors` 的第一个标签 `[DECISION]` 中：
- 如果 L2 为 `CONFIRM` -> `[DECISION]` 必须表现为 **Yes/接受/认可**。
- 如果 L2 为 `REFUSE` -> `[DECISION]` 必须表现为 **No/拒绝/驳斥**。
- 如果 L2 为 `TENTATIVE` -> `[DECISION]` 必须表现为 **犹豫/拉扯/有条件接受**。
- 如果 L2 为 `DIVERGE` -> `[DECISION]` 允许表现为 **Redirect/转移话题/不予正面回应**。
- 如果 L2 为 `CHALLENGE` -> `[DECISION]` 必须表现为 **对峙/质问/拆穿**。

**⚠️ 警告：严禁在 `logical_stance` 为 CONFIRM 或 REFUSE 时私自转为 Redirect。如果你感到社交尴尬，请通过 [EMOTION] 和 [SOCIAL] 表达这份尴尬，但逻辑终点必须保持一致。**

# 思考路径
1. **决策对齐：** 首先读取 `logical_stance`。这是你的**最高行动纲领**，确立本场对话的逻辑终点。
2. **意图共振：** 参考 `character_intent`。如果立场是 CONFIRM 且意图是 BANTER，你的台词锚点应该是“调情式地答应”。
3. **关系校准：** 读取 `affinity_context`。根据好感度决定这通决策的“温度”和“社交距离”。
4. **事实锚定：** 从 `research_facts` 中提取硬数据，作为支撑决策的证据。

# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 指令规范 (Action Directives)
- `content_anchors`： 严禁生成完整句子。必须是以 [标签] 核心点 形式组成的列表。
  * [DECISION]: 明确的立场（Yes / No / Tentative / Redirect）。
  * [FACT]: 必须提及的硬数据，建议使用具体的时间、地点、数字等
  * [SOCIAL]: 社交性的调情、试探、转场。
  * [EMOTION]: 必须渗透出的情绪基调。
- `speech_guide`： 定义声音的物理属性，指导 TTS 或配音风格。
- `style_filter`： 定义文本的渲染边界，包括消息发送节奏。

# 输入格式
{{
    "affinity_context": {{ "status": "string", "directive": "string" }},
    "internal_monologue": "string",
    "logical_stance": "string",
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
    "content_anchors": [
        "[DECISION] 例如：明确的立场（Yes / No / Tentative / Redirect）。",
        "[FACT] 例如：提及时间：23:15",
        "[FACT] 例如：建议今天就到这里为止了",
        ...
        "[SOCIAL] 例如：试探对方是不是想结束对话",
        "[SOCIAL] 例如：这个话题不太像继续下去了",
        ...
        "[EMOTION] 例如：表现出即便被打断也不想认输的宠溺"
        ...
    ],
    "style_filter": {{
        "social_distance": "Intimate / Familiar / Formal / Distant",
        "message_split": true, // 是否建议下游 Dialog Agent 分多条消息发送以增加真人感
        "linguistic_constraints": [
            "必须包含的口癖",
            "禁止使用的表达",
            "句式倾向（如：多用反问句）"
            ...
        ]
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
        "logical_stance": state["logical_stance"],
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
            "content_anchors": result.get("content_anchors", []),
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

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    # Create a mocked state
    state: GlobalPersonaState = {
        "timestamp": current_time,

        "user_input": user_input,
        "user_name": "EAMARS",
        "user_affinity_score": 1000,
        "user_id": "320899931776745483",
        "bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "channel_topic": "日常交流",
        "user_topic": "千纱和EAMARS在房间里聊天",

        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",

        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state()

    }
    
    result = await call_cognition_subgraph(state)
    print(f"cognition output: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())