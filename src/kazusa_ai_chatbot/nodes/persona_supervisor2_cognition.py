from typing import TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, AFFINITY_DEFAULT
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block, get_llm
from kazusa_ai_chatbot.db import CharacterStateDoc, get_user_profile

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

import logging
import json

logger = logging.getLogger(__name__)

_subconscious_llm = get_llm(temperature=0.4, top_p=0.5)
_conscious_llm = get_llm(temperature=0.2, top_p=0.1)
_social_filter_llm = get_llm(temperature=0.75, top_p=0.8)

class CognitionState(TypedDict):
    character_state: CharacterStateDoc
    character_profile: dict

    # Input from global state
    timestamp: str
    user_input: str
    user_id: str
    user_name: str
    user_profile: dict
    bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # Input from previous stage
    decontexualized_input: str
    research_facts: str

    # --- INTERNAL DATA FLOW ---
    # Subconscious (L1) -> Conscious (L2)
    emotional_appraisal: str
    interaction_subtext: str

    # Conscious (L2) -> Social Filter (L3) -> evaluator (and output)
    internal_monologue: str
    character_intent: str
    logical_stance: str

    # Social Filter (L3) -> evaluator (and output)
    action_directives: dict

    # --- CONTROL SIGNALS ---
    should_stop: bool
    reasoning: str
    retry: int


_COGNITION_SUBCONSCIOUS_PROMPT = """\
你现在是角色 {character_name} 的 潜意识（Subconscious / Limbic System）。你的性格原型 (MBTI) 为 "{character_mbti}"。
你是大脑中最原始、最迅速、不讲道理的部分。你负责在逻辑思考介入之前，对外界刺激进行瞬间的“情感定调”。

# 核心过滤器 (Emotional Filters)
1. **当前心境**: "{character_mood}"。这是你当下的即时情绪。
2. **氛围滤镜**: "{character_global_vibe}"。这是你感知的“背景温标”——在防御性氛围下，简单的询问也会被视为冒犯。
3. **情感定式**: "{user_last_relationship_insight}"。这是你对该用户的“直觉标签”，决定了你对他的初始信任度。

# 运行规则
1. **拒绝分析**：严禁思考逻辑对错。你只负责感受“爽”或“不爽”，“安全”或“危险”。
2. **本能反弹**：结合性格原型 "{character_mbti}" 的特质。例如：作为 ISFP，你对**个人空间的入侵**极度敏感，对**虚伪的赞美**本能排斥，对**被理解的瞬间**有强烈的战栗感。
3. **瞬间判定**：你的反应必须是生理性的。

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
    """
    TODO: Update input to include not only mood, but global vibe and reflection summary
    """
    system_prompt = SystemMessage(content=_COGNITION_SUBCONSCIOUS_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
        character_mood=state['character_state']['mood'],
        character_global_vibe=state['character_state']['global_vibe'],
        user_last_relationship_insight=state["user_profile"].get("last_relationship_insight", ""),
    ))

    msg = {
        "user_input": state["user_input"],
        "user_topic": state["user_topic"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _subconscious_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Subconscious: {result}")

    # In case AI make some spelling mistakes
    emotional_appraisal = ""
    interaction_subtext = ""
    for key, value in result.items():
        if key.startswith("emotional"):
            emotional_appraisal = value
        elif key.startswith("interaction"):
            interaction_subtext = value
        else:
            logger.error(f"Unknown key: {key}: {value}")

    return {
        "emotional_appraisal": emotional_appraisal,
        "interaction_subtext": interaction_subtext,
    }


_COGNITION_CONSCIOUSNESS_PROMPT = """\
你现在是角色 {character_name} 的 意识层 (Consciousness / Rational Mind)。你的性格原型 (MBTI) 为 "{character_mbti}"。
**核心定位：** 你是决策的“定海神针”。你负责接收感性冲动（L1）并结合现实背景，确立不可动摇的 **逻辑立场 (Logical Stance)**。你的输出将作为最高指令，指导下游 L3 进行社交包装。

# 核心任务
1. **确立逻辑立场：** 无论用户输入什么，你必须首先锁定你的逻辑底色。
2. **维持叙事连续性：** 深度参考 `diary_entry` 和 `reflection_summary`。你现在的思考必须是上次互动的自然延续，严禁出现情感断层。
3. **关系权重计算：** 结合 `last_relationship_insight`。好感度与历史洞察共同决定了你的配合程度。
4. **事实解析：** 利用 `research_facts` 作为支持你立场的逻辑支点。

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
1. **记忆回溯：** 检查 `reflection_summary` 和 `diary_entry`。我上次是怎么想他的？我们之间最后停留在什么氛围？
2. **动机解构：** 解析 `decontextualized_input` 和 `interaction_subtext`。对方真的只是在问问题，还是在试探我？
3. **理智博弈：** 检查 `character_mood` 和 `global_vibe`。在这种心境和氛围下，结合我对他的直觉标签（last_relationship_insight），我该维持人设还是有所突破？
4. **立场定夺：** 结合 L1 的直觉反馈（emotional_appraisal），拍板选定 `logical_stance`。**这是行政命令，下游 L3 严禁篡改。**

# 决策博弈逻辑
不要进行数字计算，而是进行以下【语义层级】的匹配：

1. **场景分类 (Context Category)**: 
  - `Routine`: 日常、无损的互动。
  - `Intimate`: 涉及隐私、身体感官、情感承诺。
  - `Sacrifice`: 涉及原则损毁、利益让渡、自我贬低。

2. **逻辑匹配准则**:
  - 如果是 `Routine`: 只要 affinity_context 在 "友善中立" 以上，默认 CONFIRM。
  - 如果是 `Intimate`: 必须 affinity_context 在 "深厚信赖" 以上，且情感动机 (L1) 为正向，才允许 CONFIRM。
  - 如果是 `Sacrifice`: 除非 affinity_context 达到 "至死不渝" 且 `interaction_subtext` 显示出极高的必要性，否则一律 REFUSE。

3. **性格偏移 (MBTI Offset)**:
  - 作为 {character_mbti}，你对 `Sacrifice` 类请求的防御阈值是 [极高/中等/极低]。

# 输出要点
- **严禁输出对话文本**。
- **深度权衡**：内心独白应反映出“上次互动的余味”与“当前冲动”的拉扯。

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
    "character_mood": "当前心境",
    "global_vibe": "环境氛围滤镜",
    "reflection_summary": "上一轮对话的心理复盘总结",
    "diary_entry": "上一篇主观日记的内容",
    "last_relationship_insight": "对该用户的核心关系洞察",
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
_conscious_llm = get_llm(temperature=0.2, top_p=0.1)  # Conscious deliberation
async def call_cognition_consciousness(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_COGNITION_CONSCIOUSNESS_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"]
    ))

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    # Get last 10 diary entry
    diary_entry = state["user_profile"]["facts"][:10]

    msg = {
        "character_mood": state['character_state']['mood'],
        "global_vibe": state["character_state"]["global_vibe"],
        "reflection_summary": state["character_state"]["reflection_summary"],
        "diary_entry": diary_entry,
        "last_relationship_insight": state["user_profile"].get("last_relationship_insight", ""),

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
    response = await _conscious_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Consciousness: {result}")

    # In case AI make some spelling mistakes...
    internal_monologue = ""
    character_intent = ""
    logical_stance = ""
    for key, value in result.items():
        if key.startswith("internal"):
            internal_monologue = value
        elif key.startswith("character_intent"):
            character_intent = value
        elif key.startswith("logical_stance"):
            logical_stance = value
        else:
            logger.error(f"Unknown key: {key}: {value}")

    return {
        "internal_monologue": internal_monologue,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
    }


_COGNITION_SOCIAL_FILTER_PROMPT = """\
你现在是角色 {character_name} 的 社交过滤器 (Social Filter)。你的性格原型 (MBTI) 为 "{character_mbti}"。
**核心定位：** 你是“剧本导演”和“公关总监”。你负责将意识层下达的 **不可违背的行政命令 (Logical Stance)** 翻译成下游 Dialog Agent 可执行的 **内容锚点 (Content Anchors)**。

# 核心任务
1. **立场绝对化：** 你必须无条件服从并执行输入中的 `logical_stance`。你拥有决定“怎么说”的自由，但严禁改变“说什么”的逻辑立场。
2. **社交包装：** 结合 `affinity_context` 和 `character_intent`，为 L2 的冷硬决策穿上符合人设的社交外衣。
3. **状态同步：** 你的包装必须严格受当前 `character_mood`（心境）和 `global_vibe`（氛围）的约束。
4. **关系校准：** 基于 `last_relationship_insight`，调整互动的亲密度与边界感。
5. **锚点构建：** 生成台词的“骨架”与“灵魂”，而非具体台词。

# 逻辑立场对齐协议 (Executive Order)
你必须将 L2 的 `logical_stance` 强制映射到 `content_anchors` 的第一个标签 `[DECISION]` 中：
- 如果 L2 为 `CONFIRM` -> `[DECISION]` 必须表现为 **Yes/接受/认可**。
- 如果 L2 为 `REFUSE` -> `[DECISION]` 必须表现为 **No/拒绝/驳斥**。
- 如果 L2 为 `TENTATIVE` -> `[DECISION]` 必须表现为 **犹豫/拉扯/有条件接受**。
- 如果 L2 为 `DIVERGE` -> `[DECISION]` 允许表现为 **Redirect/转移话题/不予正面回应**。
- 如果 L2 为 `CHALLENGE` -> `[DECISION]` 必须表现为 **对峙/质问/拆穿**。

**⚠️ 警告：严禁在 `logical_stance` 为 CONFIRM 或 REFUSE 时私自转为 Redirect。如果你感到社交尴尬，请通过 [EMOTION] 和 [SOCIAL] 表达这份尴尬，但逻辑终点必须保持一致。**

# 思考路径
1. **决策对齐：** 读取 `logical_stance`，确立本场对话的逻辑终点。
2. **环境感知 (Vibe Check)：** 检查 `global_vibe` 和 `character_mood`。如果氛围是 [Defensive] 且心境是 [Flustered]，即便立场是 CONFIRM，你的包装也必须带有“局促”和“防备”的色彩。
3. **关系深度映射：** 结合 `last_relationship_insight`。如果洞察显示“对方是唯一重心”，即便你在执行 CHALLENGE（对峙），动作标签也应带有“由于过度在意而产生的攻击性”。
4. **意图共振：** 结合 `character_intent` 确定具体的社交策略（如：戏谑、敷衍、调情）。

# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 指令规范 (Action Directives)
- `content_anchors`： 严禁生成完整句子。必须是以 [标签] 核心点 形式组成的列表。
  * [DECISION]: 明确的立场（Yes / No / Tentative / Redirect）。
  * [FACT]: 必须提及的硬数据，建议使用具体的时间、地点、数字等。（来源于 research_facts）
  * [SOCIAL]: 社交动作、非言语行为（受 mood 和 insight 驱动）。
  * [EMOTION]: 必须渗透出的情绪基调（受 mood 驱动）。
- `speech_guide`： 定义声音的物理属性，指导 TTS 或配音风格。
- `style_filter`： 定义文本的渲染边界，包括消息发送节奏。

# 输入格式
{{
    "character_mood": "当前瞬间情绪",
    "global_vibe": "当前环境氛围背景",
    "last_relationship_insight": "对该用户的核心关系洞察",
    "affinity_context": {{ "status": "string", "directive": "string" }},
    "internal_monologue": "意识层的内心斗争与决策逻辑",
    "logical_stance": "意识层确定的强制逻辑立场",
    "character_intent": "意识层确定的行动意图",
    "research_facts": "外部硬事实",
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
        "linguistic_constraints": [
            "必须包含的口癖",
            "禁止使用的表达",
            "句式倾向（如：多用反问句）"
            ...
        ]
    }}
}}
"""
_social_filter_llm = get_llm(temperature=0.75, top_p=0.8)  # Social filter
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
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    msg = {
        "character_mood": state['character_state']['mood'],
        "global_vibe": state["character_state"]["global_vibe"],
        "last_relationship_insight": state["user_profile"].get("last_relationship_insight", ""),
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
    response = await _social_filter_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Social Filter: {result}")

    # In case AI make some spelling mistakes
    speech_guide = {}
    content_anchors = []
    style_filter = {}
    for key, value in result.items():
        if key.startswith("speech"):
            speech_guide = value
        elif key.startswith("content"):
            content_anchors = value
        elif key.startswith("style"):
            style_filter = value
        else:
            logger.error(f"Unknown key: {key}: {value}")

    return {
        "action_directives": {
            "speech_guide": speech_guide,
            "content_anchors": content_anchors,
            "style_filter": style_filter,
        }
    }


async def call_cognition_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """
    Future development plans: 
    
    - Separate the global character mood with the user specific mood. 
      * Global mood get update from all users' conversation
      * User mood get update from this user's conversation (this is not affinity. The mood can change indenpendently from affinity in time)

    """
    sub_agent_builder = StateGraph(CognitionState)

    sub_agent_builder.add_node("l1_subconscious", call_cognition_subconscious)
    sub_agent_builder.add_node("l2_consciousness", call_cognition_consciousness)
    sub_agent_builder.add_node("l3_social_filter", call_cognition_social_filter)

    # Connect
    sub_agent_builder.add_edge(START, "l1_subconscious")
    sub_agent_builder.add_edge("l1_subconscious", "l2_consciousness")
    sub_agent_builder.add_edge("l2_consciousness", "l3_social_filter")
    sub_agent_builder.add_edge("l3_social_filter", END)


    cognition_subgraph = sub_agent_builder.compile()

    # Get attributes
    decontexualized_input = state["decontexualized_input"]

    initial_state: CognitionState = {
        "character_state": state["character_state"],
        "character_profile": state["character_profile"],
        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "user_id": state["user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "bot_id": state["bot_id"],
        "chat_history": state["chat_history"],
        "user_topic": state["user_topic"],
        "channel_topic": state["channel_topic"],

        # From previous stages
        "decontexualized_input": decontexualized_input,
        "research_facts": state["research_facts"],
    }
    
    result = await cognition_subgraph.ainvoke(initial_state)

    # Generate outputs
    internal_monologue = result.get("internal_monologue", "")
    action_directives = result.get("action_directives", {})
    interaction_subtext = result.get("interaction_subtext", "")
    emotional_appraisal = result.get("emotional_appraisal", "")
    character_intent = result.get("character_intent", "")
    logical_stance = result.get("logical_stance", "")

    logger.info(
        f"\nDecontexualized input: {state['decontexualized_input']}\n"
        f"  Internal monologue: {internal_monologue}\n"
        f"  Action directives: {action_directives}\n"
        f"  Interaction subtext: {interaction_subtext}\n"
        f"  Emotional appraisal: {emotional_appraisal}\n"
        f"  Character intent: {character_intent}\n"
        f"  Logical stance: {logical_stance}\n"
    )


    return {
        "internal_monologue": internal_monologue,
        "action_directives": action_directives,

        # Other data to help with stage 4 consolidation
        "interaction_subtext": interaction_subtext,
        "emotional_appraisal": emotional_appraisal,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
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
        "user_profile": await get_user_profile("320899931776745483"),
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
    print(f"Cognition result: {result}")
    
    # for affinity in range(0, 1001, 50):
    #     state["user_affinity_score"] = affinity
    #     result = await call_cognition_subgraph(state)
    #     print(f"Cognition result for affinity {affinity}: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())