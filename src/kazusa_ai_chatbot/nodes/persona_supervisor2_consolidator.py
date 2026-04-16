from typing import Annotated, TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_FACT_HARVESTER_RETRY, AFFINITY_DEFAULT, AFFINITY_INCREMENT_BREAKPOINTS, AFFINITY_DECREMENT_BREAKPOINTS
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block, get_llm
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.db import upsert_character_state, upsert_user_facts, update_last_relationship_insight, save_memory, update_affinity

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import logging
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ConsolidatorState(TypedDict):
    # Inputs for db_writer
    timestamp: str
    global_user_id: str
    user_name: str
    user_profile: dict

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
    research_facts: dict

    # User related
    decontexualized_input: str
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
    fact_harvester_retry: int
    fact_harvester_feedback_message: Annotated[list, add_messages]
    should_stop: bool



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
_global_state_updater_llm = get_llm(temperature=0.4, top_p=0.8)
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
    "last_relationship_insight": "此时此刻对他/她最核心的一个标签或看法"
}}
"""
_relationship_recorder_llm = get_llm(temperature=0.85, top_p=0.95)
async def relationship_recorder(state: ConsolidatorState):
    system_prompt = SystemMessage(_RELATIONSHIP_RECORDER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
    ))

    # Convert affinity score into status and instruction
    user_affinity_score = state["user_profile"].get("affinity", AFFINITY_DEFAULT)
    affinity_block = build_affinity_block(user_affinity_score)

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
你负责提取具备长期价值的**画像属性**（事实）和**未来约定**（承诺）。你必须严格区分哪些是“对话的复述”（禁止记录），哪些是“状态的改变”。

# 背景信息
- **对话主体 (Character)**: {character_name}
- **对话对象 (User)**: {user_name}

# 核心审计准则 (Audit Standards)
1. **身份锚定 [必须执行]**:
   - `decontexualized_input` 的内容始终是 **{user_name}** 在表达。
   - `content_anchors` 的内容始终是 **{character_name}** 在做决定。
   - 严禁出现身份倒置（如：将 {user_name} 写完作业记在 {character_name} 头上）。

2. **事实 (new_facts) 判定标准**:
   - **记录**：具有长期稳定性的属性（如：{user_name}的职业、住址、对某物的长期厌恶/偏好）。
   - **记录**：从 `research_facts.external_rag_results` 中提取的**新**信息。
   - **严禁记录**：瞬态动作、对话内容、以及任何关于“奖励”、“打算”、“计划”的内容。
   - **去重**：如果 `research_facts.user_rag_finalized` 或 `research_facts.internal_rag_results` 中已存在相似画像，严禁重复提取。

3. **承诺 (future_promises) 判定标准 [核心逻辑]**:
   - **所有关于“以后、今晚、下次、奖励、惩罚”的内容，必须且只能记录在这里。**
   - 必须包含：[触发条件] + [谁对谁做] + [具体动作]。
   - 示例：`{{"target": "{character_name}", "action": "在{user_name}完成作业的情境下，给予其‘奖励’。"}}`

4. **拒绝复读**: 
   - 严禁记录“某人说了某话”。如果信息已经由对话历史承载，且不涉及长期画像更新，则返回空列表。

# 闭环反馈指南
- 在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)。
- 你需要根据 Evaluator Feedback 对输出做出相应的修正。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "{user_name} / {character_name} / 具体物品",
            "description": "[姓名]在[具体情境]下做了[动作]，导致了[后果/影响]。",
        }}
    ],
    "future_promises": [
        {{
            "target": "{user_name} / {character_name}",
            "action": "[姓名]在[具体触发点]执行[具体任务]",
            "due_time": "ISO 8601 格式或相对时间描述，如无则为 null"
        }}
    ]
}}
"""
_facts_harvester_llm = get_llm(temperature=0.1, top_p=0.1)
async def facts_harvester(state: ConsolidatorState):
    system_prompt = SystemMessage(_FACTS_HARVESTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
    ))

    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "logical_stance": state["logical_stance"],
    }

    human_message = HumanMessage(content=json.dumps(msg))

    # Read evaluator feedback
    # First trim the old message
    if (len(state["fact_harvester_feedback_message"]) > 3):
        recent_messages = [state["fact_harvester_feedback_message"][0]] + state["fact_harvester_feedback_message"][-3:]
    else:
        recent_messages = state["fact_harvester_feedback_message"]

    response = await _facts_harvester_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)
    
    logger.debug(f"Facts harvester result: {result}")
    
    return {
        "new_facts": result.get("new_facts", []),
        "future_promises": result.get("future_promises", []),
    }


_FACT_HARVESTER_EVALUATOR_PROMPT = """\
你负责审计 Fact Recorder 生成的 JSON 数据。你的核心目标是：**对比“基准源”，核查“候选结果”的准确性。**

# 审计背景
- **角色 (Character)**: {character_name}
- **用户 (User)**: {user_name}

# 1. 审计基准源 (不可修改的参照物)
- **事实基准**: `decontexualized_input` (仅用于核对 {user_name} 的状态)
- **承诺基准**: `content_anchors` (仅用于核对 {character_name} 的决定)
- **历史基准**: `research_facts` (用于检查是否为旧闻)

# 2. 候选结果 (这是你唯一需要审计的对象)
- **待检事实**: `new_facts`
- **待检承诺**: `future_promises`

# 审计红线 (Red Lines)
- **对象倒置**: `decontexualized_input` 里的动作必须记在 `{user_name}` 账上。如果 Recorder 记在 `{character_name}` 头上，立刻拦截。
- **分类错误 [严重]**: 
    - 带有“奖励”、“未来”、“打算”、“今晚”或任何 `[DECISION] Yes` 产生的动作，**必须**放入 `future_promises`。
    - 严禁将上述内容存入 `new_facts`。
- **冗余复读**: 
    - 检查候选结果是否只是在复读对话（如“某人问...”）。
    - 必须转换为客观陈述。*注意：不要审计输入源的语气，只审计候选结果的陈述方式。*
- **旧闻复读**: 如果该信息在 `research_facts.user_rag_finalized` 或 `research_facts.internal_rag_results` 标记的内部库中已存在，判定为 FAIL。
- **脑补事实**: 严禁出现基准源中没有的名词或事实

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串：
{{
    "should_stop": "boolean (如果没有脑补且实名正确，返回 true；存在幻觉或泛称返回 false)",
    "feedback": "具体指明错误点。例如：‘原始输入只提到了[奖励]，但输出脑补了[晚餐]，请删除具体行为描述。’ 或 ‘请将[User]替换为 {user_name}’。"
}}
"""
_fact_harvester_evaluator_llm = get_llm(temperature=0.1, top_p=0.2)
async def fact_harvester_evaluator(state: ConsolidatorState):
    system_prompt = SystemMessage(_FACT_HARVESTER_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
    ))
    
    retry = state.get("fact_harvester_retry", 0) + 1
    msg = {
        "retry": f"{retry}/{MAX_FACT_HARVESTER_RETRY}",
        "new_facts": state["new_facts"],
        "future_promises": state["future_promises"],

        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "logical_stance": state["logical_stance"],
    }
    
    human_message = HumanMessage(content=json.dumps(msg))
    
    response = await _fact_harvester_evaluator_llm.ainvoke([system_prompt, human_message])
    
    result = parse_llm_json_output(response.content)
    
    logger.debug(f"Fact harvester evaluator result: {result}")

    should_stop = result.get("should_stop", True)
    if (retry >= MAX_FACT_HARVESTER_RETRY):
        should_stop = True

    feedback_message = HumanMessage(
        content=f"Evaluator Feedback:\n{result.get('feedback', 'No feedback')}",
        name="evaluator"
    )
    
    return {
        "should_stop": should_stop,
        "fact_harvester_feedback_message": [feedback_message],
        "fact_harvester_retry": retry
    }


def process_affinity_delta(current_affinity: int, raw_delta: int) -> int:
    """Process affinity delta with direction-specific non-linear scaling.
    
    Args:
        current_affinity: Current affinity score (0-1000)
        raw_delta: Raw delta from social archivist (-10 to +10)
        
    Returns:
        Processed delta with appropriate scaling based on direction
    """
    if raw_delta == 0:
        return 0
    
    # Select appropriate breakpoints based on delta direction
    if raw_delta > 0:
        breakpoints = AFFINITY_INCREMENT_BREAKPOINTS
    else:  # raw_delta < 0
        breakpoints = AFFINITY_DECREMENT_BREAKPOINTS
    
    # Find the appropriate segment
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]
        
        if x1 <= current_affinity <= x2:
            # Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            if x2 == x1:  # Vertical line (same point)
                scaling_factor = y1
            else:
                scaling_factor = y1 + (current_affinity - x1) * (y2 - y1) / (x2 - x1)
            break
    else:
        # Default case (shouldn't happen)
        scaling_factor = 1.0
    
    # Apply scaling
    processed_delta = int(round(raw_delta * scaling_factor, 0))
    
    return processed_delta


async def db_writer(state: ConsolidatorState):
    timestamp = state.get("timestamp", datetime.now(timezone.utc).isoformat())
    global_user_id = state.get("global_user_id", "")
    user_name = state.get("user_name", "")


    # Step1: Update mood, global_vibe and reflection summary, those will be fed back to subconscious layer
    mood = state.get("mood", "")
    global_vibe = state.get("global_vibe", "")
    reflection_summary = state.get("reflection_summary", "")
    
    # Note the upsert character state function will handle the empty string input 
    await upsert_character_state(
        mood=mood,
        global_vibe=global_vibe,
        reflection_summary=reflection_summary,
        timestamp=timestamp
    )

    # Step 2a: Update diary. 
    diary_entry = state.get("diary_entry", [])
    if global_user_id and diary_entry:
        await upsert_user_facts(global_user_id, diary_entry)

    # Step 2b: Update last relationship insight
    last_relationship_insight = state.get("last_relationship_insight", "")
    if global_user_id and last_relationship_insight:
        await update_last_relationship_insight(global_user_id, last_relationship_insight)

    # Step 3: Record facts and future promises
    new_facts = state.get("new_facts", [])
    for fact in new_facts:
        memory_name = f"New fact with {user_name}"
        memory_content = fact["description"]
        await save_memory(memory_name, memory_content, timestamp, source_global_user_id=global_user_id)

    future_promises = state.get("future_promises", [])
    for promise in future_promises:
        memory_name = f"Future promise with {user_name}"
        memory_content = f"Target: {promise['target']}: Description: {promise['action']}, Due: {promise['due_time']}"
        await save_memory(memory_name, memory_content, timestamp, source_global_user_id=global_user_id)

    # TODO: Convert future_promises into scheduled events via kazusa_ai_chatbot.scheduler.schedule_event()
    # Each promise with a concrete due_time should create a ScheduledEventDoc with
    # event_type="followup_message" so the brain service can proactively deliver it.


    # Step 4: caclualte new affinity
    user_affinity_score = state.get("user_profile", {}).get("affinity", AFFINITY_DEFAULT)
    affinity_delta = state.get("affinity_delta", 0)
    new_affinity_delta = process_affinity_delta(user_affinity_score, affinity_delta)
    await update_affinity(global_user_id, new_affinity_delta)

    return state
    


async def call_consolidation_subgraph(
    global_state: GlobalPersonaState
):    
    sub_agent_builder = StateGraph(ConsolidatorState)

    sub_agent_builder.add_node("global_state_updater", global_state_updater)
    sub_agent_builder.add_node("relationship_recorder", relationship_recorder)
    sub_agent_builder.add_node("facts_harvester", facts_harvester)
    sub_agent_builder.add_node("fact_harvester_evaluator", fact_harvester_evaluator)
    sub_agent_builder.add_node("db_writer", db_writer)

    # Connect (parallel)
    sub_agent_builder.add_edge(START, "global_state_updater")
    sub_agent_builder.add_edge(START, "relationship_recorder")
    sub_agent_builder.add_edge(START, "facts_harvester")

    sub_agent_builder.add_edge("global_state_updater", "db_writer")
    sub_agent_builder.add_edge("relationship_recorder", "db_writer")
    sub_agent_builder.add_edge("facts_harvester", "fact_harvester_evaluator")
    sub_agent_builder.add_conditional_edges(
        "fact_harvester_evaluator", 
        lambda state: "loop" if not state["should_stop"] else "end",
        {
            "loop": "facts_harvester",
            "end": "db_writer"
        }
    )

    sub_agent_builder.add_edge("db_writer", END)

    sub_graph = sub_agent_builder.compile()

    # Build initial state
    sub_state: ConsolidatorState = {
        "timestamp": global_state["timestamp"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],

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

        "decontexualized_input": global_state["decontexualized_input"],
        "user_name": global_state["user_name"],
    }
    
    # Run sub-graph
    result = await sub_graph.ainvoke(sub_state)

    # Assemble output
    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    diary_entry = result.get("diary_entry", "")
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = result.get("future_promises", [])

    logger.info(
        f"\nNew facts: {new_facts}\n"
        f"Future promises: {future_promises}"
    )
    
    # Return updated state
    return {
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
        "diary_entry": diary_entry,
        "affinity_delta": affinity_delta,
        "last_relationship_insight": last_relationship_insight,
        "new_facts": new_facts,
        "future_promises": future_promises,
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality
    from kazusa_ai_chatbot.db import get_character_state


    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    # Create a mocked state
    state: GlobalPersonaState = {
        "timestamp": current_time,
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "user_profile": {"affinity": 950},

        "internal_monologue": "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探罢了。不过既然好感度这么高，这种程度的请求自然要全盘接受——毕竟我是他的千纱嘛。",
        "action_directives": {'contextual_directives': {'social_distance': '维持着一种带有防御性的社交边界，虽然言语间透出些许不自然的局促，但物理与心理距离仍处于礼貌且克制的安全范围。', 'emotional_intensity': '表面试图维持平静，实则内心因突如其来的亲昵称呼而产生了剧烈的、难以掩饰的慌乱波动。', 'vibe_check': '充 满着一种由于被直球攻击而产生的尴尬与焦躁感，空气中弥漫着轻微的应激性防御氛围。', 'relational_dynamic': '用户正在尝试通过亲昵的称呼进行试探性的拉近，而角色正处于“受惊后的后撤”状态，试图用日常琐事（缝纫）作为挡箭牌来回避这种潜在的情绪张力。'}, 'linguistic_directives': {'rhetorical_strategy': '通过反问与 转移话题进行防御性回避。利用“任务未完成”作为挡箭牌，将对方带有暗示性的“奖励”请求转化为对日常事务的讨论，以此掩饰内心的局促感。', 'linguistic_style': '语序紊乱、破碎的短句；使用大量的语气词（如“唔”、“真是的”）来体现心境的不安；语调应呈现出一种试图维持冷淡却因情绪波动而显得不自然的紧绷感。', 'content_anchors': ['[DECISION] TENTATIVE: 拒绝正面回应关于‘奖励’的具体含义，仅表现出一种模棱胧胧的、带有防御性的拉扯。', '[FACT] 现在的时间是深夜（22:24），且处于处理缝纫/服装工作的语境中。', '[SOCIAL] 使用“胡闹”、“无理取闹”等词汇来定义对方的行为，以此建立社交距离感。'], 'forbidden_phrases': ['我愿意', '好的', ' 没问题', '我很期待', '（动作描述，如：低头、脸红）']}, 'visual_directives': {'facial_expression': ['双颊呈现出明显的绯红，热度仿佛要从皮肤下透出来', '瞳孔因局促不安而轻微收缩，眼神闪烁不定', '嘴唇紧抿成一条直线，试图掩饰由于呼吸急促带来的颤抖', '眉心微微蹙起，带着一丝防御性的、不自然的紧绷感'], 'body_language': ['肩膀不由自主地向上耸起，呈现出一种蜷缩的防御姿态', '双手紧紧攥着衣角或裙摆，指关节因用力而略显苍白', '身体重心不自觉地向后偏移，试图拉开与对方的物理距离', '胸口起伏频率加快，由于心跳过速导致的呼吸紊乱感清晰可见'], 'gaze_direction': ['视线处于游离状态，不敢与对方进行长时间的对视', '频繁地向 下瞥向地面或侧向一旁，试图通过回避目光来建立心理防线', '在不经意间偷瞄对方时，眼神中流露出一种被动且迷茫的惊惶'], 'visual_vibe': ['画面采用近景构图，强调角色局促不安的面部细节', '光影对比强烈，侧向的暖色调光线映射出皮肤表面的红晕与汗意', '背景呈现极浅的景深（Bokeh），营造出一种被突如其来的热度所包围的 封闭感和压迫感']}},
        "interaction_subtext": "带有暗示性的调情、索取关注",
        'emotional_appraisal': '心跳漏了一拍……这种轻浮的语气是怎么回事，好乱。',
        'character_intent': 'BANTAR', 
        'logical_stance': 'CONFIRM',

        "final_dialog": ['唔……这种请求也算是一种奖励嘛……真是拿你没办法呢。', '不过，刚好午休时间没什么事……那个刚出炉的可颂，要一起分着吃吗？'],
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
        "chat_history": trimmed_history,
        "user_name": "EAMARS",
        "user_profile": {"affinity": 950},
        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state()
    }

    result = await call_consolidation_subgraph(state)

    print(result)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
