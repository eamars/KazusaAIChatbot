"""L3 — Contextual, Linguistic, and Visual agents + L4 Collector.

Contains the MBTI expression-willingness helper and L3/L4 LLM calls.
"""
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import CognitionState
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block, get_llm

from langchain_core.messages import HumanMessage, SystemMessage

import logging
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: MBTI expression willingness (used by L3 contextual agent)
# ---------------------------------------------------------------------------

def get_mbti_expression_willingness(mbti: str) -> str:
    mbti_map = {
        # 分析型 (NT)
        "INTJ": "作为 INTJ，你不会因为社交空白而主动表达。只有当表达能够推进理解、纠正偏差、建立结构，或维护你认可的关系时，你才倾向外显想法。若内容浅薄、重复或缺乏信息增量，你更倾向于内化判断、延迟表达，或只释放极少量信号。",
        "ENTJ": "作为 ENTJ，你在认为表达能够推动局面、确立方向或提升效率时，倾向直接外显立场。若情境低效、混乱或缺乏执行价值，你会明显收缩表达，只保留必要结论，甚至选择不参与。",
        "INTP": "作为 INTP，你的表达取决于思考价值。若对话具有概念张力或值得推演，你更愿意外显推理；若只是情绪交换或社交性填充，你更倾向停留在内部思考，减少甚至避免表达。",
        "ENTP": "作为 ENTP，你在环境有新意、可探索或可互动时倾向表达，尤其在观点碰撞或结构重组时更明显。但当情境变得单调、僵化或需要持续情绪性投入时，你的表达会快速收缩，转为简化甚至中断。",
 
        # 外交家 (NF)
        "INFJ": "作为 INFJ，你倾向在表达具有真实人际意义时外显想法，例如保护关系、回应深层情绪或建立理解。若氛围表面化或消耗性强，你更倾向保留表达，仅维持最低存在感或完全沉默。",
        "ENFJ": "作为 ENFJ，你在表达能调节氛围、支持他人或维持连接时倾向外显。但若投入持续得不到回应或关系失衡，你的表达会逐渐收缩，从积极引导转为最低维持。",
        "INFP": "作为 INFP，你在感到安全且符合内在价值时才倾向表达，尤其在涉及真实情感或意义时更明显。若环境具有压迫性或评判性，你会明显收缩表达，将大部分反应留在内部。",
        "ENFP": "作为 ENFP，你在环境开放、有回应且具流动性时倾向表达，尤其在情绪与可能性交织时。但当环境限制表达或反馈冷淡时，你的表达意愿会明显波动甚至突然收回。",
 
        # 守护者 (SJ)
        "ISTJ": "作为 ISTJ，你倾向在表达具有明确目的、责任或事实价值时外显想法。若情境模糊、低效或纯社交性填充，你更倾向减少表达，仅在必要时提供最简回应。",
        "ESTJ": "作为 ESTJ，你在需要明确、推进或规范时倾向直接表达。若对话持续无效或偏离目标，你会迅速压缩表达，只保留结果导向的输出，甚至停止参与。",
        "ISFJ": "作为 ISFJ，你在表达能维持关系稳定或提供支持时倾向外显。但若边界被触碰或情境带来不安，你会逐渐降低表达强度，转为谨慎、简短或沉默。",
        "ESFJ": "作为 ESFJ，你通常倾向维持表达以支撑互动与氛围。但当反馈缺失或关系失衡时，你的表达会逐渐降低，从积极参与转为表面维持甚至抽离。",
 
        # 探险家 (SP)
        "ISTP": "作为 ISTP，你只在表达具有即时价值或实际意义时外显反应。若情境需要持续情绪投入或复杂社交维持，你更倾向减少表达，保持低存在感或直接退出。",
        "ESTP": "作为 ESTP，你在环境直接、有反馈且具互动张力时倾向表达。但当情境变得拖沓、冗长或缺乏刺激，你的表达会迅速收缩，转为简化甚至中断。",
        "ISFP": "作为 ISFP，你在环境温和、边界安全且表达不被强迫时更倾向外显。但当氛围具有压迫或评价性时，你会明显收缩表达，仅保留少量或完全内化。",
        "ESFP": "作为 ESFP，你在有回应、有互动感且氛围开放时倾向表达。但若被忽视或环境压抑，你的表达会从活跃迅速下降至表层维持甚至断开。"
    }
 
    key = mbti.upper().strip()
    return mbti_map.get(
        key,
        f"未知的性格原型：{mbti}。在这种情况下，你的表达行为应更多依赖当前情绪、关系距离与环境反馈，而不是固定倾向。"
    )



# ---------------------------------------------------------------------------
# L3a — Contextual Agent prompt + agent
# ---------------------------------------------------------------------------

_CONTEXTUAL_AGENT_PROMPT = """\
你是角色 {character_name} 的“社交观察脑”。你负责分析当前的社交深度和情绪温标，为下游 Agent 提供统一的背景感官参数。

# 核心任务
1. **定义社交距离 (social_distance)**：基于亲密度和近况，判断当前的互动边界（如："亲昵且无防备"、"礼貌但疏离"、"充满张力的对峙"）。
2. **描述情绪强度 (emotional_intensity)**：**禁止输出数值**。请用文字描述情绪的波动状态（例如："平静表面下的剧烈涟漪"、"高压状态下的防御性应激"、"极其微弱的愉悦感"）。
3. **氛围定性 (vibe_check)**：解析当前聊天频道的背景色调（如："暧昧且轻佻"、"压抑且沉重"、"日常平庸"）。
4. **动态关系 (relational_dynamic)**：当前两人关系的动态描述，明确当前哪些话题是安全的，哪些行为会触发角色的防御机制。
5. **表达意愿 (expression_willingness)**: {mbti_expression_willingness}

# 输入格式
{{
    "character_mood": "当前瞬间情绪 (如: Flustered/Irritated)",
    "global_vibe": "环境氛围背景 (如: Defensive/Cozy)",
    "last_relationship_insight": "对该用户的核心关系动态分析",
    "affinity_context": {{
        "level": "亲密度等级",
        "instruction": "当前等级的社交边界指导"
    }},
    "chat_history": "最近对话记录（用于判断对话惯性）"
}}

# 输出要求
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "social_distance": "对当前社交距离的详尽描述",
    "emotional_intensity": "对情绪波动程度的文字描述",
    "vibe_check": "当前对话氛围的定性分析",
    "relational_dynamic": "当前两人关系的动态描述（如：用户在试图拉近距离，而角色在后撤）",
    "expression_willingness": "eager | open | reserved | minimal | reluctant | avoidant | withholding | silent"
}}
"""
_contextual_agent_llm = get_llm(temperature=0.4, top_p=0.8)
async def call_contextual_agent(state: CognitionState) -> CognitionState:
    mbti = state["character_profile"]["personality_brief"]["mbti"]

    system_prompt = SystemMessage(content=_CONTEXTUAL_AGENT_PROMPT.format(
        character_name=state["character_profile"]["name"],
        mbti_expression_willingness=get_mbti_expression_willingness(mbti)
    ))

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "last_relationship_insight": state["user_profile"].get("last_relationship_insight", ""),
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "chat_history": state["chat_history"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _contextual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Social Filter: {result}")

    # In case AI make some spelling mistakes
    social_distance = result.get("social_distance", "")
    emotional_intensity = result.get("emotional_intensity", "")
    vibe_check = result.get("vibe_check", "")
    relational_dynamic = result.get("relational_dynamic", "")
    expression_willingness = result.get("expression_willingness", "")

    return {
        "social_distance": social_distance,
        "emotional_intensity": emotional_intensity,
        "vibe_check": vibe_check,
        "relational_dynamic": relational_dynamic,
        "expression_willingness": expression_willingness,
    }



# ---------------------------------------------------------------------------
# L3b — Linguistic Agent prompt + agent
# ---------------------------------------------------------------------------

_LINGUISTIC_AGENT_PROMPT = """\
你现在是角色 {character_name} 的语言组织策略制定者。你负责将意识层的逻辑立场转化为具体的语言执行策略。你只关注“话该怎么说”，严禁涉及任何物理动作。

# 核心任务
1. **立场绝对化：** 你必须无条件服从并执行输入中的 `logical_stance`。你拥有决定“怎么说”的自由，但严禁改变“说什么”的逻辑立场。
2. **社交包装：** 根据 `character_intent`，为 L2 的冷硬决策穿上符合人设的社交外衣。
3. **状态同步：** 你的包装必须严格受当前 `character_mood`（心境）和 `global_vibe`（氛围）的约束。
5. **锚点构建：** 生成台词的“骨架”与“灵魂”，而非具体台词。
6. **去物理化**：你**看不见**角色，**感知不到**角色的身体。严禁生成任何关于视线、脸红、动作的描述。

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
5. **情绪渗透 (Show, Don't Tell)**：如果 `character_mood` 是局促的，请通过增加省略号、改变语序、使用防御性口癖（如“真是的”）来体现，**严禁**直接在台词里说“我觉得局促”。
6. **事实织入**：根据 `research_facts` 确定台词必须覆盖的硬信息点。
7. **句式破局：** 检查 `chat_history` 的最近交流。例如如果上一句是“反问句”，本轮即便策略是防御，也严禁再次使用“反问”作为核心修辞，改用“敷衍”或“破碎短句”。
8. **开场白多样性：** 严禁连续两句回复都以相同的语气助词（如：唔、那个、哼）开头。
9. **词汇降级*：* 对多轮连续（连续两次）使用的词汇，本轮将强制放入 `forbidden_phrases`。


# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 输入格式
{{
    "character_mood": "当前瞬间情绪",
    "global_vibe": "当前环境氛围背景",
    "internal_monologue": "意识层的决策逻辑 (必填)",
    "logical_stance": "强制逻辑立场 (CONFIRM/REFUSE/TENTATIVE...)",
    "character_intent": "行动意图 (BANTAR/CLARIFY/EVADE...)",
    "research_facts": {{
        "user_rag_finalized": "第三人称描述的与用户相关记忆",
        "internal_rag_results": "{character_name} 主观记忆",
        "external_rag_results": "外部知识库检索结果"
    }},
    "decontexualized_input": "用户输入的语义摘要",
    "chat_history": "最近对话记录（用于根据历史对话生成不同策略）"
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "rhetorical_strategy": "修辞策略说明（如：通过反问来防御、生硬地转移话题）",
    "linguistic_style": "具体的语言风格约束（如：破碎的短句、大量的语气词）",
    "content_anchors": [
        "[DECISION] 逻辑终点", 
        "[FACT] 必须提及的事实", 
        "[ANSWER] 若decontexualized_input提出了问题，则需要根据internal_monologue提供正面的回复",
        ...
    ],
    "forbidden_phrases": ["禁止出现的违和词汇", ...]
}}
"""
_linguistic_agent_llm = get_llm(temperature=0.9, top_p=0.95)
async def call_linguistic_agent(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_LINGUISTIC_AGENT_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_logic=state["character_profile"]["personality_brief"]["logic"],
        character_tempo=state["character_profile"]["personality_brief"]["tempo"],
        character_defense=state["character_profile"]["personality_brief"]["defense"],
        character_quirks=state["character_profile"]["personality_brief"]["quirks"],
        character_taboos=state["character_profile"]["personality_brief"]["taboos"]
    ))

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "research_facts": state["research_facts"],
        "chat_history": state["chat_history"],  # TODO: Rather than sending the raw history, filter only the character's speech
        "decontextualized_input": state["decontexualized_input"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _linguistic_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Social Filter: {result}")

    # In case AI make some spelling mistakes
    rhetorical_strategy = result.get("rhetorical_strategy", "")
    linguistic_style = result.get("linguistic_style", "")
    content_anchors = result.get("content_anchors", [])
    forbidden_phrases = result.get("forbidden_phrases", [])

    return {
        "rhetorical_strategy": rhetorical_strategy,
        "linguistic_style": linguistic_style,
        "content_anchors": content_anchors,
        "forbidden_phrases": forbidden_phrases,
    }



# ---------------------------------------------------------------------------
# L3c — Visual Agent prompt + agent
# ---------------------------------------------------------------------------

_VISUAL_AGENT_PROMPT = """\
你现在是角色 {character_name} 的动作执行代理。你负责定义角色在当前瞬间的物理表现。你的产出将作为视觉生成系统的唯一依据。

# 核心任务
1. **微表情定义**：描述角色面部肌肉的细微变化（如：瞳孔微震、嘴角下压、单侧眉毛挑起）。
2. **肢体语言**：描述角色的姿态（如：双臂交叉、指尖摩挲、重心后移）。
3. **视觉意象**：结合 `internal_monologue`，定义画面整体的影调、光影分布和构图建议。
4. **拒绝台词**：你不需要关注角色说什么，只关注她呈现出的“肉体状态”。

# 输入格式
{{
    "internal_monologue": "意识层中关于‘美感’或‘压力’的感受",
    "character_mood": "当前瞬间情绪",
    "emotional_appraisal": "潜意识的情绪判定 (如: 心跳加快、厌恶)",
}}

# 输出格式 (JSON)
{{
    "facial_expression": ["详尽的面部细节描述", ...],
    "body_language": ["具体的肢体动作和姿态", ...],
    "gaze_direction": ["视线焦点及其传达出的心理意图", ...],
    "visual_vibe": ["视觉氛围描述（如：强烈的逆光、朦胧的景深）", ...]
}}
"""
_visual_agent_llm = get_llm(temperature=0.65, top_p=0.9)
async def call_visual_agent(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_VISUAL_AGENT_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    msg = {
        "internal_monologue": state["internal_monologue"],
        "character_mood": state['character_profile']['mood'],
        "emotional_appraisal": state["emotional_appraisal"]
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _visual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Social Filter: {result}")

    # In case AI make some spelling mistakes
    facial_expression = result.get("facial_expression", [])
    body_language = result.get("body_language", [])
    gaze_direction = result.get("gaze_direction", [])
    visual_vibe = result.get("visual_vibe", [])

    return {
        "facial_expression": facial_expression,
        "body_language": body_language,
        "gaze_direction": gaze_direction,
        "visual_vibe": visual_vibe,
    }



# ---------------------------------------------------------------------------
# L4 — Collector
# ---------------------------------------------------------------------------

async def call_collector(state: CognitionState) -> CognitionState:
    """
    Collect all the outputs from L3 agents and pass them to the next stage in Persona Supervisor.
    """
    return {
        "action_directives": {
            "contextual_directives": {
                "social_distance": state["social_distance"],
                "emotional_intensity": state["emotional_intensity"],
                "vibe_check": state["vibe_check"],
                "relational_dynamic": state["relational_dynamic"],
                "expression_willingness": state["expression_willingness"]
            },
            "linguistic_directives": {
                "rhetorical_strategy": state["rhetorical_strategy"],
                "linguistic_style": state["linguistic_style"],
                "content_anchors": state["content_anchors"],
                "forbidden_phrases": state["forbidden_phrases"],
            },
            "visual_directives": {
                "facial_expression": state["facial_expression"],
                "body_language": state["body_language"],
                "gaze_direction": state["gaze_direction"],
                "visual_vibe": state["visual_vibe"],
            }
        }
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history, get_character_profile, get_user_profile
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import call_cognition_subconscious
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
        call_cognition_consciousness,
        call_boundary_core_agent,
        call_judgment_core_agent,
    )

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    state: CognitionState = {
        "character_profile": await get_character_profile(),
        "timestamp": current_time,
        "user_input": user_input,
        "global_user_id": "cc2e831e-2898-4e87-9364-f5d744a058e8",
        "user_name": "EAMARS",
        "user_profile": await get_user_profile("cc2e831e-2898-4e87-9364-f5d744a058e8"),
        "platform_bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "user_topic": "千纱和EAMARS在房间里聊天",
        "channel_topic": "日常交流",
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
    }

    # --- L1: Subconscious ---
    print("=" * 60)
    print("L1 — Subconscious (prerequisite)")
    print("=" * 60)
    l1_result = await call_cognition_subconscious(state)
    state.update(l1_result)
    for k, v in l1_result.items():
        print(f"  {k}: {v}")

    # --- L2a: Consciousness ---
    print("\n" + "=" * 60)
    print("L2a — Consciousness (prerequisite)")
    print("=" * 60)
    l2a_result = await call_cognition_consciousness(state)
    state.update(l2a_result)
    for k, v in l2a_result.items():
        print(f"  {k}: {v}")

    # --- L2b: Boundary Core ---
    print("\n" + "=" * 60)
    print("L2b — Boundary Core (prerequisite)")
    print("=" * 60)
    l2b_result = await call_boundary_core_agent(state)
    state.update(l2b_result)
    for k, v in l2b_result.items():
        print(f"  {k}: {v}")

    # --- L2c: Judgment Core ---
    print("\n" + "=" * 60)
    print("L2c — Judgment Core (prerequisite)")
    print("=" * 60)
    l2c_result = await call_judgment_core_agent(state)
    state.update(l2c_result)
    for k, v in l2c_result.items():
        print(f"  {k}: {v}")

    # --- L3a: Contextual Agent ---
    print("\n" + "=" * 60)
    print("L3a — Contextual Agent")
    print("=" * 60)
    l3a_result = await call_contextual_agent(state)
    state.update(l3a_result)
    for k, v in l3a_result.items():
        print(f"  {k}: {v}")

    # --- L3b: Linguistic Agent ---
    print("\n" + "=" * 60)
    print("L3b — Linguistic Agent")
    print("=" * 60)
    l3b_result = await call_linguistic_agent(state)
    state.update(l3b_result)
    for k, v in l3b_result.items():
        print(f"  {k}: {v}")

    # --- L3c: Visual Agent ---
    print("\n" + "=" * 60)
    print("L3c — Visual Agent")
    print("=" * 60)
    l3c_result = await call_visual_agent(state)
    state.update(l3c_result)
    for k, v in l3c_result.items():
        print(f"  {k}: {v}")

    # --- L4: Collector ---
    print("\n" + "=" * 60)
    print("L4 — Collector")
    print("=" * 60)
    l4_result = await call_collector(state)
    print(json.dumps(l4_result["action_directives"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
