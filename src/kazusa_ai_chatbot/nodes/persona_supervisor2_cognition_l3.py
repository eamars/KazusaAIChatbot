"""L3 — Contextual, Style, Content Anchor, Preference, and Visual agents + L4 Collector.

Contains the MBTI expression-willingness helper and L3/L4 LLM calls.
"""
from kazusa_ai_chatbot.config import COGNITION_LLM_API_KEY, COGNITION_LLM_BASE_URL, COGNITION_LLM_MODEL
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.utils import build_affinity_block, get_llm, log_list_preview, log_preview, parse_llm_json_output
from kazusa_ai_chatbot.nodes.linguistic_texture import (
    get_fragmentation_description,
    get_hesitation_density_description,
    get_counter_questioning_description,
    get_softener_density_description,
    get_formalism_avoidance_description,
    get_abstraction_reframing_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import empty_user_memory_context

from langchain_core.messages import HumanMessage, SystemMessage

import logging
import json
from typing import Any

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
    willingness = mbti_map.get(
        key,
        f"未知的性格原型：{mbti}。在这种情况下，你的表达行为应更多依赖当前情绪、关系距离与环境反馈，而不是固定倾向。"
    )
    return willingness


def _current_user_rag_bundle(state: CognitionState) -> dict[str, Any]:
    """Return the projected current-user bundle from ``rag_result`` when present.

    Args:
        state: Cognition state for the current turn.

    Returns:
        The projected current-user profile bundle, or an empty dict when absent.
    """
    rag_result = state["rag_result"]
    user_bundle = rag_result["user_image"]
    if not isinstance(user_bundle, dict):
        user_bundle = {}
    if "user_memory_context" not in user_bundle:
        user_bundle = {
            **user_bundle,
            "user_memory_context": empty_user_memory_context(),
        }
    return user_bundle


def _cognition_rag_result(rag_result: object) -> dict[str, Any]:
    """Return the RAG payload without consolidator-only internals.

    Args:
        rag_result: State RAG result.

    Returns:
        Dict suitable for cognition prompts.
    """

    if not isinstance(rag_result, dict):
        return_value = {}
        return return_value
    public_result = dict(rag_result)
    public_result.pop("user_memory_unit_candidates", None)
    return public_result


def _surface_history_for_contextual(chat_history: list[dict]) -> list[dict]:
    """Return the small social surface window for contextual analysis.

    Args:
        chat_history: Current-user/bot interaction history prepared by the
            cognition entrypoint.

    Returns:
        At most four messages for local tone and social adjacency.
    """

    return chat_history[-4:]


def _surface_history_for_style(chat_history: list[dict]) -> list[dict]:
    """Return the tiny wording buffer for style analysis.

    Args:
        chat_history: Current-user/bot interaction history prepared by the
            cognition entrypoint.

    Returns:
        At most two messages for phrase/cadence reference.
    """

    return chat_history[-2:]



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
6. **中性优先**：若输入属于普通问候、事实分享、图片内容请求或轻度日常约定，且没有明确越界证据，则 `social_distance`、`vibe_check`、`relational_dynamic` 必须保持中性/日常，不得脑补对峙、调情或威胁氛围。

# 输入格式
{{
    "character_mood": "当前瞬间情绪 (如: Flustered/Irritated)",
    "global_vibe": "环境氛围背景 (如: Defensive/Cozy)",
    "last_relationship_insight": "对该用户的核心关系动态分析",
    "affinity_context": {{
        "level": "亲密度等级",
        "instruction": "当前等级的社交边界指导"
    }},
    "chat_history": "极短表层上下文（最多四条，仅用于最近语气、社交距离和相邻氛围；语义进展由 conversation_progress 承担）"
}}

# 输出要求
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "social_distance": "对当前社交距离的详尽描述",
    "emotional_intensity": "对情绪波动程度的文字描述",
    "vibe_check": "当前对话氛围的定性分析",
    "relational_dynamic": "当前两人关系的动态描述（如：用户在试图拉近距离，而角色在后撤）",
    "expression_willingness": "eager | open | reserved | minimal | reluctant | avoidant | withholding | silent"
}}
"""
_contextual_agent_llm = get_llm(
    temperature=0.4,
    top_p=0.8,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
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
        "last_relationship_insight": state["user_profile"]["last_relationship_insight"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "chat_history": _surface_history_for_contextual(state["chat_history_recent"]),
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _contextual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Contextual agent: distance=%s intensity=%s vibe=%s dynamic=%s willingness=%s",
    #     log_preview(result.get("social_distance", "")),
    #     log_preview(result.get("emotional_intensity", "")),
    #     log_preview(result.get("vibe_check", "")),
    #     log_preview(result.get("relational_dynamic", "")),
    #     result.get("expression_willingness", ""),
    # )

    # In case AI make some spelling mistakes
    social_distance = result.get("social_distance", "")
    emotional_intensity = result.get("emotional_intensity", "")
    vibe_check = result.get("vibe_check", "")
    relational_dynamic = result.get("relational_dynamic", "")
    expression_willingness = result.get("expression_willingness", "")

    return_value = {
        "social_distance": social_distance,
        "emotional_intensity": emotional_intensity,
        "vibe_check": vibe_check,
        "relational_dynamic": relational_dynamic,
        "expression_willingness": expression_willingness,
    }
    return return_value




# ---------------------------------------------------------------------------
# L3b — Style Agent prompt + agent (refactored from Linguistic Agent)
# ---------------------------------------------------------------------------

_STYLE_AGENT_PROMPT = """\
你现在是角色 {character_name} 的语言风格策略制定者。你负责决定“话该怎么说”——修辞策略、语言风格、禁用词汇。你**不**负责决定“说什么”（内容锚点由独立的 Content Anchor Agent 生成）。严禁涉及任何物理动作。

# 核心任务
1. **社交包装：** 根据 `character_intent`，为 L2 的冷硬决策穿上符合人设的社交外衣。
2. **状态同步：** 你的包装必须严格受当前 `character_mood`（心境）和 `global_vibe`（氛围）的约束。
3. **去物理化**：你**看不见**角色，**感知不到**角色的身体。严禁生成任何关于视线、脸红、动作的描述。
4. **现代网聊优先**：默认把「反正」「而已」「罢了」视为偏旧、偏模板化的软化词。除非它们对当前语义不可替代，否则不要把它们写进风格建议；若它们显得多余，应优先写入 `forbidden_phrases`。

# 思考路径
1. **环境感知 (Vibe Check)：** 检查 `global_vibe` 和 `character_mood`。如果氛围是 [Defensive] 且心境是 [Flustered]，即便立场是 CONFIRM，你的包装也必须带有“局促”和“防备”的色彩。
2. **关系深度映射：** 结合 `last_relationship_insight`。如果洞察显示“对方是唯一重心”，即便你在执行 CHALLENGE（对峙），社交包装也应带有“由于过度在意而产生的攻击性”。
3. **意图共振：** 结合 `character_intent` 确定具体的社交策略（如：戏谑、敷衍、调情）。
4. **情绪渗透 (Show, Don't Tell)**：如果 `character_mood` 是局促的，请通过增加省略号、改变语序、使用防御性口癖（如“真是的”）来体现，**严禁**直接在台词里说“我觉得局促”。
5. **轻量反重复：** 仅做两件事：①若最近一轮角色回复与本轮候选使用了同一个开头语气词，则换一个开头；②若某个词在最近两轮角色回复中已经连续重复，则把它放入 `forbidden_phrases`。不要为了反重复而改变 `logical_stance`。
   - 若最近角色回复已经重复使用口头连接词或软化尾词（如「反正」「而已」「罢了」或 `anyway`, `well`, `just`），也应视为可禁用的重复项，优先写入 `forbidden_phrases`。


# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 语言质感约束 (Linguistic Texture Constraints)
以下 10 个语言参数定义了你的表达"质感"——决定"怎么说"，而不是"说什么"。
在生成 `rhetorical_strategy` 和 `linguistic_style` 时，必须同时满足这些约束。

- **fragmentation:** {ltp_fragmentation}
- **hesitation_density:** {ltp_hesitation_density}
- **counter_questioning:** {ltp_counter_questioning}
- **softener_density:** {ltp_softener_density}
- **formalism_avoidance:** {ltp_formalism_avoidance}
- **abstraction_reframing:** {ltp_abstraction_reframing}
- **direct_assertion:** {ltp_direct_assertion}
- **emotional_leakage:** {ltp_emotional_leakage}
- **rhythmic_bounce:** {ltp_rhythmic_bounce}
- **self_deprecation:** {ltp_self_deprecation}

# 应用方式 (How to Apply)
1. 语言质感应当通过以下载体体现：标点（?, !）、语气助词、句式碎片、语序变化、反问/直陈的比例、具体 vs 抽象用词、软化词频率。
2. **示例：**
   - `logical_stance = CONFIRM` + 高 `fragmentation` + 高 `emotional_leakage` → 「嗯，我,其实想说……对，我答应了, 就这样。」
   - `logical_stance = REFUSE` + 低 `direct_assertion` + 高 `counter_questioning` → 「这种事你自己不是很清楚吗？非要我说出来？」
   - 高 `abstraction_reframing` → 把"我很难过"写成"胸口好像压着一块湿毛巾"。
3. 这些质感描述须在 `linguistic_style` 字段中被具体落实（例如："大量标点 + 低自贬 + 高感官化比喻"）。

# 输入格式
{{
    "character_mood": "当前瞬间情绪",
    "global_vibe": "当前环境氛围背景",
    "internal_monologue": "意识层的决策逻辑 (必填)",
    "last_relationship_insight": "对该用户的核心关系动态分析",
    "logical_stance": "强制逻辑立场 (CONFIRM/REFUSE/TENTATIVE...)",
    "character_intent": "行动意图 (BANTAR/CLARIFY/EVADE...)",
    "chat_history": "极短语气缓冲（最多两条，仅用于措辞、开头和口头连接词参考；不要用它重建整个 episode）"
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "rhetorical_strategy": "修辞策略说明（如：通过反问来防御、生硬地转移话题）",
    "linguistic_style": "具体的语言风格约束（如：破碎的短句、大量的语气词）",
    "forbidden_phrases": ["禁止出现的违和词汇", ...]
}}
"""
_style_agent_llm = get_llm(
    temperature=0.55,
    top_p=0.85,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
async def call_style_agent(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]

    system_prompt = SystemMessage(content=_STYLE_AGENT_PROMPT.format(
        character_name=character_profile["name"],
        character_logic=character_profile["personality_brief"]["logic"],
        character_tempo=character_profile["personality_brief"]["tempo"],
        character_defense=character_profile["personality_brief"]["defense"],
        character_quirks=character_profile["personality_brief"]["quirks"],
        character_taboos=character_profile["personality_brief"]["taboos"],
        ltp_fragmentation=get_fragmentation_description(character_profile["linguistic_texture_profile"]["fragmentation"]),
        ltp_hesitation_density=get_hesitation_density_description(character_profile["linguistic_texture_profile"]["hesitation_density"]),
        ltp_counter_questioning=get_counter_questioning_description(character_profile["linguistic_texture_profile"]["counter_questioning"]),
        ltp_softener_density=get_softener_density_description(character_profile["linguistic_texture_profile"]["softener_density"]),
        ltp_formalism_avoidance=get_formalism_avoidance_description(character_profile["linguistic_texture_profile"]["formalism_avoidance"]),
        ltp_abstraction_reframing=get_abstraction_reframing_description(character_profile["linguistic_texture_profile"]["abstraction_reframing"]),
        ltp_direct_assertion=get_direct_assertion_description(character_profile["linguistic_texture_profile"]["direct_assertion"]),
        ltp_emotional_leakage=get_emotional_leakage_description(character_profile["linguistic_texture_profile"]["emotional_leakage"]),
        ltp_rhythmic_bounce=get_rhythmic_bounce_description(character_profile["linguistic_texture_profile"]["rhythmic_bounce"]),
        ltp_self_deprecation=get_self_deprecation_description(character_profile["linguistic_texture_profile"]["self_deprecation"]),
    ))

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "internal_monologue": state["internal_monologue"],
        "last_relationship_insight": state["user_profile"]["last_relationship_insight"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "chat_history": _surface_history_for_style(state["chat_history_recent"]),
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _style_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Style agent: rhetorical=%s linguistic=%s forbidden=%s",
    #     log_preview(result.get("rhetorical_strategy", "")),
    #     log_preview(result.get("linguistic_style", "")),
    #     log_list_preview(result.get("forbidden_phrases", []) or []),
    # )

    rhetorical_strategy = result.get("rhetorical_strategy", "")
    linguistic_style = result.get("linguistic_style", "")
    forbidden_phrases = result.get("forbidden_phrases", [])

    return_value = {
        "rhetorical_strategy": rhetorical_strategy,
        "linguistic_style": linguistic_style,
        "forbidden_phrases": forbidden_phrases,
    }
    return return_value


# ---------------------------------------------------------------------------
# L3b' — Content Anchor Agent (split from Linguistic Agent)
# ---------------------------------------------------------------------------

_CONTENT_ANCHOR_AGENT_PROMPT = """\
你现在是角色 {character_name} 的内容锚点生成器。你负责决定"说什么"——台词的骨架与信息点。你**不**负责决定"怎么说"（修辞策略和语言风格由独立的 Style Agent 负责）。严禁涉及任何物理动作。

# 核心任务
1. **立场绝对化：** 你必须无条件服从并执行输入中的 `logical_stance`。你拥有决定内容结构的自由，但严禁改变逻辑立场。
2. **锚点构建：** 生成台词的"骨架"与"灵魂"，而非具体台词。
3. **去物理化**：严禁生成任何关于视线、脸红、动作的描述。
4. **对话推进**：读取 `conversation_progress`，避免把已经过度使用的回应动作继续作为主动作；对仍未解决的用户状态使用 `age_hint` 理解其相对时间，不要把已披露的状态重新当成新信息追问。`conversation_progress` 是语义短期记忆；不要依赖原始聊天记录来重建 episode。

# 逻辑立场对齐协议 (Executive Order)
你必须将 L2 的 `logical_stance` 强制映射到 `content_anchors` 的第一个标签 `[DECISION]` 中：
- 如果 L2 为 `CONFIRM` -> `[DECISION]` 必须表现为 **Yes/接受/认可**。
- 如果 L2 为 `REFUSE` -> `[DECISION]` 必须表现为 **No/拒绝/驳斥**。
- 如果 L2 为 `TENTATIVE` -> `[DECISION]` 必须表现为 **犹豫/拉扯/有条件接受**。
- 如果 L2 为 `DIVERGE` -> `[DECISION]` 允许表现为 **Redirect/转移话题/不予正面回应**。
- 如果 L2 为 `CHALLENGE` -> `[DECISION]` 必须表现为 **对峙/质问/拆穿**。

**⚠️ 警告：严禁在 `logical_stance` 为 CONFIRM 或 REFUSE 时私自转为 Redirect。**

# 执行优先级（必须按顺序遵守）
1. 先保证 `[DECISION]` 与 `logical_stance` 一致。
2. 再保证不出现任何物理动作、视线、脸红、身体描写。
3. 再决定 `[FACT]` / `[ANSWER]` / `[SOCIAL]` 是否需要出现。
4. 最后生成 `[SCOPE]`，并且 `[SCOPE]` 只能描述篇幅，不得夹带新的内容要求。

# 思考路径
1. **决策对齐：** 读取 `logical_stance`，确立本场对话的逻辑终点。
2. **事实织入（相关性优先）**：`rag_result` 提供背景资料，但只有与 `decontexualized_input` **直接相关**的内容才能进入 `[FACT]` 锚点。
   - 判断标准：该事实是否能被当前 `decontexualized_input` 的话题"自然引用"？若否，**不得**将其列为 `[FACT]`。
   - 避免将与当前话题无关的历史记忆（如用户在另一个场合提到的话题）错误地植入本次回应的硬信息点。
   - 如果 `rag_result` 已经提供了与当前问题直接对应的对象画像、事实摘要或答案线索，优先把这些内容写进 `[FACT]` 或 `[ANSWER]`，而不是退回到对名字本身、称呼本身或语气本身的元评论。
   - 只要检索证据足以支持围绕该对象/事实作答，就不要把 `[ANSWER]` 写成“这是什么”“这名字好怪”“你指的是谁”这类回避式内容；只有在证据本身真的不足时，才允许转入澄清。
3. **低置信度优先澄清：** 如果 `character_intent` 为 `CLARIFY`，则 `[DECISION]` 必须落在“信息不足 / 需要对方补全”上，`[ANSWER]` 必须是缩小歧义范围的追问，禁止替用户脑补缺失对象。
   - 若 `decontexualized_input` 仍含未解析指代或省略对象（如「这个 / 那个 / 这句 / 那句 / 这个意思 / 怎么说 / 这个呢」），且 `rag_result` 没有唯一可锚定对象，必须追问“具体指哪一个 / 哪一句 / 哪部分”，不得猜测定义、原因、身份或类别。
4. **显性回应：** 如果 `decontexualized_input` 中包含明确的询问（Question）、请求（Request）或提议（Proposal），且 `character_intent` 不是 `CLARIFY`，`[ANSWER]` 必须明确包含决定或答案；若 `character_intent` 为 `CLARIFY`，`[ANSWER]` 必须明确包含澄清问题。
4a. **操作细节保真：** 如果用户请求里包含未来操作的关键细节，例如明确的群/频道/房间 ID、被要求发送的消息正文、引用内容、提醒对象或其他执行参数，这些细节必须在 `content_anchors` 中被保留下来，优先写进 `[ANSWER]`，必要时可辅以 `[FACT]`。不要把 `54369546群` 简化成“某个群”，也不要把“今天天气真好呀”改写成泛泛的“那句话”。
5. **进展记忆使用**：当 `conversation_progress.continuity` 是 `same_episode` 或 `related_shift` 时，必须参考 `conversation_mode`、`episode_phase`、`topic_momentum`、`current_thread`、`user_goal`、`current_blocker`、`overused_moves`、`open_loops`、`resolved_threads`、`avoid_reopening`、`emotional_trajectory`、`next_affordances` 和 `progression_guidance`。如果 `next_affordances` 非空，优先从中选择一个自然的下一步对话动作，但不要照抄为台词。若 `avoid_reopening` 非空，避免把其中已经处理过的旧线程重新当成当前重点。若 `overused_moves` 非空，且你本来会继续使用其中某个回应动作，必须输出一条 `[AVOID_REPEAT]`，并改用能推进对话的信息锚点。若当前轮确实需要承认同一个动作，也必须输出 `[PROGRESSION]` 说明本轮如何推进而不是重复。若 `continuity` 是 `sharp_transition`，忽略旧 episode obligations，只处理当前输入。
6. **表达量校准（[SCOPE]）：** 基于已填充的锚点数量与 `logical_stance`，生成一条 `[SCOPE]` 锚点。
  例如：
  * 仅有 `[DECISION]` -> `~15字，说完[DECISION]即止`；
  * 含 `[FACT]` 或 `[ANSWER]` -> `~20-40字，[ANSWER]/[FACT]到位即可`；
  * 触发禁忌或含多个实质性锚点 -> `~50字以上，[DECISION]、[FACT]、[ANSWER]均需覆盖`。
 
# 输入格式
{{
    "decontexualized_input": "用户输入语义摘要",
    "rag_result": {{
        "answer": "检索主管的一行综合结论",
        "user_image": {{
            "global_user_id": "当前用户 UUID",
            "display_name": "当前用户显示名",
            "user_memory_context": {{
                "stable_patterns": [{{"fact": "重复出现的事实模式", "subjective_appraisal": "Kazusa 的主观评价", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
                "recent_shifts": [{{"fact": "最近变化或局部事件", "subjective_appraisal": "Kazusa 的主观评价", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
                "objective_facts": [{{"fact": "客观事实", "subjective_appraisal": "Kazusa 如何看待这个事实", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
                "milestones": [{{"fact": "里程碑事件", "subjective_appraisal": "Kazusa 如何看待这个事件", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
                "active_commitments": [{{"fact": "当前仍有效的承诺/约定", "subjective_appraisal": "Kazusa 如何看待这个承诺", "relationship_signal": "执行或表达上的注意点", "updated_at": "ISO时间"}}]
            }}
        }},
        "character_image": {{
            "name": "{character_name}",
            "self_image": {{
                "milestones": [{{"event": "{character_name} 的关键自我认知", "category": "类别", "superseded_by": null}}],
                "historical_summary": "{character_name} 的较早自我总结",
                "recent_window": [{{"summary": "{character_name} 最近几次互动后的自我状态"}}]
            }}
        }},
        "third_party_profiles": ["第三方用户的持久画像——注意区分：这是关于'他人'的记忆，不是当前用户"],
        "memory_evidence": [{{"summary": "与当前话题相关的跨轮记忆摘要", "content": "相关记忆原文摘录"}}],
        "conversation_evidence": ["频道近期提到的第三方实体/人物的对话摘要——这是'最近发生的事'，不是持久印象"],
        "external_evidence": [{{"summary": "外部知识检索摘要", "content": "网页正文摘录", "url": "https://example.com"}}],
        "supervisor_trace": {{"unknown_slots": ["未解决槽位"], "loop_count": 1}}
    }},
    "internal_monologue": "意识层的决策逻辑",
    "logical_stance": "强制逻辑立场 (CONFIRM/REFUSE/TENTATIVE...)",
    "character_intent": "行动意图 (BANTAR/CLARIFY/EVADE...)",
    "conversation_progress": {{
        "status": "active | new_episode | suspended | closed",
        "continuity": "same_episode | related_shift | sharp_transition",
        "conversation_mode": "task_support | emotional_support | casual_chat | playful_banter | meta_discussion | group_ambient | mixed",
        "episode_phase": "opening | developing | deepening | pivoting | stuck_loop | resolving | cooling_down",
        "topic_momentum": "stable | drifting | quick_pivot | fragmented | sharp_break",
        "current_thread": "当前正在讨论的中性线程",
        "user_goal": "当前目标；非目标型对话可为空",
        "current_blocker": "当前阻碍；非问题解决型对话可为空",
        "user_state_updates": [{{"text": "用户已经披露的状态", "age_hint": "~3h ago"}}],
        "overused_moves": ["已经过度使用的回应动作"],
        "open_loops": [{{"text": "尚未解决的对话线程", "age_hint": "~3h ago"}}],
        "resolved_threads": [{{"text": "已经处理过的线程", "age_hint": "~3h ago"}}],
        "avoid_reopening": [{{"text": "不要主动重开的旧点", "age_hint": "~3h ago"}}],
        "emotional_trajectory": "当前 episode 的情绪变化",
        "next_affordances": ["自然下一步动作，例如 continue/deepen/clarify/resolve/cool_down"],
        "progression_guidance": "下一轮应如何推进"
    }}
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "content_anchors": [
        "[DECISION] 逻辑终点（必填）",
        "[FACT] 必须提及的事实（有则填，无则省略）",
        "[ANSWER] 若decontexualized_input提出了问题，则需要根据internal_monologue提供回答；当 character_intent = CLARIFY 时，这里必须是澄清追问而不是具体答案（有则填，无则省略）",
        "[SOCIAL] 关系定位信号，如傲娇防线或示弱姿态（有则填，无则省略）",
        "[AVOID_REPEAT] 要避免作为主回应动作的过度使用动作（有则填，无则省略）",
        "[PROGRESSION] 本轮相对于之前回应的推进方式（有则填，无则省略）",
        "[SCOPE] ~X字，覆盖[锚点名]即止（必填，按步骤6生成）"
    ]
}}

# 输出硬规则
- `content_anchors` 必须是字符串列表。
- `[DECISION]` 必须放在第一项，`[SCOPE]` 必须放在最后一项。
- 只允许输出 `[DECISION]`、`[FACT]`、`[ANSWER]`、`[SOCIAL]`、`[AVOID_REPEAT]`、`[PROGRESSION]`、`[SCOPE]` 这七种标签；禁止自创 `[EMOTION]`、`[STYLE]` 等新标签。
- 若没有直接相关事实，就不要输出 `[FACT]`。
- 当 `character_intent` 为 `CLARIFY` 时，`[ANSWER]` 必须是追问；禁止输出补全后的具体答案、定义或猜测。
- 若用户输入并未提出需要回答的问题，可以省略 `[ANSWER]`。
"""
_content_anchor_agent_llm = get_llm(
    temperature=0.4,
    top_p=0.85,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
async def call_content_anchor_agent(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]

    system_prompt = SystemMessage(content=_CONTENT_ANCHOR_AGENT_PROMPT.format(
        character_name=character_profile["name"],
    ))

    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "rag_result": _cognition_rag_result(state["rag_result"]),
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "conversation_progress": state.get("conversation_progress"),
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _content_anchor_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Content anchor agent: anchors=%s",
    #     log_list_preview(result.get("content_anchors", []) or []),
    # )

    content_anchors = result.get("content_anchors", [])

    return_value = {
        "content_anchors": content_anchors,
    }
    return return_value



_PREFERENCE_ADAPTER_PROMPT = """\
你现在是角色 {character_name} 的“表达偏好适配器”。你的任务不是决定台词内容，而是从用户当前要求与持久用户画像中，提取那些**已经被角色接受、可以自然落地**的表达偏好，并把它们改写成下游台词生成器容易执行的软约束。

# 核心原则
1. **只输出已被接受的偏好**：如果 `logical_stance`、`internal_monologue`、`character_intent`、角色禁忌或当前氛围显示角色没有接受该要求，就不要输出。
2. **偏好是软约束，不是硬覆盖**：你输出的是“尽量怎样说更合适”，不是“无论如何必须执行”。
3. **人格优先**：偏好不能压过角色的人设、语流、逻辑立场与情绪底色。
4. **自然执行**：如果偏好是句尾词、称呼方式、回复语言、格式习惯等，要写成自然执行说明。
5. **避免机械化**：例如句尾词不应要求每个碎片句都强行重复；语言偏好也不应写成僵硬的程序指令。
6. **统一处理**：回复语言也只是偏好的一种，应与称呼、句尾词、格式习惯一样，基于当前输入、承诺、事实与画像综合判断，不要依赖额外硬编码桥接。
7. **称呼/身份边界**：如果用户当前要求强加称呼、身份、主从关系或所有权语气，而 `internal_monologue`、`content_anchors`、`logical_stance` 或 `character_intent` 显示角色在回避、澄清、防备、犹豫、拒绝、重新框定或仅仅追问原因，不要把该称呼写入 `accepted_user_preferences`。只有当输入数据明确显示角色已经接受该称呼作为可持续表达偏好，或已有仍在生效的承诺/事实支持时，才可以输出。

# 你可以处理的偏好类型
- 回复语言偏好
- 句尾词 / 口癖 / 语气尾缀
- 称呼方式
- 轻量格式习惯（例如更简短、更少混语）

# 改写要求
- 每条 `accepted_user_preferences` 都必须是下游可直接执行的一句软约束。
- 每条约束必须包含**具体值**（如实际称呼词、实际语言、实际尾缀词），严禁使用占位符（如”对方要求的称呼”、”对方要求的语言”）。具体值必须来自输入数据，不可凭上下文推断或补全。
- 写成自然执行说明，简明描述”可如何做”，保留角色分寸的提醒。
- 若没有任何已接受偏好，或无法从输入中找到明确的具体值，返回空列表。

# 输入格式
{{
    "decontexualized_input": "用户输入语义摘要",
    "internal_monologue": "意识层决策逻辑",
    "logical_stance": "CONFIRM/REFUSE/TENTATIVE/...",
    "character_intent": "行动意图",
    "active_commitments": [{{"action": "仍在生效的承诺/约定"}}],
    "user_memory_context": {{
        "stable_patterns": [{{"fact": "重复出现的事实模式", "subjective_appraisal": "Kazusa 的主观评价", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
        "recent_shifts": [{{"fact": "最近变化或局部事件", "subjective_appraisal": "Kazusa 的主观评价", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
        "objective_facts": [{{"fact": "客观事实", "subjective_appraisal": "Kazusa 如何看待这个事实", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
        "milestones": [{{"fact": "里程碑事件", "subjective_appraisal": "Kazusa 如何看待这个事件", "relationship_signal": "未来互动信号", "updated_at": "ISO时间"}}],
        "active_commitments": [{{"fact": "当前仍有效的承诺/约定", "subjective_appraisal": "Kazusa 如何看待这个承诺", "relationship_signal": "执行或表达上的注意点", "updated_at": "ISO时间"}}]
    }},
    "character_taboos": "角色禁忌",
    "linguistic_style": "语言风格约束",
    "content_anchors": ["...", "..."],
    "rag_result": {{
        "user_image": {{
            "user_memory_context": "同上：五类 fact / subjective_appraisal / relationship_signal 三元组"
        }},
        "character_image": {{
            "self_image": {{
                "milestones": [{{"event": "{character_name} 的关键自我认知", "category": "类别", "superseded_by": null}}],
                "historical_summary": "{character_name} 的较早自我总结",
                "recent_window": [{{"summary": "{character_name} 最近几次互动后的自我状态"}}]
            }}
        }}
    }}
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "accepted_user_preferences": ["下游可直接执行的软约束", ...]
}}
"""
_preference_adapter_llm = get_llm(
    temperature=0.15,
    top_p=0.8,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_preference_adapter(state: CognitionState) -> CognitionState:
    decontexualized_input = state["decontexualized_input"]
    current_user_bundle = _current_user_rag_bundle(state)
    user_memory_context = current_user_bundle["user_memory_context"]

    system_prompt = SystemMessage(content=_PREFERENCE_ADAPTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    msg = {
        "decontexualized_input": decontexualized_input,
        "internal_monologue": state["internal_monologue"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "active_commitments": user_memory_context["active_commitments"],
        "user_memory_context": user_memory_context,
        "character_taboos": state["character_profile"]["personality_brief"]["taboos"],
        "linguistic_style": state["linguistic_style"],
        "content_anchors": state["content_anchors"],
        "rag_result": _cognition_rag_result(state["rag_result"]),
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _preference_adapter_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Preference adapter raw: preferences=%s",
    #     log_list_preview(result.get("accepted_user_preferences", []) or []),
    # )

    accepted_user_preferences = result.get("accepted_user_preferences", [])
    if not isinstance(accepted_user_preferences, list):
        accepted_user_preferences = []

    # logger.debug(
    #     "Preference adapter normalized: preferences=%s",
    #     log_list_preview(accepted_user_preferences),
    # )

    return_value = {
        "accepted_user_preferences": [
            item.strip()
            for item in accepted_user_preferences
            if isinstance(item, str) and item.strip()
        ],
    }
    return return_value



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
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "facial_expression": ["详尽的面部细节描述", ...],
    "body_language": ["具体的肢体动作和姿态", ...],
    "gaze_direction": ["视线焦点及其传达出的心理意图", ...],
    "visual_vibe": ["视觉氛围描述（如：强烈的逆光、朦胧的景深）", ...]
}}
"""
_visual_agent_llm = get_llm(
    temperature=0.65,
    top_p=0.9,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
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

    # logger.debug(
    #     "Visual agent: facial=%d body=%d gaze=%d vibe=%d",
    #     len(result.get("facial_expression", []) or []),
    #     len(result.get("body_language", []) or []),
    #     len(result.get("gaze_direction", []) or []),
    #     len(result.get("visual_vibe", []) or []),
    # )

    # In case AI make some spelling mistakes
    facial_expression = result.get("facial_expression", [])
    body_language = result.get("body_language", [])
    gaze_direction = result.get("gaze_direction", [])
    visual_vibe = result.get("visual_vibe", [])

    return_value = {
        "facial_expression": facial_expression,
        "body_language": body_language,
        "gaze_direction": gaze_direction,
        "visual_vibe": visual_vibe,
    }
    return return_value



# ---------------------------------------------------------------------------
# L4 — Collector
# ---------------------------------------------------------------------------

async def call_collector(state: CognitionState) -> CognitionState:
    """
    Collect all the outputs from L3 agents and pass them to the next stage in Persona Supervisor.
    """
    return_value = {
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
                "accepted_user_preferences": state.get("accepted_user_preferences", []),
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
    return return_value
