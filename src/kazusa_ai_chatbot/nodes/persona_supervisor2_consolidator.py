"""Stage 4: consolidator subgraph.

Wraps the post-dialog reflection pipeline. Runs three parallel reflection
nodes (``global_state_updater``, ``relationship_recorder``, ``facts_harvester``),
an evaluator loop over ``facts_harvester``, and a single ``db_writer`` that
commits everything to MongoDB and invalidates the RAG cache.

Stage-4a additions:

* A unified ``metadata`` bundle threaded through every node, seeded from
  the RAG metadata produced in Stage 3 and accumulated at each step.
* The ``db_writer`` now routes diary entries / objective facts through the
  new structured helpers (``upsert_character_diary`` / ``upsert_objective_facts``),
  invalidates the matching RAG cache namespaces after a successful commit,
  bumps the per-user RAG version, and schedules ``future_promise`` events
  through ``kazusa_ai_chatbot.scheduler`` so the bot can act on promises when
  they come due.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import (
    AFFINITY_DECREMENT_BREAKPOINTS,
    AFFINITY_DEFAULT,
    AFFINITY_INCREMENT_BREAKPOINTS,
    MAX_FACT_HARVESTER_RETRY,
)
from kazusa_ai_chatbot.db import (
    CharacterDiaryEntry,
    MemoryDoc,
    ObjectiveFactEntry,
    ScheduledEventDoc,
    build_memory_doc,
    increment_rag_version,
    save_memory,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_diary,
    upsert_character_state,
    upsert_objective_facts,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.scheduler import schedule_event
from kazusa_ai_chatbot.utils import build_affinity_block, get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


# ── Cache-invalidation thresholds ───────────────────────────────────
# |affinity_delta| > this clears all of the user's cache entries (major mood
# shift → every cached view is now suspect).
AFFINITY_CACHE_NUKE_THRESHOLD = 50


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values overwriting a's."""
    result = dict(a)
    result.update(b)
    return result


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
    character_profile: dict

    # Facts
    research_facts: dict

    # User related
    decontexualized_input: str

    # Stage-4a metadata bundle (seeded from RAG metadata, accumulated per node).
    metadata: Annotated[dict, _merge_dicts]

    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    diary_entry: list[str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: list[dict]
    future_promises: list[dict]
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
async def global_state_updater(state: ConsolidatorState) -> dict:
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
async def relationship_recorder(state: ConsolidatorState) -> dict:
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
- **系统时间**: {timestamp}

# 核心审计准则 (Audit Standards)
1. **身份锚定 [必须执行]**:
   - `decontexualized_input` 的内容始终是 **{user_name}** 在表达。
   - `content_anchors` 的内容始终是 **{character_name}** 在做决定。
   - 严禁出现身份倒置（如：将 {user_name} 写完作业记在 {character_name} 头上）。

2. **事实 (new_facts) 判定标准**:
   - **记录**：具有长期稳定性的**属性级陈述**（如：{user_name}的职业、住址、对某物的长期厌恶/偏好）。
   - **记录**：从 `research_facts.external_rag_results` 中提取的**新**信息。
   - **严禁记录**：瞬态动作、对话内容、以及任何关于“奖励”、“打算”、“计划”的内容。
   - **去重**：如果 `research_facts.user_rag_finalized` 或 `research_facts.internal_rag_results` 中已存在相似画像，严禁重复提取。

3. **承诺 (future_promises) 判定标准 [核心逻辑]**:
   - **所有关于“以后、今晚、下次、奖励、惩罚”的内容，必须且只能记录在这里。**
   - 必须包含：[触发条件] + [谁对谁做] + [具体动作]。
   - 承诺时间计算：若角色提出，或者答应了一个将来会发生的事件，则需要根据当前时间推算 `due_time`。
   - `due_time` 推算规则：若输入出现明确时间线索（如“今晚/明早/明天早上/下周末”），必须换算为 ISO 8601 时间戳；仅在完全缺失时间线索时才允许 `null`。
   - **硬约束**：当 `decontexualized_input` 或 `content_anchors` 含有“今晚/晚上/明天/明早/早上/下午/下周”等时间词时，`due_time` 禁止为 `null`。
   - 相对时间默认映射（用于降低歧义）：
     - “今晚/晚上” -> 当日 21:00（若当前已过 21:00，则次日 21:00）
     - “明早/明天早上” -> 次日 08:00
     - “明天下午” -> 次日 15:00
     - “明天晚上” -> 次日 21:00
   - 计算时区使用系统时间 `{timestamp}` 的本地时区；输出必须为 ISO 8601。
   - 若 `decontexualized_input` 是用户对未来行为的请求（如“晚上奖励我”），且 `logical_stance` 为 `CONFIRM`（或等价的认可态），默认视为可落库的“待履行承诺”，不要漏记。
   - 若出现冲突信号（如 `content_anchors` 明确拒绝且 `logical_stance` 非确认），再选择不记录。
   - 对“软性答应/模糊同意”（例如“拿你没办法”“那就这样吧”）也可记录为 promise，动作写成中性可执行表述。
   - `action` 只写“可执行承诺本体”，不得写时间推测或计划态词汇。**禁止出现**：`计划`、`打算`、`准备`、`想要`、`可能`、`也许`、`明天`、`今晚`、`下次` 等时间/意图词。
   - 时间信息统一写入 `due_time`；若无法确定具体时间可为 `null`，但不要把时间短语塞进 `action`。
   - `action` 必须是去叙事的承诺句，推荐模板：`[执行者]在[触发条件]对[对象]执行[动作]` 或 `[执行者]将对[对象]执行[动作]`。
   - `action` 的触发条件允许写“完成作业后/满足约定后”等条件，但不允许写具体时间点。
   - 若同时存在“触发条件 + 时间线索”，两者都保留：触发条件留在 `action`，时间线索写入 `due_time`。
   - 只有在“纯提问且无任何认可信号（含 logical_stance 与 content_anchors）”时，才不要生成 promise。

5. **future_promises 示例（必须遵守）**:
   - ✅ 正确：
     - `action`: `杏山千纱在EAMARS完成作业后对EAMARS执行奖励`
     - `due_time`: `2026-04-19T06:00:00+12:00`
   - ❌ 错误：
     - `action`: `杏山千纱今晚对EAMARS执行奖励`
     - `action`: `杏山千纱计划于明天早上奖励EAMARS`
     - `action`: `杏山千纱打算下次奖励EAMARS`

6. **拒绝复读**:
   - 严禁记录“某人说了某话”。如果信息已经由对话历史承载，且不涉及长期画像更新，则返回空列表。

# 闭环反馈指南
- 在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)。
- 你需要根据 Evaluator Feedback 对输出做出相应的修正。
- 对未提及内容不要做修改

# RAG 元信息（仅供参考）
- `rag_metadata` 提供了上游 RAG 的 cache_hit / depth / confidence 等信号。当 `cache_hit=true` 或 `depth=SHALLOW` 时，`research_facts` 的覆盖面可能有限，谨慎判断是否“已知画像”。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "事实所属的主语实名（如 {user_name}、{character_name}、或具体物品/地点名）",
            "category": "事实类别标签（如 occupation、location、preference、hobby、relationship、health、schedule、personality 等）",
            "description": "属性级的客观陈述。格式：'[主语] + [属性/状态]'。示例：'{user_name}住在新西兰奥克兰' 或 '{user_name}对猫毛过敏'。严禁使用叙事句式（如'在某情境下做了某事'）。"
        }}
    ],
    "future_promises": [
        {{
            "target": "{user_name} / {character_name}",
            "action": "[姓名]将对[对象]执行[具体动作]（仅承诺本体，不含计划/时间词）",
            "due_time": "ISO 8601 时间戳（如 2026-04-19T06:00:00+12:00），无法确定则为 null"
        }}
    ]
}}
"""
_facts_harvester_llm = get_llm(temperature=0.0, top_p=0.95)
async def facts_harvester(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACTS_HARVESTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        timestamp=state["timestamp"],
    ))

    metadata = state.get("metadata", {}) or {}
    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "logical_stance": state["logical_stance"],
        "rag_metadata": {
            "cache_hit": metadata.get("cache_hit", False),
            "depth": metadata.get("depth", "DEEP"),
            "depth_confidence": metadata.get("depth_confidence", 0.0),
            "rag_sources_used": metadata.get("rag_sources_used", []),
            "response_confidence": metadata.get("response_confidence", {}),
        },
    }

    human_message = HumanMessage(content=json.dumps(msg))

    # Trim evaluator feedback to the first + latest three messages.
    feedback = state.get("fact_harvester_feedback_message", []) or []
    if len(feedback) > 3:
        recent_messages = [feedback[0]] + feedback[-3:]
    else:
        recent_messages = feedback

    response = await _facts_harvester_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)

    logger.debug(f"Facts harvester result: {result}")

    return {
        "new_facts": result.get("new_facts", []),
        "future_promises": result.get("future_promises", []),
    }


_FACT_HARVESTER_EVALUATOR_PROMPT = """\
你负责审计 Fact Recorder 生成的 JSON 数据。你的核心目标是：**对比“基准源”，核查“候选结果”的准确性和格式合规性。**

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

# 2.1 与 Harvester 对齐的判定口径（必须遵守）
- `new_facts` 与 `future_promises` 是两个独立通道：
  - `new_facts` 要求“属性级事实陈述”。
  - `future_promises.action` 要求“可执行承诺陈述”，**不适用** `new_facts.description` 的属性句式审计规则。
- 当输入没有新增稳定事实时，`new_facts: []` 是合法结果，**不得仅因为空判定失败**。
- 当输入没有明确未来承诺时，`future_promises: []` 也是合法结果。

# 审计红线 (Red Lines)
- **对象倒置**:
  * `decontexualized_input` 里的动作必须记在 `{user_name}` 账上。如果 Recorder 记在 `{character_name}` 头上，立刻拦截。
- **分类错误 [严重]**:
    - 例如：将带有 “未来”、“打算”、“今晚”或任何有许诺时间性质的行为作为 `new_facts`。这是禁止的行为。
    - 例如：将过去发生的事实存入 `future_promises` 也同样是禁止的行为
- **冗余复读**:
    - 检查候选结果是否只是在复读对话（如“某人问...”）。
    - 必须转换为客观陈述。*注意：不要审计输入源的语气，只审计候选结果的陈述方式。*
- **旧闻复读**: 如果该信息在 `research_facts.user_rag_finalized` 或 `research_facts.internal_rag_results` 标记的内部库中已存在，判定为 FAIL。
- **脑补事实**: 严禁出现基准源中没有的名词或事实
- **描述格式违规**: `new_facts` 中的 `description` 必须是属性级陈述（如 '{user_name}住在奥克兰'），严禁使用叙事句式（如 '在某情境下做了某事'）。若发现叙事句式，要求改写为属性陈述。
- **类别缺失或不当**: `new_facts` 中的 `category` 必须是有意义的英文标签（如 occupation、location、preference、hobby 等）。若缺失、为空、或为无意义的 "general"，要求补充具体类别。
- **承诺 action 审计标准（专用于 `future_promises`）**:
  - 合格条件：表达“谁对谁做什么”的可执行承诺，不是对话复读（如“他说/她问/我觉得”）。
  - 可以接受两种写法：
    1) 不含时间词的承诺本体（推荐）；
    2) 含时间词（如“今晚/明早”）的承诺句，但语义仍是承诺执行动作。
  - 仅在以下情况判 FAIL：
    - `action` 是纯计划/猜测（如“可能、也许、打算、准备”）且无明确执行动作；
    - `action` 只是复述对话或主观感受；
    - `action` 与 `target`/基准源主体明显不一致。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串：
{{
    "should_stop": "boolean (如果没有脑补且实名正确且格式合规，返回 true；仅在违反上述明确红线时返回 false。注意：new_facts 为空本身不构成错误。)",
    "feedback": "具体指明错误点。若无实质错误，返回 '通过审计，无需修改'。禁止输出'请确认是否没有新事实'这类非错误性质建议。",
    "contradiction_flags": "可选字符串列表，列举与 research_facts 直接冲突的条目 id 或描述；无冲突则返回 []"
}}
"""
_fact_harvester_evaluator_llm = get_llm(temperature=0.1, top_p=0.5)
async def fact_harvester_evaluator(state: ConsolidatorState) -> dict:
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
    if retry >= MAX_FACT_HARVESTER_RETRY:
        should_stop = True

    feedback_message = HumanMessage(
        content=f"Evaluator Feedback:\n{result.get('feedback', 'No feedback')}",
        name="evaluator",
    )

    # Propagate evaluator metadata so db_writer can see contradiction flags + retry count.
    contradiction_flags = result.get("contradiction_flags") or []
    existing_meta = state.get("metadata", {}) or {}
    metadata = {
        **existing_meta,
        "fact_harvester_retry": retry,
        "evaluator_passes": existing_meta.get("evaluator_passes", 0) + 1,
        "contradiction_flags": contradiction_flags,
    }

    return {
        "should_stop": should_stop,
        "fact_harvester_feedback_message": [feedback_message],
        "fact_harvester_retry": retry,
        "metadata": metadata,
    }


def process_affinity_delta(current_affinity: int, raw_delta: int) -> int:
    """Scale a raw affinity delta by direction-specific breakpoints.

    Args:
        current_affinity: Current affinity score (0-1000).
        raw_delta: Raw delta from the relationship recorder (-10..+10).

    Returns:
        Scaled delta with sign preserved.
    """
    if raw_delta == 0:
        return 0

    if raw_delta > 0:
        breakpoints = AFFINITY_INCREMENT_BREAKPOINTS
    else:
        breakpoints = AFFINITY_DECREMENT_BREAKPOINTS

    scaling_factor = 1.0
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]

        if x1 <= current_affinity <= x2:
            if x2 == x1:
                scaling_factor = y1
            else:
                scaling_factor = y1 + (current_affinity - x1) * (y2 - y1) / (x2 - x1)
            break

    return int(round(raw_delta * scaling_factor, 0))


def _build_diary_entries(
    diary_strings: list[str],
    *,
    timestamp: str,
    interaction_subtext: str,
) -> list[CharacterDiaryEntry]:
    """Convert raw diary strings into ``CharacterDiaryEntry`` dicts."""
    entries: list[CharacterDiaryEntry] = []
    for text in diary_strings or []:
        if not text:
            continue
        entry: CharacterDiaryEntry = {
            "entry": text,
            "timestamp": timestamp,
            "confidence": 0.8,
            "context": interaction_subtext or "",
        }
        entries.append(entry)
    return entries


def _build_objective_fact_entries(
    new_facts: list[dict],
    *,
    timestamp: str,
) -> list[ObjectiveFactEntry]:
    """Convert harvester ``new_facts`` rows into ``ObjectiveFactEntry`` dicts."""
    entries: list[ObjectiveFactEntry] = []
    for fact in new_facts or []:
        description = fact.get("description", "")
        if not description:
            continue
        entry: ObjectiveFactEntry = {
            "fact": description,
            "category": fact.get("category", "general"),
            "timestamp": timestamp,
            "source": "conversation_extracted",
            "confidence": 0.85,
        }
        entries.append(entry)
    return entries


async def _schedule_future_promises(
    promises: list[dict],
    *,
    global_user_id: str,
    user_name: str,
    character_name: str,
    decontexualized_input: str,
) -> list[str]:
    """Persist each promise as a ``future_promise`` scheduled event.

    Returns the list of event_ids scheduled. Events with no ``due_time`` are
    skipped — there's nothing to fire on.
    """
    scheduled: list[str] = []
    for promise in promises or []:
        due_time = promise.get("due_time")
        if not due_time:
            continue
        try:
            datetime.fromisoformat(due_time)
        except ValueError:
            logger.warning("Skipping promise with unparseable due_time: %r", due_time)
            continue

        event: ScheduledEventDoc = {
            "event_type": "future_promise",
            "target_platform": "",
            "target_channel_id": "",
            "target_global_user_id": global_user_id,
            "payload": {
                "promise_text": promise.get("action", ""),
                "target": promise.get("target", user_name),
                "character_name": character_name,
                "original_input": decontexualized_input,
                "context_summary": f"promise by {character_name} to {promise.get('target', user_name)}",
            },
            "scheduled_at": due_time,
        }
        try:
            event_id = await schedule_event(event)
            scheduled.append(event_id)
        except PyMongoError:
            logger.exception("Failed to persist future_promise event for user %s", global_user_id)
    return scheduled


async def db_writer(state: ConsolidatorState) -> dict:
    timestamp = state.get("timestamp") or datetime.now(timezone.utc).isoformat()
    global_user_id = state.get("global_user_id", "")
    user_name = state.get("user_name", "")
    character_name = state.get("character_profile", {}).get("name", "")

    metadata = dict(state.get("metadata", {}) or {})
    write_log: dict[str, bool] = {}
    cache_invalidated: list[str] = []

    # ── Step 1: character_state (mood / vibe / reflection) ──────────
    mood = state.get("mood", "")
    global_vibe = state.get("global_vibe", "")
    reflection_summary = state.get("reflection_summary", "")
    try:
        await upsert_character_state(
            mood=mood,
            global_vibe=global_vibe,
            reflection_summary=reflection_summary,
            timestamp=timestamp,
        )
        write_log["character_state"] = True
    except PyMongoError:
        logger.exception("db_writer: failed to upsert character_state")
        write_log["character_state"] = False

    # ── Step 2a: character diary (subjective per-user notes) ────────
    diary_entries = _build_diary_entries(
        state.get("diary_entry") or [],
        timestamp=timestamp,
        interaction_subtext=state.get("interaction_subtext", ""),
    )
    if global_user_id and diary_entries:
        try:
            await upsert_character_diary(global_user_id, diary_entries)
            write_log["character_diary"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to upsert character_diary")
            write_log["character_diary"] = False

    # ── Step 2b: last relationship insight ──────────────────────────
    last_relationship_insight = state.get("last_relationship_insight", "")
    if global_user_id and last_relationship_insight:
        try:
            await update_last_relationship_insight(global_user_id, last_relationship_insight)
            write_log["relationship_insight"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to update_last_relationship_insight")
            write_log["relationship_insight"] = False

    # ── Step 3a: objective facts (structured) + memory (searchable) ─
    new_facts = state.get("new_facts") or []
    objective_facts = _build_objective_fact_entries(new_facts, timestamp=timestamp)
    if global_user_id and objective_facts:
        try:
            await upsert_objective_facts(global_user_id, objective_facts)
            write_log["objective_facts"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to upsert_objective_facts")
            write_log["objective_facts"] = False

    for fact in new_facts:
        entity = fact.get("entity", user_name)
        category = fact.get("category", "general")
        description = fact.get("description", "")
        if not description:
            continue
        doc = build_memory_doc(
            memory_name=f"[{entity}] {category}",
            content=description,
            source_global_user_id=global_user_id,
            memory_type="fact",
            source_kind="conversation_extracted",
            confidence_note="This is extracted as a stable factual memory and may be used as background support.",
        )
        try:
            await save_memory(cast(MemoryDoc, doc), timestamp)
        except PyMongoError:
            logger.exception("db_writer: failed to save fact memory")

    # ── Step 3b: future promises (memory + scheduled event) ─────────
    future_promises = state.get("future_promises") or []
    for promise in future_promises:
        target = promise.get("target", user_name)
        action = promise.get("action", "")
        due_time = promise.get("due_time")
        if not action:
            continue
        doc = build_memory_doc(
            memory_name=f"[Promise] {target}",
            content=action,
            source_global_user_id=global_user_id,
            memory_type="promise",
            source_kind="conversation_extracted",
            confidence_note="This is an unfulfilled or future-oriented commitment and should be treated as pending until resolved.",
            expiry_timestamp=due_time,
        )
        try:
            await save_memory(cast(MemoryDoc, doc), timestamp)
        except PyMongoError:
            logger.exception("db_writer: failed to save promise memory")

    scheduled_event_ids = await _schedule_future_promises(
        future_promises,
        global_user_id=global_user_id,
        user_name=user_name,
        character_name=character_name,
        decontexualized_input=state.get("decontexualized_input", ""),
    )

    # ── Step 4: affinity (direction-scaled) ─────────────────────────
    user_affinity_score = state.get("user_profile", {}).get("affinity", AFFINITY_DEFAULT)
    raw_affinity_delta = state.get("affinity_delta", 0) or 0
    processed_affinity_delta = process_affinity_delta(user_affinity_score, raw_affinity_delta)
    if global_user_id:
        try:
            await update_affinity(global_user_id, processed_affinity_delta)
            write_log["affinity"] = True
        except PyMongoError:
            logger.exception("db_writer: failed to update_affinity")
            write_log["affinity"] = False

    logger.debug(
        "User %s(@%s) affinity %s -> %s",
        user_name, global_user_id,
        user_affinity_score, user_affinity_score + processed_affinity_delta,
    )

    # ── Step 5: cache invalidation (best-effort, after writes) ──────
    # The RAG cache is the hot read-path; stale entries now outweigh recency.
    if global_user_id:
        try:
            rag_cache = await _get_rag_cache()
            if diary_entries:
                await rag_cache.invalidate_pattern(
                    cache_type="character_diary",
                    global_user_id=global_user_id,
                )
                cache_invalidated.append("character_diary")
            if objective_facts:
                await rag_cache.invalidate_pattern(
                    cache_type="objective_user_facts",
                    global_user_id=global_user_id,
                )
                cache_invalidated.append("objective_user_facts")
            if future_promises:
                await rag_cache.invalidate_pattern(
                    cache_type="user_promises",
                    global_user_id=global_user_id,
                )
                cache_invalidated.append("user_promises")
            if abs(processed_affinity_delta) > AFFINITY_CACHE_NUKE_THRESHOLD:
                await rag_cache.clear_all_user(global_user_id)
                cache_invalidated.append("ALL_USER")

            if cache_invalidated:
                await increment_rag_version(global_user_id)
        except PyMongoError:
            logger.exception("db_writer: cache invalidation failed")

    metadata.update({
        "write_success": write_log,
        "cache_invalidation_scope": cache_invalidated,
        "scheduled_event_ids": scheduled_event_ids,
        "affinity_before": user_affinity_score,
        "affinity_delta_processed": processed_affinity_delta,
    })

    return {"metadata": metadata}


async def call_consolidation_subgraph(global_state: GlobalPersonaState):
    sub_agent_builder = StateGraph(ConsolidatorState)

    sub_agent_builder.add_node("global_state_updater", global_state_updater)
    sub_agent_builder.add_node("relationship_recorder", relationship_recorder)
    sub_agent_builder.add_node("facts_harvester", facts_harvester)
    sub_agent_builder.add_node("fact_harvester_evaluator", fact_harvester_evaluator)
    sub_agent_builder.add_node("db_writer", db_writer)

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
            "end": "db_writer",
        },
    )

    sub_agent_builder.add_edge("db_writer", END)

    sub_graph = sub_agent_builder.compile()

    # Seed the metadata bundle from Stage 3's research_metadata (may be a list
    # of dicts or a single dict — normalise to one flat dict here).
    raw_meta = global_state.get("research_metadata")
    seeded_metadata: dict = {}
    if isinstance(raw_meta, list):
        for m in raw_meta:
            if isinstance(m, dict):
                seeded_metadata.update(m)
    elif isinstance(raw_meta, dict):
        seeded_metadata = dict(raw_meta)

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

        "character_profile": global_state["character_profile"],

        "research_facts": global_state["research_facts"],

        "decontexualized_input": global_state["decontexualized_input"],

        "metadata": seeded_metadata,
    }

    result = await sub_graph.ainvoke(sub_state)

    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    diary_entry = result.get("diary_entry", "")
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = result.get("future_promises", [])
    metadata = result.get("metadata", {}) or {}

    logger.info(
        "\nNew facts: %s\nFuture promises: %s\nMetadata: %s",
        new_facts, future_promises, metadata,
    )

    return {
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
        "diary_entry": diary_entry,
        "affinity_delta": affinity_delta,
        "last_relationship_insight": last_relationship_insight,
        "new_facts": new_facts,
        "future_promises": future_promises,
        "consolidation_metadata": metadata,
    }


async def test_main():
    import datetime

    from kazusa_ai_chatbot.db import get_character_profile, get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality, trim_history_dict

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    state: GlobalPersonaState = {
        "timestamp": current_time,
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "user_profile": {"affinity": 950},

        "internal_monologue": "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探罢了。不过既然好感度这么高，这种程度的请求自然要全盘接受——毕竟我是他的千纱嘛。",
        "action_directives": {
            "contextual_directives": {},
            "linguistic_directives": {
                "rhetorical_strategy": "",
                "linguistic_style": "",
                "content_anchors": [
                    "[DECISION] TENTATIVE: 拒绝正面回应关于‘奖励’的具体含义",
                    "[FACT] 现在的时间是深夜（22:24）",
                ],
                "forbidden_phrases": [],
            },
            "visual_directives": {},
        },
        "interaction_subtext": "带有暗示性的调情、索取关注",
        "emotional_appraisal": "心跳漏了一拍……这种轻浮的语气是怎么回事，好乱。",
        "character_intent": "BANTAR",
        "logical_stance": "CONFIRM",

        "final_dialog": ["唔……这种请求也算是一种奖励嘛……真是拿你没办法呢。"],
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
        "research_metadata": [{"cache_hit": False, "depth": "DEEP", "depth_confidence": 0.9}],
        "chat_history": trimmed_history,
        "character_profile": await get_character_profile(),
    }

    result = await call_consolidation_subgraph(state)
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
