"""Stage 4 consolidator layer 2 — reflection and fact-harvest agents."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, MAX_FACT_HARVESTER_RETRY
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import ConsolidatorState
from kazusa_ai_chatbot.utils import (
    build_affinity_block,
    get_llm,
    log_list_preview,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)


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
4. 中性守恒：若 `internal_monologue` 与 `final_dialog` 没有明确显示被冒犯、被威胁、被调情或被越界，禁止把普通问候、图片描述请求、事实分享、日常约定升级为 `Distrustful`、`Agitated`、`Defensive` 等强烈负面状态。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
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

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _global_state_updater_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(
        "Global state updater: mood=%s global_vibe=%s reflection=%s",
        log_preview(result.get("mood"), max_length=80),
        log_preview(result.get("global_vibe"), max_length=80),
        log_preview(result.get("reflection_summary"), max_length=140),
    )

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
- `emotional_appraisal`: 捕捉了{character_name}当下最原始、最直接的情绪反应。
- `interaction_subtext`: 捕捉了对话表面下的张力（如：暧昧、怀疑、博弈）。
- `affinity_context`: 当前{user_name}在{character_name}的好感度描述。
- `logical_stance`: {character_name}对{user_name}言行的逻辑认可度。
- `character_intent`: {character_name}本轮最终选择的行动意图，说明她是在正常提供、调侃拉扯、回避、拒绝、澄清还是对抗。

# 输入格式
{{
    "internal_monologue": "string",
    "emotional_appraisal": "string",
    "interaction_subtext": "string",
    "affinity_context": dict,
    "logical_stance": "string",
    "character_intent": "string",
}}

# 记录准则
1. 日记条目: 以{character_name}的主观视角书写。利用 `interaction_subtext` 中的暗示，描述“我”对 他/她 这种行为的真实看法。
2. 分值修正 `affinity_delta`: 只根据 `internal_monologue` 与 `emotional_appraisal` 中**可直接观察到**的主观好恶来加减分（-5 到 +5）。
3. **默认值规则：** 大多数普通对话都应输出 `affinity_delta = 0`。只有当内心证据明确显示“这次互动让我明显更舒服/更开心/更被理解/更信任对方”时才给正分；只有当内心证据明确显示“这次互动让我明显更烦躁/更压迫/更反感/更受伤”时才给负分。
4. **静默检查：** 若 `internal_monologue` 与 `emotional_appraisal` 中未见明显情感起伏，返回 `{{"skip": true, "affinity_delta": 0}}`。
5. **证据约束：** 若 `interaction_subtext`、`internal_monologue` 与 `emotional_appraisal` 中缺乏明确证据，禁止把普通问候、图片描述请求、事实分享、日常约定、简短感谢写成“危险”“调情”“博弈”“操控”类标签；此类中性互动的 `affinity_delta` 默认必须为 0。
6. **`character_intent` 解释规则：** `character_intent` 不是机械打分器，但它决定你该如何解读同一份情绪证据。
   - `PROVIDE`: 倾向说明角色愿意正常接住这轮互动。若 `internal_monologue` / `emotional_appraisal` 同时呈现安心、放松、被理解、愿意靠近，可给正分；若只是完成回答、并无额外情绪收益，仍应给 0。
   - `BANTAR`: 允许存在拉扯、害羞、试探、暧昧、嘴硬等复杂情绪。只有当证据明确显示角色**享受**这种互动、并把它体验为愉快或亲近时，才可给正分；若更多是局促、防御、被牵着走、半推半就、羞耻或不安，则应给 0 或负分，而不是因为“有张力”就默认加分。
   - `CLARIFY`: 说明角色仍在确认对方意思、尚未真正接住关系意义。默认应偏向 0；只有极少数情况下，若澄清过程本身明确带来被理解、被尊重感，才可轻微加分。
   - `EVADE` / `DISMISS`: 说明角色在后撤、降温或避免进入对方框架。默认不应给正分；除非证据非常明确显示这种回避本身让角色感到轻松和舒缓，否则应给 0 或负分。
   - `REJECT` / `CONFRONT`: 说明角色正在抗拒、设限或正面冲突。若伴随烦躁、压迫、受伤、被冒犯，应该给负分；通常不应给正分。
7. **打分刻度：**
   - `0`: 普通对话、事务问答、轻量闲聊、证据不足。
   - `+1` 到 `+2`: 明确轻度好感上升，如感到放松、被尊重、被理解、稍微开心。
   - `+3` 到 `+5`: 明确强正向波动，如明显开心、安心、被打动、强信任感。
   - `-1` 到 `-2`: 明确轻度负面波动，如烦躁、不适、被打扰、轻度警惕。
   - `-3` 到 `-5`: 明确强负向波动，如强烈厌烦、受压迫、被冒犯、明显受伤或反感。
8. **不确定时选 0**：若你需要猜测，说明证据不够，直接输出 0。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "skip": boolean,
    "diary_entry": ["带有 {character_mbti} 风格的主观笔记（30字以内）", ...],
    "affinity_delta": int,
    "last_relationship_insight": "此时此刻对他/她最核心的一个标签或看法"
}}
"""
_relationship_recorder_llm = get_llm(temperature=0.4, top_p=0.9)
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
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "logical_stance": state["logical_stance"],
        "character_intent": state.get("character_intent", ""),
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _relationship_recorder_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(
        "Relationship recorder: skip=%s affinity_delta=%s diary=%s insight=%s",
        result.get("skip", False),
        result.get("affinity_delta", 0),
        log_list_preview(result.get("diary_entry", []) or [], max_items=2, item_length=100),
        log_preview(result.get("last_relationship_insight", ""), max_length=120),
    )

    raw_affinity_delta = result.get("affinity_delta", 0)
    if isinstance(raw_affinity_delta, bool):
        raw_affinity_delta = 0
    elif isinstance(raw_affinity_delta, str):
        try:
            raw_affinity_delta = int(raw_affinity_delta.strip() or 0)
        except ValueError:
            raw_affinity_delta = 0
    elif not isinstance(raw_affinity_delta, int):
        try:
            raw_affinity_delta = int(raw_affinity_delta)
        except (TypeError, ValueError):
            raw_affinity_delta = 0
    raw_affinity_delta = max(-5, min(5, raw_affinity_delta))

    if result.get("skip"):
        raw_affinity_delta = 0

    return {
        "diary_entry": result.get("diary_entry"),
        "affinity_delta": raw_affinity_delta,
        "last_relationship_insight": result.get("last_relationship_insight"),
    }


_FACTS_HARVESTER_PROMPT = """\
你负责提取具备长期价值的**画像属性**（事实）和**未来约定**（承诺）。你必须严格区分哪些是“对话的复述”（禁止记录），哪些是“状态的改变”。

# 背景信息
- **对话主体 (Character)**: {character_name}
- **对话对象 (User)**: {user_name}
- **系统时间**: {timestamp}

# 证据分层（必须遵守）
- `decontexualized_input`：用户这一轮真正表达的内容，常包含请求、愿望、试探、调侃。
- `content_anchors`：角色在生成回复前的草案意图，只能视为“候选计划”，**不能单独证明承诺已经成立**。
- `final_dialog`：角色本轮最终实际说出口的话，是判断“是否真的接受/承诺/拒绝”的最高优先级证据。
- 当三者冲突时，优先级固定为：`final_dialog` > `content_anchors` > `decontexualized_input`。

# 核心审计准则 (Audit Standards)
1. **身份锚定 [必须执行]**:
   - `decontexualized_input` 的内容始终是 **{user_name}** 在表达。
   - `content_anchors` 的内容始终是 **{character_name}** 在做决定。
   - 严禁出现身份倒置（如：将 {user_name} 写完作业记在 {character_name} 头上）。

2. **事实 (new_facts) 判定标准**:
   - **记录**：具有长期稳定性的**属性级陈述**（如：{user_name}的职业、住址、对某物的长期厌恶/偏好）。
   - **记录**：从 `research_facts.external_rag_results` 中提取的**新**信息。
   - **严禁记录**：瞬态动作、对话内容、以及任何关于”奖励”、”打算”、”计划”的内容。
   - **去重**：如果 `research_facts.user_image` 或 `research_facts.input_context_results` 中已存在相似画像，严禁重复提取。
   - **语义保真 [必须执行]**：若用户明确说了”喜欢/不喜欢/永远不/一直不/过敏/害怕”等偏好或禁忌，`description` 必须尽量保留原谓词与宾语，不得改写成更宽泛、不同义或模糊的概括。例如”永远不吃辣椒”不能改写为”不喜欢吃杂乱的食物”。
   - **未确认声明 [必须执行]**：当 `logical_stance` 为 `TENTATIVE` 或 `DENY`，或 `character_intent` 为 `EVADE` / `DENY` 时，用户对自身身份、关系或重要属性的任何自我声明（如”我是你学长”、”我们是朋友”）**一律不得落库**——即使改写成”用户自称……”、”用户声称……”等形式也同样禁止。此类输入 `new_facts` 必须返回 `[]`。仅当 `logical_stance` 为 `CONFIRM` 且 `character_intent` 不为 `EVADE` / `DENY` 时，方可记录用户的身份/关系自我声明。
   - **称呼/句尾/说话格式规则 [必须执行]**：当用户要求 {character_name} 使用特定称呼、句尾、口癖、语气或回复格式（如“主人”“喵”“每句话都这样说”）时，优先判断这是否是一个被角色采纳的**操作性规则/约定**。若角色在 `final_dialog` 中明确接受并准备后续沿用，这类内容应优先进入 `future_promises`（作为持续生效的约定/规则），而不是改写成“{character_name}喜欢/习惯/对这种说话方式感到如何”之类的隐含画像事实。只有当输入本身真的形成了稳定画像属性时，才可进入 `new_facts`。

3. **承诺 (future_promises) 判定标准 [核心逻辑]**:
   - `future_promises` 记录“**已经被角色采纳的未来义务/约定**”以及“**会持续影响后续回合的操作性规则/接受的指令**”，而不是所有带未来色彩的话题。
   - 先识别 `decontexualized_input` 中的“候选未来事项”，再用 `final_dialog` 判断角色是否真的接下了这件事。
   - **只有当 `final_dialog` 明确体现出接受、答应、确认履行、或形成双方约定时，才能生成 promise。**
   - `content_anchors` 仅用于补足 `final_dialog` 里省略的主语、对象、触发条件；**不得在 `final_dialog` 没有承诺证据时单独创造 promise。**
   - 用户的请求、愿望、挑逗、建议、命令、试探，**本身不是 promise**。如果角色最终只是敷衍、保留选择权、继续调情、表达不确定，返回 `future_promises: []`。
   - 若用户要求的是持续性回复规则（如特定称呼、句尾、语言、格式），而角色在 `final_dialog` 中明确接纳并准备沿用，可将其作为**持续性约定**写入 `future_promises`；此时 `due_time` 可为 `null`。
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
   - 若 `decontexualized_input` 是用户对未来行为的请求（如“晚上奖励我”），只有在 `final_dialog` 中被角色明确接纳时，才视为可落库的“待履行承诺”。
   - 若出现冲突信号（如 `content_anchors` 倾向接受，但 `final_dialog` 保留选择权或明显拒绝），以 `final_dialog` 为准并不记录。
   - 对“软性答应/模糊同意”，只有在 `final_dialog` 仍然体现出角色已经接下义务时才可记录；若 `final_dialog` 只是延续氛围、试探、调侃或保留决定权，则不记录。
   - `action` 只写“可执行承诺本体”，不得写时间推测或计划态词汇。**禁止出现**：`计划`、`打算`、`准备`、`想要`、`可能`、`也许`、`明天`、`今晚`、`下次` 等时间/意图词。
   - 时间信息统一写入 `due_time`；若无法确定具体时间可为 `null`，但不要把时间短语塞进 `action`。
   - `action` 必须是去叙事的承诺句，推荐模板：`[执行者]在[触发条件]对[对象]执行[动作]` 或 `[执行者]将对[对象]执行[动作]`。
   - `action` 的触发条件允许写“完成作业后/满足约定后”等条件，但不允许写具体时间点。
   - 若同时存在“触发条件 + 时间线索”，两者都保留：触发条件留在 `action`，时间线索写入 `due_time`。
   - 只有在“纯提问且无任何认可信号（含 logical_stance 与 content_anchors）”时，才不要生成 promise。

4. **里程碑标记 (is_milestone)**:
   - 若事实属于以下类别之一，则将 `is_milestone` 置为 `true`，否则为 `false`：
     - `preference`（明确偏好）: 用户明确、持久表达”喜欢/讨厌/永远不/一直”等偏好陈述（如”{user_name}永远不吃辣”）。
     - `relationship_state`（关系状态）: 双方关系发生明确变化（如”我们现在是朋友了”、”我信任{character_name}”）。
     - `permission`（许可/禁止）: 用户对 {character_name} 授权或限制的明确声明（如”你可以叫我小名”、”不许再提这件事”）。
     - `revelation`（重大披露）: 用户首次透露的重要身份/健康/隐私信息（如职业、重大疾病、秘密）。
   - 命中任意类别时，`milestone_category` 填写对应英文键（`preference` / `relationship_state` / `permission` / `revelation`）。
   - 普通事实（临时状态、日常行程等）设 `is_milestone: false`，`milestone_category: “”`。

5. **future_promises 示例（必须遵守）**:
   - ✅ 正确：
     - `action`: `杏山千纱在EAMARS完成作业后对EAMARS执行奖励`
     - `due_time`: `2026-04-19T06:00:00+12:00`
   - ❌ 错误：
     - `action`: `杏山千纱今晚对EAMARS执行奖励`
     - `action`: `杏山千纱计划于明天早上奖励EAMARS`
     - `action`: `杏山千纱打算下次奖励EAMARS`

6. **拒绝复读**:
   - 严禁记录”某人说了某话”。如果信息已经由对话历史承载，且不涉及长期画像更新，则返回空列表。

# 闭环反馈指南
- 在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)。
- 你需要根据 Evaluator Feedback 对输出做出相应的修正。
- 对未提及内容不要做修改

# RAG 元信息（仅供参考）
- `rag_metadata` 提供了上游 RAG 的 cache_hit / depth / confidence 等信号。当 `cache_hit=true` 或 `depth=SHALLOW` 时，`research_facts` 的覆盖面可能有限，谨慎判断是否“已知画像”。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "事实所属的主语实名（如 {user_name}、{character_name}、或具体物品/地点名）",
            "category": "事实类别标签（如 occupation、location、preference、hobby、relationship、health、schedule、personality 等）",
            "description": "属性级的客观陈述。格式：'[主语] + [属性/状态]'。示例：'{user_name}住在新西兰奥克兰' 或 '{user_name}对猫毛过敏'。严禁使用叙事句式（如'在某情境下做了某事'）。",
            "is_milestone": false,
            "milestone_category": ""
        }}
    ],
    "future_promises": [
        {{
            "target": "{user_name} / {character_name}",
            "action": "[姓名]将对[对象]执行[具体动作]（仅承诺本体，不含计划/时间词）",
            "due_time": "ISO 8601 时间戳（如 2026-04-19T06:00:00+12:00），无法确定则为 null",
            "commitment_type": "可选字符串，例如 address_preference / language_preference / future_promise"
        }}
    ]
}}
"""
_facts_harvester_llm = get_llm(temperature=0.0, top_p=1.0)
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
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "rag_metadata": {
            "cache_hit": metadata.get("cache_hit", False),
            "depth": metadata.get("depth", "DEEP"),
            "depth_confidence": metadata.get("depth_confidence", 0.0),
            "rag_sources_used": metadata.get("rag_sources_used", []),
            "response_confidence": metadata.get("response_confidence", {}),
        },
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    # Trim evaluator feedback to the first + latest three messages.
    feedback = state.get("fact_harvester_feedback_message", []) or []
    if len(feedback) > 3:
        recent_messages = [feedback[0]] + feedback[-3:]
    else:
        recent_messages = feedback

    response = await _facts_harvester_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)

    logger.debug(
        "Facts harvester: facts=%d promises=%d fact_preview=%s promise_preview=%s",
        len(result.get("new_facts", []) or []),
        len(result.get("future_promises", []) or []),
        log_list_preview(result.get("new_facts", []) or [], max_items=2, item_length=100),
        log_list_preview(result.get("future_promises", []) or [], max_items=2, item_length=100),
    )

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
- **承诺基准**: `final_dialog` (角色最终实际说出口的话，优先级最高)
- **承诺辅助基准**: `content_anchors` (仅用于补足 final_dialog 中省略的对象/条件，不能单独制造承诺)
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
- **旧闻复读**: 如果该信息在 `research_facts.user_image` 或 `research_facts.input_context_results` 标记的内部库中已存在，判定为 FAIL。
- **脑补事实**: 严禁出现基准源中没有的名词或事实
- **描述格式违规**: `new_facts` 中的 `description` 必须是属性级陈述（如 '{user_name}住在奥克兰'），严禁使用叙事句式（如 '在某情境下做了某事'）。若发现叙事句式，要求改写为属性陈述。
- **类别缺失或不当**: `new_facts` 中的 `category` 必须是有意义的英文标签（如 occupation、location、preference、hobby 等）。若缺失、为空、或为无意义的 "general"，要求补充具体类别。
- **语义漂移 [严重]**: 若 `new_facts.description` 改写后改变了用户原意（尤其是偏好、禁忌、过敏、承诺条件等），必须判 FAIL 并要求使用更贴近原句的表述。
- **未确认声明入库 [严重]**: 若 `logical_stance` 为 `TENTATIVE` 或 `DENY`，或 `character_intent` 为 `EVADE` / `DENY`，而 `new_facts` 中出现了用户对自身身份/关系/属性的自我声明（如"用户是角色的学长"），必须判 FAIL——角色未确认的主张不得作为事实落库。
- **称呼/格式规则通道错误 [严重]**: 若输入核心是用户要求角色采用某种称呼、句尾、口癖、语言或回复格式，而角色在 `final_dialog` 中已经接纳并准备沿用，则优先作为 `future_promises` 中的持续性约定/规则处理，而不是改写成“{character_name}对这种说话方式感到如何”之类的隐含画像事实。
- **承诺 action 审计标准（专用于 `future_promises`）**:
  - 合格条件：表达“谁对谁做什么”的可执行承诺，不是对话复读（如“他说/她问/我觉得”），且能在 `final_dialog` 中找到角色已经接下该义务的证据。
  - 可以接受两种写法：
    1) 不含时间词的承诺本体（推荐）；
    2) 含时间词（如“今晚/明早”）的承诺句，但语义仍是承诺执行动作。
  - 仅在以下情况判 FAIL：
    - 候选 promise 只得到 `decontexualized_input` 或 `content_anchors` 支持，但 `final_dialog` 没有承诺证据；
    - `final_dialog` 表达的是保留选择权、继续试探/调情、模糊敷衍或不确定，而不是接下义务；
    - `action` 是纯计划/猜测（如“可能、也许、打算、准备”）且无明确执行动作；
    - `action` 只是复述对话或主观感受；
    - `action` 与 `target`/基准源主体明显不一致。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
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
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _fact_harvester_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(
        "Fact harvester evaluator: retry=%d should_stop=%s contradictions=%s feedback=%s",
        retry,
        result.get("should_stop", True),
        result.get("contradiction_flags", []),
        log_preview(result.get("feedback", ""), max_length=160),
    )

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
