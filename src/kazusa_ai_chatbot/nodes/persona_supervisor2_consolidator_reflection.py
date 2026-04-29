"""Stage 4 consolidator reflection agents."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    AFFINITY_DEFAULT,
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
    normalize_subjective_appraisals,
)
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
_global_state_updater_llm = get_llm(
    temperature=0.4,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)
async def global_state_updater(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_GLOBAL_STATE_UPDATER_PROMPT.format(character_name=state["character_profile"]["name"]))

    msg = {
        "internal_monologue": state["internal_monologue"],
        "final_dialog": state["final_dialog"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _global_state_updater_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(f'Global state updater: mood={log_preview(result.get("mood"))} global_vibe={log_preview(result.get("global_vibe"))} reflection={log_preview(result.get("reflection_summary"))}')

    return_value = {
        "mood": result.get("mood"),
        "global_vibe": result.get("global_vibe"),
        "reflection_summary": result.get("reflection_summary"),
    }
    return return_value


_RELATIONSHIP_RECORDER_PROMPT = """\
你负责为角色 `{character_name}` 与特定用户 `{user_name}` 的用户记忆单元提取主观评价证据。重点在于“主观体感”，而非对话复述。

# 核心任务
将瞬时的思考转化为可被下游 memory-unit consolidator 使用的 `subjective_appraisals`。

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
1. 主观评价证据: 以{character_name}的主观视角书写，描述“我”如何理解他/她这次行为背后的关系意义。不要写成日记，不要复述对话流程。
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
    "subjective_appraisals": ["带有 {character_mbti} 风格的主观关系判断（30字以内）", ...],
    "affinity_delta": int,
    "last_relationship_insight": "此时此刻对他/她最核心的一个标签或看法"
}}
"""
_relationship_recorder_llm = get_llm(
    temperature=0.4,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)
async def relationship_recorder(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_RELATIONSHIP_RECORDER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        character_mbti=state["character_profile"]["personality_brief"]["mbti"],
    ))

    # Convert affinity score into status and instruction
    user_affinity_score = state["user_profile"]["affinity"]
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
        "character_intent": state["character_intent"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _relationship_recorder_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    subjective_appraisals = normalize_subjective_appraisals(result.get("subjective_appraisals"))
    logger.debug(f'Relationship recorder: skip={result.get("skip", False)} affinity_delta={result.get("affinity_delta", 0)} appraisals={log_list_preview(subjective_appraisals)} insight={log_preview(result.get("last_relationship_insight", ""))}')

    raw_affinity_delta = result.get("affinity_delta", 0)
    if isinstance(raw_affinity_delta, bool):
        raw_affinity_delta = 0
    elif isinstance(raw_affinity_delta, str):
        try:
            raw_affinity_delta = int(raw_affinity_delta.strip() or 0)
        except ValueError as exc:
            logger.debug(f"Handled exception in relationship_recorder: {exc}")
            raw_affinity_delta = 0
    elif not isinstance(raw_affinity_delta, int):
        try:
            raw_affinity_delta = int(raw_affinity_delta)
        except (TypeError, ValueError) as exc:
            logger.debug(f"Handled exception in relationship_recorder: {exc}")
            raw_affinity_delta = 0
    raw_affinity_delta = max(-5, min(5, raw_affinity_delta))

    if result.get("skip"):
        raw_affinity_delta = 0

    return_value = {
        "subjective_appraisals": subjective_appraisals,
        "affinity_delta": raw_affinity_delta,
        "last_relationship_insight": result.get("last_relationship_insight"),
    }
    return return_value
