"""Evaluation, continuation, and finalization for the RAG supervisor."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.continuation import (
    MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    RAGContinuationDecision,
    empty_continuation_decision,
    validate_refined_query,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_initializer import (
    _normalize_initializer_slots,
    rag_initializer,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_prompt_views import (
    _clip_llm_summary_text,
    _compact_raw_result_for_llm,
    _known_facts_llm_view,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_types import (
    _MAX_LOOP_COUNT,
    ProgressiveRAGState,
)
from kazusa_ai_chatbot.utils import (
    get_llm,
    log_list_preview,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)


# ── Evaluator ──────────────────────────────────────────────────────

_EVALUATOR_SUMMARIZER_PROMPT = '''\
你是一个槽位结果提炼器。给定槽位任务描述和原始工具结果，提炼出一段简洁的中文事实摘要，供后续检索代理和最终回答器使用。

# 生成步骤
1. 先读取 `slot` 和 `agent`，确认本次工具结果回答的是哪一个槽位。
2. 读取 `raw_result`，只提炼其中已经存在的事实、标识符和可引用来源。
3. 参考 `known_facts`，避免重复已经总结过的槽位结论。
4. 如果 `resolved` 为 false 或 `raw_result` 缺少可用信息，只说明本次来源没有返回什么，不要扩大结论。

# 摘要要求
- 保留对后续步骤有用的关键标识符（global_user_id、display_name、URL 等）
- 如果内容是对话记录，列出 1-5 条最相关的消息摘要（说话人 + 关键内容）
- 如果内容是用户画像或持久记忆，提炼关键事实
- 如果槽位未解决（resolved: false），简洁说明本次检索的来源没有返回什么
- 如果 raw_result 为空，不要推断先前槽位失败；只有 known_facts 明确显示先前槽位 unresolved 时才可这样说

# 输入格式
human payload 是以下 JSON：
{
    "slot": "当前槽位任务描述",
    "agent": "执行该槽位的 agent 名称",
    "resolved": true,
    "raw_result": "工具原始输出，可以是 dict/list/string/null",
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# Recall 结果
- 如果 `agent` 是 `recall_agent` 且 `raw_result.selected_summary` 存在，必须保留该 selected_summary 的核心内容。
- 可补充 `primary_source` 与 `supporting_sources`，但不要把 progress-only recall 当成长期事实来源。

# 输出格式
- 不超过 200 字，纯文本，无 JSON 外壳
'''

_EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT = '''\
你是一个用户/角色画像槽位结果提炼器。给定 Profile 槽位、原始画像结果、以及先前身份解析结果，提炼一段简洁中文事实摘要。

# 字段语义（必须遵守）
- user_memory_context：当前用户的统一记忆投影。每条记录都包含 fact、subjective_appraisal、relationship_signal，可分别作为事实锚点、角色的主观评价、未来互动信号。
- objective_facts、milestones、active_commitments 若出现，应来自 user_memory_context 内部分类，不再作为旧的独立画像来源处理。
- 如果 raw_result 包含 name/description/gender/age/birthday/backstory/self_image，且不包含 user_memory_context：
  这是角色自己的公开资料或自我画像。按“角色自身资料”总结，不要当成第三方用户画像。

# 生成步骤
1. 先读取 `slot`、`agent` 与 `known_facts`，确认本次 profile 结果对应当前用户、第三方用户还是角色自身。
2. 如果 `raw_result` 包含 `user_memory_context`，按五类记忆单元总结：先写事实锚点，再写角色的主观评价和关系信号。
3. 如果 `raw_result` 是角色公开资料或 `self_image`，按角色自身资料总结，不要写成用户画像。
4. 只使用 `raw_result` 中已有的信息；未知字段保持未知，不要补全。

# 摘要要求
- 保留 global_user_id、display_name 等对后续步骤有用的标识。
- 明确区分“目标用户是谁”、事实锚点是什么、角色的主观评价是什么。
- 只总结 raw_result 中已有的信息，不要补全未知信息。

# 输入格式
human payload 是以下 JSON：
{
    "slot": "当前 Profile 槽位任务描述",
    "agent": "user_profile_agent",
    "resolved": true,
    "raw_result": {
        "global_user_id": "用户 UUID",
        "display_name": "用户显示名",
        "user_memory_context": {
            "stable_patterns": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "recent_shifts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "objective_facts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "milestones": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "active_commitments": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM", "due_at": "可选本地到期时间YYYY-MM-DD HH:MM", "due_state": "no_due_date | future_due | due_today | past_due | unknown_due_date"}]
        }
    },
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# 输出格式
- 不超过 220 字，纯文本，无 JSON 外壳。
'''

_evaluator_summarizer_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

async def _summarize_agent_result(
    slot: str,
    agent_name: str,
    resolved: bool,
    raw_result: object,
    known_facts: list[dict],
) -> str:
    """Distil a resolved agent result into a concise fact summary for downstream agents.

    Only called when resolved=True. Unresolved slots receive a deterministic
    template in rag_evaluator and never reach this function.

    Args:
        slot: The slot description that was being resolved.
        agent_name: Inner-loop agent that produced the raw result.
        resolved: Whether the inner-loop agent judged the slot as resolved.
        raw_result: Native tool output from the inner-loop agent (dict, list, str, or None).
        known_facts: Facts resolved before this slot.

    Returns:
        A concise Chinese-language summary of the key facts extracted.
    """
    if agent_name == "user_profile_agent":
        prompt = _EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT
    else:
        prompt = _EVALUATOR_SUMMARIZER_PROMPT

    compact_raw_result = _compact_raw_result_for_llm(raw_result)
    compact_known_facts = _known_facts_llm_view(known_facts)
    system_prompt = SystemMessage(content=prompt)
    human_message = HumanMessage(
        content=json.dumps(
            {
                "slot": slot,
                "agent": agent_name,
                "resolved": resolved,
                "raw_result": compact_raw_result,
                "known_facts": compact_known_facts,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _evaluator_summarizer_llm.ainvoke([system_prompt, human_message])
    return_value = response.content.strip()
    return return_value


def _unresolved_summary(slot: str, raw_result: object) -> str:
    """Build deterministic unresolved text without hiding candidate rows."""

    if isinstance(raw_result, dict):
        observation_candidates = _compact_continuation_items(
            raw_result.get('observation_candidates')
        )
        if observation_candidates:
            preview_parts = []
            for candidate in observation_candidates[:3]:
                content = candidate.get('content') or candidate.get('summary')
                source = candidate.get('source')
                if content and source:
                    preview_parts.append(f'{source}: {content}')
                    continue
                if content:
                    preview_parts.append(content)
            preview = '；'.join(preview_parts)
            if preview:
                summary = (
                    f'检索到候选结果，但未确认足以解决槽位。'
                    f'槽位: {slot}。候选: {preview}'
                )
                return summary
            summary = f'检索到候选结果，但未确认足以解决槽位。槽位: {slot}'
            return summary

    summary = f'检索未返回相关结果。槽位: {slot}'
    return summary


_CONTINUATION_OBSERVATION_LIMIT = 5
_CONTINUATION_OBSERVATION_TEXT_LIMIT = 500
_CONTINUATION_KNOWN_FACT_LIMIT = 5


def _compact_continuation_mapping(value: object) -> dict[str, str]:
    """Build a compact observation row for the continuation assessor."""
    if not isinstance(value, dict):
        text = _clip_llm_summary_text(
            value,
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        compact_text = {"content": text} if text else {}
        return compact_text

    compact: dict[str, str] = {}
    for key in (
        "kind",
        "source",
        "reason",
        "content",
        "summary",
        "text",
        "fact",
        "description",
        "source_kind",
        "source_system",
        "memory_name",
        "missing_context",
    ):
        if key not in value:
            continue
        text = _clip_llm_summary_text(
            value[key],
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        if text:
            compact[key] = text

    return_value = compact
    return return_value


def _compact_continuation_items(value: object) -> list[dict[str, str]]:
    """Clip observation-like rows before they enter the assessor prompt."""
    if isinstance(value, list):
        raw_items = value[:_CONTINUATION_OBSERVATION_LIMIT]
    elif value:
        raw_items = [value]
    else:
        raw_items = []

    compact_items: list[dict[str, str]] = []
    for raw_item in raw_items:
        compact_item = _compact_continuation_mapping(raw_item)
        if compact_item:
            compact_items.append(compact_item)

    return_value = compact_items
    return return_value


def _compact_continuation_text_list(value: object) -> list[str]:
    """Build a compact list of source-scoped hint strings."""
    if isinstance(value, list):
        raw_items = value[:_CONTINUATION_OBSERVATION_LIMIT]
    elif value:
        raw_items = [value]
    else:
        raw_items = []

    compact_items: list[str] = []
    for raw_item in raw_items:
        text = _clip_llm_summary_text(
            raw_item,
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        if text:
            compact_items.append(text)

    return_value = compact_items
    return return_value


def _continuation_observation_payload(
    *,
    state: ProgressiveRAGState,
    slot: str,
    agent_name: str,
    raw_result: object,
    remaining_slots: list[str],
) -> dict[str, object]:
    """Project unresolved retrieval output into assessor-visible observations."""
    observation_candidates: list[dict[str, str]] = []
    source_hints: list[dict[str, str]] = []
    missing_context: list[str] = []
    conflicts: list[str] = []
    source_policy = ""
    user_resolution_hints: list[str] = []

    if isinstance(raw_result, dict):
        observation_candidates = _compact_continuation_items(
            raw_result.get("observation_candidates"),
        )
        source_hints = _compact_continuation_items(
            raw_result.get("source_hints"),
        )
        missing_context = _compact_continuation_text_list(
            raw_result.get("missing_context"),
        )
        conflicts = _compact_continuation_text_list(raw_result.get("conflicts"))
        source_policy = _clip_llm_summary_text(
            raw_result.get("source_policy", ""),
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        user_resolution_hints = _compact_continuation_text_list(
            raw_result.get("user_resolution_hints"),
        )

    payload = {
        "original_query": state.get("original_query", ""),
        "current_slot": slot,
        "agent": agent_name,
        "resolved": False,
        "source_policy": source_policy,
        "missing_context": missing_context,
        "conflicts": conflicts,
        "observation_candidates": observation_candidates,
        "source_hints": source_hints,
        "user_resolution_hints": user_resolution_hints,
        "known_facts": _known_facts_llm_view(
            state.get("known_facts", []),
        )[-_CONTINUATION_KNOWN_FACT_LIMIT:],
        "pending_slots": remaining_slots[:_CONTINUATION_OBSERVATION_LIMIT],
    }
    return payload


def _has_continuation_observation(payload: dict[str, object]) -> bool:
    """Return whether the assessor has enough material to justify one LLM call."""
    has_material = bool(
        payload["observation_candidates"]
        or payload["source_hints"]
        or payload["user_resolution_hints"]
    )
    return has_material


def _accepted_continuation_count(known_facts: list[dict]) -> int:
    """Count prior accepted refined-query re-entries in this RAG run."""
    count = 0
    for fact in known_facts:
        continuation = fact.get("continuation")
        if not isinstance(continuation, dict):
            continue
        if (
            continuation.get("should_continue") is True
            and continuation.get("refined_query")
        ):
            count += 1
    return count


def _previous_refined_queries(known_facts: list[dict]) -> list[str]:
    """Collect accepted refined queries already used in this RAG run."""
    refined_queries: list[str] = []
    for fact in known_facts:
        continuation = fact.get("continuation")
        if not isinstance(continuation, dict):
            continue

        refined_query = continuation.get("refined_query")
        if isinstance(refined_query, str) and refined_query.strip():
            refined_queries.append(refined_query.strip())

    return refined_queries


def _refined_initializer_context(
    context: dict,
    refined_query: str,
) -> dict:
    """Build active-query context for a refined initializer re-entry."""
    refined_context = copy.deepcopy(context)
    prompt_message_context = refined_context.get("prompt_message_context")
    if isinstance(prompt_message_context, dict):
        prompt_message_context["body_text"] = refined_query

    return refined_context


_CONTINUATION_ASSESSOR_PROMPT = '''\
你只负责判断是否需要把上一轮未解决检索的观察材料合并成一个新的用户查询，再交给现有初始化器重新规划。
不要回答用户。不要生成槽位、前缀、agent 名、工具名、数据库参数或下一步列表。

# 决策顺序
1. 先判断是否必须停止。命中任一停止条件时，should_continue=false 且 refined_query=""。
2. 停止条件：观察材料无关、噪声、只说明失败，或没有提供能改善下一轮查询的信息。
3. 停止条件：观察材料说答案取决于用户尚未提供的用途、预算、偏好、范围、权限、时间或地点。
4. 停止条件：下一轮查询只能写成"根据我的用途/预算/偏好..."这类占位请求。
5. 未命中停止条件时，再判断能否继续。只有观察材料提供了现在就能使用的知识、来源方向、约束或检索策略，才允许 should_continue=true。
6. true 时 refined_query 必须是自包含、现在可执行的自然语言查询：包含原问题目标和观察材料中的有用知识；不要依赖隐藏上下文。

# 输入格式
用户消息是 JSON：
{
  "original_query": "去上下文化后的用户问题",
  "current_slot": "刚失败的槽位",
  "agent": "产生未解决结果的 agent",
  "resolved": false,
  "source_policy": "当前来源选择说明",
  "missing_context": ["当前来源缺失的上下文"],
  "conflicts": ["当前来源发现的冲突"],
  "observation_candidates": [{"content": "未能回答但可能有用的观察材料"}],
  "source_hints": [{"kind": "线索类型", "source": "线索来源"}],
  "user_resolution_hints": ["已发现需要用户补充的约束"],
  "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "..."}],
  "pending_slots": ["已经在等待的槽位"]
}

# 输出格式
只返回合法 JSON：
{
  "should_continue": true,
  "refined_query": "继续时填写自包含自然语言查询；停止时为空字符串",
  "reason": "简短诊断说明，仅用于日志和 trace"
}
'''
_continuation_assessor_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _assess_continuation(
    *,
    observation_payload: dict[str, object],
    original_query: str,
    previous_refined_queries: list[str],
    continuation_count: int,
) -> RAGContinuationDecision:
    """Classify an unresolved observation and validate refined-query re-entry."""
    if not _has_continuation_observation(observation_payload):
        decision = empty_continuation_decision()
        return decision

    if continuation_count >= MAX_CONTINUATION_DECISIONS_PER_RAG_RUN:
        decision = empty_continuation_decision()
        return decision

    system_prompt = SystemMessage(content=_CONTINUATION_ASSESSOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(
            observation_payload,
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _continuation_assessor_llm.ainvoke(
        [system_prompt, human_message],
    )
    raw_decision = parse_llm_json_output(response.content)
    decision = validate_refined_query(
        raw_decision,
        original_query=original_query,
        previous_refined_queries=previous_refined_queries,
        continuation_count=continuation_count,
        max_continuations=MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    )
    logger.info(
        f"RAG2 continuation assessor output: "
        f"should_continue={decision['should_continue']} "
        f"refined_query={log_preview(decision['refined_query'])} "
        f"reason={log_preview(decision['reason'])}"
    )
    logger.debug(
        f"RAG2 continuation assessor metadata: "
        f"payload={log_preview(observation_payload)} "
        f"raw_decision={log_preview(raw_decision)} "
        f"validated={log_preview(decision)}"
    )
    return decision


async def _initialize_refined_query_slots(
    *,
    state: ProgressiveRAGState,
    refined_query: str,
) -> list[str]:
    """Run the existing initializer path on a self-contained refined query."""
    refined_context = _refined_initializer_context(
        state.get("context", {}),
        refined_query,
    )
    initializer_state: ProgressiveRAGState = {
        "original_query": refined_query,
        "character_name": state.get("character_name", ""),
        "context": refined_context,
        "unknown_slots": [],
        "current_slot": "",
        "known_facts": state.get("known_facts", []),
        "messages": [],
        "initializer_cache": {},
        "current_dispatch": {},
        "last_agent_result": {},
        "loop_count": state.get("loop_count", 0),
        "final_answer": "",
    }
    initializer_update = await rag_initializer(initializer_state)
    refined_slots = _normalize_initializer_slots(
        initializer_update.get("unknown_slots", []),
    )
    logger.info(
        f"RAG2 continuation initializer output: "
        f"refined_query={log_preview(refined_query)} "
        f"unknown_slots={log_list_preview(refined_slots)}"
    )
    logger.debug(
        f"RAG2 continuation initializer metadata: "
        f"cache={log_preview(initializer_update.get('initializer_cache', {}))}"
    )
    return refined_slots


async def rag_evaluator(state: ProgressiveRAGState) -> dict:
    """Record the inner-agent verdict for the current slot, then drain it.

    The slot is always removed from unknown_slots regardless of success. The
    evaluator normalizes the agent result, calls the summarizer to produce a
    concise fact summary, and stores both the summary and raw result.

    Args:
        state: Current state after the executor ran.

    Returns:
        Partial state update with the current slot removed from unknown_slots
        and the agent verdict (with summary and raw_result) appended to known_facts.
    """
    agent_result = state.get("last_agent_result", {})
    if not isinstance(agent_result, dict):
        agent_result = {}

    slot = state.get("current_slot", "")
    resolved = bool(agent_result.get("resolved", False))
    raw_result = agent_result.get("result")
    known_facts = state.get("known_facts", [])
    agent_name = str(agent_result.get("agent", ""))
    remaining_slots = list(state.get("unknown_slots", []))[1:]

    if resolved:
        summary = await _summarize_agent_result(
            slot,
            agent_name,
            resolved,
            raw_result,
            known_facts,
        )
    else:
        summary = _unresolved_summary(slot, raw_result)

    continuation_decision: RAGContinuationDecision | None = None
    loop_count = int(state.get("loop_count", 0) or 0)
    if not resolved:
        continuation_decision = empty_continuation_decision()
        observation_payload = _continuation_observation_payload(
            state=state,
            slot=slot,
            agent_name=agent_name,
            raw_result=raw_result,
            remaining_slots=remaining_slots,
        )
        if loop_count < _MAX_LOOP_COUNT:
            continuation_decision = await _assess_continuation(
                observation_payload=observation_payload,
                original_query=state.get("original_query", ""),
                previous_refined_queries=_previous_refined_queries(known_facts),
                continuation_count=_accepted_continuation_count(known_facts),
            )

    new_fact = {
        "slot": slot,
        "agent": agent_name,
        "resolved": resolved,
        "summary": summary,
        "raw_result": raw_result,
        "attempts": int(agent_result.get("attempts", 0) or 0),
    }
    if continuation_decision is not None:
        new_fact["continuation"] = continuation_decision

    if (
        continuation_decision is not None
        and continuation_decision["should_continue"]
        and continuation_decision["refined_query"]
    ):
        refined_slots = await _initialize_refined_query_slots(
            state=state,
            refined_query=continuation_decision["refined_query"],
        )
        remaining_slots = refined_slots + remaining_slots

    logger.info(
        f'RAG2 fact: slot={log_preview(slot)} agent={agent_name or "<none>"} '
        f"resolved={resolved} summary={log_preview(summary)}"
    )
    logger.debug(
        f'RAG2 fact metadata: attempts={new_fact["attempts"]} '
        f"remaining_slots={len(remaining_slots)} "
        f"continuation={log_preview(continuation_decision)}"
    )
    logger.debug(f"RAG2 fact detail: raw_result={log_preview(raw_result)}")

    return_value = {
        "unknown_slots": remaining_slots,
        "known_facts": known_facts + [new_fact],
    }
    return return_value


# ── Finalizer ──────────────────────────────────────────────────────

_FINALIZER_PROMPT = '''\
你是一个事实总结员。请根据 `known_facts` 生成简短事实摘要。

# 生成步骤
1. 先读取 `original_query`，确认本次摘要需要覆盖的事实类型。
2. 读取 `time_context`，将“今天/昨天/前天”等相对日期解释为角色本地日期。
3. 按顺序读取 `known_facts`，只使用 resolved 槽位中的 summary 和 raw_result。
4. 如果 user_profile_agent 的 raw_result 包含 user_memory_context，区分 fact、subjective_appraisal、relationship_signal 三种语义。
5. 如果某个必要槽位 unresolved，只说明缺少该槽位信息。
6. 如果 agent="recall_agent"，优先使用 raw_result.selected_summary 总结约定/承诺/进度事实。
7. 输出一段短的事实摘要；说话人、来源、时间和引用都应来自 `known_facts` 中可见内容。

# 准则
- 围绕 `original_query` 需要的事实组织摘要，不要复述查找过程。
- 如果 known_facts 为空，说明本次 RAG 没有需要检索的外部/内部事实；不要说“缺少关于该问题的具体信息”。
- 如果某个槽位未能解决（resolved: false），如实告知缺少哪一部分信息。
- 不要把某个来源没有检索结果扩大成“没有任何记录/没有互动记录”；只能说明实际查询过的来源没有返回结果。
- 引用来源 URL 或对话来源时尽量保留。
- 对 conversation evidence，按“可见来源/说话人标签 + 时间 + 内容”的方式摘要；没有可见标签时使用“说话人”。
- 引用对话原文时，保留原文内部的人称，不要改写引用内容。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 包含 user_memory_context：
  fact 是事实锚点，subjective_appraisal 是画像来源的主观评价，relationship_signal 是未来互动信号。
  回答时不要把 subjective_appraisal 误写成目标用户自己的感受。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 是公开资料或 self_image：
  这是 self_image 或公开资料对应的主体资料。回答自我资料问题时，可以使用这些公开资料；
  不要误写成第三方用户画像。
- 当 known_facts 中 agent="recall_agent" 且 raw_result 包含 selected_summary：
  这是当前约定/承诺/进度的已仲裁回忆结果。直接使用 selected_summary 回答，不要改搜关键字或把它改写成长期设定。

# 输入格式
{
    "original_query": "用户原始问题",
    "time_context": {"current_local_datetime": "YYYY-MM-DD HH:MM", "current_local_weekday": "Weekday"},
    "known_facts": [{"slot": ..., "agent": ..., "resolved": ..., "summary": "简洁事实摘要", "raw_result": "原始工具输出（如需引用原文）", "attempts": ...}, ...]
}

# 输出格式
请直接返回一段自然语言事实摘要（纯文本，无 JSON 外壳）。
- no markdown formatting
- preserve visible source/speaker labels
- no broad interpretation beyond short extractive summaries
'''
_finalizer_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


def _finalizer_time_context(state: ProgressiveRAGState) -> dict[str, object]:
    """Return prompt-facing local time context for the finalizer."""

    context = state.get("context", {})
    if not isinstance(context, dict):
        return_value: dict[str, object] = {}
        return return_value

    time_context = context.get("time_context")
    if not isinstance(time_context, dict):
        return_value = {}
        return return_value

    allowed_keys = ("current_local_datetime", "current_local_weekday")
    return_value = {
        key: time_context[key]
        for key in allowed_keys
        if isinstance(time_context.get(key), str)
    }
    return return_value


async def rag_finalizer(state: ProgressiveRAGState) -> dict:
    """Synthesise the final answer from all collected slot results.

    Args:
        state: Final state after all slots have been processed.

    Returns:
        Partial state update with final_answer set.
    """
    system_prompt = SystemMessage(content=_FINALIZER_PROMPT)
    finalizer_input = {
        "original_query": state["original_query"],
        "time_context": _finalizer_time_context(state),
        "known_facts": _known_facts_llm_view(state.get("known_facts", [])),
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False, default=str))

    response = await _finalizer_llm.ainvoke([system_prompt, human_message])
    logger.info(f"RAG2 finalizer output: answer={log_preview(response.content)}")
    logger.debug(
        f'RAG2 finalizer metadata: query={log_preview(state["original_query"])} '
        f'facts={len(state.get("known_facts", []))}'
    )
    return_value = {"final_answer": response.content}
    return return_value
