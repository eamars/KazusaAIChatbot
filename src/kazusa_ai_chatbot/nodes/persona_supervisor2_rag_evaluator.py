"""Evaluation, continuation, and finalization for the RAG supervisor."""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import event_logging
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
from kazusa_ai_chatbot.rag.evidence_formatting import (
    sanitize_public_rag_evidence_text,
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

MILLISECONDS_PER_SECOND = 1000
RAG_EVALUATOR_COMPONENT = "nodes.persona_supervisor2_rag_evaluator"


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _state_correlation_id(state: ProgressiveRAGState) -> str:
    """Build a non-content correlation id for RAG evaluator work."""

    context = state.get("context", {})
    if isinstance(context, dict):
        platform = str(context.get("platform", ""))
        message_ref = str(context.get("platform_message_id", "") or "")
    else:
        platform = ""
        message_ref = ""
    correlation_id = f"rag:{platform}:{message_ref or 'no-message-id'}"
    return correlation_id


# ── Evaluator ──────────────────────────────────────────────────────

_EVALUATOR_SUMMARIZER_PROMPT = '''\
你把一个 RAG2 槽位结果压缩成简短事实摘要，供后续检索代理和最终事实综合器使用。

# 生成步骤
1. 读取 `slot` 和 `agent`，识别本结果回答的证据目标。
2. 读取 `raw_result`，只抽取其中已经存在的事实和可读来源。
3. `known_facts` 只用于避免重复早前槽位结论。
4. 如果 `resolved` 为 false，或 `raw_result` 没有可用证据，说明该来源没有确认什么，
   不要扩大结论。
5. 生成的控制/状态说明使用中文。保留 display names、quotes、URLs、filenames、
   code/model labels 和 message text 的原始语言。

# 摘要要求
- 仅当有助于理解事实时保留 display_name、URL 或说话人线索。不要包含 global_user_id 或来源标识。
- 对 conversation rows，列出 1-10 条最相关消息摘要，格式为 speaker 加关键内容。
- 对 profile 或 durable memory payloads，抽取关键事实。
- 对 unresolved slots，简短说明该检索来源未确认槽位。
- 如果 raw_result 为空，不要推断之前槽位失败，除非 known_facts 明确显示。

# 输入格式
human payload 是 JSON：
{
    "slot": "当前槽位描述",
    "agent": "执行该槽位的代理名称",
    "resolved": true,
    "raw_result": "原始工具输出；可以是 dict/list/string/null",
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# Recall 结果
- 如果 `agent` 是 `recall_agent` 且 `raw_result.selected_summary` 存在，
  保留其核心内容。
- 可以补充 `primary_source` 和 `supporting_sources`，但不要把仅属于 progress 的 recall
  当作长期事实存储。

# 输出格式
- 只输出纯文本，不要 JSON wrapper。
- 控制在 200 个中文字符或 120 个英文词以内。
'''

_EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT = '''\
你把 Profile/Person-context 槽位结果压缩成简短事实摘要。结果可能描述当前用户、
第三方用户，或活跃角色自己的公开资料。

# 字段语义
- `user_memory_context` 是统一的当前用户记忆投影。每行可能包含 `fact`、
  `subjective_appraisal` 和 `relationship_signal`；分别视为事实锚点、
  角色侧评价和未来互动信号。
- `objective_facts`、`milestones` 和 `active_commitments` 若存在，都是
  `user_memory_context` 内的分区，不是旧式独立 profile 来源。
- 如果 raw_result 包含 name/description/gender/age/birthday/backstory 或 self_image，
  且没有 user_memory_context，它就是活跃角色自己的公开 profile 或 self-image。
  摘要为角色 profile 数据，不要当作第三方用户 profile。

# 生成步骤
1. 读取 `slot`、`agent` 和 `known_facts`，判断该 profile 结果属于当前用户、
   第三方用户还是活跃角色。
2. 如果 `raw_result` 包含 `user_memory_context`，先摘要 memory-unit facts，
   再摘要角色侧评价和关系信号。
3. 如果 `raw_result` 是公开角色 profile 或 `self_image`，摘要为角色自己的 profile。
4. 只使用 raw_result 中存在的信息；未知字段保持未知。
5. 生成的控制/状态说明使用中文。保留 source content、names 和 quotes 的原始语言。

# 摘要要求
- 不要在摘要中保留 global_user_id、UUID 或来源标识；用 display_name 或槽位引用表达人物。
- 区分目标用户、事实锚点和角色侧评价。
- 不要填补未知信息。

# 输入格式
human payload 是 JSON：
{
    "slot": "当前 Profile 槽位描述",
    "agent": "user_profile_agent",
    "resolved": true,
    "raw_result": {
        "global_user_id": "用户 UUID",
        "display_name": "用户显示名",
        "user_memory_context": {
            "stable_patterns": [{"fact": "事实锚点", "subjective_appraisal": "角色侧评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间 YYYY-MM-DD HH:MM"}],
            "recent_shifts": [{"fact": "事实锚点", "subjective_appraisal": "角色侧评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间 YYYY-MM-DD HH:MM"}],
            "objective_facts": [{"fact": "事实锚点", "subjective_appraisal": "角色侧评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间 YYYY-MM-DD HH:MM"}],
            "milestones": [{"fact": "事实锚点", "subjective_appraisal": "角色侧评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间 YYYY-MM-DD HH:MM"}],
            "active_commitments": [{"fact": "事实锚点", "subjective_appraisal": "角色侧评价", "relationship_signal": "未来互动信号", "updated_at": "本地时间 YYYY-MM-DD HH:MM", "due_at": "可选本地到期时间 YYYY-MM-DD HH:MM", "due_state": "no_due_date | future_due | due_today | past_due | unknown_due_date"}]
        }
    },
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# 输出格式
- 只输出纯文本，不要 JSON wrapper。
- 控制在 220 个中文字符或 140 个英文词以内。
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
        A concise factual summary with Chinese RAG2 control prose.
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
    started_at = time.perf_counter()
    response = await _evaluator_summarizer_llm.ainvoke([system_prompt, human_message])
    await event_logging.record_llm_stage_event(
        component=RAG_EVALUATOR_COMPONENT,
        stage_name="rag_result_summarizer",
        route_name=agent_name,
        model_name=RAG_SUBAGENT_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="not_json",
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
    )
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
            preview = '; '.join(preview_parts)
            if preview:
                summary = (
                    f'检索到了候选证据，但没有足够确认当前槽位。'
                    f'槽位：{slot}。候选：{preview}'
                )
                return summary
            summary = (
                f'检索到了候选证据，但没有足够确认当前槽位。槽位：{slot}'
            )
            return summary

        missing_context = _compact_continuation_text_list(
            raw_result.get('missing_context')
        )
        if missing_context:
            incompatible_routes = []
            for item in missing_context:
                if not item.startswith('incompatible_intent:'):
                    continue
                _, _, route = item.partition(':')
                if route:
                    incompatible_routes.append(route)

            if incompatible_routes:
                route_text = ', '.join(incompatible_routes)
                summary = (
                    f'检索来源不匹配；这个槽位应由 {route_text} 处理。'
                    f'这不等于没有记录。槽位：{slot}'
                )
                return summary

            context_text = ', '.join(missing_context)
            summary = (
                f'检索缺少必要上下文，未能确认槽位。'
                f'槽位：{slot}。缺少：{context_text}'
            )
            return summary

    summary = f'检索没有返回相关的已确认结果。槽位：{slot}'
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


def _has_promotion_candidates(payload: dict[str, object]) -> bool:
    """Return whether observation candidates can be cited for promotion."""

    candidates = payload["observation_candidates"]
    has_candidates = isinstance(candidates, list) and bool(candidates)
    return has_candidates


def _slot_prefix(slot: object) -> str:
    """Return the semantic prefix before a planned RAG slot separator."""

    slot_text = str(slot).strip()
    prefix, _, _ = slot_text.partition(":")
    normalized_prefix = prefix.casefold()
    return normalized_prefix


def _pending_slots_can_absorb_continuation(
    payload: dict[str, object],
) -> bool:
    """Return whether queued evidence slots should run before re-entry."""

    pending_slots = payload["pending_slots"]
    if not isinstance(pending_slots, list):
        return_value = False
        return return_value

    evidence_prefixes = {"conversation-evidence", "memory-evidence"}
    can_absorb = any(
        _slot_prefix(slot) in evidence_prefixes
        for slot in pending_slots
    )
    return can_absorb


def _has_resolved_known_fact(known_facts: list[dict]) -> bool:
    """Return whether the current RAG run already found usable evidence."""

    has_resolved_fact = any(
        fact.get("resolved") is True
        for fact in known_facts
    )
    return has_resolved_fact


def _memory_miss_after_recall_should_finalize(
    payload: dict[str, object],
) -> bool:
    """Return whether Recall plus memory miss is enough to finalize."""

    if payload["agent"] != "memory_evidence_agent":
        return_value = False
        return return_value

    known_facts = payload["known_facts"]
    if not isinstance(known_facts, list):
        return_value = False
        return return_value

    has_unresolved_recall = False
    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        if fact.get("resolved") is True:
            return_value = False
            return return_value
        if fact.get("agent") == "recall_agent" and fact.get("resolved") is False:
            has_unresolved_recall = True

    return_value = has_unresolved_recall
    return return_value


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
你判断一次未解决检索里的观察材料应该如何处理：候选证据是否已经直接回答当前槽位，或是否需要把观察材料折成新的自包含自然语言查询再交回现有 initializer。
不要替用户作答。不要生成 slots、prefixes、agent names、tool names、数据库参数或步骤列表。

# 判断顺序
1. 先检查 `observation_candidates`。如果某个候选项已经直接回答 `current_slot`，返回 `promote_candidate=true`，给出候选索引和中文 `promotion_summary`；同时保持 `should_continue=false`、`refined_query=""`。
2. 只有候选项覆盖当前槽位的关键对象、时间、来源或事件时，才能 promotion。不要因为同类、相近、相关、猜测性材料就 promotion。
3. 如果候选项不能直接回答，再判断是否需要 continuation。若任何停止条件成立，返回 `should_continue=false`、`refined_query=""`。
4. 当观察材料无关、嘈杂、只有失败信息，或不能改善下一次查询时停止。
5. 当答案仍依赖用户未提供的用途、预算、偏好、范围、权限、时间或地点时停止。
6. 当下一次查询只能写成“根据我的用途/预算/偏好”这类占位语时停止。
7. 当观察材料只是同一大类里的旁支候选，例如其他承诺、其他食物、其他设备、其他图片、其他历史事件，但没有覆盖原目标的关键对象、时间、来源或事件时停止。不要为了证明不存在而枚举。
8. 当已搜索请求来源，只剩相关/附近/未确认候选，且没有新的来源、消息位置、时间范围、人物身份或可执行约束时停止。
9. 当当前来源是 memory_evidence，而槽位目标明显是对话中分享过的图片、照片、截图、附件、插图或 OCR 描述时，可以 continuation 到 conversation evidence。
10. 其他情况只有在观察材料提供可立即使用的知识、来源方向、约束或检索策略时才 continuation。
11. 如果 continuation 为 true，`refined_query` 必须是自包含自然语言，包含原目标和有用观察材料；不要依赖隐藏上下文。
12. 普通诊断和生成说明使用中文。保留 names、quotes、URLs、filenames、code/model labels 和 source text 的原文。

# 输入格式
用户消息是 JSON：
{
  "original_query": "去上下文化后的用户问题",
  "current_slot": "刚刚失败的槽位",
  "agent": "产生未解决结果的代理",
  "resolved": false,
  "source_policy": "当前来源选择说明",
  "missing_context": ["当前来源缺少的上下文"],
  "conflicts": ["当前来源发现的冲突"],
  "observation_candidates": [{"content": "未直接回答但可能有帮助的观察材料"}],
  "source_hints": [{"kind": "提示类型", "source": "提示来源"}],
  "user_resolution_hints": ["需要用户输入才能解决的约束"],
  "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "..."}],
  "pending_slots": ["已经等待处理的槽位"]
}

# 输出格式
只返回有效 JSON：
{
  "promote_candidate": false,
  "promoted_candidate_indexes": [0],
  "promotion_summary": "候选证据直接回答当前槽位的中文事实摘要；不 promotion 时为空字符串",
  "should_continue": true,
  "refined_query": "continuation 时使用的自包含自然语言查询；停止时为空字符串",
  "reason": "中文短诊断，仅用于日志和 trace"
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

    if _pending_slots_can_absorb_continuation(observation_payload):
        decision: RAGContinuationDecision = {
            "should_continue": False,
            "refined_query": "",
            "reason": (
                "已有待处理证据槽位，应先执行它们再考虑 initializer re-entry。"
            ),
            "promote_candidate": False,
            "promoted_candidate_indexes": [],
            "promotion_summary": "",
        }
        logger.info(
            f"RAG2 continuation skipped: "
            f"reason={log_preview(decision['reason'])}"
        )
        return decision

    if _memory_miss_after_recall_should_finalize(observation_payload):
        decision = {
            "should_continue": False,
            "refined_query": "",
            "reason": (
                "Recall 和 memory evidence 都未解决；应收束负结果而不是扩大查询。"
            ),
            "promote_candidate": False,
            "promoted_candidate_indexes": [],
            "promotion_summary": "",
        }
        logger.info(
            f"RAG2 continuation skipped: "
            f"reason={log_preview(decision['reason'])}"
        )
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
    started_at = time.perf_counter()
    response = await _continuation_assessor_llm.ainvoke(
        [system_prompt, human_message],
    )
    raw_decision = parse_llm_json_output(response.content)
    parse_status = "succeeded" if isinstance(raw_decision, dict) else "failed"
    observation_candidates = observation_payload["observation_candidates"]
    if isinstance(observation_candidates, list):
        candidate_count = len(observation_candidates)
    else:
        candidate_count = 0
    decision = validate_refined_query(
        raw_decision,
        original_query=original_query,
        previous_refined_queries=previous_refined_queries,
        continuation_count=continuation_count,
        candidate_count=candidate_count,
        max_continuations=MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    )
    logger.info(
        f"RAG2 continuation assessor output: "
        f"promote_candidate={decision['promote_candidate']} "
        f"promoted_candidate_indexes={decision['promoted_candidate_indexes']} "
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
    await event_logging.record_llm_stage_event(
        component=RAG_EVALUATOR_COMPONENT,
        stage_name="continuation_assessor",
        route_name="continuation",
        model_name=RAG_SUBAGENT_LLM_MODEL,
        status="succeeded" if parse_status == "succeeded" else "failed",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if parse_status == "succeeded" else "warning",
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


def _candidate_content(candidate: object) -> str:
    """Return prompt-safe text from one assessor-visible candidate."""

    if isinstance(candidate, dict):
        for key in ("content", "summary", "claim", "fact", "text"):
            if key not in candidate:
                continue
            content = sanitize_public_rag_evidence_text(candidate[key])
            if content:
                return content
        return_value = ""
        return return_value

    content = sanitize_public_rag_evidence_text(candidate)
    return content


def _selected_promotion_candidates(
    observation_payload: dict[str, object],
    decision: RAGContinuationDecision,
) -> list[dict[str, str]]:
    """Select LLM-cited observation candidates by validated indexes."""

    candidates = observation_payload["observation_candidates"]
    if not isinstance(candidates, list):
        selected_candidates: list[dict[str, str]] = []
        return selected_candidates

    selected_candidates = []
    for index in decision["promoted_candidate_indexes"]:
        if index < 0 or index >= len(candidates):
            continue
        candidate = candidates[index]
        content = _candidate_content(candidate)
        if not content:
            continue
        selected_candidates.append({"content": content})
    return selected_candidates


def _promoted_projection_payload(
    agent_name: str,
    promotion_summary: str,
    candidates: list[dict[str, str]],
) -> dict[str, object]:
    """Build a projection payload from LLM-promoted observation candidates."""

    candidate_rows = [
        {"summary": candidate["content"], "content": candidate["content"]}
        for candidate in candidates
    ]
    if not candidate_rows and promotion_summary:
        candidate_rows = [
            {"summary": promotion_summary, "content": promotion_summary}
        ]

    if agent_name == "memory_evidence_agent":
        payload = {
            "memory_rows": [
                {
                    "content": row["content"],
                    "source_system": "promoted_candidate",
                }
                for row in candidate_rows
            ],
        }
        return payload

    payload = {
        "summaries": [promotion_summary] if promotion_summary else [],
        "rows": candidate_rows,
    }
    return payload


def _promoted_raw_result(
    raw_result: object,
    *,
    agent_name: str,
    observation_payload: dict[str, object],
    decision: RAGContinuationDecision,
) -> object:
    """Attach LLM-approved candidate evidence to the existing result shape."""

    promotion_summary = decision["promotion_summary"]
    promoted_candidates = _selected_promotion_candidates(
        observation_payload,
        decision,
    )
    if isinstance(raw_result, dict):
        promoted_result = copy.deepcopy(raw_result)
    else:
        promoted_result = {}

    promoted_result["selected_summary"] = promotion_summary
    promoted_result["promotion_source"] = "candidate_evidence"
    promoted_result["promotion_reason"] = (
        decision["reason"] or "候选证据直接回答当前槽位。"
    )
    promoted_result["promoted_candidate_indexes"] = list(
        decision["promoted_candidate_indexes"],
    )
    promoted_result["promoted_candidates"] = promoted_candidates

    existing_payload = promoted_result.get("projection_payload")
    if isinstance(existing_payload, dict):
        projection_payload = copy.deepcopy(existing_payload)
    else:
        projection_payload = {}
    promoted_payload = _promoted_projection_payload(
        agent_name,
        promotion_summary,
        promoted_candidates,
    )
    projection_payload.update(promoted_payload)
    promoted_result["projection_payload"] = projection_payload
    return promoted_result


def _stop_refined_query_after_assessment(
    decision: RAGContinuationDecision,
    reason: str,
) -> RAGContinuationDecision:
    """Preserve promotion while suppressing refined-query re-entry."""

    if decision["promote_candidate"]:
        return decision

    stopped_decision: RAGContinuationDecision = {
        "should_continue": False,
        "refined_query": "",
        "reason": reason,
        "promote_candidate": False,
        "promoted_candidate_indexes": [],
        "promotion_summary": "",
    }
    return stopped_decision


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
    observation_payload: dict[str, object] | None = None
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
            if _has_resolved_known_fact(known_facts):
                continuation_decision = {
                    "should_continue": False,
                    "refined_query": "",
                    "reason": "已有已解决证据，应先 finalizer 收束。",
                    "promote_candidate": False,
                    "promoted_candidate_indexes": [],
                    "promotion_summary": "",
                }
                logger.info(
                    f"RAG2 continuation skipped: "
                    f"reason={log_preview(continuation_decision['reason'])}"
                )
            else:
                continuation_decision = await _assess_continuation(
                    observation_payload=observation_payload,
                    original_query=state.get("original_query", ""),
                    previous_refined_queries=_previous_refined_queries(known_facts),
                    continuation_count=_accepted_continuation_count(known_facts),
                )
                if (
                    _has_resolved_known_fact(known_facts)
                    and continuation_decision["should_continue"]
                ):
                    continuation_decision = _stop_refined_query_after_assessment(
                        continuation_decision,
                        "已有已解决证据，应先 finalizer 收束，不做二次 query expansion。",
                    )
                    logger.info(
                        f"RAG2 continuation refined-query suppressed: "
                        f"reason={log_preview(continuation_decision['reason'])}"
                    )

    if (
        not resolved
        and continuation_decision is not None
        and observation_payload is not None
        and continuation_decision["promote_candidate"]
    ):
        resolved = True
        summary = continuation_decision["promotion_summary"]
        raw_result = _promoted_raw_result(
            raw_result,
            agent_name=agent_name,
            observation_payload=observation_payload,
            decision=continuation_decision,
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
你是事实综合器。根据 `known_facts` 生成简短事实摘要。

# 生成步骤
1. 读取 `original_query`，识别本摘要必须覆盖的事实类型。
2. 读取 `time_context`；使用角色本地日期解释今天、昨天、前两天等相对日期。
3. 按顺序读取 `known_facts`。只使用 resolved 槽位的 summaries 和 raw_result。
4. 如果 user_profile_agent 的 raw_result 包含 user_memory_context，区分 fact、
   subjective_appraisal 和 relationship_signal。
5. 如果必要槽位 unresolved，如实说明缺少哪一部分。
6. 如果 agent="recall_agent"，对 agreement、commitment 或 episode-position facts
   优先使用 raw_result.selected_summary。
7. 生成一段简短事实摘要。说话人、来源、时间和引文必须来自可见 known_facts 内容。
8. 生成的控制/状态说明使用中文。保留 source message text、display names、
   quoted text、URLs、filenames 和 code/model labels 的原始语言。

# 规则
- 围绕 `original_query` 所需事实组织摘要；不要复述搜索过程。
- 如果 known_facts 为空，说明本次 RAG 没有需要检索的外部/内部事实。
  不要声称某个具体信息缺失。
- 如果某个槽位 unresolved，如实说明缺失部分。
- 不要把一个来源未命中扩大成 "no record exists" 或 "no interaction exists"；
  只能说明被搜索来源返回了什么。
- 仅当有助于理解事实时保留可读 URL、说话人、时间或来源线索。
- 不要复述 global_user_id、UUID、platform_message_id、conversation_row_id、
  message ID 或类似来源标识；需要指代来源时只写 "消息记录" 或直接省略。
- 对 conversation evidence，摘要为可见 source/speaker label + time + content。
  如果没有可见 label，使用 "speaker"。
- 引用 message text 时，保留引文原始代词。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 包含 user_memory_context，
  fact 是事实锚点，subjective_appraisal 是 profile 来源的角色侧评价，
  relationship_signal 是未来互动信号。不要把 subjective_appraisal 写成目标用户自己的感受。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 是 public profile 或 self_image，
  那是主体自己的 profile data。用于 self-profile 问题，不要当作第三方用户 profile memory。
- 当 known_facts 中 agent="recall_agent" 且 raw_result 包含 selected_summary，
  它是对当前 agreement、commitment 或 progress 的仲裁 recall 结果。
  直接使用 selected_summary，不要改写成长期设定或搜索关键词。

# 输入格式
{
    "original_query": "用户原始问题",
    "time_context": {"current_local_datetime": "YYYY-MM-DD HH:MM", "current_local_weekday": "星期几"},
    "known_facts": [{"slot": "槽位描述", "agent": "agent_name", "resolved": true, "summary": "简短事实摘要", "raw_result": "需要引用时的原始工具输出", "attempts": 1}]
}

# 输出格式
返回一段自然语言事实摘要，纯文本，不要 JSON wrapper。
- 不要 markdown formatting
- 当可见 source/speaker labels 对意义有必要时保留它们
- 生成的状态说明使用中文
- 不要超出简短抽取式摘要做宽泛解释
'''
_finalizer_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)
_FINALIZER_UNRESOLVED_STATUS_LIMIT = 4
_FINALIZER_UNRESOLVED_CANDIDATE_LIMIT = 6
_FINALIZER_UNRESOLVED_CANDIDATE_TEXT_LIMIT = 180


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


def _all_known_facts_unresolved(known_facts: object) -> bool:
    """Return whether every collected RAG fact is unresolved."""

    if not isinstance(known_facts, list):
        return_value = False
        return return_value

    if not known_facts:
        return_value = False
        return return_value

    for fact in known_facts:
        if not isinstance(fact, dict):
            return_value = False
            return return_value
        if fact.get("resolved") is not False:
            return_value = False
            return return_value

    return_value = True
    return return_value


def _unresolved_fact_label(fact: dict) -> str:
    """Return a compact source label for an unresolved fact."""

    agent = _clip_llm_summary_text(
        fact.get("agent", ""),
        limit=80,
    )
    if agent:
        return agent

    slot = _clip_llm_summary_text(
        fact.get("slot", ""),
        limit=80,
    )
    if slot:
        return slot

    return_value = "unknown_source"
    return return_value


def _unresolved_fact_status(fact: dict) -> str:
    """Describe why one fact is still unresolved without dumping candidates."""

    label = _unresolved_fact_label(fact)
    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        missing_context = _compact_continuation_text_list(
            raw_result.get("missing_context"),
        )
        incompatible_routes = []
        for item in missing_context:
            if not item.startswith("incompatible_intent:"):
                continue
            _, _, route = item.partition(":")
            if route:
                incompatible_routes.append(route)

        if incompatible_routes:
            route_text = ", ".join(incompatible_routes)
            status = f"{label}：来源不匹配，应由 {route_text} 处理"
            return status

        has_candidates = bool(
            raw_result.get("observation_candidates")
            or raw_result.get("candidates")
        )
        if has_candidates:
            status = f"{label}：找到了候选证据，但不足以确认答案"
            return status

        if missing_context:
            context_text = ", ".join(missing_context)
            status = f"{label}：缺少 {context_text}"
            return status

    status = f"{label}：没有返回已确认结果"
    return status


def _candidate_preview_text(candidate: object) -> str:
    """Return one concise candidate text from a raw unresolved candidate row."""

    if isinstance(candidate, dict):
        text = ""
        for key in ("content", "summary", "claim", "fact", "text"):
            if key in candidate:
                text = _clip_llm_summary_text(
                    candidate[key],
                    limit=_FINALIZER_UNRESOLVED_CANDIDATE_TEXT_LIMIT,
                )
                break
        if not text:
            return_value = ""
            return return_value

        return text

    preview = _clip_llm_summary_text(
        candidate,
        limit=_FINALIZER_UNRESOLVED_CANDIDATE_TEXT_LIMIT,
    )
    return preview


def _candidate_previews_from_facts(known_facts: list[dict]) -> list[str]:
    """Collect unresolved candidate text from the provided facts."""

    previews: list[str] = []
    seen: set[str] = set()
    for fact in known_facts:
        raw_result = fact.get("raw_result")
        if not isinstance(raw_result, dict):
            continue

        raw_candidates = raw_result.get("observation_candidates")
        if not raw_candidates:
            raw_candidates = raw_result.get("candidates")
        if not isinstance(raw_candidates, list):
            continue

        for candidate in raw_candidates:
            preview = _candidate_preview_text(candidate)
            if not preview:
                continue
            if preview in seen:
                continue
            seen.add(preview)
            previews.append(preview)
            if len(previews) >= _FINALIZER_UNRESOLVED_CANDIDATE_LIMIT:
                return previews

    return previews


def _unresolved_candidate_previews(known_facts: list[dict]) -> list[str]:
    """Collect the most relevant nearby candidates for a negative result."""

    recall_facts = [
        fact
        for fact in known_facts
        if fact.get("agent") == "recall_agent"
    ]
    recall_previews = _candidate_previews_from_facts(recall_facts)
    if recall_previews:
        return recall_previews

    previews = _candidate_previews_from_facts(known_facts)
    return previews


def _unresolved_finalizer_answer(known_facts: list[dict]) -> str:
    """Summarize all-unresolved RAG output without raw slot dumps."""

    status_lines = [
        _unresolved_fact_status(fact)
        for fact in known_facts[:_FINALIZER_UNRESOLVED_STATUS_LIMIT]
    ]
    status_lines = [line for line in status_lines if line]
    candidate_previews = _unresolved_candidate_previews(known_facts)

    final_parts = ["本次 RAG 没有找到已确认事实。"]
    if status_lines:
        status_text = "; ".join(status_lines)
        final_parts.append(f" 已检查来源：{status_text}。")
    if candidate_previews:
        candidate_text = "; ".join(candidate_previews)
        final_parts.append(f" 附近但未确认的候选：{candidate_text}。")

    final_answer = "".join(final_parts)
    return final_answer


def _finalizer_known_facts_llm_view(
    known_facts: object,
) -> list[dict[str, object]]:
    """Build finalizer input without unresolved candidate evidence leakage."""

    compact_facts = _known_facts_llm_view(known_facts)
    finalizer_facts: list[dict[str, object]] = []
    for fact in compact_facts:
        finalizer_fact = dict(fact)
        if finalizer_fact.get("resolved") is not False:
            finalizer_facts.append(finalizer_fact)
            continue

        raw_result = finalizer_fact.get("raw_result")
        missing_context: list[str] = []
        selected_summary = ""
        if isinstance(raw_result, dict):
            missing_context = _compact_continuation_text_list(
                raw_result.get("missing_context"),
            )
            selected_summary = _clip_llm_summary_text(
                raw_result.get("selected_summary", ""),
            )

        finalizer_fact["raw_result"] = {
            "missing_context": missing_context,
            "selected_summary": selected_summary,
        }
        finalizer_facts.append(finalizer_fact)

    return_value = finalizer_facts
    return return_value


async def rag_finalizer(state: ProgressiveRAGState) -> dict:
    """Synthesise the final answer from all collected slot results.

    Args:
        state: Final state after all slots have been processed.

    Returns:
        Partial state update with final_answer set.
    """
    known_facts = state.get("known_facts", [])
    if _all_known_facts_unresolved(known_facts):
        final_answer = _unresolved_finalizer_answer(known_facts)
        final_answer = sanitize_public_rag_evidence_text(final_answer)
        logger.info(
            "RAG2 finalizer deterministic unresolved output: "
            f"answer={log_preview(final_answer)}"
        )
        return_value = {"final_answer": final_answer}
        return return_value

    system_prompt = SystemMessage(content=_FINALIZER_PROMPT)
    finalizer_input = {
        "original_query": state["original_query"],
        "time_context": _finalizer_time_context(state),
        "known_facts": _finalizer_known_facts_llm_view(known_facts),
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False, default=str))

    started_at = time.perf_counter()
    response = await _finalizer_llm.ainvoke([system_prompt, human_message])
    logger.info(f"RAG2 finalizer output: answer={log_preview(response.content)}")
    logger.debug(
        f'RAG2 finalizer metadata: query={log_preview(state["original_query"])} '
        f'facts={len(state.get("known_facts", []))}'
    )
    await event_logging.record_llm_stage_event(
        component=RAG_EVALUATOR_COMPONENT,
        stage_name="rag_finalizer",
        route_name="finalize",
        model_name=RAG_SUBAGENT_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="not_json",
        retry_count=0,
        json_repair_used=False,
        correlation_id=_state_correlation_id(state),
        duration_ms=_elapsed_ms(started_at),
    )
    final_answer = sanitize_public_rag_evidence_text(response.content)
    return_value = {"final_answer": final_answer}
    return return_value
