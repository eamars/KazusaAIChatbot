"""Top-level RAG capability agent for conversation-history evidence."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
    RAG_SEARCH_SELECTED_SUMMARY_LIMIT,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.conversation_aggregate_agent import (
    ConversationAggregateAgent,
)
from kazusa_ai_chatbot.rag.conversation_filter_agent import ConversationFilterAgent
from kazusa_ai_chatbot.rag.conversation_search_agent import ConversationSearchAgent
from kazusa_ai_chatbot.rag.evidence_coverage import (
    EvidenceCoverage,
    assess_evidence_coverage,
    coverage_allows_resolution,
    evidence_buckets_for_coverage,
    requested_coverage_items,
    task_requires_value_evidence,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.hybrid_retrieval import candidate_prompt_text
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "conversation_evidence"
_AGENT_NAME = "conversation_evidence_agent"
_UNCACHED_REASON = "capability_orchestrator_uncached"
_URL_PATTERN = re.compile(r"https?://[^\s)>\]}\"']+")
_COUNT_INTENT_RE = re.compile(
    r"\b(?:count|counts|counted|counting)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_SPEAKER_SCOPE_PATTERN = re.compile(
    r"\bspeaker\s*=\s*(current_user|active_character|any_speaker)\b",
    re.IGNORECASE,
)
_PERSON_SPEAKER_SCOPE_PATTERN = re.compile(
    r"\bspeaker\s*=\s*person\s+resolved\s+in\s+slot\s+\d+\b",
    re.IGNORECASE,
)
_SCOPE_CURRENT_USER = "current_user"
_SCOPE_ACTIVE_CHARACTER = "active_character"
_SCOPE_ANY_SPEAKER = "any_speaker"
_SCOPE_PERSON_RESOLVED = "person_resolved"
_KNOWN_WORKERS = {
    "conversation_search_agent",
    "conversation_filter_agent",
    "conversation_aggregate_agent",
    "incompatible",
}
_RECALL_OWNED_TASK_MARKERS = (
    "active agreement",
    "active promise",
    "active commitment",
    "ongoing agreement",
    "ongoing promise",
    "ongoing commitment",
    "current plan",
    "current plans",
    "open loop",
    "open loops",
    "unresolved loop",
    "unresolved loops",
    "next step",
    "episode position",
    "episode state",
    "current episode state",
    "where the current episode left off",
    "where current episode left off",
)
_RETRIEVAL_CONFIRMATION_MARKERS = (
    "retrieve messages",
    "retrieve recent messages",
    "find context",
    "messages around",
    "messages near",
    "context mentioning",
    "mentioning",
    "containing",
)
_VALUE_IDENTIFICATION_MARKERS = (
    "identify",
    "who said",
    "which speaker",
    "which user",
    "which person",
    "what time",
    "when was",
    "when did",
    "count ",
    "how many",
    "ranking",
    "grouped",
    "aggregate",
)
_EXACT_ANCHOR_CHARS = (
    '"',
    "'",
    "\u2018",
    "\u2019",
    "\u201c",
    "\u201d",
    "\u300c",
    "\u300d",
    "\u300e",
    "\u300f",
)


class _ConversationProjection(TypedDict):
    """Canonical evidence shape exposed by the conversation capability."""

    summaries: list[str]
    rows: list[dict[str, Any]]
    resolved_refs: list[dict[str, Any]]


class _ActiveTurnExclusionCounts(TypedDict):
    """Counts of active-turn rows removed by deterministic identity type."""

    conversation_row_id: int
    platform_message_id: int


def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact prompt-facing text.

    Args:
        value: Source value to convert to text.
        limit: Maximum number of characters.

    Returns:
        Stripped and clipped text.
    """

    text = text_or_empty(value)
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rstrip()
    return_value = f"{clipped}…"
    return return_value


def _cache_status() -> dict[str, Any]:
    """Build no-cache metadata for top-level capability orchestration."""
    status = {
        "enabled": False,
        "hit": False,
        "cache_name": "",
        "reason": _UNCACHED_REASON,
    }
    return status


def _result_payload(
    *,
    selected_summary: object = "",
    primary_worker: str = "",
    supporting_workers: list[str] | None = None,
    source_policy: object = "",
    resolved_refs: list[dict[str, Any]] | None = None,
    projection_payload: dict[str, Any] | None = None,
    worker_payloads: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    missing_context: list[str] | None = None,
    conflicts: list[str] | None = None,
    observation_candidates: list[dict[str, Any]] | None = None,
    source_hints: list[dict[str, Any]] | None = None,
    coverage: EvidenceCoverage | None = None,
    confirmed_evidence: list[str] | None = None,
    partial_evidence: list[str] | None = None,
    nearby_evidence: list[str] | None = None,
) -> dict[str, Any]:
    """Build the standard top-level conversation capability payload."""

    coverage_payload = coverage or assess_evidence_coverage(
        task="",
        evidence_items=[],
        worker_resolved=False,
    )
    payload = {
        "selected_summary": _clip_text(selected_summary),
        "capability": _CAPABILITY_NAME,
        "primary_worker": primary_worker,
        "supporting_workers": list(supporting_workers or []),
        "source_policy": _clip_text(source_policy, limit=400),
        "resolved_refs": list(resolved_refs or []),
        "projection_payload": dict(projection_payload or {}),
        "worker_payloads": dict(worker_payloads or {}),
        "evidence": list(evidence or []),
        "missing_context": list(missing_context or []),
        "conflicts": list(conflicts or []),
        "observation_candidates": list(observation_candidates or []),
        "source_hints": list(source_hints or []),
        "coverage": coverage_payload,
        "confirmed_evidence": list(confirmed_evidence or []),
        "partial_evidence": list(partial_evidence or []),
        "nearby_evidence": list(nearby_evidence or []),
    }
    return payload


def _agent_result(*, resolved: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the outer helper result for one capability-agent execution."""
    result = {
        "resolved": resolved,
        "result": payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return result


def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value


def _coverage_fields(
    *,
    task: str,
    evidence_items: list[str],
    worker_resolved: bool,
    requires_value_evidence: bool | None = None,
) -> tuple[EvidenceCoverage, dict[str, list[str]]]:
    """Build coverage and quality-specific evidence buckets."""

    coverage = assess_evidence_coverage(
        task=task,
        evidence_items=evidence_items,
        worker_resolved=worker_resolved,
        requires_value_evidence=requires_value_evidence,
    )
    buckets = evidence_buckets_for_coverage(coverage, evidence_items)
    return_value = (coverage, buckets)
    return return_value


def _coverage_confirms_retrieval_task(
    task: str,
    coverage: EvidenceCoverage,
) -> bool:
    """Return whether deterministic coverage confirms a retrieval slot.

    Args:
        task: Conversation-evidence slot text.
        coverage: Deterministic target coverage for projected evidence.

    Returns:
        True when the slot asks to retrieve matching conversation messages and
        all required anchors are covered. Slots asking to identify or extract
        a missing value still require the worker's own resolved judgment.
    """

    if coverage["evidence_quality"] != "partial":
        return_value = False
        return return_value
    if not coverage["covered_items"]:
        return_value = False
        return return_value

    coverage_requirement = coverage["coverage_requirement"]
    missing_items = coverage["missing_items"]
    if coverage_requirement == "all" and missing_items:
        return_value = False
        return return_value

    task_body = _strip_prefix(task).lower()
    if any(marker in task_body for marker in _VALUE_IDENTIFICATION_MARKERS):
        return_value = False
        return return_value

    confirms_retrieval = any(
        marker in task_body
        for marker in _RETRIEVAL_CONFIRMATION_MARKERS
    )
    return_value = confirms_retrieval
    return return_value


def _confirmed_retrieval_coverage(
    coverage: EvidenceCoverage,
) -> EvidenceCoverage:
    """Promote fully covered retrieval evidence to confirmed coverage."""

    confirmed_coverage: EvidenceCoverage = {
        "requested_items": list(coverage["requested_items"]),
        "covered_items": list(coverage["covered_items"]),
        "missing_items": list(coverage["missing_items"]),
        "evidence_quality": "confirmed",
        "confidence": coverage["confidence"],
        "reason": "Deterministic coverage confirmed the retrieval evidence.",
        "coverage_requirement": coverage["coverage_requirement"],
    }
    return confirmed_coverage


def _speaker_scope(task: str) -> str:
    """Extract the optional conversation author scope from a slot.

    Args:
        task: Conversation-evidence slot text.

    Returns:
        One approved scope constant, or an empty string for legacy unscoped
        slots.
    """

    task_body = _strip_prefix(task)
    if _PERSON_SPEAKER_SCOPE_PATTERN.search(task_body):
        return_value = _SCOPE_PERSON_RESOLVED
        return return_value

    explicit_match = _EXPLICIT_SPEAKER_SCOPE_PATTERN.search(task_body)
    if explicit_match is None:
        return_value = ""
        return return_value

    return_value = explicit_match.group(1).lower()
    return return_value


def _requires_person_ref(task: str) -> bool:
    """Return whether a slot needs a prior structured person reference.

    Args:
        task: Conversation-evidence slot text.

    Returns:
        True when the slot explicitly depends on a person from an earlier slot.
    """

    if _speaker_scope(task) == _SCOPE_PERSON_RESOLVED:
        return True

    normalized = _strip_prefix(task).lower()
    return_value = "resolved in slot" in normalized
    return return_value


def _recall_owned_task_reason(task_body: str) -> str:
    """Return Recall when a conversation task asks for episode state.

    Args:
        task_body: Conversation-evidence slot text without the capability
            prefix.

    Returns:
        ``"Recall"`` when the task belongs to active episode recall; otherwise
        an empty string.
    """

    normalized = task_body.lower()
    is_recall_task = any(
        marker in normalized
        for marker in _RECALL_OWNED_TASK_MARKERS
    )
    reason = "Recall" if is_recall_task else ""
    return reason


def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured conversation-evidence slots without selector LLM."""

    task_body = _strip_prefix(task)
    normalized = task_body.lower()
    speaker_scope = _speaker_scope(task)
    requires_person_ref = _requires_person_ref(task)
    incompatible_reason = _recall_owned_task_reason(task_body)

    if incompatible_reason:
        plan = {
            "worker": "incompatible",
            "reason": incompatible_reason,
            "requires_person_ref": False,
        }
        return plan

    if "durable" in normalized or "official address" in normalized:
        plan = {
            "worker": "incompatible",
            "reason": "Memory-evidence",
            "requires_person_ref": False,
        }
        return plan

    if (
        _COUNT_INTENT_RE.search(normalized)
        or "how many" in normalized
        or "ranking" in normalized
        or "grouped" in normalized
        or "aggregate" in normalized
    ):
        plan = {
            "worker": "conversation_aggregate_agent",
            "reason": "aggregate/count conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        any(anchor in task_body for anchor in _EXACT_ANCHOR_CHARS)
        or "exact phrase" in normalized
        or "exact term" in normalized
        or "url" in normalized
        or "filename" in normalized
        or "literal" in normalized
    ):
        plan = {
            "worker": "conversation_search_agent",
            "reason": "hybrid literal phrase, URL, filename, or exact anchor",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "topic" in normalized
        or "about " in normalized
        or "semantic" in normalized
    ):
        plan = {
            "worker": "conversation_search_agent",
            "reason": "semantic or fuzzy topic conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "resolved in slot" in normalized
        or "recent messages from" in normalized
        or (
            speaker_scope
            and "recent messages" in normalized
        )
        or "date-window" in normalized
        or "from timestamp" in normalized
        or "to timestamp" in normalized
    ):
        plan = {
            "worker": "conversation_filter_agent",
            "reason": "structured conversation filter",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    return None


def _normalize_selector_plan(
    raw_plan: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Normalize selector output while keeping dependency checks deterministic.

    Args:
        raw_plan: Parsed selector LLM output.
        task: Conversation-evidence slot text being planned.

    Returns:
        Approved worker plan with person-reference requirements derived from
        the slot text instead of the selector payload.
    """
    worker = text_or_empty(raw_plan.get("worker"))
    if worker == "conversation_keyword_agent":
        worker = "conversation_search_agent"
    if worker not in _KNOWN_WORKERS:
        worker = "conversation_search_agent"
    reason = text_or_empty(raw_plan.get("reason"))
    requires_person_ref = worker != "incompatible" and _requires_person_ref(task)
    plan = {
        "worker": worker,
        "reason": reason,
        "requires_person_ref": requires_person_ref,
    }
    return plan


_SELECTOR_PROMPT = '''\
你要为一个 RAG 证据槽位选择一个有边界的 conversation-history worker 路径。
不要用聊天历史路径回答 durable memory、活跃 episode 进度、用户资料或网页内容。

# 生成步骤
1. 如果任务询问活跃/当前约定或 episode 状态，输出
   worker="incompatible"，reason="Recall"。
2. 如果任务询问 durable world fact，输出 worker="incompatible"，
   reason="Memory-evidence"。
3. 模糊话题、语义消息证据、精确短语、URLs、filenames、字面词和引用消息来源，
   使用 conversation_search_agent。该 worker 执行语义加字面锚点的混合检索。
4. 已知用户、时间或日期窗口检索使用 conversation_filter_agent。
5. 计数、排名或分组统计使用 conversation_aggregate_agent。
6. 不要判断是否需要结构化人物引用；该依赖由槽位文本的确定性校验负责。

# 输入格式
{
  "task": "Conversation-evidence 槽位文本",
  "original_query": "可用时的去上下文化用户问题",
  "current_slot": "槽位标签",
  "known_facts": "之前 RAG2 槽位得到的有序事实"
}

# 输出格式
只返回有效 JSON：
{
  "worker": "conversation_search_agent | conversation_filter_agent | conversation_aggregate_agent | incompatible",
  "reason": "简短来源选择说明"
}
'''
_selector_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _select_plan(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Select the bounded conversation worker path for one slot."""
    deterministic_plan = _deterministic_plan(task)
    if deterministic_plan is not None:
        return deterministic_plan

    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )
    response = await _selector_llm.ainvoke([system_prompt, human_message])
    raw_plan = parse_llm_json_output(response.content)
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    plan = _normalize_selector_plan(raw_plan, task)
    return plan


def _iter_known_refs(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect structured refs from previous RAG2 known facts."""
    refs: list[dict[str, Any]] = []
    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        return refs

    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        raw_result = fact.get("raw_result")
        if not isinstance(raw_result, dict):
            continue
        raw_refs = raw_result.get("resolved_refs")
        if not isinstance(raw_refs, list):
            continue
        for ref in raw_refs:
            if isinstance(ref, dict):
                refs.append(ref)
    return refs


def _coverage_requires_value_evidence(task: str, context: dict[str, Any]) -> bool:
    """Preserve value-evidence intent across narrowed continuation slots."""

    if task_requires_value_evidence(task):
        return_value = True
        return return_value

    current_items = requested_coverage_items(task)
    if not current_items:
        return_value = False
        return return_value

    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        return_value = False
        return return_value

    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        if _known_fact_carries_value_intent(fact, current_items):
            return_value = True
            return return_value

    return_value = False
    return return_value


def _known_fact_carries_value_intent(
    fact: dict[str, Any],
    current_items: list[str],
) -> bool:
    """Return whether one prior fact preserves value intent for current items."""

    if fact.get("agent") != _AGENT_NAME:
        return_value = False
        return return_value
    if bool(fact.get("resolved")):
        return_value = False
        return return_value

    slot = text_or_empty(fact.get("slot"))
    if not task_requires_value_evidence(slot):
        return_value = False
        return return_value

    prior_items = _prior_requested_items(fact, slot)
    if not prior_items:
        return_value = False
        return return_value

    current_item_set = set(current_items)
    prior_item_set = set(prior_items)
    return_value = bool(current_item_set & prior_item_set)
    return return_value


def _prior_requested_items(fact: dict[str, Any], slot: str) -> list[str]:
    """Read requested coverage items from a prior fact or its slot text."""

    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        coverage = raw_result.get("coverage")
        if isinstance(coverage, dict):
            requested_items = coverage.get("requested_items")
            if isinstance(requested_items, list):
                items = [
                    item
                    for value in requested_items
                    if (item := text_or_empty(value))
                ]
                return items

    items = requested_coverage_items(slot)
    return items


def _first_person_ref(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first structured person reference from previous slots."""
    for ref in _iter_known_refs(context):
        if ref.get("ref_type") == "person":
            return ref
    return_value = None
    return return_value


def _worker_context(
    context: dict[str, Any],
    person_ref: dict[str, Any] | None,
    speaker_scope: str,
) -> dict[str, Any]:
    """Build worker context with approved structured handoff values.

    Args:
        context: Runtime context from the RAG supervisor.
        person_ref: Prior resolved person reference, when the slot depends on
            one.
        speaker_scope: Optional author scope declared by the conversation slot.

    Returns:
        Worker context with only the user filter implied by the author scope.
    """
    worker_context = dict(context)
    worker_context["exclude_current_question"] = True
    if speaker_scope:
        worker_context["conversation_user_scope"] = speaker_scope
    else:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)

    if speaker_scope == _SCOPE_ANY_SPEAKER:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)

    if speaker_scope == _SCOPE_ACTIVE_CHARACTER:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)
        character_profile = context.get("character_profile")
        resolved_global_user_id = ""
        resolved_display_name = ""
        if isinstance(character_profile, dict):
            resolved_global_user_id = text_or_empty(
                character_profile.get("global_user_id")
            )
            resolved_display_name = text_or_empty(character_profile.get("name"))
            if resolved_global_user_id:
                worker_context["global_user_id"] = resolved_global_user_id
            if resolved_display_name:
                worker_context["display_name"] = resolved_display_name
        if not resolved_global_user_id or not resolved_display_name:
            platform_channel_id = text_or_empty(context.get("platform_channel_id"))
            logger.warning(
                "conversation_evidence: speaker=active_character requested "
                "without character_profile "
                f"platform_channel_id={platform_channel_id}"
            )
        else:
            logger.debug(
                "conversation_evidence active-character scope resolved: "
                f"global_user_id={resolved_global_user_id}"
            )

    if person_ref is not None:
        global_user_id = text_or_empty(person_ref.get("global_user_id"))
        display_name = text_or_empty(person_ref.get("display_name"))
        if global_user_id:
            worker_context["global_user_id"] = global_user_id
        if display_name:
            worker_context["display_name"] = display_name
    return worker_context


def _projection_from_worker(
    worker_name: str,
    worker_result: dict[str, Any],
    context: dict[str, Any],
) -> tuple[_ConversationProjection, _ActiveTurnExclusionCounts]:
    """Project one worker's raw contract into capability evidence.

    Args:
        worker_name: Approved conversation worker name.
        worker_result: Worker result containing the raw tool payload.
        context: Runtime context used to remove active-turn source messages
            before evidence is exposed downstream.

    Returns:
        Pair of canonical conversation projection and active-turn exclusion
        counts.
    """

    raw_result = worker_result.get("result")

    if worker_name == "conversation_search_agent":
        rows = _semantic_message_rows(raw_result)
        filtered_rows, exclusion_counts = _filter_active_turn_rows(
            rows,
            context,
        )
        projection = _message_projection(filtered_rows)
        return_value = (projection, exclusion_counts)
        return return_value

    if worker_name == "conversation_filter_agent":
        rows = _plain_message_rows(raw_result)
        filtered_rows, exclusion_counts = _filter_active_turn_rows(
            rows,
            context,
        )
        projection = _message_projection(filtered_rows)
        return_value = (projection, exclusion_counts)
        return return_value

    if worker_name == "conversation_aggregate_agent":
        projection = _aggregate_projection(raw_result)
        return_value = (projection, {
            "conversation_row_id": 0,
            "platform_message_id": 0,
        })
        return return_value

    projection = _empty_projection()
    return_value = (projection, {
        "conversation_row_id": 0,
        "platform_message_id": 0,
    })
    return return_value


def _empty_projection() -> _ConversationProjection:
    """Build an empty canonical conversation evidence projection."""
    projection: _ConversationProjection = {
        "summaries": [],
        "rows": [],
        "resolved_refs": [],
    }
    return projection


def _semantic_message_rows(value: object) -> list[dict[str, Any]]:
    """Extract message rows from semantic search ``(score, message)`` results."""
    rows: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return rows

    for item in value:
        if isinstance(item, dict):
            rows.append(dict(item))
            continue

        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        score, message = item
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and isinstance(message, dict)
        ):
            rows.append(message)

    return rows


def _plain_message_rows(value: object) -> list[dict[str, Any]]:
    """Extract message rows from keyword and structured-filter results."""
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows = [
        row
        for row in value
        if isinstance(row, dict)
    ]
    return rows


def _active_turn_message_ids(context: dict[str, Any]) -> set[str]:
    """Return platform message IDs that belong to the active graph turn.

    Args:
        context: Runtime context passed to the conversation capability.

    Returns:
        Non-empty platform message IDs for the message or collapsed messages
        currently being answered.
    """

    raw_ids = context.get("active_turn_platform_message_ids")
    if not isinstance(raw_ids, list):
        return_value: set[str] = set()
        return return_value

    message_ids = {
        message_id
        for raw_id in raw_ids
        if (message_id := text_or_empty(raw_id))
    }
    return_value = message_ids
    return return_value


def _active_turn_conversation_row_ids(context: dict[str, Any]) -> set[str]:
    """Return conversation row IDs that belong to the active graph turn.

    Args:
        context: Runtime context passed to the conversation capability.

    Returns:
        Non-empty conversation-history row IDs for source messages currently
        being answered.
    """

    raw_ids = context.get("active_turn_conversation_row_ids")
    if not isinstance(raw_ids, list):
        return_value: set[str] = set()
        return return_value

    row_ids = {
        row_id
        for raw_id in raw_ids
        if (row_id := text_or_empty(raw_id))
    }
    return_value = row_ids
    return return_value


def _row_scope_matches_context(
    row: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    """Return whether an active-row match stays within the current scope.

    Args:
        row: Conversation message row returned by a worker.
        context: Runtime context passed to the conversation capability.

    Returns:
        True when row platform/channel metadata does not contradict the
        current RAG context.
    """

    row_platform = text_or_empty(row.get("platform"))
    context_platform = text_or_empty(context.get("platform"))
    if row_platform and row_platform != context_platform:
        return_value = False
        return return_value

    row_channel = text_or_empty(row.get("platform_channel_id"))
    context_channel = text_or_empty(context.get("platform_channel_id"))
    if row_channel and row_channel != context_channel:
        return_value = False
        return return_value

    return_value = True
    return return_value


def _active_turn_exclusion_reason(
    row: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Return why a message row belongs to the active graph turn.

    Args:
        row: Conversation message row returned by a worker.
        context: Runtime context passed to the conversation capability.

    Returns:
        ``conversation_row_id`` or ``platform_message_id`` when the row should
        be excluded; otherwise an empty string.
    """

    active_row_ids = _active_turn_conversation_row_ids(context)
    row_id = text_or_empty(row.get("conversation_row_id"))
    if row_id and row_id in active_row_ids:
        return_value = "conversation_row_id"
        return return_value

    active_ids = _active_turn_message_ids(context)
    if not active_ids:
        return_value = ""
        return return_value

    row_message_id = text_or_empty(row.get("platform_message_id"))
    if not row_message_id or row_message_id not in active_ids:
        return_value = ""
        return return_value

    if not _row_scope_matches_context(row, context):
        return_value = ""
        return return_value

    return_value = "platform_message_id"
    return return_value


def _filter_active_turn_rows(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], _ActiveTurnExclusionCounts]:
    """Remove active-turn source rows before evidence projection.

    Args:
        rows: Conversation message rows extracted from a worker result.
        context: Runtime context passed to the conversation capability.

    Returns:
        Pair of filtered rows and removed active-turn row counts by identity.
    """

    filtered_rows: list[dict[str, Any]] = []
    exclusion_counts: _ActiveTurnExclusionCounts = {
        "conversation_row_id": 0,
        "platform_message_id": 0,
    }
    for row in rows:
        exclusion_reason = _active_turn_exclusion_reason(row, context)
        if exclusion_reason == "conversation_row_id":
            exclusion_counts["conversation_row_id"] += 1
            continue
        if exclusion_reason == "platform_message_id":
            exclusion_counts["platform_message_id"] += 1
            continue
        filtered_rows.append(row)

    return_value = (filtered_rows, exclusion_counts)
    return return_value


def _message_projection(rows: list[dict[str, Any]]) -> _ConversationProjection:
    """Project typed conversation message rows into summaries and refs."""
    summaries: list[str] = []
    projected_rows: list[dict[str, Any]] = []
    for row in rows:
        summary = _clip_text(
            _message_row_text(row),
            limit=RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
        )
        summary = _dedupe_speaker_prefix(summary, row)
        if not summary:
            continue
        summaries.append(summary)
        projected_rows.append(_projection_row(row, summary))
    resolved_refs = _refs_from_message_rows(rows)
    projection: _ConversationProjection = {
        "summaries": summaries,
        "rows": projected_rows,
        "resolved_refs": resolved_refs,
    }
    return projection


def _dedupe_speaker_prefix(summary: str, row: dict[str, Any]) -> str:
    """Collapse repeated speaker prefixes in projected conversation text."""

    display_name = text_or_empty(row.get("display_name"))
    if not display_name:
        return summary
    doubled_prefix = f"{display_name}: {display_name}: "
    if summary.startswith(doubled_prefix):
        deduped = f"{display_name}: {summary[len(doubled_prefix):]}"
        return deduped
    return summary


def _projection_row(row: dict[str, Any], summary: str) -> dict[str, Any]:
    """Build inspectable row provenance for one projected message."""

    conversation_row_id = text_or_empty(row.get("conversation_row_id"))
    if not conversation_row_id:
        conversation_row_id = text_or_empty(row.get("_id"))
    methods_value = row.get("methods")
    methods: list[str] = []
    if isinstance(methods_value, list):
        methods = [
            text
            for item in methods_value
            if (text := text_or_empty(item))
        ]
    method = text_or_empty(row.get("method"))
    if method and method not in methods:
        methods.append(method)
    score_value = row.get("score")
    if isinstance(score_value, (int, float)) and not isinstance(score_value, bool):
        score: float | None = float(score_value)
    else:
        score = None

    projected_row = {
        "summary": summary,
        "timestamp": text_or_empty(row.get("timestamp")),
        "display_name": text_or_empty(row.get("display_name")),
        "platform_message_id": text_or_empty(row.get("platform_message_id")),
        "conversation_row_id": conversation_row_id,
        "methods": methods,
        "score": score,
    }
    return projected_row


def _conversation_projection_source(row: dict[str, Any]) -> str:
    """Build a compact source label for a projected conversation row."""

    platform_message_id = text_or_empty(row.get("platform_message_id"))
    if platform_message_id:
        source = f"conversation:platform_message_id:{platform_message_id}"
        return source

    conversation_row_id = text_or_empty(row.get("conversation_row_id"))
    if conversation_row_id:
        source = f"conversation:row_id:{conversation_row_id}"
        return source

    timestamp = text_or_empty(row.get("timestamp"))
    display_name = text_or_empty(row.get("display_name"))
    if timestamp or display_name:
        source = f"conversation:{display_name}:{timestamp}"
        return source

    source = "conversation:unknown"
    return source


def _message_row_text(row: dict[str, Any]) -> str:
    """Extract prompt-facing text from one canonical message row."""
    text = candidate_prompt_text(
        row,
        source="conversation",
        text_limit=RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
    )
    return text


def _refs_from_message_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured speaker, message, and URL refs from message rows."""
    refs: list[dict[str, Any]] = []
    for row in rows:
        global_user_id = text_or_empty(row.get("global_user_id"))
        display_name = text_or_empty(row.get("display_name"))
        if global_user_id or display_name:
            refs.append(
                {
                    "ref_type": "person",
                    "role": "speaker",
                    "global_user_id": global_user_id,
                    "display_name": display_name,
                }
            )

        platform_message_id = text_or_empty(row.get("platform_message_id"))
        timestamp = text_or_empty(row.get("timestamp"))
        if platform_message_id or timestamp:
            refs.append(
                {
                    "ref_type": "message",
                    "platform_message_id": platform_message_id,
                    "timestamp": timestamp,
                    "global_user_id": global_user_id,
                    "display_name": display_name,
                }
            )

        text = _message_row_text(row)
        refs.extend(_url_refs_from_text(text))
    return refs


def _aggregate_projection(value: object) -> _ConversationProjection:
    """Project aggregate worker payloads into canonical conversation evidence."""
    if not isinstance(value, dict):
        projection = _empty_projection()
        return projection

    rows_value = value.get("rows")
    rows = rows_value if isinstance(rows_value, list) else []
    row_summaries = [
        row_summary
        for row in rows
        if isinstance(row, dict)
        if (row_summary := _aggregate_row_summary(row))
    ]
    total_count = value.get("total_count")
    total_text = _aggregate_total_text(total_count)
    aggregate = text_or_empty(value.get("aggregate")) or "conversation aggregate"
    time_window = text_or_empty(value.get("time_window"))

    summary_parts = [aggregate]
    if time_window:
        summary_parts.append(f"window={time_window}")
    if total_text:
        summary_parts.append(f"total={total_text}")
    if row_summaries:
        summary_parts.append(
            "top rows: "
            + "; ".join(row_summaries[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT])
        )

    if len(summary_parts) == 1:
        summaries: list[str] = []
    else:
        summary = ", ".join(summary_parts)
        summaries = [summary]

    projection: _ConversationProjection = {
        "summaries": summaries,
        "rows": [],
        "resolved_refs": _refs_from_aggregate_rows(rows),
    }
    return projection


def _aggregate_total_text(value: object) -> str:
    """Render aggregate total count without assuming a raw payload type."""
    if isinstance(value, int) and not isinstance(value, bool):
        return_value = str(value)
        return return_value
    return_value = text_or_empty(value)
    return return_value


def _aggregate_row_summary(row: dict[str, Any]) -> str:
    """Render one aggregate row for the RAG finalizer."""
    display_name = _first_display_name(row.get("display_names"))
    if not display_name:
        display_name = text_or_empty(row.get("platform_user_id"))
    if not display_name:
        display_name = text_or_empty(row.get("global_user_id"))

    message_count = row.get("message_count")
    if isinstance(message_count, int) and not isinstance(message_count, bool):
        count_text = str(message_count)
    else:
        count_text = text_or_empty(message_count)

    last_timestamp = format_storage_utc_for_llm(
        text_or_empty(row.get("last_timestamp"))
    )
    parts = []
    if display_name:
        parts.append(display_name)
    if count_text:
        parts.append(f"{count_text} messages")
    if last_timestamp:
        parts.append(f"last={last_timestamp}")

    summary = ", ".join(parts)
    return summary


def _first_display_name(value: object) -> str:
    """Return the first non-empty display name from an aggregate row."""
    if not isinstance(value, list):
        return_value = text_or_empty(value)
        return return_value

    for item in value:
        display_name = text_or_empty(item)
        if display_name:
            return display_name

    return_value = ""
    return return_value


def _refs_from_aggregate_rows(rows: list[object]) -> list[dict[str, Any]]:
    """Extract person refs from aggregate result rows when available."""
    refs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        global_user_id = text_or_empty(row.get("global_user_id"))
        display_name = _first_display_name(row.get("display_names"))
        if not global_user_id and not display_name:
            continue

        refs.append(
            {
                "ref_type": "person",
                "role": "aggregate_subject",
                "global_user_id": global_user_id,
                "display_name": display_name,
            }
        )

    return refs


def _url_refs_from_text(text: str) -> list[dict[str, str]]:
    """Extract URL refs from one text block."""
    refs = [
        {
            "ref_type": "url",
            "role": "posted_url",
            "url": match.group(0).rstrip(".,"),
        }
        for match in _URL_PATTERN.finditer(text)
    ]
    return refs


class ConversationEvidenceAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for evidence from conversation history."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached conversation-evidence capability agent."""

        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.search_agent = ConversationSearchAgent(cache_runtime=cache_runtime)
        self.filter_agent = ConversationFilterAgent(cache_runtime=cache_runtime)
        self.aggregate_agent = ConversationAggregateAgent(
            cache_runtime=cache_runtime
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one conversation evidence slot through one worker path."""

        del max_attempts

        plan = await _select_plan(task, context)
        primary_worker = plan["worker"]
        speaker_scope = _speaker_scope(task)
        if primary_worker == "incompatible":
            reason = text_or_empty(plan["reason"]) or "unsupported"
            result = self._unresolved(
                task=task,
                source_policy=f"incompatible intent; use {reason}",
                missing_context=[f"incompatible_intent:{reason}"],
                primary_worker="",
                worker_payloads={},
            )
            return result

        person_ref = _first_person_ref(context)
        if bool(plan["requires_person_ref"]) and person_ref is None:
            result = self._unresolved(
                task=task,
                source_policy="structured person ref required but missing",
                missing_context=["person_ref"],
                primary_worker=primary_worker,
                worker_payloads={},
            )
            return result

        effective_person_ref = person_ref if bool(plan["requires_person_ref"]) else None
        worker = self._worker_for_name(primary_worker)
        context_for_worker = _worker_context(
            context,
            effective_person_ref,
            speaker_scope,
        )
        worker_result = await worker.run(
            task,
            context_for_worker,
            max_attempts=1,
        )
        worker_payloads = {primary_worker: worker_result}
        projection, exclusion_counts = _projection_from_worker(
            primary_worker,
            worker_result,
            context_for_worker,
        )
        excluded_count = (
            exclusion_counts["conversation_row_id"]
            + exclusion_counts["platform_message_id"]
        )
        summaries = projection["summaries"]
        projection_rows = projection["rows"]
        worker_resolved = bool(worker_result.get("resolved"))
        requires_value_evidence = _coverage_requires_value_evidence(
            task,
            context,
        )
        coverage, evidence_buckets = _coverage_fields(
            task=task,
            evidence_items=summaries,
            worker_resolved=worker_resolved,
            requires_value_evidence=requires_value_evidence,
        )
        coverage_confirms_retrieval = _coverage_confirms_retrieval_task(
            task,
            coverage,
        )
        if coverage_confirms_retrieval:
            coverage = _confirmed_retrieval_coverage(coverage)
            evidence_buckets = evidence_buckets_for_coverage(
                coverage,
                summaries,
            )
        confirmed_evidence = evidence_buckets["confirmed_evidence"]
        partial_evidence = evidence_buckets["partial_evidence"]
        nearby_evidence = evidence_buckets["nearby_evidence"]
        legacy_evidence = confirmed_evidence
        if not legacy_evidence:
            legacy_evidence = partial_evidence or nearby_evidence
        selected_summary = "\n".join(
            legacy_evidence[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT],
        )
        resolved_refs = projection["resolved_refs"]
        resolved = (
            bool(summaries)
            and coverage_allows_resolution(coverage)
            and (worker_resolved or coverage_confirms_retrieval)
        )
        missing_context = [] if resolved else ["conversation_evidence"]
        observation_candidates = []
        source_hints: list[dict[str, Any]] = []
        if summaries and not resolved:
            observation_candidates = [
                {
                    "content": row["summary"],
                    "source": _conversation_projection_source(row),
                }
                for row in projection_rows[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT]
            ]
            source_hints = [
                {
                    "kind": "conversation",
                    "source": candidate["source"],
                }
                for candidate in observation_candidates
                if candidate.get("source")
            ]
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload={
                "summaries": summaries,
                "rows": projection_rows,
            },
            worker_payloads=worker_payloads,
            evidence=legacy_evidence,
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=observation_candidates,
            source_hints=source_hints,
            coverage=coverage,
            confirmed_evidence=confirmed_evidence,
            partial_evidence=partial_evidence,
            nearby_evidence=nearby_evidence,
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        if excluded_count:
            logger.info(
                f"{_AGENT_NAME} active-turn rows excluded: "
                f"primary_worker={primary_worker} "
                f"excluded_active_turn_rows={excluded_count} "
                "excluded_by_conversation_row_id="
                f"{exclusion_counts['conversation_row_id']} "
                "excluded_by_platform_message_id="
                f"{exclusion_counts['platform_message_id']}"
            )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs={resolved_refs} "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result

    def _worker_for_name(self, worker_name: str) -> BaseRAGHelperAgent:
        """Return the configured worker instance for an approved name."""
        if worker_name == "conversation_filter_agent":
            return self.filter_agent
        if worker_name == "conversation_aggregate_agent":
            return self.aggregate_agent
        return self.search_agent

    def _unresolved(
        self,
        *,
        task: str,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling another source."""
        coverage, evidence_buckets = _coverage_fields(
            task=task,
            evidence_items=[],
            worker_resolved=False,
        )
        payload = _result_payload(
            selected_summary="",
            primary_worker=primary_worker,
            supporting_workers=[],
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={"summaries": []},
            worker_payloads=worker_payloads,
            evidence=[],
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=[],
            source_hints=[],
            coverage=coverage,
            confirmed_evidence=evidence_buckets["confirmed_evidence"],
            partial_evidence=evidence_buckets["partial_evidence"],
            nearby_evidence=evidence_buckets["nearby_evidence"],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=False "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} selected_summary='' "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=False, payload=payload)
        return result
