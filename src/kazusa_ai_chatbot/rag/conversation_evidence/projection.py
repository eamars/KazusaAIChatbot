"""Conversation worker result projection."""

from __future__ import annotations

import re
from typing import Any

from kazusa_ai_chatbot.config import (
    RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
    RAG_SEARCH_SELECTED_SUMMARY_LIMIT,
)
from kazusa_ai_chatbot.rag.conversation_evidence.active_turn_filter import (
    _filter_active_turn_rows,
)
from kazusa_ai_chatbot.rag.conversation_evidence.contracts import (
    _ActiveTurnExclusionCounts,
    _ConversationProjection,
    _clip_text,
)
from kazusa_ai_chatbot.rag.conversation_evidence.selector import (
    _RETRIEVAL_CONFIRMATION_MARKERS,
    _VALUE_IDENTIFICATION_MARKERS,
    _strip_prefix,
)
from kazusa_ai_chatbot.rag.evidence_coverage import (
    EvidenceCoverage,
    assess_evidence_coverage,
    evidence_buckets_for_coverage,
)
from kazusa_ai_chatbot.rag.hybrid_retrieval import candidate_prompt_text
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import text_or_empty

_URL_PATTERN = re.compile(r"https?://[^\s)>\]}\"']+")

_COUNT_INTENT_RE = re.compile(
    r"\b(?:count|counts|counted|counting)\b",
    flags=re.IGNORECASE,
)

_RELATION_REQUIREMENT_RE = re.compile(
    r"\brelation\s*=\s*"
    r"(previous_message|next_message|reply_parent)\b",
    re.IGNORECASE,
)

_RELATION_LABELS = {
    "previous_message": "上一条",
    "next_message": "下一条",
    "reply_parent": "回复对象",
}

_MAX_PACKET_RELATIONS = 3

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

def _relation_requirement(task: str) -> str:
    """Read the stable relation contract token from a conversation slot."""

    task_body = _strip_prefix(task)
    match = _RELATION_REQUIREMENT_RE.search(task_body)
    if match is None:
        return_value = ""
        return return_value

    relation = match.group(1).lower()
    return relation

def _projection_has_relation(
    projection: _ConversationProjection,
    relation: str,
) -> bool:
    """Return whether a projected packet includes the required relation."""

    if not relation:
        return_value = True
        return return_value

    for packet in projection["packets"]:
        relation_types = packet.get("relation_types")
        if not isinstance(relation_types, list):
            continue
        if relation in relation_types:
            return_value = True
            return return_value

    return_value = False
    return return_value

def _projection_evidence_items(
    projection: _ConversationProjection,
    *,
    packets: list[dict[str, Any]],
) -> list[str]:
    """Return packet summaries when available, otherwise row summaries."""

    packet_summaries = [
        summary
        for packet in packets
        if (summary := text_or_empty(packet.get("summary")))
    ]
    if packet_summaries:
        return_value = packet_summaries
        return return_value

    return_value = projection["summaries"]
    return return_value

def _projection_evidence_packets(
    projection: _ConversationProjection,
    *,
    relation_required: bool,
) -> list[dict[str, Any]]:
    """Return packets selected for cognition-facing evidence."""

    if relation_required:
        packets = list(projection["packets"])
        return packets

    packets = [
        packet
        for packet in projection["packets"]
        if _packet_has_keyword_support(packet)
    ]
    return packets

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
        "packets": [],
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
    packets = _conversation_packets(projected_rows)
    resolved_refs = _refs_from_message_rows(rows)
    projection: _ConversationProjection = {
        "summaries": summaries,
        "rows": projected_rows,
        "packets": packets,
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
    reply_context = row.get("reply_context")
    if not isinstance(reply_context, dict):
        reply_context = {}

    projected_row = {
        "summary": summary,
        "timestamp": text_or_empty(row.get("timestamp")),
        "display_name": text_or_empty(row.get("display_name")),
        "platform_message_id": text_or_empty(row.get("platform_message_id")),
        "conversation_row_id": conversation_row_id,
        "methods": methods,
        "score": score,
        "relation_to_seed": text_or_empty(row.get("relation_to_seed")),
        "seed_platform_message_id": text_or_empty(
            row.get("seed_platform_message_id")
        ),
        "seed_conversation_row_id": text_or_empty(
            row.get("seed_conversation_row_id")
        ),
        "seed_timestamp": text_or_empty(row.get("seed_timestamp")),
        "reply_parent_summary": text_or_empty(
            reply_context.get("reply_excerpt")
        ),
        "reply_parent_display_name": text_or_empty(
            reply_context.get("reply_to_display_name")
        ),
    }
    return projected_row

def _conversation_packets(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build compact seed-plus-relation packets from projected rows."""

    seed_rows = [
        row
        for row in rows
        if not _row_relation_type(row) or _row_has_direct_retrieval_method(row)
    ]
    if not seed_rows:
        return_value: list[dict[str, Any]] = []
        return return_value

    packets: list[dict[str, Any]] = []
    for seed in seed_rows[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT]:
        relations = _relations_for_seed(seed, rows)
        if not relations:
            continue
        relation_types = [
            relation["relation_type"]
            for relation in relations
            if relation["relation_type"]
        ]
        packet = {
            "summary": _packet_summary(seed, relations),
            "seed": seed,
            "relations": relations,
            "relation_types": relation_types,
        }
        packets.append(packet)

    return packets

def _relations_for_seed(
    seed: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return bounded relation rows attached to one seed message."""

    relations: list[dict[str, Any]] = []
    reply_parent = _reply_parent_relation(seed)
    seen_types: set[str] = set()
    if reply_parent is not None:
        relations.append(reply_parent)
        seen_types.add("reply_parent")

    for row in rows:
        relation_type = _row_relation_type(row)
        if not relation_type:
            continue
        if not _row_attaches_to_seed(row, seed):
            continue
        if relation_type in seen_types:
            continue
        relations.append(
            {
                "relation_type": relation_type,
                "summary": row["summary"],
                "row": row,
            }
        )
        seen_types.add(relation_type)
        if len(relations) >= _MAX_PACKET_RELATIONS:
            break

    return relations

def _reply_parent_relation(
    seed: dict[str, Any],
) -> dict[str, Any] | None:
    """Return reply-parent relation from prompt-safe reply metadata."""

    summary = text_or_empty(seed.get("reply_parent_summary"))
    if not summary:
        return None

    display_name = text_or_empty(seed.get("reply_parent_display_name"))
    if display_name:
        relation_summary = f"{display_name}: {summary}"
    else:
        relation_summary = summary
    relation = {
        "relation_type": "reply_parent",
        "summary": relation_summary,
        "row": {},
    }
    return relation

def _row_relation_type(row: dict[str, Any]) -> str:
    """Return a stable relation type from a projected row."""

    relation_type = text_or_empty(row.get("relation_to_seed")).lower()
    if relation_type in _RELATION_LABELS:
        return relation_type
    return_value = ""
    return return_value

def _row_has_direct_retrieval_method(row: dict[str, Any]) -> bool:
    """Return whether a projected row was retrieved directly, not only nearby."""

    methods = row.get("methods")
    if not isinstance(methods, list):
        return_value = False
        return return_value

    for method in methods:
        method_text = text_or_empty(method)
        if method_text == "semantic" or method_text.startswith("keyword:"):
            return_value = True
            return return_value

    return_value = False
    return return_value

def _packet_has_keyword_support(packet: dict[str, Any]) -> bool:
    """Return whether a packet seed came from exact keyword retrieval."""

    seed = packet.get("seed")
    if not isinstance(seed, dict):
        return_value = False
        return return_value

    methods = seed.get("methods")
    if not isinstance(methods, list):
        return_value = False
        return return_value

    for method in methods:
        if text_or_empty(method).startswith("keyword:"):
            return_value = True
            return return_value

    return_value = False
    return return_value

def _row_attaches_to_seed(
    row: dict[str, Any],
    seed: dict[str, Any],
) -> bool:
    """Return whether a relation row is attached to the seed row."""

    seed_platform_message_id = text_or_empty(seed.get("platform_message_id"))
    row_seed_platform_message_id = text_or_empty(
        row.get("seed_platform_message_id")
    )
    if seed_platform_message_id and row_seed_platform_message_id:
        return_value = seed_platform_message_id == row_seed_platform_message_id
        return return_value

    seed_conversation_row_id = text_or_empty(seed.get("conversation_row_id"))
    row_seed_conversation_row_id = text_or_empty(
        row.get("seed_conversation_row_id")
    )
    if seed_conversation_row_id and row_seed_conversation_row_id:
        return_value = seed_conversation_row_id == row_seed_conversation_row_id
        return return_value

    seed_timestamp = text_or_empty(seed.get("timestamp"))
    row_seed_timestamp = text_or_empty(row.get("seed_timestamp"))
    return_value = bool(seed_timestamp and seed_timestamp == row_seed_timestamp)
    return return_value

def _packet_summary(
    seed: dict[str, Any],
    relations: list[dict[str, Any]],
) -> str:
    """Reduce a conversation packet into one cognition-facing fact."""

    parts = [f"命中消息：{seed['summary']}"]
    for relation in relations[:_MAX_PACKET_RELATIONS]:
        relation_type = text_or_empty(relation.get("relation_type"))
        label = _RELATION_LABELS.get(relation_type, "相关消息")
        summary = text_or_empty(relation.get("summary"))
        if not summary:
            continue
        parts.append(f"{label}：{summary}")
    summary = "；".join(parts)
    return summary

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
        "packets": [],
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
