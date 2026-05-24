"""Projection helpers for converting RAG2 facts into persona-stage context."""

from __future__ import annotations

import re
from typing import Any

from kazusa_ai_chatbot.config import RAG_SEARCH_SELECTED_SUMMARY_LIMIT
from kazusa_ai_chatbot.rag.evidence_formatting import (
    ensure_public_rag_evidence_prompt_safe,
    format_evidence_block,
    sanitize_public_rag_evidence_text,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import empty_user_memory_context
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm_seconds
from kazusa_ai_chatbot.utils import text_or_empty

_URL_RE = re.compile(r"https?://\S+")
_SLOT_REF_RE = re.compile(r"slot\s+(\d+)", flags=re.IGNORECASE)
_MAX_RECALL_EVIDENCE = 3
_MAX_CONVERSATION_EVIDENCE_ITEMS = RAG_SEARCH_SELECTED_SUMMARY_LIMIT


def _clip_text(text: str, *, limit: int) -> str:
    """Apply a uniform character limit to evidence text.

    Args:
        text: Text payload from a retrieval helper.
        limit: Maximum number of characters to retain.

    Returns:
        The stripped text, clipped uniformly when needed.
    """
    clipped = text.strip()
    if len(clipped) <= limit:
        return clipped
    clipped_text = clipped[: limit - 1].rstrip() + "…"
    return clipped_text


def _as_dict(value: object) -> dict[str, Any]:
    """Return ``value`` when it is a dict, otherwise an empty dict."""
    if isinstance(value, dict):
        return value
    return_value = {}
    return return_value


def _as_list(value: object) -> list[Any]:
    """Return ``value`` when it is a list, otherwise an empty list."""
    if isinstance(value, list):
        return value
    return_value = []
    return return_value


def _strip_internal_profile_fields(profile: dict[str, Any]) -> dict[str, Any]:
    """Remove RAG-internal helper fields before cognition sees a profile.

    Args:
        profile: Raw user-profile agent result.

    Returns:
        Public profile payload suitable for cognition prompts.
    """

    stripped = dict(profile)
    stripped.pop("_user_memory_units", None)
    return stripped


def _extract_slot_reference(slot: str) -> int | None:
    """Parse a structured ``slot N`` reference from a RAG2 slot label."""
    match = _SLOT_REF_RE.search(slot or "")
    if match is None:
        return None
    slot_reference = int(match.group(1))
    return slot_reference


def _resolve_profile_owner_id(fact: dict[str, Any], known_facts: list[dict[str, Any]]) -> str:
    """Resolve the owner UUID for a ``user_profile_agent`` fact.

    Args:
        fact: Current RAG2 fact row.
        known_facts: Full ordered fact list, used for slot-reference lookup.

    Returns:
        The resolved owner UUID, or an empty string when unknown.
    """
    raw_result = _as_dict(fact.get("raw_result"))
    direct_id = text_or_empty(raw_result.get("global_user_id"))
    if direct_id:
        return direct_id

    slot_reference = _extract_slot_reference(text_or_empty(fact.get("slot")))
    if slot_reference is None:
        return ""
    if slot_reference < 1 or slot_reference > len(known_facts):
        return ""

    referenced = _as_dict(known_facts[slot_reference - 1])
    referenced_raw = _as_dict(referenced.get("raw_result"))
    owner_id = text_or_empty(referenced_raw.get("global_user_id"))
    return owner_id


def _extract_memory_content(raw_result: object, *, evidence_char_limit: int) -> str:
    """Extract clipped memory snippets from persistent-memory raw results.

    Args:
        raw_result: Helper-agent raw result, expected to be a list of rows.
        evidence_char_limit: Per-row character limit for copied evidence.

    Returns:
        Newline-joined clipped memory content.
    """
    entries = _as_list(raw_result)
    snippets: list[str] = []
    for entry in entries[:5]:
        if not isinstance(entry, dict):
            continue
        content = text_or_empty(entry.get("content"))
        if not content:
            continue
        snippets.append(_clip_text(content, limit=evidence_char_limit))
    memory_content = "\n".join(snippets)
    return memory_content


def _scoped_user_memory_rows(
    rows: list[dict[str, Any]],
    *,
    current_user_id: str,
) -> list[dict[str, Any]]:
    """Return scoped current-user continuity rows from memory-evidence payloads."""
    scoped_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if text_or_empty(row.get("source_system")) != "user_memory_units":
            continue
        if text_or_empty(row.get("scope_type")) != "user_continuity":
            continue
        scope_global_user_id = text_or_empty(row.get("scope_global_user_id"))
        if scope_global_user_id != current_user_id:
            continue
        scoped_rows.append(dict(row))
    return scoped_rows


def _append_user_memory_unit_candidates(
    existing_candidates: list[dict[str, Any]],
    new_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge user-memory candidates while preserving first-seen order."""
    merged_candidates: list[dict[str, Any]] = []
    seen_unit_ids: set[str] = set()
    for candidate in existing_candidates + new_candidates:
        if not isinstance(candidate, dict):
            continue
        unit_id = text_or_empty(candidate.get("unit_id"))
        if unit_id:
            if unit_id in seen_unit_ids:
                continue
            seen_unit_ids.add(unit_id)
        merged_candidates.append(candidate)
    return merged_candidates


def _conclusion_line(summary: str) -> str:
    """Return one prompt-facing conclusion line."""

    summary_text = sanitize_public_rag_evidence_text(summary)
    if summary_text.startswith("结论："):
        return summary_text
    if summary_text.startswith("Conclusion: "):
        summary_text = summary_text.removeprefix("Conclusion: ").strip()
    conclusion = f"结论：{summary_text}"
    return conclusion


def _evidence_summary_content(
    evidence_items: list[str],
    *,
    empty_uncertainty: str,
) -> str:
    """Return evidence-summary text without a repeated conclusion line."""

    clean_items = [
        item_text
        for item in evidence_items
        if (item_text := sanitize_public_rag_evidence_text(item))
    ]
    if not clean_items:
        uncertainty_text = (
            sanitize_public_rag_evidence_text(empty_uncertainty)
            or "无"
        )
        content = f"不确定性：{uncertainty_text}"
        return content

    lines = ["上下文："]
    for item in clean_items:
        lines.append(f"- {item}")
    lines.append("不确定性：无")
    content = "\n".join(lines)
    return content


def _first_local_second_time(
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> str:
    """Return the first valid local-second timestamp from known row fields."""

    for field in fields:
        value = text_or_empty(row.get(field))
        if not value:
            continue
        projected_time = format_storage_utc_for_llm_seconds(value)
        if projected_time:
            return projected_time
    return ""


def _memory_evidence_items(
    rows: list[dict[str, Any]],
    *,
    evidence_char_limit: int,
) -> list[str]:
    """Project memory rows into prompt-facing evidence lines."""

    evidence_items: list[str] = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        content = text_or_empty(row.get("content"))
        if not content:
            continue
        clipped_content = _clip_text(content, limit=evidence_char_limit)
        evidence_time = _first_local_second_time(
            row,
            ("updated_at", "timestamp", "created_at", "last_seen_at"),
        )
        if evidence_time:
            evidence_item = f"记忆（{evidence_time}）：{clipped_content}"
        else:
            evidence_item = f"记忆：{clipped_content}"
        evidence_items.append(evidence_item)
    return evidence_items


def _conversation_row_text(row: dict[str, Any]) -> str:
    """Return the prompt-facing text from one conversation evidence row."""

    for field in ("summary", "body_text", "content"):
        content = text_or_empty(row.get(field))
        if content:
            return content
    return ""


def _conversation_evidence_items(
    rows: list[dict[str, Any]],
    *,
    evidence_char_limit: int,
) -> list[str]:
    """Project conversation rows into speaker/time evidence lines."""

    evidence_items: list[str] = []
    for row in rows[:_MAX_CONVERSATION_EVIDENCE_ITEMS]:
        if not isinstance(row, dict):
            continue
        content = _conversation_row_text(row)
        if not content:
            continue
        clipped_content = _clip_text(content, limit=evidence_char_limit)
        speaker = text_or_empty(row.get("display_name"))
        if not speaker:
            speaker = text_or_empty(row.get("role")) or "对话记录"
        evidence_time = _first_local_second_time(row, ("timestamp",))
        if evidence_time:
            evidence_item = f"{speaker}（{evidence_time}）：{clipped_content}"
        else:
            evidence_item = f"{speaker}：{clipped_content}"
        evidence_items.append(evidence_item)
    return evidence_items


def _recall_candidate_items(
    candidates: list[Any],
    *,
    evidence_char_limit: int,
) -> list[str]:
    """Project recall candidates into prompt-facing evidence lines."""

    evidence_items: list[str] = []
    for candidate in candidates[:5]:
        if not isinstance(candidate, dict):
            continue
        claim = text_or_empty(candidate.get("claim"))
        if not claim:
            claim = text_or_empty(candidate.get("summary"))
        if not claim:
            continue
        clipped_claim = _clip_text(claim, limit=evidence_char_limit)
        evidence_time = _first_local_second_time(
            candidate,
            ("evidence_time", "timestamp", "updated_at"),
        )
        if evidence_time:
            evidence_item = f"召回记录（{evidence_time}）：{clipped_claim}"
        else:
            evidence_item = f"召回记录：{clipped_claim}"
        evidence_items.append(evidence_item)
    return evidence_items


def _source_ref_value(row: dict[str, Any], field: str) -> str:
    """Return a trace-only source-ref value from a raw row field."""

    value = row.get(field)
    if field == "_id" and value is not None:
        return_value = str(value)
        return return_value
    return_value = text_or_empty(value)
    return return_value


def _source_refs_from_rows(rows: list[Any]) -> list[dict[str, str]]:
    """Collect compact trace-only refs from raw evidence rows."""

    source_refs: list[dict[str, str]] = []
    source_fields = (
        "conversation_row_id",
        "platform_message_id",
        "_id",
        "unit_id",
        "source",
        "source_system",
        "timestamp",
        "evidence_time",
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_ref: dict[str, str] = {}
        for field in source_fields:
            value = _source_ref_value(row, field)
            if value:
                source_ref[field] = value
        if source_ref:
            source_refs.append(source_ref)
    return source_refs


def _attach_source_refs(
    dispatched_entry: dict[str, Any],
    rows: list[Any],
) -> None:
    """Attach trace-only source refs to one dispatched supervisor row."""

    source_refs = _source_refs_from_rows(rows)
    if not source_refs:
        return
    dispatched_entry["source_refs"] = source_refs


def _public_string_list(value: object) -> list[str] | None:
    """Return a list of strings when the value is a prompt-safe string list."""

    if not isinstance(value, list):
        return None
    string_items = [
        item
        for item in value
        if isinstance(item, str)
    ]
    if len(string_items) != len(value):
        return None
    return string_items


def _recall_evidence_entry(
    recall_payload: dict[str, Any],
    *,
    summary: str,
    evidence_char_limit: int,
) -> dict[str, Any]:
    """Project one Recall payload into cognition-ready public evidence."""

    selected_summary = (
        text_or_empty(recall_payload.get("selected_summary")) or summary
    )
    entry: dict[str, Any] = {
        "selected_summary": _conclusion_line(selected_summary),
    }

    for field in ("recall_type", "primary_source", "freshness_basis"):
        value = text_or_empty(recall_payload.get(field))
        if value:
            entry[field] = value

    for field in ("supporting_sources", "conflicts"):
        value = _public_string_list(recall_payload.get(field))
        if value is not None:
            entry[field] = value

    candidates = _as_list(recall_payload.get("candidates"))
    evidence_items = _recall_candidate_items(
        candidates,
        evidence_char_limit=evidence_char_limit,
    )
    entry["evidence_summary"] = _evidence_summary_content(
        evidence_items,
        empty_uncertainty="没有可用于提示的召回证据。",
    )
    return entry


def _memory_evidence_entry(
    *,
    summary: str,
    rows: list[dict[str, Any]],
    current_user_id: str,
    evidence_char_limit: int,
) -> dict[str, Any]:
    """Project one memory-evidence item and preserve scoped continuity metadata."""
    evidence_items = _memory_evidence_items(
        rows,
        evidence_char_limit=evidence_char_limit,
    )
    entry: dict[str, Any] = {
        "summary": _conclusion_line(summary),
        "content": _evidence_summary_content(
            evidence_items,
            empty_uncertainty=(
                "没有可用于提示的记忆证据。"
            ),
        ),
    }
    scoped_rows = _scoped_user_memory_rows(rows, current_user_id=current_user_id)
    if not scoped_rows:
        return entry

    scoped_row = scoped_rows[0]
    for field in (
        "source_system",
        "scope_type",
        "scope_global_user_id",
        "authority",
        "truth_status",
        "origin",
    ):
        value = text_or_empty(scoped_row.get(field))
        if value:
            entry[field] = value
    return entry


def _extract_external_content(raw_result: object, *, evidence_char_limit: int) -> tuple[str, str]:
    """Extract clipped external evidence text and its first visible URL.

    Args:
        raw_result: Web-search raw result payload.
        evidence_char_limit: Character limit for copied evidence text.

    Returns:
        Tuple of clipped text and first detected URL, if any.
    """
    text = _clip_text(text_or_empty(raw_result), limit=evidence_char_limit)
    url_match = _URL_RE.search(text)
    url = url_match.group(0) if url_match else ""
    external_content = (text, url)
    return external_content


def _projection_payload(raw_result: object) -> dict[str, Any]:
    """Read a normalized top-level capability projection payload."""
    raw_dict = _as_dict(raw_result)
    payload = _as_dict(raw_dict.get("projection_payload"))
    return payload


def _project_top_level_summaries(
    payload: dict[str, Any],
    *,
    key: str,
    evidence_char_limit: int,
) -> list[str]:
    """Project string summaries from a normalized capability payload."""
    raw_summaries = _as_list(payload.get(key))
    summaries = [
        _clip_text(text, limit=evidence_char_limit)
        for item in raw_summaries
        if (text := text_or_empty(item))
    ]
    return summaries


def project_known_facts(
    known_facts: list[dict],
    *,
    current_user_id: str,
    character_user_id: str,
    evidence_char_limit: int = 800,
    answer: str = "",
    unknown_slots: list[str] | None = None,
    loop_count: int = 0,
) -> dict[str, Any]:
    """Project RAG2 ``known_facts`` into the hybrid cognition payload.

    Args:
        known_facts: Raw fact rows emitted by ``call_rag_supervisor``.
        current_user_id: Global user id for the current speaker.
        character_user_id: Global user id for the character profile.
        evidence_char_limit: Uniform per-hit evidence text limit.
        answer: Final supervisor synthesis.
        unknown_slots: Slots left unresolved by the supervisor.
        loop_count: Number of supervisor dispatch loops.

    Returns:
        ``rag_result`` dict consumed by cognition and consolidation stages.
    """
    rag_result: dict[str, Any] = {
        "answer": sanitize_public_rag_evidence_text(answer),
        "user_image": {"user_memory_context": empty_user_memory_context()},
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": int(loop_count or 0),
            "unknown_slots": list(unknown_slots or []),
            "dispatched": [],
        },
    }

    third_party_profiles: list[str] = []
    memory_evidence: list[dict[str, Any]] = []
    recall_evidence: list[dict[str, Any]] = []
    conversation_evidence: list[str] = []
    external_evidence: list[dict[str, str]] = []
    dispatched: list[dict[str, Any]] = []

    for fact in known_facts:
        if not isinstance(fact, dict):
            continue

        slot = text_or_empty(fact.get("slot"))
        agent = text_or_empty(fact.get("agent"))
        resolved = bool(fact.get("resolved", False))
        summary = text_or_empty(fact.get("summary"))
        raw_result = fact.get("raw_result")

        dispatched_entry: dict[str, Any] = {
            "slot": slot,
            "agent": agent,
            "resolved": resolved,
        }
        continuation = fact.get("continuation")
        if isinstance(continuation, dict):
            dispatched_entry["continuation"] = {
                "promote_candidate": bool(
                    continuation.get("promote_candidate", False)
                ),
                "promoted_candidate_indexes": [
                    index
                    for index in continuation.get(
                        "promoted_candidate_indexes",
                        [],
                    )
                    if isinstance(index, int) and not isinstance(index, bool)
                ],
                "promotion_summary": text_or_empty(
                    continuation.get("promotion_summary")
                ),
                "should_continue": bool(
                    continuation.get("should_continue", False)
                ),
                "refined_query": text_or_empty(
                    continuation.get("refined_query")
                ),
                "reason": text_or_empty(continuation.get("reason")),
            }
        dispatched.append(dispatched_entry)

        if not resolved:
            continue

        if agent == "live_context_agent":
            payload = _projection_payload(raw_result)
            content = _clip_text(
                text_or_empty(payload.get("external_text")),
                limit=evidence_char_limit,
            )
            url = text_or_empty(payload.get("url"))
            if not url:
                url_match = _URL_RE.search(content)
                url = url_match.group(0) if url_match else ""
            external_evidence.append({
                "summary": sanitize_public_rag_evidence_text(summary),
                "content": sanitize_public_rag_evidence_text(content),
                "url": url,
            })
            continue

        if agent == "conversation_evidence_agent":
            payload = _projection_payload(raw_result)
            rows = [
                row
                for row in _as_list(payload.get("rows"))
                if isinstance(row, dict)
            ]
            _attach_source_refs(dispatched_entry, rows)
            row_items = _conversation_evidence_items(
                rows,
                evidence_char_limit=evidence_char_limit,
            )
            if row_items:
                conversation_evidence.append(
                    format_evidence_block(
                        conclusion=summary,
                        evidence_items=row_items,
                    )
                )
                continue

            summaries = _project_top_level_summaries(
                payload,
                key="summaries",
                evidence_char_limit=evidence_char_limit,
            )
            for item_summary in summaries:
                conversation_evidence.append(
                    format_evidence_block(
                        conclusion=item_summary,
                        evidence_items=[],
                    )
                )
            if not summaries and summary:
                conversation_evidence.append(
                    format_evidence_block(
                        conclusion=summary,
                        evidence_items=[],
                    )
                )
            continue

        if agent == "memory_evidence_agent":
            payload = _projection_payload(raw_result)
            memory_rows = [
                row
                for row in _as_list(payload.get("memory_rows"))
                if isinstance(row, dict)
            ]
            _attach_source_refs(dispatched_entry, memory_rows)
            memory_evidence.append(
                _memory_evidence_entry(
                    summary=summary,
                    rows=memory_rows,
                    current_user_id=current_user_id,
                    evidence_char_limit=evidence_char_limit,
                )
            )
            rag_result["user_memory_unit_candidates"] = _append_user_memory_unit_candidates(
                rag_result["user_memory_unit_candidates"],
                _scoped_user_memory_rows(memory_rows, current_user_id=current_user_id),
            )
            continue

        if agent == "person_context_agent":
            payload = _projection_payload(raw_result)
            profile_kind = text_or_empty(payload.get("profile_kind"))
            raw_profile = _as_dict(payload.get("profile"))
            payload_summary = text_or_empty(payload.get("summary")) or summary
            if profile_kind == "current_user":
                rag_result["user_image"] = _strip_internal_profile_fields(raw_profile)
                rag_result["user_memory_unit_candidates"] = _append_user_memory_unit_candidates(
                    rag_result["user_memory_unit_candidates"],
                    [
                        row
                        for row in _as_list(raw_profile.get("_user_memory_units"))
                        if isinstance(row, dict)
                    ],
                )
                continue
            if profile_kind == "active_character":
                rag_result["character_image"] = raw_profile
                continue
            public_summary = sanitize_public_rag_evidence_text(payload_summary)
            if public_summary:
                third_party_profiles.append(public_summary)
            continue

        if agent == "user_profile_agent":
            raw_profile = _as_dict(raw_result)
            owner_id = _resolve_profile_owner_id(fact, known_facts)
            if owner_id == current_user_id:
                rag_result["user_image"] = _strip_internal_profile_fields(raw_profile)
                rag_result["user_memory_unit_candidates"] = _append_user_memory_unit_candidates(
                    rag_result["user_memory_unit_candidates"],
                    [
                        row
                        for row in _as_list(raw_profile.get("_user_memory_units"))
                        if isinstance(row, dict)
                    ],
                )
                continue
            if owner_id == character_user_id or (not owner_id and raw_profile.get("self_image") is not None):
                rag_result["character_image"] = raw_profile
                continue
            public_summary = sanitize_public_rag_evidence_text(summary)
            if public_summary:
                third_party_profiles.append(public_summary)
            continue

        if agent in {"user_lookup_agent", "user_list_agent", "relationship_agent"}:
            public_summary = sanitize_public_rag_evidence_text(summary)
            if public_summary:
                third_party_profiles.append(public_summary)
            continue

        if agent in {"persistent_memory_search_agent", "persistent_memory_keyword_agent"}:
            memory_rows = [
                row
                for row in _as_list(raw_result)
                if isinstance(row, dict)
            ]
            _attach_source_refs(dispatched_entry, memory_rows)
            memory_evidence.append(
                _memory_evidence_entry(
                    summary=summary,
                    rows=memory_rows,
                    current_user_id=current_user_id,
                    evidence_char_limit=evidence_char_limit,
                )
            )
            continue

        if agent == "recall_agent":
            if len(recall_evidence) >= _MAX_RECALL_EVIDENCE:
                continue
            recall_payload = _as_dict(raw_result)
            if recall_payload:
                _attach_source_refs(
                    dispatched_entry,
                    _as_list(recall_payload.get("candidates")),
                )
                recall_evidence.append(
                    _recall_evidence_entry(
                        recall_payload,
                        summary=summary,
                        evidence_char_limit=evidence_char_limit,
                    )
                )
            elif summary:
                recall_evidence.append(
                    {
                        "selected_summary": _conclusion_line(summary),
                        "evidence_summary": _evidence_summary_content(
                            [],
                            empty_uncertainty=(
                                "没有可用于提示的召回证据。"
                            ),
                        ),
                    }
                )
            continue

        if agent in {
            "conversation_search_agent",
            "conversation_filter_agent",
            "conversation_keyword_agent",
            "conversation_aggregate_agent",
        }:
            if summary:
                rows = [
                    row
                    for row in _as_list(raw_result)
                    if isinstance(row, dict)
                ]
                _attach_source_refs(dispatched_entry, rows)
                conversation_evidence.append(
                    format_evidence_block(
                        conclusion=summary,
                        evidence_items=[],
                    )
                )
            continue

        if agent == "web_search_agent2":
            content, url = _extract_external_content(raw_result, evidence_char_limit=evidence_char_limit)
            external_evidence.append({
                "summary": sanitize_public_rag_evidence_text(summary),
                "content": sanitize_public_rag_evidence_text(content),
                "url": url,
            })
            continue

        if summary:
            conversation_evidence.append(
                format_evidence_block(
                    conclusion=summary,
                    evidence_items=[],
                )
            )

    rag_result["third_party_profiles"] = third_party_profiles
    rag_result["memory_evidence"] = memory_evidence
    rag_result["recall_evidence"] = recall_evidence
    rag_result["conversation_evidence"] = conversation_evidence
    rag_result["external_evidence"] = external_evidence
    rag_result["supervisor_trace"]["dispatched"] = dispatched
    ensure_public_rag_evidence_prompt_safe(rag_result)
    return rag_result
