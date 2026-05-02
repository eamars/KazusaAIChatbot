"""Projection helpers for converting RAG2 facts into persona-stage context."""

from __future__ import annotations

import re
from typing import Any

from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import empty_user_memory_context
from kazusa_ai_chatbot.utils import text_or_empty

_URL_RE = re.compile(r"https?://\S+")
_SLOT_REF_RE = re.compile(r"slot\s+(\d+)", flags=re.IGNORECASE)
_MAX_RECALL_EVIDENCE = 3


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
        "answer": text_or_empty(answer),
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
    memory_evidence: list[dict[str, str]] = []
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

        dispatched.append({
            "slot": slot,
            "agent": agent,
            "resolved": resolved,
        })

        if not resolved:
            continue

        if agent == "user_profile_agent":
            raw_profile = _as_dict(raw_result)
            owner_id = _resolve_profile_owner_id(fact, known_facts)
            if owner_id == current_user_id:
                rag_result["user_image"] = _strip_internal_profile_fields(raw_profile)
                rag_result["user_memory_unit_candidates"] = _as_list(
                    raw_profile.get("_user_memory_units")
                )
                continue
            if owner_id == character_user_id or (not owner_id and raw_profile.get("self_image") is not None):
                rag_result["character_image"] = raw_profile
                continue
            if summary:
                third_party_profiles.append(summary)
            continue

        if agent in {"user_lookup_agent", "user_list_agent", "relationship_agent"}:
            if summary:
                third_party_profiles.append(summary)
            continue

        if agent in {"persistent_memory_search_agent", "persistent_memory_keyword_agent"}:
            memory_evidence.append({
                "summary": summary,
                "content": _extract_memory_content(raw_result, evidence_char_limit=evidence_char_limit),
            })
            continue

        if agent == "recall_agent":
            if len(recall_evidence) >= _MAX_RECALL_EVIDENCE:
                continue
            recall_payload = _as_dict(raw_result)
            if recall_payload:
                recall_evidence.append(recall_payload)
            elif summary:
                recall_evidence.append({"selected_summary": summary})
            continue

        if agent in {
            "conversation_search_agent",
            "conversation_filter_agent",
            "conversation_keyword_agent",
            "conversation_aggregate_agent",
        }:
            if summary:
                conversation_evidence.append(summary)
            continue

        if agent == "web_search_agent2":
            content, url = _extract_external_content(raw_result, evidence_char_limit=evidence_char_limit)
            external_evidence.append({
                "summary": summary,
                "content": content,
                "url": url,
            })
            continue

        if summary:
            conversation_evidence.append(summary)

    rag_result["third_party_profiles"] = third_party_profiles
    rag_result["memory_evidence"] = memory_evidence
    rag_result["recall_evidence"] = recall_evidence
    rag_result["conversation_evidence"] = conversation_evidence
    rag_result["external_evidence"] = external_evidence
    rag_result["supervisor_trace"]["dispatched"] = dispatched
    return rag_result
