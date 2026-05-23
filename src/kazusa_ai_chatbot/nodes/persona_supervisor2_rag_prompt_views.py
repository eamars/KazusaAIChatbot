"""Prompt-facing compact views for RAG supervisor LLM calls."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.config import RAG_SEARCH_SELECTED_SUMMARY_LIMIT
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm


_LLM_SUMMARY_TEXT_LIMIT = 400
_LLM_SUMMARY_LIST_LIMIT = min(RAG_SEARCH_SELECTED_SUMMARY_LIMIT, 10)
_LLM_SUMMARY_REF_LIMIT = 8
_LLM_SUMMARY_PROFILE_UNIT_LIMIT = 4


def _clip_llm_summary_text(
    value: object,
    *,
    limit: int = _LLM_SUMMARY_TEXT_LIMIT,
) -> str:
    """Clip a field before sending it to a local summarizer/finalizer LLM.

    Args:
        value: Arbitrary evidence field to render.
        limit: Maximum number of characters kept from the rendered value.

    Returns:
        A string capped to the requested length.
    """

    text = str(value)
    if len(text) > limit:
        text = f"{text[:limit]}..."
    return_value = text
    return return_value


def _compact_memory_unit_rows(rows: object) -> list[object]:
    """Project memory/profile rows to the fields useful for summary prompts.

    Args:
        rows: Optional list of user-memory rows or memory-like dictionaries.

    Returns:
        A capped list with heavy metadata removed and text fields clipped.
    """

    if not isinstance(rows, list):
        return_value: list[object] = []
        return return_value

    _TIME_KEYS = {"updated_at", "timestamp"}
    compact_rows: list[object] = []
    for row in rows[:_LLM_SUMMARY_PROFILE_UNIT_LIMIT]:
        if not isinstance(row, dict):
            compact_rows.append(_clip_llm_summary_text(row))
            continue

        compact_row: dict[str, object] = {}
        for key in (
            "unit_type",
            "fact",
            "subjective_appraisal",
            "relationship_signal",
            "status",
            "updated_at",
            "timestamp",
            "memory_name",
            "content",
        ):
            if key in row:
                value = _clip_llm_summary_text(row[key])
                if key in _TIME_KEYS:
                    value = format_storage_utc_for_llm(str(value))
                compact_row[key] = value
        compact_rows.append(compact_row)

    return_value = compact_rows
    return return_value


def _compact_user_memory_context(memory_context: object) -> dict[str, object]:
    """Cap user-memory context sections for evaluator/finalizer prompts.

    Args:
        memory_context: Profile memory context from a profile payload.

    Returns:
        A dictionary with the standard memory sections and clipped rows.
    """

    if not isinstance(memory_context, dict):
        return_value: dict[str, object] = {}
        return return_value

    compact_context: dict[str, object] = {}
    for section in (
        "stable_patterns",
        "recent_shifts",
        "objective_facts",
        "milestones",
        "active_commitments",
    ):
        if section in memory_context:
            compact_context[section] = _compact_memory_unit_rows(
                memory_context[section],
            )

    return_value = compact_context
    return return_value


def _compact_profile_for_llm(profile: object) -> dict[str, object]:
    """Build a bounded profile view for local LLM summary stages.

    Args:
        profile: Raw profile or character image payload.

    Returns:
        Profile identity fields plus capped memory sections.
    """

    if not isinstance(profile, dict):
        return_value: dict[str, object] = {}
        return return_value

    compact_profile: dict[str, object] = {}
    for key in (
        "global_user_id",
        "display_name",
        "name",
        "description",
        "gender",
        "age",
        "birthday",
        "backstory",
    ):
        if key in profile:
            compact_profile[key] = _clip_llm_summary_text(profile[key])

    if "self_image" in profile:
        compact_profile["self_image"] = _clip_llm_summary_text(
            profile["self_image"],
        )

    memory_context = profile.get("user_memory_context")
    compact_context = _compact_user_memory_context(memory_context)
    if compact_context:
        compact_profile["user_memory_context"] = compact_context
    else:
        memory_units = profile.get("_user_memory_units")
        compact_units = _compact_memory_unit_rows(memory_units)
        if compact_units:
            compact_profile["_user_memory_units"] = compact_units

    return_value = compact_profile
    return return_value


def _compact_projection_payload_for_llm(payload: object) -> dict[str, object]:
    """Build a summary-safe view of a top-level capability projection payload.

    Args:
        payload: The capability result projection payload.

    Returns:
        A compact payload that preserves prompt-facing facts without heavy raw
        worker/profile internals.
    """

    if not isinstance(payload, dict):
        return_value: dict[str, object] = {}
        return return_value

    compact_payload: dict[str, object] = {}
    for key in (
        "profile_kind",
        "owner_global_user_id",
        "summary",
        "external_text",
        "url",
    ):
        if key in payload:
            compact_payload[key] = _clip_llm_summary_text(payload[key])

    if "profile" in payload:
        compact_payload["profile"] = _compact_profile_for_llm(
            payload["profile"],
        )

    summaries = payload.get("summaries")
    if isinstance(summaries, list):
        compact_payload["summaries"] = [
            _clip_llm_summary_text(item)
            for item in summaries[:_LLM_SUMMARY_LIST_LIMIT]
        ]

    memory_rows = payload.get("memory_rows")
    compact_memory_rows = _compact_memory_unit_rows(memory_rows)
    if compact_memory_rows:
        compact_payload["memory_rows"] = compact_memory_rows

    return_value = compact_payload
    return return_value


def _compact_raw_result_for_llm(raw_result: object) -> object:
    """Remove heavy internals before a raw result enters a local LLM prompt.

    Args:
        raw_result: Agent raw result stored in known_facts.

    Returns:
        The original raw result for ordinary worker payloads, or a compact
        view for top-level capability/profile payloads.
    """

    projected_result = project_tool_result_for_llm(raw_result)
    if not isinstance(projected_result, dict):
        return projected_result

    if "capability" in projected_result:
        compact_result: dict[str, object] = {
            "capability": projected_result.get("capability", ""),
            "primary_worker": projected_result.get("primary_worker", ""),
            "supporting_workers": projected_result.get("supporting_workers", []),
            "source_policy": projected_result.get("source_policy", ""),
            "missing_context": projected_result.get("missing_context", []),
            "conflicts": projected_result.get("conflicts", []),
            "selected_summary": projected_result.get("selected_summary", ""),
        }

        evidence = projected_result.get("evidence")
        if isinstance(evidence, list):
            compact_result["evidence"] = [
                _clip_llm_summary_text(item)
                for item in evidence[:_LLM_SUMMARY_LIST_LIMIT]
            ]

        refs = projected_result.get("resolved_refs")
        if isinstance(refs, list):
            compact_result["resolved_refs"] = refs[:_LLM_SUMMARY_REF_LIMIT]

        projection_payload = projected_result.get("projection_payload")
        compact_result["projection_payload"] = _compact_projection_payload_for_llm(
            projection_payload,
        )
        return_value: object = compact_result
        return return_value

    if (
        "user_memory_context" in projected_result
        or "_user_memory_units" in projected_result
        or "self_image" in projected_result
    ):
        return_value = _compact_profile_for_llm(projected_result)
        return return_value

    return projected_result


def _known_facts_llm_view(known_facts: object) -> list[dict[str, object]]:
    """Compact previous facts before sending them back through an LLM.

    Args:
        known_facts: Facts accumulated by the evaluator.

    Returns:
        A list preserving slot summaries and compact raw results.
    """

    if not isinstance(known_facts, list):
        return_value: list[dict[str, object]] = []
        return return_value

    compact_facts: list[dict[str, object]] = []
    for fact in known_facts:
        if not isinstance(fact, dict):
            continue

        compact_fact = {
            "slot": fact.get("slot", ""),
            "agent": fact.get("agent", ""),
            "resolved": bool(fact.get("resolved", False)),
            "summary": _clip_llm_summary_text(fact.get("summary", "")),
            "raw_result": _compact_raw_result_for_llm(fact.get("raw_result")),
            "attempts": fact.get("attempts", 0),
        }
        compact_facts.append(compact_fact)

    return_value = compact_facts
    return return_value
