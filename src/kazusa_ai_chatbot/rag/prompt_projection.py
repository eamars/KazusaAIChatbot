"""Explicit LLM-facing projections for RAG prompt payloads."""

from __future__ import annotations

import copy
from typing import Any

from kazusa_ai_chatbot.time_context import (
    format_history_for_llm,
    format_time_fields_for_llm,
)

_TIME_FIELDS = (
    "timestamp",
    "created_at",
    "updated_at",
    "first_seen_at",
    "last_seen_at",
    "last_timestamp",
    "from_timestamp",
    "to_timestamp",
    "expiry_timestamp",
    "expires_at",
    "execute_at",
    "due_at",
    "due_time",
    "completed_at",
    "cancelled_at",
    "current_turn_timestamp",
    "existing_updated_at",
    "existing_last_seen_at",
)

_NESTED_LIST_FIELDS = (
    "rows",
    "messages",
    "results",
    "known_facts",
    "candidates",
    "resolved_refs",
    "source_refs",
    "evidence_refs",
    "memory_rows",
    "user_memory_unit_candidates",
    "stable_patterns",
    "recent_shifts",
    "objective_facts",
    "milestones",
    "active_commitments",
    "memory_evidence",
    "recall_evidence",
    "conversation_evidence",
    "external_evidence",
    "third_party_profiles",
    "recent_window",
    "user_state_updates",
    "open_loops",
    "resolved_threads",
    "avoid_reopening",
)

_NESTED_OBJECT_FIELDS = (
    "raw_result",
    "result",
    "projection_payload",
    "user_memory_context",
    "user_image",
    "character_image",
    "self_image",
    "supervisor_trace",
    "profile",
)

_RUNTIME_DIRECT_KEYS = (
    "platform",
    "platform_channel_id",
    "channel_type",
    "global_user_id",
    "platform_user_id",
    "platform_bot_id",
    "user_name",
    "display_name",
    "original_query",
    "current_slot",
    "channel_topic",
    "reply_context",
    "indirect_speech_context",
    "exclude_current_question",
)

_RUNTIME_STRUCTURED_KEYS = (
    "prompt_message_context",
    "referents",
    "conversation_progress",
    "conversation_episode_state",
)

_USER_PROFILE_FIELDS = (
    "global_user_id",
    "display_name",
    "user_name",
    "platform",
    "platform_user_id",
    "affinity",
    "last_relationship_insight",
    "relationship_rank",
)

_TIME_CONTEXT_FIELDS = (
    "current_local_datetime",
    "current_local_weekday",
)


def project_tool_result_for_llm(value: object) -> object:
    """Project known tool-result structures into prompt-facing local time.

    Args:
        value: Tool result or nested result fragment from a known RAG helper.

    Returns:
        A copied value with timestamp fields converted only on source-owned
        row and result fields.
    """
    if isinstance(value, dict):
        projected = _project_dict_for_llm(value)
        return projected

    if isinstance(value, list):
        projected_list = [
            project_tool_result_for_llm(item)
            for item in value
        ]
        return_value: object = projected_list
        return return_value

    if isinstance(value, tuple):
        projected_tuple = [
            project_tool_result_for_llm(item)
            for item in value
        ]
        return_value = projected_tuple
        return return_value

    return_value = copy.deepcopy(value)
    return return_value


def project_known_facts_for_llm(known_facts: object) -> list[dict[str, object]]:
    """Project RAG known facts before another LLM sees them.

    Args:
        known_facts: Accumulated known-fact rows from RAG supervisor state.

    Returns:
        Prompt-facing fact rows with compact source metadata and local time.
    """
    if not isinstance(known_facts, list):
        return_value: list[dict[str, object]] = []
        return return_value

    projected_facts: list[dict[str, object]] = []
    for fact in known_facts:
        if not isinstance(fact, dict):
            continue

        raw_result = fact.get("raw_result")
        projected_fact = {
            "slot": fact.get("slot", ""),
            "agent": fact.get("agent", ""),
            "resolved": bool(fact.get("resolved", False)),
            "summary": fact.get("summary", ""),
            "raw_result": project_tool_result_for_llm(raw_result),
            "attempts": fact.get("attempts", 0),
        }
        projected_facts.append(projected_fact)

    return_value = projected_facts
    return return_value


def project_runtime_context_for_llm(context: dict[str, Any]) -> dict[str, Any]:
    """Build the explicit LLM view of a RAG runtime context.

    Args:
        context: Internal RAG context. It may contain raw UTC clocks for
            deterministic tools.

    Returns:
        A prompt-facing context. Root machine clocks are omitted; the model sees
        only ``time_context`` for current time and local times in projected
        history/evidence rows.
    """
    projected: dict[str, Any] = {}

    for key in _RUNTIME_DIRECT_KEYS:
        if key in context:
            projected[key] = copy.deepcopy(context[key])

    for key in _RUNTIME_STRUCTURED_KEYS:
        if key in context:
            projected[key] = project_tool_result_for_llm(context[key])

    time_context = context.get("time_context")
    if isinstance(time_context, dict):
        projected["time_context"] = _project_time_context(time_context)

    user_profile = context.get("user_profile")
    if isinstance(user_profile, dict):
        projected["user_profile"] = _project_user_profile_for_llm(user_profile)

    for history_key in ("chat_history_recent", "chat_history_wide"):
        history_rows = context.get(history_key)
        if isinstance(history_rows, list):
            projected[history_key] = _project_history_for_llm(history_rows)

    known_facts = context.get("known_facts")
    if isinstance(known_facts, list):
        projected["known_facts"] = project_known_facts_for_llm(known_facts)

    return_value = projected
    return return_value


def project_selector_input_for_llm(
    task: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build the common top-level RAG selector payload.

    Args:
        task: Slot text being routed by a top-level capability selector.
        context: Runtime context supplied by the outer RAG supervisor.

    Returns:
        The exact selector JSON shape with projected known facts.
    """
    selector_input = {
        "task": task,
        "original_query": context.get("original_query"),
        "current_slot": context.get("current_slot"),
        "known_facts": project_known_facts_for_llm(context.get("known_facts")),
    }
    return selector_input


def _project_dict_for_llm(row: dict[str, Any]) -> dict[str, Any]:
    """Project one known result row and its source-owned nested fields."""
    projected = format_time_fields_for_llm(row, _TIME_FIELDS)

    for field in _NESTED_LIST_FIELDS:
        value = projected.get(field)
        if isinstance(value, list):
            projected[field] = [
                project_tool_result_for_llm(item)
                for item in value
            ]

    for field in _NESTED_OBJECT_FIELDS:
        value = projected.get(field)
        if isinstance(value, (dict, list, tuple)):
            projected[field] = project_tool_result_for_llm(value)

    return_value = projected
    return return_value


def _project_history_for_llm(rows: list[object]) -> list[object]:
    """Project chat-history rows while preserving non-dict entries."""
    dict_rows = [
        row
        for row in rows
        if isinstance(row, dict)
    ]
    formatted_rows = format_history_for_llm(dict_rows)
    formatted_iter = iter(formatted_rows)

    projected_rows: list[object] = []
    for row in rows:
        if isinstance(row, dict):
            projected_rows.append(next(formatted_iter))
        else:
            projected_rows.append(copy.deepcopy(row))

    return_value = projected_rows
    return return_value


def _project_time_context(time_context: dict[str, Any]) -> dict[str, Any]:
    """Keep only the prompt contract fields from time context."""
    projected = {
        key: time_context[key]
        for key in _TIME_CONTEXT_FIELDS
        if key in time_context
    }
    return projected


def _project_user_profile_for_llm(profile: dict[str, Any]) -> dict[str, Any]:
    """Project user profile fields used as prompt hints."""
    projected = {
        key: copy.deepcopy(profile[key])
        for key in _USER_PROFILE_FIELDS
        if key in profile
    }
    projected = format_time_fields_for_llm(projected, _TIME_FIELDS)
    return projected
