"""Active-turn exclusion helpers for conversation evidence."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.rag.conversation_evidence.contracts import (
    _ActiveTurnExclusionCounts,
)
from kazusa_ai_chatbot.utils import text_or_empty

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
