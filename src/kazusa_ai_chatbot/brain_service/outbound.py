"""Shared write-ahead persistence for assistant-authored outbound text."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime


EnsureCharacterIdentity = Callable[..., Awaitable[str]]
SaveConversation = Callable[[dict], Awaitable[str | None]]


class ConversationHistoryWriteError(RuntimeError):
    """Raised when outbound history persistence does not commit a row."""


def _dedupe_non_empty(values: Sequence[str]) -> list[str]:
    """Return non-empty strings in first-seen order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean_value = str(value or "").strip()
        if not clean_value or clean_value in seen:
            continue
        seen.add(clean_value)
        result.append(clean_value)
    return result


async def record_assistant_outbound_message(
    *,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    platform_bot_id: str,
    character_name: str,
    body_text: str,
    addressed_to_global_user_ids: Sequence[str],
    broadcast: bool,
    fallback_addressed_global_user_id: str = "",
    delivery_tracking_id: str = "",
    storage_timestamp_utc: str,
    ensure_character_global_identity_func: EnsureCharacterIdentity,
    save_conversation_func: SaveConversation,
) -> str:
    """Persist one assistant outbound row before it can be delivered.

    Args:
        platform: Target runtime platform.
        platform_channel_id: Target channel, group, or private thread.
        channel_type: Platform-neutral channel class.
        platform_bot_id: Platform account id of the active character.
        character_name: Display name of the active character.
        body_text: Exact text intended for user-visible delivery.
        addressed_to_global_user_ids: Explicit assistant-row addressees.
        broadcast: Whether this row is visible as a channel broadcast.
        fallback_addressed_global_user_id: User id used when a direct
            assistant row has no explicit addressee.
        delivery_tracking_id: Optional local id for delivery receipts.
        storage_timestamp_utc: Storage UTC timestamp for the persisted row.
        ensure_character_global_identity_func: Identity resolver/backfiller.
        save_conversation_func: Conversation-history persistence function.

    Returns:
        Inserted conversation-history row id.

    Raises:
        ConversationHistoryWriteError: If persistence does not return a row id.
    """

    target_addressed_user_ids = _dedupe_non_empty(
        addressed_to_global_user_ids,
    )
    fallback_user_id = str(fallback_addressed_global_user_id or "").strip()
    if not target_addressed_user_ids and not broadcast and fallback_user_id:
        target_addressed_user_ids = [fallback_user_id]

    character_global_user_id = await ensure_character_global_identity_func(
        platform=platform,
        platform_bot_id=platform_bot_id,
        character_name=character_name,
    )
    assistant_doc = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "role": "assistant",
        "platform_user_id": platform_bot_id,
        "global_user_id": character_global_user_id,
        "display_name": character_name,
        "body_text": body_text,
        "raw_wire_text": body_text,
        "content_type": "text",
        "addressed_to_global_user_ids": target_addressed_user_ids,
        "mentions": [],
        "broadcast": broadcast,
        "attachments": [],
        "timestamp": storage_timestamp_utc,
    }
    clean_tracking_id = str(delivery_tracking_id or "").strip()
    if clean_tracking_id:
        assistant_doc["delivery_tracking_id"] = clean_tracking_id
        assistant_doc["delivery_status"] = "pending"

    conversation_row_id = await save_conversation_func(assistant_doc)
    if not conversation_row_id:
        raise ConversationHistoryWriteError(
            "assistant outbound message was not committed to conversation_history"
        )
    return_value = conversation_row_id
    return return_value


def utc_timestamp(now: Callable[[], datetime]) -> str:
    """Return an ISO timestamp from an injected clock."""

    return_value = now().isoformat()
    return return_value
