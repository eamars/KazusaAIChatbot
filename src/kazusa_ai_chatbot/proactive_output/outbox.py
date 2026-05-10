"""Record builders and fake-transport boundary for proactive output."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_episode import (
    OutputMode,
    TriggerSource,
    Visibility,
)
from kazusa_ai_chatbot.dispatcher.adapter_iface import MessagingAdapter
from kazusa_ai_chatbot.proactive_output.contracts import (
    ProactiveOutboxRecord,
    ProactiveOutboxStateError,
    ProactiveOutboxStatus,
    ProactivePermissionRecord,
    ProactivePreviewRecord,
)

__all__ = [
    "build_proactive_outbox_record",
    "build_proactive_preview_record",
    "mark_proactive_outbox_denied",
    "send_ready_proactive_outbox",
]


def build_proactive_preview_record(
    *,
    preview_id: str,
    episode_id: str,
    trigger_source: TriggerSource,
    output_mode: OutputMode,
    visibility: Visibility,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    target_global_user_id: str,
    target_platform_user_id: str,
    preview_text: str,
    idempotency_key: str,
    created_at: str,
    audit_reason: str,
) -> ProactivePreviewRecord:
    """Build a proactive preview record from approved candidate text.

    Args:
        preview_id: Stable preview identifier.
        episode_id: Source cognitive episode identifier.
        trigger_source: Source that produced the candidate preview.
        output_mode: Output mode requested by the source episode.
        visibility: Visibility class of the preview content.
        platform: Target platform key.
        platform_channel_id: Target channel or private thread id.
        channel_type: Target channel class.
        target_global_user_id: Internal target user id.
        target_platform_user_id: Platform-native target user id.
        preview_text: Candidate public text.
        idempotency_key: Duplicate-suppression key for the target preview.
        created_at: Preview creation timestamp.
        audit_reason: Human-readable reason this preview exists.

    Returns:
        Proactive preview record.
    """

    preview: ProactivePreviewRecord = {
        "preview_id": preview_id,
        "episode_id": episode_id,
        "trigger_source": trigger_source,
        "output_mode": output_mode,
        "visibility": visibility,
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "target_global_user_id": target_global_user_id,
        "target_platform_user_id": target_platform_user_id,
        "preview_text": preview_text,
        "idempotency_key": idempotency_key,
        "created_at": created_at,
        "audit_reason": audit_reason,
    }
    return preview


def build_proactive_outbox_record(
    *,
    outbox_id: str,
    preview: ProactivePreviewRecord,
    permission: ProactivePermissionRecord,
    status: ProactiveOutboxStatus,
    created_at: str,
) -> ProactiveOutboxRecord:
    """Build an auditable outbox record for a proactive preview.

    Args:
        outbox_id: Stable outbox identifier.
        preview: Approved preview accepted into the outbox.
        permission: Explicit permission record used for this outbox item.
        status: Initial outbox status, limited to ``dry_run`` or ``ready``.
        created_at: Outbox creation timestamp.

    Returns:
        Proactive outbox record.

    Raises:
        ProactiveOutboxStateError: If the initial status is not allowed.
    """

    if status not in ("dry_run", "ready"):
        raise ProactiveOutboxStateError(
            f"initial proactive outbox status is not allowed: {status}"
        )
    target_matches = (
        preview["platform"] == permission["platform"]
        and preview["platform_channel_id"] == permission["platform_channel_id"]
        and preview["channel_type"] == permission["channel_type"]
        and preview["target_global_user_id"] == permission["target_global_user_id"]
        and preview["target_platform_user_id"] == permission["target_platform_user_id"]
    )
    if not target_matches:
        raise ProactiveOutboxStateError(
            "proactive outbox permission target does not match preview"
        )

    outbox: ProactiveOutboxRecord = {
        "outbox_id": outbox_id,
        "preview_id": preview["preview_id"],
        "permission_id": permission["permission_id"],
        "idempotency_key": preview["idempotency_key"],
        "platform": preview["platform"],
        "platform_channel_id": preview["platform_channel_id"],
        "channel_type": preview["channel_type"],
        "target_global_user_id": preview["target_global_user_id"],
        "target_platform_user_id": preview["target_platform_user_id"],
        "preview_text": preview["preview_text"],
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "transport_attempt_count": 0,
        "last_failure_reason": "",
        "sent_at": "",
        "platform_message_id": "",
        "delivery_adapter": "",
        "origin_kind": "proactive_preview",
    }
    return outbox


def mark_proactive_outbox_denied(
    *,
    outbox: ProactiveOutboxRecord,
    reason: str,
    updated_at: str,
) -> ProactiveOutboxRecord:
    """Copy an outbox record into a denied audit state.

    Args:
        outbox: Existing proactive outbox record.
        reason: Stable policy denial reason.
        updated_at: Timestamp for the state transition.

    Returns:
        Copied outbox record marked denied.
    """

    updated_outbox = outbox.copy()
    updated_outbox["status"] = "denied"
    updated_outbox["updated_at"] = updated_at
    updated_outbox["last_failure_reason"] = reason
    return updated_outbox


async def send_ready_proactive_outbox(
    *,
    outbox: ProactiveOutboxRecord,
    adapter: MessagingAdapter,
) -> ProactiveOutboxRecord:
    """Send a ready outbox record through the supplied messaging adapter.

    Args:
        outbox: Proactive outbox record to send.
        adapter: Messaging adapter used by the test-only transport boundary.

    Returns:
        Copied outbox record marked sent with transport metadata.

    Raises:
        ProactiveOutboxStateError: If the outbox is not in ``ready`` state.
    """

    if outbox["status"] != "ready":
        raise ProactiveOutboxStateError(
            f'proactive outbox is not ready: {outbox["status"]}'
        )
    if adapter.platform != outbox["platform"]:
        raise ProactiveOutboxStateError(
            "proactive outbox adapter platform does not match target platform"
        )

    send_result = await adapter.send_message(
        channel_id=outbox["platform_channel_id"],
        text=outbox["preview_text"],
        channel_type=outbox["channel_type"],
        reply_to_msg_id=None,
    )
    sent_at = send_result.sent_at.isoformat()

    updated_outbox = outbox.copy()
    updated_outbox["status"] = "sent"
    updated_outbox["transport_attempt_count"] = (
        outbox["transport_attempt_count"] + 1
    )
    updated_outbox["sent_at"] = sent_at
    updated_outbox["platform_message_id"] = send_result.message_id
    updated_outbox["delivery_adapter"] = send_result.platform
    updated_outbox["origin_kind"] = "proactive_sent"
    updated_outbox["updated_at"] = sent_at
    return updated_outbox
