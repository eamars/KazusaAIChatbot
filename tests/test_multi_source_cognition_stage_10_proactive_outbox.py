"""Deterministic tests for proactive output outbox records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pytest

from kazusa_ai_chatbot.dispatcher.adapter_iface import SendResult
from kazusa_ai_chatbot.proactive_output.contracts import (
    ProactiveOutboxRecord,
    ProactiveOutboxStateError,
    ProactiveOutboxStatus,
    ProactivePermissionRecord,
    ProactivePreviewRecord,
)
from kazusa_ai_chatbot.proactive_output.outbox import (
    build_proactive_outbox_record,
    build_proactive_preview_record,
    mark_proactive_outbox_denied,
    send_ready_proactive_outbox,
)


CREATED_AT = "2026-05-10T00:00:00+00:00"


class _FakeAdapter:
    """Capture fake sends without touching runtime adapter registration."""

    platform = "qq"

    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        """Record a fake send and return deterministic delivery metadata.

        Args:
            channel_id: Target channel identifier.
            text: Message body requested by the outbox.
            channel_type: Target channel class.
            reply_to_msg_id: Optional reply target, expected to be absent.

        Returns:
            Stable fake delivery result for assertions.
        """

        self.calls.append(
            {
                "channel_id": channel_id,
                "text": text,
                "channel_type": channel_type,
                "reply_to_msg_id": reply_to_msg_id,
            }
        )
        return_value = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id="platform-message-1",
            sent_at=datetime(2026, 5, 10, 0, 1, tzinfo=timezone.utc),
        )
        return return_value


def _permission(**overrides: Any) -> ProactivePermissionRecord:
    """Build a valid permission record with optional field overrides.

    Args:
        **overrides: Field values that replace the valid baseline record.

    Returns:
        A proactive permission record suitable for outbox tests.
    """

    permission = {
        "permission_id": "perm-1",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "private",
        "target_global_user_id": "global-user-1",
        "target_platform_user_id": "platform-user-1",
        "allowed_trigger_sources": ["scheduled_recall"],
        "allowed_output_modes": ["preview"],
        "quiet_hours": {
            "enabled": False,
            "start_local_time": "22:00",
            "end_local_time": "07:00",
        },
        "expires_at": "2026-05-11T00:00:00+00:00",
        "enabled": True,
        "created_at": "2026-05-09T00:00:00+00:00",
        "audit_reason": "explicit test fixture permission",
    }
    permission.update(overrides)
    return_value = cast(ProactivePermissionRecord, permission)
    return return_value


def _preview(**overrides: Any) -> ProactivePreviewRecord:
    """Build a valid preview record through the public preview helper.

    Args:
        **overrides: Field values that replace the valid baseline record.

    Returns:
        A proactive preview record suitable for outbox tests.
    """

    preview_fields = {
        "preview_id": "preview-1",
        "episode_id": "episode-1",
        "trigger_source": "scheduled_recall",
        "output_mode": "preview",
        "visibility": "model_visible",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "private",
        "target_global_user_id": "global-user-1",
        "target_platform_user_id": "platform-user-1",
        "preview_text": "Time to follow up.",
        "idempotency_key": "episode-1:global-user-1",
        "created_at": CREATED_AT,
        "audit_reason": "approved preview fixture",
    }
    preview_fields.update(overrides)
    preview = build_proactive_preview_record(
        preview_id=preview_fields["preview_id"],
        episode_id=preview_fields["episode_id"],
        trigger_source=preview_fields["trigger_source"],
        output_mode=preview_fields["output_mode"],
        visibility=preview_fields["visibility"],
        platform=preview_fields["platform"],
        platform_channel_id=preview_fields["platform_channel_id"],
        channel_type=preview_fields["channel_type"],
        target_global_user_id=preview_fields["target_global_user_id"],
        target_platform_user_id=preview_fields["target_platform_user_id"],
        preview_text=preview_fields["preview_text"],
        idempotency_key=preview_fields["idempotency_key"],
        created_at=preview_fields["created_at"],
        audit_reason=preview_fields["audit_reason"],
    )
    return preview


def _outbox(status: ProactiveOutboxStatus = "dry_run") -> ProactiveOutboxRecord:
    """Build a proactive outbox record for one valid preview.

    Args:
        status: Initial outbox status requested by the test.

    Returns:
        Outbox record created through the public builder.
    """

    outbox = build_proactive_outbox_record(
        outbox_id="outbox-1",
        preview=_preview(),
        permission=_permission(),
        status=status,
        created_at=CREATED_AT,
    )
    return outbox


def test_preview_record_keeps_public_text_separate_from_outbox() -> None:
    preview = _preview()
    outbox = build_proactive_outbox_record(
        outbox_id="outbox-1",
        preview=preview,
        permission=_permission(),
        status="dry_run",
        created_at=CREATED_AT,
    )

    assert preview["episode_id"] == "episode-1"
    assert outbox["preview_id"] == preview["preview_id"]
    assert outbox["preview_text"] == preview["preview_text"]
    assert outbox["origin_kind"] == "proactive_preview"
    assert "episode_id" not in outbox
    assert "trigger_source" not in outbox


def test_outbox_builder_rejects_permission_target_mismatch() -> None:
    preview = _preview()
    permission = _permission(platform_channel_id="chan-2")

    with pytest.raises(ProactiveOutboxStateError):
        build_proactive_outbox_record(
            outbox_id="outbox-1",
            preview=preview,
            permission=permission,
            status="dry_run",
            created_at=CREATED_AT,
        )


@pytest.mark.asyncio
async def test_dry_run_outbox_does_not_call_adapter() -> None:
    adapter = _FakeAdapter()

    with pytest.raises(ProactiveOutboxStateError):
        await send_ready_proactive_outbox(
            outbox=_outbox("dry_run"),
            adapter=adapter,
        )

    assert adapter.calls == []


@pytest.mark.asyncio
async def test_transport_refuses_adapter_platform_mismatch() -> None:
    adapter = _FakeAdapter()
    adapter.platform = "discord"

    with pytest.raises(ProactiveOutboxStateError):
        await send_ready_proactive_outbox(
            outbox=_outbox("ready"),
            adapter=adapter,
        )

    assert adapter.calls == []


@pytest.mark.asyncio
async def test_ready_outbox_fake_transport_marks_sent_with_audit_metadata() -> None:
    adapter = _FakeAdapter()
    outbox = _outbox("ready")

    sent = await send_ready_proactive_outbox(
        outbox=outbox,
        adapter=adapter,
    )

    assert outbox["status"] == "ready"
    assert sent is not outbox
    assert sent["status"] == "sent"
    assert sent["transport_attempt_count"] == 1
    assert sent["sent_at"] == "2026-05-10T00:01:00+00:00"
    assert sent["updated_at"] == sent["sent_at"]
    assert sent["platform_message_id"] == "platform-message-1"
    assert sent["delivery_adapter"] == "qq"
    assert sent["origin_kind"] == "proactive_sent"
    assert adapter.calls == [
        {
            "channel_id": "chan-1",
            "text": "Time to follow up.",
            "channel_type": "private",
            "reply_to_msg_id": None,
        }
    ]


@pytest.mark.asyncio
async def test_transport_refuses_dry_run_denied_or_sent_status() -> None:
    adapter = _FakeAdapter()
    dry_run = _outbox("dry_run")
    denied = dict(_outbox("ready"))
    denied["status"] = "denied"
    sent = dict(_outbox("ready"))
    sent["status"] = "sent"

    with pytest.raises(ProactiveOutboxStateError):
        await send_ready_proactive_outbox(outbox=dry_run, adapter=adapter)
    with pytest.raises(ProactiveOutboxStateError):
        await send_ready_proactive_outbox(
            outbox=cast(ProactiveOutboxRecord, denied),
            adapter=adapter,
        )
    with pytest.raises(ProactiveOutboxStateError):
        await send_ready_proactive_outbox(
            outbox=cast(ProactiveOutboxRecord, sent),
            adapter=adapter,
        )

    assert adapter.calls == []


def test_denied_outbox_records_failure_reason_without_transport() -> None:
    outbox = _outbox("dry_run")
    denied = mark_proactive_outbox_denied(
        outbox=outbox,
        reason="quiet_hours",
        updated_at="2026-05-10T00:02:00+00:00",
    )

    assert outbox["status"] == "dry_run"
    assert denied is not outbox
    assert denied["status"] == "denied"
    assert denied["last_failure_reason"] == "quiet_hours"
    assert denied["updated_at"] == "2026-05-10T00:02:00+00:00"
    assert denied["transport_attempt_count"] == 0
