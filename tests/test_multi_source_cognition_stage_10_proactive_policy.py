"""Deterministic tests for proactive output permission decisions."""

from __future__ import annotations

from typing import Any, cast

import pytest

from kazusa_ai_chatbot.proactive_output.contracts import (
    ProactivePermissionRecord,
    ProactivePreviewRecord,
)
from kazusa_ai_chatbot.proactive_output.policy import (
    evaluate_proactive_permission,
    is_local_time_in_quiet_hours,
)


CURRENT_TIMESTAMP = "2026-05-10T00:00:00+00:00"
CURRENT_LOCAL_TIME = "12:00"


def _permission(**overrides: Any) -> ProactivePermissionRecord:
    """Build a valid permission record with optional field overrides.

    Args:
        **overrides: Field values that replace the valid baseline record.

    Returns:
        A proactive permission record suitable for policy tests.
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
    """Build a valid preview record with optional field overrides.

    Args:
        **overrides: Field values that replace the valid baseline record.

    Returns:
        A proactive preview record suitable for policy tests.
    """

    preview = {
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
        "created_at": "2026-05-10T00:00:00+00:00",
        "audit_reason": "approved preview fixture",
    }
    preview.update(overrides)
    return_value = cast(ProactivePreviewRecord, preview)
    return return_value


def test_missing_permission_is_denied() -> None:
    decision = evaluate_proactive_permission(
        preview=_preview(),
        permission=None,
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )

    assert decision == {"allowed": False, "reason": "missing_permission"}


def test_disabled_and_expired_permissions_are_denied() -> None:
    disabled = evaluate_proactive_permission(
        preview=_preview(),
        permission=_permission(enabled=False),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )
    expired = evaluate_proactive_permission(
        preview=_preview(),
        permission=_permission(expires_at=CURRENT_TIMESTAMP),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )

    assert disabled == {"allowed": False, "reason": "permission_disabled"}
    assert expired == {"allowed": False, "reason": "permission_expired"}


def test_wrong_target_and_unapproved_trigger_are_denied() -> None:
    user_message = evaluate_proactive_permission(
        preview=_preview(trigger_source="user_message"),
        permission=_permission(allowed_trigger_sources=["user_message"]),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )
    wrong_trigger = evaluate_proactive_permission(
        preview=_preview(trigger_source="internal_thought"),
        permission=_permission(allowed_trigger_sources=["scheduled_recall"]),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )
    wrong_target = evaluate_proactive_permission(
        preview=_preview(platform_channel_id="chan-2"),
        permission=_permission(),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )

    assert user_message == {"allowed": False, "reason": "user_message_not_proactive"}
    assert wrong_trigger == {"allowed": False, "reason": "trigger_source_not_allowed"}
    assert wrong_target == {"allowed": False, "reason": "target_mismatch"}


def test_quiet_hours_denies_even_with_valid_permission() -> None:
    quiet_hours = {
        "enabled": True,
        "start_local_time": "22:00",
        "end_local_time": "07:00",
    }
    decision = evaluate_proactive_permission(
        preview=_preview(),
        permission=_permission(quiet_hours=quiet_hours),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time="23:30",
    )

    assert is_local_time_in_quiet_hours(
        current_local_time="23:30",
        quiet_hours=quiet_hours,
    ) is True
    assert is_local_time_in_quiet_hours(
        current_local_time="12:00",
        quiet_hours=quiet_hours,
    ) is False
    assert is_local_time_in_quiet_hours(
        current_local_time="08:00",
        quiet_hours={
            "enabled": True,
            "start_local_time": "08:00",
            "end_local_time": "08:00",
        },
    ) is True
    with pytest.raises(ValueError):
        is_local_time_in_quiet_hours(
            current_local_time="bad-time",
            quiet_hours=quiet_hours,
        )
    assert decision == {"allowed": False, "reason": "quiet_hours"}


def test_adapter_unavailable_and_duplicate_idempotency_are_denied() -> None:
    unavailable = evaluate_proactive_permission(
        preview=_preview(),
        permission=_permission(),
        existing_idempotency_keys=set(),
        adapter_platforms={"discord"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )
    duplicate = evaluate_proactive_permission(
        preview=_preview(),
        permission=_permission(),
        existing_idempotency_keys={"episode-1:global-user-1"},
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )

    assert unavailable == {"allowed": False, "reason": "adapter_unavailable"}
    assert duplicate == {"allowed": False, "reason": "duplicate_idempotency_key"}


def test_private_or_unsafe_preview_content_is_denied() -> None:
    unsafe_mode = evaluate_proactive_permission(
        preview=_preview(output_mode="silent"),
        permission=_permission(allowed_output_modes=["silent"]),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )
    internal_content = evaluate_proactive_permission(
        preview=_preview(visibility="internal_only"),
        permission=_permission(),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )
    empty_text = evaluate_proactive_permission(
        preview=_preview(preview_text="   "),
        permission=_permission(),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )

    assert unsafe_mode == {"allowed": False, "reason": "unsafe_output_mode"}
    assert internal_content == {"allowed": False, "reason": "content_not_public"}
    assert empty_text == {"allowed": False, "reason": "empty_preview_text"}


def test_valid_permission_allows_preview() -> None:
    decision = evaluate_proactive_permission(
        preview=_preview(),
        permission=_permission(),
        existing_idempotency_keys=set(),
        adapter_platforms={"qq"},
        current_timestamp_utc=CURRENT_TIMESTAMP,
        current_local_time=CURRENT_LOCAL_TIME,
    )

    assert decision == {"allowed": True, "reason": "allowed"}
