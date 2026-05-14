"""Delivery mention contract tests."""

from __future__ import annotations

from typing import Any, get_type_hints

from kazusa_ai_chatbot.self_cognition import models, tracking


def _case_with_scope(target_scope: dict[str, Any]) -> dict[str, Any]:
    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "commitment:promise-001",
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "contact_is_socially_available",
        "target_scope": target_scope,
        "source_refs": [
            {
                "source_kind": "future_promise",
                "source_id": "promise-001",
                "due_at": "2026-05-10T00:00:00+00:00",
                "summary": "The user expected a follow-up.",
            }
        ],
        "visible_context": [],
    }
    return case


def _candidate_for_scope(
    target_scope: dict[str, Any],
    *,
    mention_target_user: bool = False,
) -> dict[str, Any] | None:
    case = _case_with_scope(target_scope)
    trigger_record = tracking.build_trigger_record(case)
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        existing_attempts=[],
    )
    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Checking in now.",
        mention_target_user=mention_target_user,
    )
    return action_candidate


def test_delivery_mention_typed_shape_is_declared() -> None:
    assert get_type_hints(models.DeliveryMention) == {
        "entity_kind": str,
        "placement": str,
        "platform_user_id": str | None,
        "global_user_id": str | None,
        "display_name": str,
        "requested_by": str,
    }


def test_group_delivery_scope_omits_mentions_without_dialog_flag() -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": "global-target-1",
            "display_name": "Target User",
        }
    )

    assert action_candidate is not None
    assert "delivery_mentions" not in action_candidate


def test_group_delivery_mention_preserves_missing_platform_user_id() -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": "global-target-1",
            "display_name": "Target User",
        },
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": None,
            "global_user_id": "global-target-1",
            "display_name": "Target User",
            "requested_by": "dialog.mention_target_user",
        }
    ]


def test_private_delivery_scope_keeps_dialog_mention_request_for_adapter_noop(
) -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "user_id": "global-target-1",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        },
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert action_candidate["delivery_mentions"] == [
        {
            "entity_kind": "user",
            "placement": "prefix",
            "platform_user_id": "qq-target",
            "global_user_id": "global-target-1",
            "display_name": "Target User",
            "requested_by": "dialog.mention_target_user",
        }
    ]


def test_group_delivery_scope_without_semantic_target_omits_mentions() -> None:
    action_candidate = _candidate_for_scope(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": None,
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        },
        mention_target_user=True,
    )

    assert action_candidate is not None
    assert "delivery_mentions" not in action_candidate
