"""Tests for the generic action-attempt ledger compatibility layer."""

from __future__ import annotations

from kazusa_ai_chatbot.action_spec.attempt_ledger import (
    ACTION_ATTEMPT_LEDGER_COLLECTION,
    build_action_attempt_record,
    build_action_idempotency_key,
    read_action_attempt_compat,
)


def _source_ref() -> dict:
    return {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "memory_unit",
        "ref_id": "promise-001",
        "owner": "user_memory_units",
        "relationship": "basis",
        "evidence_refs": [],
    }


def _action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "send_message",
        "cognition_mode": "deliberative",
        "source_refs": [_source_ref()],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "dispatcher",
            "scope": {"channel_relation": "same"},
        },
        "params": {
            "target_channel": "same",
            "text": "Checking in now.",
            "execute_at": None,
            "delivery_mentions": [],
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The character selected this send.",
    }


def test_old_self_cognition_attempt_row_remains_readable() -> None:
    """Existing rows should survive the generic ledger compatibility layer."""

    old_row = {
        "attempt_id": "self_cognition_attempt:promise-001",
        "run_id": "self_cognition_run:promise-001",
        "trigger_id": "self_cognition_trigger:promise-001",
        "source_kind": "user_memory_unit",
        "source_id": "promise-001",
        "target_scope": {"platform": "qq", "platform_channel_id": "673225019"},
        "action_kind": "send_message",
        "due_at": "2026-05-07T00:00:00+00:00",
        "idempotency_key": "legacy-key",
        "status": "scheduled",
        "dispatch_status": "accepted",
        "scheduled_event_ids": ["event-001"],
        "recorded_at": "2026-05-15T00:00:00+00:00",
    }

    normalized = read_action_attempt_compat(old_row)

    assert ACTION_ATTEMPT_LEDGER_COLLECTION == "self_cognition_action_attempts"
    assert normalized["idempotency_key"] == "legacy-key"
    assert normalized["action_kind"] == "send_message"
    assert normalized["status"] == "scheduled"
    assert normalized["cognition_mode"] is None
    assert normalized["handler_owner"] is None
    assert normalized["validation_status"] == "legacy_unvalidated"


def test_action_idempotency_key_is_stable_for_semantic_identity() -> None:
    """Reordered params should not change the action idempotency key."""

    action_spec = _action_spec()
    same_action = _action_spec()
    same_action["params"] = {
        "delivery_mentions": [],
        "execute_at": None,
        "text": "Checking in now.",
        "target_channel": "same",
    }

    first_key = build_action_idempotency_key(action_spec)
    second_key = build_action_idempotency_key(same_action)

    assert first_key == second_key
    assert first_key.startswith("action_spec:v1:")


def test_new_action_attempt_record_extends_existing_collection_shape() -> None:
    """New action metadata should be additive on the existing attempt ledger."""

    action_spec = _action_spec()
    eval_result = {
        "ok": True,
        "action_spec": action_spec,
        "capability": {"capability_kind": "send_message"},
        "idempotency_key": build_action_idempotency_key(action_spec),
        "handler_owner": "dispatcher",
        "errors": [],
    }

    record = build_action_attempt_record(
        action_spec,
        eval_result,
        recorded_at="2026-05-16T00:00:00+00:00",
        execution_result={"status": "accepted", "scheduled_event_ids": ["event-001"]},
    )

    assert record["action_kind"] == "send_message"
    assert record["cognition_mode"] == "deliberative"
    assert record["validation_status"] == "accepted"
    assert record["handler_owner"] == "dispatcher"
    assert record["continuation_status"] == "none_requested"
    assert record["execution_result"]["scheduled_event_ids"] == ["event-001"]
    assert record["idempotency_key"] == eval_result["idempotency_key"]
