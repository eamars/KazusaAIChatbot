"""Tests for character-selected memory lifecycle action handling."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.action_spec.handlers.memory_lifecycle import (
    build_user_memory_lifecycle_update,
    execute_user_memory_lifecycle_action,
    map_lifecycle_decision_to_status,
    validate_memory_lifecycle_action,
)
from kazusa_ai_chatbot.action_spec.models import ActionValidationError


def _memory_target() -> dict:
    return {
        "schema_version": "action_target.v1",
        "target_kind": "memory_unit",
        "target_id": "promise-001",
        "owner": "user_memory_units",
        "scope": {"unit_type": "active_commitment"},
    }


def _action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "memory_lifecycle_update",
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "memory_unit",
                "ref_id": "promise-001",
                "owner": "user_memory_units",
                "relationship": "target",
                "evidence_refs": [],
            }
        ],
        "target": _memory_target(),
        "params": {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": "promise-001",
            "lifecycle_decision": "abandoned",
            "due_at": "2026-05-07T00:00:00+00:00",
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The character decided continuing this promise is no longer appropriate.",
    }


def test_lifecycle_status_mapping_matches_existing_user_memory_statuses() -> None:
    """Semantic lifecycle decisions should map to collection-native statuses."""

    assert map_lifecycle_decision_to_status("fulfilled") == "completed"
    assert map_lifecycle_decision_to_status("abandoned") == "cancelled"
    assert map_lifecycle_decision_to_status("obsolete") == "archived"
    assert map_lifecycle_decision_to_status("deferred") == "active"


def test_memory_lifecycle_update_builds_narrow_repository_call() -> None:
    """The handler should produce only the approved user-memory update fields."""

    action_spec = _action_spec()

    update = build_user_memory_lifecycle_update(
        action_spec,
        timestamp="2026-05-16T00:00:00+00:00",
        action_attempt_id="attempt-001",
    )

    assert update == {
        "unit_id": "promise-001",
        "status": "cancelled",
        "timestamp": "2026-05-16T00:00:00+00:00",
        "reason": action_spec["reason"],
        "action_attempt_id": "attempt-001",
        "due_at": "2026-05-07T00:00:00+00:00",
    }


def test_memory_lifecycle_rejects_wrong_target_owner() -> None:
    """Lifecycle updates must stay inside the user-memory repository owner."""

    action_spec = _action_spec()
    action_spec["target"] = _memory_target()
    action_spec["target"]["owner"] = "memory"

    with pytest.raises(ActionValidationError, match="user_memory_units"):
        validate_memory_lifecycle_action(action_spec)


def test_memory_lifecycle_rejects_missing_memory_source_ref() -> None:
    """Lifecycle actions need a matching memory-unit source reference."""

    action_spec = _action_spec()
    action_spec["source_refs"] = [
        {
            "schema_version": "action_source_ref.v1",
            "ref_kind": "cognitive_episode",
            "ref_id": "episode-001",
            "owner": "cognition",
            "relationship": "basis",
            "evidence_refs": [],
        }
    ]

    with pytest.raises(ActionValidationError, match="source_refs"):
        validate_memory_lifecycle_action(action_spec)


def test_memory_lifecycle_rejects_evolving_memory_doc_targets() -> None:
    """EvolvingMemoryDoc lifecycle mutation is outside this plan."""

    action_spec = _action_spec()
    action_spec["target"] = {
        "schema_version": "action_target.v1",
        "target_kind": "memory_unit",
        "target_id": "memory-001",
        "owner": "memory",
        "scope": {"memory_doc_type": "EvolvingMemoryDoc"},
    }
    action_spec["params"] = {
        "memory_kind": "EvolvingMemoryDoc",
        "unit_type": "active_commitment",
        "unit_id": "memory-001",
        "lifecycle_decision": "obsolete",
        "due_at": None,
    }

    with pytest.raises(ActionValidationError, match="EvolvingMemoryDoc"):
        validate_memory_lifecycle_action(action_spec)


def test_deferred_lifecycle_decision_keeps_commitment_active() -> None:
    """A deferred decision must not suppress later retrieval of the promise."""

    action_spec = _action_spec()
    action_spec["params"]["lifecycle_decision"] = "deferred"

    update = build_user_memory_lifecycle_update(
        action_spec,
        timestamp="2026-05-16T00:00:00+00:00",
        action_attempt_id="attempt-002",
    )

    assert update["status"] == "active"


@pytest.mark.asyncio
async def test_memory_lifecycle_execute_uses_repository_owner(monkeypatch) -> None:
    """Execution should delegate to the user-memory lifecycle repository."""

    captured = {}

    async def _fake_update(unit_id, **kwargs):
        captured["unit_id"] = unit_id
        captured.update(kwargs)
        return {
            "unit_id": unit_id,
            "status": kwargs["status"],
            "matched_count": 1,
            "modified_count": 1,
            "merge_history_entry": {
                "operation": "lifecycle_update",
                "action_attempt_id": kwargs["action_attempt_id"],
            },
        }

    monkeypatch.setattr(
        "kazusa_ai_chatbot.action_spec.handlers.memory_lifecycle."
        "update_user_memory_unit_lifecycle",
        _fake_update,
    )

    result = await execute_user_memory_lifecycle_action(
        _action_spec(),
        timestamp="2026-05-16T00:00:00+00:00",
        action_attempt_id="attempt-003",
    )

    assert result["status"] == "executed"
    assert captured["unit_id"] == "promise-001"
    assert captured["status"] == "cancelled"
    assert captured["reason"] == _action_spec()["reason"]
