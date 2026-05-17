"""Tests for action result and episode trace helpers."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.action_spec import execution as execution_module
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import (
    build_action_result,
    build_episode_trace,
    build_private_surface_output,
    build_text_surface_output,
    has_consolidatable_output,
    project_episode_trace_for_consolidation,
)
from kazusa_ai_chatbot.db import DatabaseOperationError


def _speak_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "episode-001",
                "owner": "cognition_episode",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {
                "decision": "visible_reply",
                "detail": "brief response",
            },
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
        "reason": "The character selected a visible text surface.",
    }


def _memory_lifecycle_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "memory_unit",
                "ref_id": "memory-unit-001",
                "owner": "user_memory_units",
                "relationship": "target",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": "memory-unit-001",
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": "memory-unit-001",
            "lifecycle_decision": "abandoned",
            "due_at": "2026-05-16T00:00:00+00:00",
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
        "reason": "The stale commitment should be abandoned.",
    }


def test_action_result_uses_evaluator_identity_without_raw_params() -> None:
    """Action results should be traceable without exposing action params."""

    action_spec = _speak_action_spec()
    eval_result = ActionSpecEvaluator().evaluate(action_spec)

    result = build_action_result(
        action_spec,
        eval_result,
        status="executed",
        result_summary="Text surface rendered.",
        completed_at="2026-05-16T00:00:00+00:00",
    )

    assert result["action_attempt_id"].startswith("action_attempt:")
    assert result["handler_owner"] == "l3_text"
    assert result["action_kind"] == "speak"
    assert result["status"] == "executed"
    assert "params" not in result
    assert "handler_id" not in result


def test_episode_trace_projection_omits_handler_ids_and_raw_params() -> None:
    """Consolidator projection should be prompt-safe action evidence."""

    action_spec = _speak_action_spec()
    eval_result = ActionSpecEvaluator().evaluate(action_spec)
    action_result = build_action_result(
        action_spec,
        eval_result,
        status="executed",
        result_summary="Text surface rendered.",
    )
    surface_output = build_text_surface_output(
        fragments=["hello"],
        created_at="2026-05-16T00:00:00+00:00",
        action_attempt_id=action_result["action_attempt_id"],
    )
    trace = build_episode_trace(
        episode_id="episode-001",
        trigger_source="user_message",
        created_at="2026-05-16T00:00:00+00:00",
        action_specs=[action_spec],
        action_results=[action_result],
        surface_outputs=[surface_output],
    )

    projection = project_episode_trace_for_consolidation(trace)
    serialized = json.dumps(projection, ensure_ascii=False)

    assert projection["action_results"][0]["action_kind"] == "speak"
    assert projection["surface_outputs"][0]["fragments"] == ["hello"]
    assert "handler_id" not in serialized
    assert "dispatcher.send_message" not in serialized
    assert "params" not in serialized
    assert "target_channel" not in serialized
    assert "mongodb" not in serialized.lower()


def test_private_surface_and_action_results_are_consolidatable() -> None:
    """Private or action-only episodes should not require visible dialog."""

    private_surface = build_private_surface_output(
        summary="Private finalization only.",
        created_at="2026-05-16T00:00:00+00:00",
    )

    assert has_consolidatable_output({"final_dialog": []}) is False
    assert has_consolidatable_output({
        "final_dialog": [],
        "surface_outputs": [private_surface],
    }) is True
    assert has_consolidatable_output({
        "final_dialog": [],
        "action_results": [{"status": "validated"}],
    }) is True


@pytest.mark.asyncio
async def test_action_execution_rejects_malformed_spec_without_crashing() -> None:
    """Rejected action rows should be returned even for malformed input."""

    results = await execution_module.execute_action_specs_for_trace(
        [{"kind": "speak"}],
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
    )

    assert len(results) == 1
    assert results[0]["status"] == "rejected"
    assert results[0]["action_kind"] == "speak"
    assert "schema_version" in results[0]["result_summary"]


@pytest.mark.asyncio
async def test_action_execution_records_attempt_ledger() -> None:
    """Execution should persist validation/execution status in the attempt ledger."""

    recorded_attempts: list[dict] = []

    async def record_attempt(record: dict) -> None:
        recorded_attempts.append(record)

    results = await execution_module.execute_action_specs_for_trace(
        [_speak_action_spec()],
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
        record_attempt_func=record_attempt,
    )

    assert results[0]["status"] == "rejected"
    assert recorded_attempts[0]["action_kind"] == "speak"
    assert recorded_attempts[0]["validation_status"] == "accepted"
    assert recorded_attempts[0]["execution_result"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_memory_lifecycle_execution_failure_returns_failed_result(
    monkeypatch,
) -> None:
    """DB lifecycle errors should become failed action results, not crashes."""

    async def fail_lifecycle_action(
        action_spec: dict,
        *,
        storage_timestamp_utc: str,
        action_attempt_id: str,
    ) -> dict:
        del action_spec, storage_timestamp_utc, action_attempt_id
        raise DatabaseOperationError("memory update unavailable")

    monkeypatch.setattr(
        execution_module,
        "execute_user_memory_lifecycle_action",
        fail_lifecycle_action,
    )

    results = await execution_module.execute_action_specs_for_trace(
        [_memory_lifecycle_action_spec()],
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
    )

    assert results[0]["status"] == "failed"
    assert results[0]["action_kind"] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert "memory update unavailable" in results[0]["result_summary"]


@pytest.mark.asyncio
async def test_memory_lifecycle_route_intent_is_not_executed_directly(
    monkeypatch,
) -> None:
    """Execution should not treat the L2d route as a DB lifecycle update."""

    executed = False

    async def fail_if_executed(
        action_spec: dict,
        *,
        storage_timestamp_utc: str,
        action_attempt_id: str,
    ) -> dict:
        nonlocal executed
        del action_spec, storage_timestamp_utc, action_attempt_id
        executed = True
        return {}

    monkeypatch.setattr(
        execution_module,
        "execute_user_memory_lifecycle_action",
        fail_if_executed,
    )
    route_spec = _memory_lifecycle_action_spec()
    route_spec["kind"] = MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    route_spec["source_refs"] = [
        {
            "schema_version": "action_source_ref.v1",
            "ref_kind": "cognitive_episode",
            "ref_id": "current_cognitive_episode",
            "owner": "cognition_episode",
            "relationship": "basis",
            "evidence_refs": [],
        }
    ]
    route_spec["target"] = {
        "schema_version": "action_target.v1",
        "target_kind": "cognitive_episode",
        "target_id": None,
        "owner": "memory_lifecycle_specialist",
        "scope": {"unit_type": "active_commitment"},
    }
    route_spec["params"] = {
        "review_kind": "active_commitment_lifecycle",
        "detail": "Review active commitments.",
    }

    results = await execution_module.execute_action_specs_for_trace(
        [route_spec],
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
    )

    assert executed is False
    assert results[0]["action_kind"] == MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert results[0]["status"] != "executed"
