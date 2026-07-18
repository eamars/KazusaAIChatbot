"""Tests for action result and episode trace helpers."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.action_spec import execution as execution_module
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import (
    build_action_result,
    build_private_surface_output,
    build_text_surface_output,
    has_consolidatable_output,
    project_episode_trace_for_consolidation,
)
from kazusa_ai_chatbot.brain_service.post_turn import settle_episode_trace
from kazusa_ai_chatbot.cognition_episode import build_user_message_episode
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


def _background_work_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": BACKGROUND_WORK_REQUEST_CAPABILITY,
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
            "target_kind": "current_user",
            "target_id": None,
            "owner": "background_work",
            "scope": {
                "source_platform": "debug",
                "source_channel_id": "debug:user:test-user",
                "source_channel_type": "private",
                "source_message_id": "message-001",
                "source_platform_bot_id": "debug-bot-001",
                "source_character_name": "Test Character",
                "source_trigger_source": "user_message",
                "requester_global_user_id": "global-user-001",
                "requester_platform_user_id": "debug-user-001",
                "requester_display_name": "Test User",
            },
        },
        "params": {
            "task_brief": "Generate a Fibonacci function snippet.",
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 3000,
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
        "reason": "The user requested bounded async text work.",
    }


def _accepted_task_status_check_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
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
            "target_kind": "current_user",
            "target_id": None,
            "owner": "accepted_task",
            "scope": {
                "source_platform": "debug",
                "source_channel_id": "debug:user:test-user",
                "source_channel_type": "private",
                "requester_global_user_id": "global-user-001",
                "requester_platform_user_id": "debug-user-001",
            },
        },
        "params": {},
        "urgency": "now",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The user asked for progress on an accepted task.",
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


@pytest.mark.parametrize(
    ("trace_status", "surface_status"),
    [
        ("executed", "executed"),
        ("scheduled", "scheduled"),
        ("pending", "pending"),
        ("failed", "failed"),
        ("rejected", "unavailable"),
        ("validated", "unavailable"),
        ("cancelled", "unavailable"),
    ],
)
def test_v2_surface_projection_preserves_action_lifecycle_authority(
    trace_status: str,
    surface_status: str,
) -> None:
    """Only executed action results authorize an executed surface claim."""

    from kazusa_ai_chatbot.action_spec.results import (
        project_trace_action_result_v2,
    )

    result = project_trace_action_result_v2({
        "action_kind": "background_work_request",
        "status": trace_status,
        "result_summary": "Lifecycle result.",
        "target_roles": [],
    })

    assert result["status"] == surface_status


def test_background_work_job_ref_projects_prompt_safe_job_ref() -> None:
    """Queued background work should expose only prompt-safe job evidence."""

    action_spec = _background_work_action_spec()
    eval_result = {
        "ok": True,
        "action_spec": action_spec,
        "capability": None,
        "idempotency_key": "action_spec:v1:background-work-001",
        "handler_owner": "background_work",
        "errors": [],
    }

    job_ref = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "system_event",
        "evidence_id": "background_work_job:job-001",
        "owner": "background_work_job",
        "excerpt": "queued accepted task background work",
        "observed_at": "2026-05-16T00:00:00+00:00",
    }

    result = build_action_result(
        action_spec,
        eval_result,
        status="pending",
        result_summary="Background work job queued.",
        result_refs=[job_ref],
    )

    trace = _settled_trace(
        action_specs=[action_spec],
        action_results=[result],
        surface_outputs=[],
    )
    projection = project_episode_trace_for_consolidation(trace)
    serialized = json.dumps(projection, ensure_ascii=False)

    assert result["result_refs"] == [job_ref]
    assert projection["action_results"][0]["evidence_refs"] == [job_ref]
    assert "params" not in serialized
    assert "source_channel_id" not in serialized
    assert "adapter" not in serialized.lower()


def _settled_trace(
    *,
    action_specs: list[dict],
    action_results: list[dict],
    surface_outputs: list[dict],
) -> dict:
    """Settle one trace through the sole public trace owner."""

    episode = build_user_message_episode(
        episode_id="episode-001",
        origin={
            "platform": "debug",
            "platform_message_id": "message-001",
        },
        target_scope={
            "platform": "debug",
            "platform_channel_id": "debug-private-1",
            "channel_type": "private",
            "current_platform_user_id": "debug-user-001",
            "current_global_user_id": "global-user-001",
            "current_display_name": "Test User",
            "target_addressed_user_ids": ["global-user-001"],
            "target_broadcast": False,
        },
        dialog_percept={
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "message-001",
            "content": {"semantic_text": "hello"},
            "observed_at": "2026-05-16T00:00:00+00:00",
        },
        media_percepts=[],
        evidence_refs=[],
        local_time_context={
            "current_local_datetime": "2026-05-16 12:00",
            "current_local_weekday": "Saturday",
        },
        created_at="2026-05-16T00:00:00+00:00",
        debug_controls={},
    )
    visible_surface = any(
        output.get("visibility") == "user_visible"
        and output.get("delivery_intent") == "deliver_now"
        for output in surface_outputs
    )
    terminal_status = (
        "completed_visible"
        if visible_surface
        else "completed_private"
        if surface_outputs
        else "completed_action"
    )
    delivery_correlation = {
        "schema_version": "delivery_correlation.v1",
        "delivery_intent": "deliver_now" if surface_outputs else "do_not_deliver",
        "tracking_id": "delivery-001" if surface_outputs else "",
        "receipt_status": "delivered" if surface_outputs else "not_applicable",
        "receipt_ref": "receipt-001" if surface_outputs else "",
    }
    return settle_episode_trace(
        episode=episode,
        cognition_output=None,
        action_specs=action_specs,
        action_results=action_results,
        surface_outputs=surface_outputs,
        terminal_status=terminal_status,
        attempt_diagnostics=[],
        delivery_correlation=delivery_correlation,
        settled_at="2026-05-16T00:00:01+00:00",
    )


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
    trace = _settled_trace(
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

    private_trace = _settled_trace(
        action_specs=[],
        action_results=[],
        surface_outputs=[private_surface],
    )
    action_trace = _settled_trace(
        action_specs=[],
        action_results=[{
            "schema_version": "action_result.v1",
            "action_attempt_id": "action_attempt:private-001",
            "action_kind": "speak",
            "handler_owner": "l3_text",
            "status": "validated",
            "visibility": "private",
            "result_summary": "validated",
            "result_refs": [],
            "continuation": {
                "schema_version": "action_continuation.v1",
                "mode": "none",
                "episode_type": None,
                "max_depth": 0,
                "include_result_as": None,
            },
            "completed_at": None,
        }],
        surface_outputs=[],
    )

    assert has_consolidatable_output({"final_dialog": []}) is False
    assert has_consolidatable_output(private_trace) is True
    assert has_consolidatable_output(action_trace) is True


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
    assert results[0]["semantic_result_v2"] == {
        "action_kind": "speak",
        "status": "unavailable",
        "semantic_result": results[0]["result_summary"],
        "target_roles": [],
    }


@pytest.mark.asyncio
async def test_background_work_execution_projects_accepted_task_fields_only(
    monkeypatch,
) -> None:
    """Delayed-work action results should expose semantic task state only."""

    async def enqueue_accepted_task(
        action_spec: dict,
        *,
        storage_timestamp_utc: str,
        action_attempt_id: str,
        enqueue_background_work_func=None,
    ) -> dict:
        del action_spec, storage_timestamp_utc, action_attempt_id
        del enqueue_background_work_func
        return {
            "status": "pending",
            "accepted_task_state": "scheduled",
            "accepted_task_summary": "Generate a Fibonacci function snippet.",
            "acknowledgement_constraint": "promise_allowed",
            "wait_guidance": "non_numeric_wait",
            "result_summary": "Accepted task scheduled.",
        }

    monkeypatch.setattr(
        execution_module,
        "enqueue_background_work_action",
        enqueue_accepted_task,
    )

    results = await execution_module.execute_action_specs_for_trace(
        [_background_work_action_spec()],
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
    )

    result = results[0]
    assert result["status"] == "pending"
    assert result["action_kind"] == BACKGROUND_WORK_REQUEST_CAPABILITY
    assert result["accepted_task_state"] == "scheduled"
    assert result["accepted_task_summary"] == (
        "Generate a Fibonacci function snippet."
    )
    assert result["acknowledgement_constraint"] == "promise_allowed"
    assert result["wait_guidance"] == "non_numeric_wait"
    for forbidden in (
        "queue_state",
        "job_ref",
        "operational_owner",
        "worker",
        "worker_metadata",
    ):
        assert forbidden not in result


@pytest.mark.asyncio
async def test_accepted_task_status_check_execution_projects_progress_state(
    monkeypatch,
) -> None:
    """Status checks should execute as prompt-safe lifecycle lookups."""

    async def check_status(action_spec: dict) -> dict:
        del action_spec
        return {
            "status": "active",
            "task": {
                "state": "pending",
                "accepted_task_summary": "Generate a Fibonacci function snippet.",
            },
        }

    monkeypatch.setattr(
        execution_module,
        "execute_accepted_task_status_check_action",
        check_status,
    )

    results = await execution_module.execute_action_specs_for_trace(
        [_accepted_task_status_check_action_spec()],
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
    )

    result = results[0]
    assert result["status"] == "executed"
    assert result["action_kind"] == ACCEPTED_TASK_STATUS_CHECK_CAPABILITY
    assert result["accepted_task_state"] == "scheduled"
    assert result["accepted_task_summary"] == (
        "Generate a Fibonacci function snippet."
    )
    assert result["acknowledgement_constraint"] == "progress_report_allowed"
    assert result["wait_guidance"] == "non_numeric_wait"
    for forbidden in (
        "queue_state",
        "job_ref",
        "operational_owner",
        "worker",
        "worker_metadata",
    ):
        assert forbidden not in result


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
