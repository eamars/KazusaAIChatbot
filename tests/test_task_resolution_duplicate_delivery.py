"""Deterministic replay contracts for duplicate visible task delivery."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.action_spec.handlers.background_work import (
    enqueue_accepted_coding_task_action,
)
from kazusa_ai_chatbot.action_spec.models import ActionValidationError
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
)
from kazusa_ai_chatbot.background_work import jobs as background_jobs
from kazusa_ai_chatbot.background_work.subagent import task_orchestrator
from kazusa_ai_chatbot.background_work import result_source
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    derive_action_route,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import CognitionExecutionError
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    build_tool_result_episode,
)
from kazusa_ai_chatbot.brain_service.post_turn import (
    settle_runtime_episode_trace,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_connector
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionContractError,
    validate_task_resolution_execution_context,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.test_task_resolution_orchestrator import (
    _context,
    _goal_continuation_ref,
)
from tests.test_background_work_delivery import _accepted_task_completed_job
from tests.test_cognition_chain_connector_mapping import _global_state
from tests.test_l2d_l3_surface_handoff import _state as surface_state
from tests.test_persona_supervisor2 import _cognition_output, _persona_state


_RESULT_EVIDENCE_SUMMARY = "The typed result contains one validated source."


def _typed_task_result(status: str) -> dict[str, object]:
    """Build one valid stored result for each terminal source state."""

    context = _context()
    factual = status in {"resolved", "partial"}
    evidence = [{
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "typed-result-evidence-1",
        "task_node_id": "typed-result-node-1",
        "specialist": "public_research",
        "summary": _RESULT_EVIDENCE_SUMMARY,
        "provenance_refs": ["https://example.com/typed-result"],
        "limitations": [],
    }] if factual else []
    remaining_needs = [] if status == "resolved" else [
        "The typed result has an objective-scoped limitation.",
    ]
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one bounded public question.",
        "status": status,
        "scene_context": context["scene_context"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "evidence_state": "complete" if status == "resolved" else (
            "partial" if status == "partial" else "blocked"
        ),
        "evidence_excerpts": [
            _RESULT_EVIDENCE_SUMMARY,
        ] if factual else [],
        "evidence_handles": [
            "typed-result-evidence-1",
        ] if factual else [],
        "prompt_safe_summary": (
            "The typed result is resolved."
            if status == "resolved"
            else "The typed result reports an objective-scoped status."
        ),
        "evidence": evidence,
        "completed_subgoals": ["Resolve one bounded public question."]
        if status == "resolved" else [],
        "remaining_needs": remaining_needs,
        "checkpoint": {},
        "coding_run_context": {},
    }


@pytest.mark.parametrize(
    ("status", "surface_status", "factual"),
    [
        ("resolved", "succeeded", True),
        ("partial", "succeeded", True),
        ("needs_user_input", "blocked", False),
        ("approval_required", "blocked", False),
        ("unavailable", "failed", False),
        ("failed", "failed", False),
    ],
)
def test_typed_task_result_state_is_authoritative_across_boundaries(
    status: str,
    surface_status: str,
    factual: bool,
) -> None:
    """Every stored result state keeps facts and status ownership bounded."""

    task_result = _typed_task_result(status)
    job = _accepted_task_completed_job()
    job["task_resolution_result"] = task_result
    episode = result_source.build_result_ready_episode_from_job(job)
    cognition_source = episode["percepts"][0]["content"]["cognition_source"]

    assert cognition_source["task_status"] == status
    assert bool(cognition_source["evidence_excerpts"]) is factual
    assert bool(cognition_source["evidence_handles"]) is factual

    cognition_state = _global_state()
    cognition_state["cognitive_episode"] = episode
    cognition_input = cognition_connector.build_cognition_input_from_global_state(
        cognition_state,
    )
    current_event = cognition_input["evidence"][0]
    assert current_event["evidence_ref"]["source_kind"] == "tool_result"
    assert current_event["authority"] == (
        "current_event" if factual else "contextual_fact_only"
    )
    assert current_event["semantic_text"] == cognition_source[
        "semantic_summary"
    ]

    state = surface_state()
    state["cognitive_episode"] = episode
    state["cognition_core_output"]["goal_continuation_ref"] = (
        task_result["goal_continuation_ref"]
    )
    state["cognition_core_output"]["intention"][
        "goal_continuation_ref"
    ] = task_result["goal_continuation_ref"]
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    resolver_result = payload["resolver_result"]
    assert resolver_result["status"] == surface_status
    assert bool(resolver_result["evidence_excerpts"]) is factual
    assert bool(resolver_result["evidence_handles"]) is factual


@pytest.mark.parametrize(
    ("status", "expected_surface_role"),
    [
        ("resolved", "task_result"),
        ("partial", "task_result"),
        ("needs_user_input", "task_status"),
        ("approval_required", "task_status"),
        ("unavailable", "task_status"),
        ("failed", "task_status"),
    ],
)
def test_settlement_surface_role_uses_typed_task_result_status(
    status: str,
    expected_surface_role: str,
) -> None:
    """Settlement role selection follows the stored result status exactly."""

    task_result = _typed_task_result(status)
    episode = result_source.build_result_ready_episode_from_job({
        **_accepted_task_completed_job(),
        "task_resolution_result": task_result,
    })
    state = surface_state()
    state["cognitive_episode"] = episode
    output = state["cognition_core_output"]

    surface_role, continuation_ref = (
        cognition_connector._continuation_surface_metadata(output, state)
    )

    assert surface_role == expected_surface_role
    assert continuation_ref == task_result["goal_continuation_ref"]


def test_goal_continuation_ref_survives_tool_result_episode() -> None:
    """A new tool-result episode preserves the original goal lineage."""

    continuation_ref = _goal_continuation_ref()
    target_scope = {
        "platform": "debug",
        "platform_channel_id": "debug:user:test-user",
        "channel_type": "private",
        "current_platform_user_id": "debug-user-1",
        "current_global_user_id": "user-1",
        "current_display_name": "Test User",
        "target_addressed_user_ids": ["user-1"],
        "target_broadcast": False,
    }
    episode = build_tool_result_episode(
        result={
            "schema_version": "tool_result_ready.v1",
            "task_id": "task-result-001",
            "task_kind": "accepted_task",
            "semantic_summary": "The bounded task result is ready.",
            "artifact_text": "",
            "failure_text": "",
            "completed_at": "2026-08-14T00:01:00+00:00",
            "target_scope": target_scope,
            "evidence_refs": [],
            "result_ref": "task-result-001",
            "source_message_id": "message-1",
            "goal_continuation_ref": continuation_ref,
        },
        evidence_refs=[],
        local_time_context=build_turn_clock(
            "2026-08-14 12:01:00"
        )["local_time_context"],
        created_at="2026-08-14T00:01:00+00:00",
    )

    assert episode["episode_id"] == "tool-result:task-result-001"
    assert episode["origin_metadata"]["goal_continuation_ref"] == (
        continuation_ref
    )
    assert episode["percepts"][0]["content"]["goal_continuation_ref"] == (
        continuation_ref
    )
    assert continuation_ref["source_episode_id"] == "episode-task-001"


def test_mixed_pending_resolver_and_factual_surface_fails_contract() -> None:
    """A same-reference pending resolver cannot derive factual speech."""

    continuation_ref = build_goal_continuation_ref(
        source_episode_id="mixed-state-episode",
        source_message_id="mixed-state-message",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "mixed-state-goal",
        },
    )

    with pytest.raises(CognitionExecutionError, match="same goal"):
        derive_action_route(
            episode={"output_mode": "visible_reply"},
            primary_bid={"branch_id": "ordinary_response"},
            action_requests=[],
            resolver_requests=[{
                "capability": "task_resolution_request",
                "goal_continuation_ref": continuation_ref,
            }],
            goal_resolution="answerable_now",
            goal_continuation_ref=continuation_ref,
        )


def test_task_context_requires_scene_and_continuation_ref() -> None:
    """Task execution context fails closed when either shared field is absent."""

    context = _context()
    context.pop("scene_context")
    with pytest.raises(TaskResolutionContractError, match="scene_context"):
        validate_task_resolution_execution_context(context)


def _materialized_coding_action_spec() -> dict[str, object]:
    """Build one producer-owned accepted coding action for boundary tests."""

    state = _persona_state()
    state["action_selection_context"] = {
        "coding_runs": [{
            "coding_run_ref": "coding_run:run-1",
            "allowed_next_actions": ["revise_proposal"],
        }],
    }
    output = _cognition_output("speech")
    output["action_requests"] = [{
        "action_kind": ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
        "decision": "revise_proposal",
        "semantic_goal": "Revise the accepted coding proposal.",
        "reason": "The user requested a revision to the existing run.",
        "context_ref": "coding_run:run-1",
        "target_roles": [],
        "evidence_handles": [],
    }]
    action_specs = cognition_connector._materialize_v2_action_requests(
        output,
        state,
    )
    return action_specs[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["surface_role", "ref_mismatch"])
async def test_invalid_coding_metadata_fails_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Reject invalid continuation metadata before accepted-task creation."""

    action_spec = deepcopy(_materialized_coding_action_spec())
    if mutation == "surface_role":
        action_spec["surface_role"] = "factual_answer"
    else:
        action_spec["goal_continuation_ref"] = build_goal_continuation_ref(
            source_episode_id="mismatched-episode",
            source_message_id="mismatched-message",
            branch_id="ordinary_response",
            goal_ref={
                "scope": "user",
                "kind": "goal",
                "entity_id": "mismatched-goal",
            },
        )

    async def must_not_create_task(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("accepted-task persistence was reached")

    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle.create_or_return_active_accepted_task",
        must_not_create_task,
    )

    with pytest.raises(ActionValidationError):
        await enqueue_accepted_coding_task_action(
            action_spec,
            storage_timestamp_utc="2026-07-14T00:00:00+00:00",
            action_attempt_id="action_attempt:invalid-coding-metadata",
            source_llm_trace_id="llmtrace-invalid-coding-metadata",
        )


@pytest.mark.asyncio
async def test_v2_coding_producer_persists_context_through_queue_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound coding action keeps its ref and typed context end to end."""

    coding_spec = _materialized_coding_action_spec()
    continuation_ref = coding_spec["goal_continuation_ref"]
    execution_context = coding_spec["params"]["task_execution_context"]

    assert coding_spec["kind"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
    assert coding_spec["surface_role"] == "task_acknowledgement"
    assert coding_spec["goal_continuation_ref"] == continuation_ref
    assert execution_context["goal_continuation_ref"] == continuation_ref
    validate_task_resolution_execution_context(execution_context)

    accepted_task = {
        "accepted_task_id": "task-coding-continuation-001",
        "task_identity_key": "accepted-task-coding-identity-001",
        "accepted_task_summary": "Revise the accepted coding proposal.",
    }
    stored_jobs: list[dict[str, object]] = []

    async def create_accepted_task(
        _request: dict[str, object],
    ) -> dict[str, object]:
        return {"status": "created", "task": accepted_task}

    async def mark_pending(
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ) -> dict[str, object]:
        del updated_at
        assert accepted_task_id == accepted_task["accepted_task_id"]
        assert executor_ref == "job-coding-continuation-001"
        return {**accepted_task, "state": "pending"}

    async def insert_job(job: dict[str, object]) -> dict[str, object]:
        stored_jobs.append(job)
        return job

    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle.create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle.mark_accepted_task_pending",
        mark_pending,
    )
    monkeypatch.setattr(
        background_jobs,
        "insert_background_work_job",
        insert_job,
    )

    queue_result = await enqueue_accepted_coding_task_action(
        coding_spec,
        storage_timestamp_utc="2026-07-14T00:00:00+00:00",
        action_attempt_id="action_attempt:coding-continuation-001",
        source_llm_trace_id="llmtrace-coding-continuation-001",
    )

    assert queue_result["job_id"] == "job-coding-continuation-001"
    assert len(stored_jobs) == 1
    job = stored_jobs[0]
    assert job["goal_continuation_ref"] == continuation_ref
    assert job["task_execution_context"] == execution_context
    assert job["worker_payload"]["operation"] == (
        "continue_bound_coding_run"
    )

    observed_context: dict[str, object] = {}
    worker_result = _typed_task_result("resolved")
    worker_result["semantic_objective"] = job["semantic_objective"]
    worker_result["scene_context"] = execution_context["scene_context"]
    worker_result["goal_continuation_ref"] = continuation_ref

    async def continue_coding_run(
        _payload: dict[str, object],
        *,
        semantic_objective: str,
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        observed_context.update(execution_context)
        assert semantic_objective == job["semantic_objective"]
        return worker_result

    monkeypatch.setattr(
        task_orchestrator,
        "_continue_bound_coding_run",
        continue_coding_run,
    )
    result = await task_orchestrator.execute_task_orchestrator_job(
        job,
        lease_owner="test-lease-owner",
    )

    assert result == worker_result
    assert observed_context == execution_context


@pytest.mark.asyncio
async def test_direct_answerable_fallback_settles_factual_surface() -> None:
    """Direct answerable cognition preserves its typed visible surface."""

    continuation_ref = build_goal_continuation_ref(
        source_episode_id="direct-answer-episode",
        source_message_id="direct-answer-message",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "direct-answer-goal",
        },
    )

    trace = await settle_runtime_episode_trace(
        episode=canonical_episode(episode_id="direct-answer-episode"),
        graph_result={
            "cognition_core_output": {
                "goal_resolution": "answerable_now",
                "goal_continuation_ref": continuation_ref,
            },
            "action_specs": [],
            "action_results": [],
        },
        response_dialog=["The grounded direct answer."],
        delivery_tracking_id="delivery-direct-answer",
        settled_at="2026-08-14T00:02:00+00:00",
    )

    assert len(trace["surface_outputs"]) == 1
    surface = trace["surface_outputs"][0]
    assert surface["surface_role"] == "factual_answer"
    assert surface["goal_continuation_ref"] == continuation_ref

    context = _context()
    context.pop("goal_continuation_ref")
    with pytest.raises(
        TaskResolutionContractError,
        match="goal_continuation_ref",
    ):
        validate_task_resolution_execution_context(context)
