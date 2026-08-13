"""Focused deterministic route and failure-mode tests for background handoff.

The suite exercises the boolean-controlled task-resolution route across action
planning, resolver recurrence, capability execution, and the durable promotion
boundary.  Live LLM and live database cases remain outside this deterministic
suite.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_ATTEMPT_LIMIT,
    plan_actions,
)
from kazusa_ai_chatbot.cognition_resolver import (
    capabilities as capabilities_module,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_OBSERVATION_VERSION,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.task_resolution import TaskResolutionContractError
from tests.test_cognition_core_v2_action_planning_bugfix import (
    _bid,
    _resolver,
)
from tests.test_cognition_resolver_loop import (
    _cognition_result,
    _resolver_request,
    _resolver_state,
    _task_result,
)
from tests.test_task_resolution_orchestrator import _context


def _background_request() -> dict:
    """Build one V1 task-resolution request with background priority."""

    request = _resolver_request()
    request["priority"] = "background"
    return request


def _deferred_result_with_evidence() -> dict[str, object]:
    """Build one deferred result carrying committed partial evidence."""

    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    request = {
        "capability": "task_resolution_request",
        "semantic_goal": "Resolve one bounded public question.",
        "reason": "The current response lacks required evidence.",
        "evidence_handles": [],
    }
    checkpoint = state.create_task_resolution_checkpoint(request, _context())
    specialist_result = {
        "schema_version": "task_specialist_result.v1",
        "specialist": "public_research",
        "status": "partial",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-bg-1",
            "task_node_id": checkpoint["active_node_id"],
            "specialist": "public_research",
            "summary": "The public source returned a grounded partial fact.",
            "provenance_refs": ["https://example.com/source"],
            "limitations": ["The scope is bounded to the current question."],
        }],
        "completed_subgoals": ["Find one public source."],
        "remaining_needs": ["Confirm the source is current."],
        "reason": "Partial evidence is ready for durable continuation.",
        "retryable": False,
    }
    updated = state.record_specialist_result(checkpoint, specialist_result)
    result = state.result_from_checkpoint(
        updated,
        status="deferred",
        prompt_safe_summary="The task needs durable continuation.",
        completed_subgoals=["Find one public source."],
        coding_run_context={},
    )
    return result


def _empty_deferred_result() -> dict[str, object]:
    """Build one deferred result with only the initial empty checkpoint."""

    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    request = {
        "capability": "task_resolution_request",
        "semantic_goal": "Resolve one bounded public question.",
        "reason": "The current response lacks required evidence.",
        "evidence_handles": [],
    }
    checkpoint = state.create_task_resolution_checkpoint(request, _context())
    result = state.result_from_checkpoint(
        checkpoint,
        status="deferred",
        prompt_safe_summary="The task needs durable continuation.",
        completed_subgoals=[],
        coding_run_context={},
    )
    return result


def _queue_result(*, job_id: str = "job-001") -> dict[str, object]:
    """Build one pending background-work queue result."""

    return {
        "status": "pending",
        "job_id": job_id,
        "job_ref": f"background_work_job:{job_id}",
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "accepted_task_summary": "Resolve one bounded public question.",
        "acknowledgement_constraint": "promise_allowed",
        "wait_guidance": "non_numeric_wait",
        "result_summary": "Accepted task continuation is durable.",
    }


def _succeeded_observation(capability_request: dict) -> dict[str, object]:
    """Build one validated succeeded resolver observation for loop tests."""

    return {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_background_acceptance",
        "capability_kind": capability_request["capability_kind"],
        "request_objective": capability_request["objective"],
        "request_reason": capability_request["reason"],
        "status": "succeeded",
        "prompt_safe_summary": (
            "The bounded task was accepted for continued work; its later "
            "result will return through the normal conversation path."
        ),
        "evidence_refs": [],
        "task_resolution_evidence_state": {
            "schema_version": "resolver_evidence_state.v1",
            "state": "pending",
            "remaining_needs": [],
        },
        "created_at_utc": "2026-08-07T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_true_direct_background_skips_inline_and_accepts_after_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Background priority creates one durable job without a specialist."""

    service = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.service",
    )
    task = {
        "accepted_task_id": "task-bg-001",
        "task_identity_key": "accepted_task:v2:bg",
        "accepted_task_summary": "Resolve one bounded public question.",
        "state": "enqueueing",
    }
    monkeypatch.setattr(
        service,
        "create_or_return_active_accepted_task",
        AsyncMock(return_value={"status": "created", "task": task}),
    )
    monkeypatch.setattr(
        service,
        "mark_accepted_task_pending",
        AsyncMock(return_value={**task, "state": "pending"}),
    )
    enqueue = AsyncMock(return_value=_queue_result(job_id="job-bg-001"))
    monkeypatch.setattr(service, "enqueue_background_work_request", enqueue)
    inline = AsyncMock()
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)

    observation = await capabilities_module.execute_resolver_capability_request(
        _background_request(),
        _resolver_state(),
    )

    assert observation["status"] == "succeeded"
    assert observation["prompt_safe_summary"] == (
        "The bounded task was accepted for continued work; its later "
        "result will return through the normal conversation path."
    )
    assert observation["evidence_refs"] == []
    assert "knowledge_projection" not in observation
    inline.assert_not_awaited()
    queued = enqueue.await_args.args[0]
    assert queued["job_id"] == "job-bg-001"
    assert queued["idempotency_key"] == "background_work:task-bg-001"
    assert queued["worker_payload"]["operation"] == "resume_task_resolution"
    assert queued["worker_payload"]["checkpoint"]["dispatch_count"] == 0
    assert enqueue.await_count == 1


@pytest.mark.asyncio
async def test_false_inline_complete_returns_validated_result_without_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """False keeps inline-first behavior and resolves without queueing."""

    inline = AsyncMock(return_value=_task_result(
        status="resolved",
        specialist="local_context",
        summary="找到一条相关证据。",
    ))
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)
    promote = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "promote_deferred_task_resolution",
        promote,
    )
    background = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "start_task_resolution_in_background",
        background,
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        _resolver_state(),
    )

    assert observation["status"] == "succeeded"
    assert observation["prompt_safe_summary"] == "找到一条相关证据。"
    assert observation["evidence_refs"][0]["owner"] == "local_context"
    inline.assert_awaited_once()
    promote.assert_not_awaited()
    background.assert_not_awaited()


@pytest.mark.asyncio
async def test_deferred_partial_evidence_projects_knowledge_in_typed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed evidence and remaining needs reach the observation."""

    deferred = _deferred_result_with_evidence()
    inline = AsyncMock(return_value=deferred)
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)
    monkeypatch.setattr(
        capabilities_module,
        "promote_deferred_task_resolution",
        AsyncMock(return_value=_queue_result()),
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        _resolver_state(),
    )

    assert observation["status"] == "succeeded"
    assert observation["prompt_safe_summary"] == (
        "The bounded task was accepted for continued work; its later "
        "result will return through the normal conversation path."
    )
    assert observation["evidence_refs"][0]["owner"] == "public_research"
    knowledge = observation["knowledge_projection"]
    projection_fields = list(knowledge)
    assert projection_fields.index(
        "knowledge_we_know_so_far",
    ) < projection_fields.index("knowledge_still_lacking")
    assert knowledge["knowledge_we_know_so_far"] == [
        "The public source returned a grounded partial fact.",
    ]
    assert knowledge["knowledge_still_lacking"] == [
        "Confirm the source is current.",
    ]
    assert knowledge["evidence_boundary_notes"] == [
        "The scope is bounded to the current question.",
    ]


@pytest.mark.asyncio
async def test_empty_evidence_deferred_handoff_invents_no_partial_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An evidence-free deferred handoff carries only the acceptance notice."""

    deferred = _empty_deferred_result()
    inline = AsyncMock(return_value=deferred)
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)
    monkeypatch.setattr(
        capabilities_module,
        "promote_deferred_task_resolution",
        AsyncMock(return_value=_queue_result()),
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        _resolver_state(),
    )

    assert observation["status"] == "succeeded"
    assert observation["prompt_safe_summary"] == (
        "The bounded task was accepted for continued work; its later "
        "result will return through the normal conversation path."
    )
    assert observation["evidence_refs"] == []
    assert "knowledge_projection" not in observation


@pytest.mark.asyncio
async def test_malformed_boolean_exhausts_bounded_regeneration_fail_closed() -> None:
    """A string boolean never selects a route and blocks the plan on cap."""

    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps({
                "action_requests": [],
                "resolver_requests": [{
                    "bid_handle": "b1",
                    "resolver_handle": "r1",
                    "semantic_goal": "resolve the bounded evidence task",
                    "reason": "the admitted motive has an evidence gap",
                    "start_in_background": "true",
                }],
                "goal_resolution": "requires_required_evidence",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-malformed-boolean",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == ACTION_PLANNING_ATTEMPT_LIMIT
    assert result["goal_resolution"] == "blocked"
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_missing_task_resolution_field_exhausts_bounded_regeneration(
) -> None:
    """A missing required field remains a bounded contract failure."""

    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps({
                "action_requests": [],
                "resolver_requests": [{
                    "bid_handle": "b1",
                    "resolver_handle": "r1",
                    "semantic_goal": "resolve the bounded evidence task",
                    "start_in_background": False,
                }],
                "goal_resolution": "requires_required_evidence",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-missing-task-field",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == ACTION_PLANNING_ATTEMPT_LIMIT
    assert result["goal_resolution"] == "blocked"
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_extra_route_field_exhausts_bounded_regeneration_fail_closed() -> None:
    """Extra route fields cannot bypass the exact task-resolution row."""

    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps({
                "action_requests": [],
                "resolver_requests": [{
                    "bid_handle": "b1",
                    "resolver_handle": "r1",
                    "semantic_goal": "resolve the bounded evidence task",
                    "reason": "the admitted motive has an evidence gap",
                    "start_in_background": True,
                    "priority": "background",
                }],
                "goal_resolution": "requires_required_evidence",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-extra-route-field",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == ACTION_PLANNING_ATTEMPT_LIMIT
    assert result["goal_resolution"] == "blocked"
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_unauthorized_task_resolution_never_accepts_background_work() -> None:
    """Authorization denial produces a blocked goal without acceptance."""

    responses = [
        {
            "action_requests": [],
            "resolver_requests": [{
                "bid_handle": "b1",
                "resolver_handle": "r1",
                "semantic_goal": "resolve the bounded evidence task",
                "reason": "the admitted motive has an evidence gap",
                "start_in_background": True,
            }],
            "goal_resolution": "requires_required_evidence",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        },
        {"decisions": {"c1": False}},
    ]
    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            nonlocal calls
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-unauthorized-background",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-unauthorized-background",
                "occurred_at": "2026-08-07T00:00:00Z",
                "semantic_summary": "the user asked for bounded research",
            },
            "semantic_text": "the user asked for bounded research",
            "visible_to": ["q:event_agency"],
            "authority": "current_event",
        }],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert result["goal_resolution"] == "blocked"
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_unavailable_inline_result_reports_failure_without_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable capability keeps a truthful failed observation."""

    inline = AsyncMock(return_value=_task_result(status="unavailable"))
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)
    promote = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "promote_deferred_task_resolution",
        promote,
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        _resolver_state(),
    )

    assert observation["status"] == "failed"
    promote.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkpoint_failure_returns_failed_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint creation failure cannot produce an acceptance statement."""

    async def fail_background(
        request: dict[str, object],
        context: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del request, context, kwargs
        raise TaskResolutionContractError("checkpoint could not be created")

    monkeypatch.setattr(
        capabilities_module,
        "start_task_resolution_in_background",
        fail_background,
    )
    inline = AsyncMock()
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)

    observation = await capabilities_module.execute_resolver_capability_request(
        _background_request(),
        _resolver_state(),
    )

    assert observation["status"] == "failed"
    assert observation["prompt_safe_summary"] == (
        "The bounded task could not complete through its available "
        "resolution path."
    )
    inline.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_failure_returns_failed_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue insertion failure reports that continuation is not durable."""

    async def fail_background(
        request: dict[str, object],
        context: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del request, context, kwargs
        raise TaskResolutionContractError(
            "task-resolution durable promotion failed"
        )

    monkeypatch.setattr(
        capabilities_module,
        "start_task_resolution_in_background",
        fail_background,
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _background_request(),
        _resolver_state(),
    )

    assert observation["status"] == "failed"
    assert observation["evidence_refs"] == []
    assert "knowledge_projection" not in observation


@pytest.mark.asyncio
async def test_duplicate_background_request_blocks_second_continuation() -> None:
    """A repeated background objective cannot create a second continuation."""

    request = _background_request()
    request["objective"] = "Resolve one bounded public question."
    executed: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        del state
        return _cognition_result(
            internal_monologue="The bounded task still needs work.",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        executed.append(capability_request)
        return _succeeded_observation(capability_request)

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(executed) == 1
    assert result["resolver_state"]["status"] == "blocked"
    assert result["resolver_state"]["terminal_reason"] == (
        "duplicate resolver capability request converted to terminal surface"
    )


@pytest.mark.asyncio
async def test_delayed_pending_reuses_existing_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending accepted task remains authoritative across a retry."""

    service = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.service",
    )
    task = {
        "accepted_task_id": "task-existing",
        "task_identity_key": "accepted_task:v2:existing",
        "accepted_task_summary": "Resolve one bounded public question.",
        "state": "pending",
        "executor_ref": "job-existing",
    }
    monkeypatch.setattr(
        service,
        "create_or_return_active_accepted_task",
        AsyncMock(return_value={"status": "already_active", "task": task}),
    )
    enqueue = AsyncMock(return_value=_queue_result(job_id="job-existing"))
    monkeypatch.setattr(service, "enqueue_background_work_request", enqueue)
    mark_pending = AsyncMock()
    monkeypatch.setattr(service, "mark_accepted_task_pending", mark_pending)

    await service.start_task_resolution_in_background(
        {
            "capability": "task_resolution_request",
            "semantic_goal": "Resolve one bounded public question.",
            "reason": "The current response lacks required evidence.",
            "evidence_handles": [],
        },
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="debug-bot",
        requester_display_name="Test User",
    )

    queued = enqueue.await_args.args[0]
    assert queued["job_id"] == "job-existing"
    assert queued["idempotency_key"] == "background_work:task-existing"
    mark_pending.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_worker_failure_reports_failed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed worker outcome never produces a success claim."""

    inline = AsyncMock(return_value=_task_result(
        status="failed",
        summary="The bounded task could not be completed.",
    ))
    monkeypatch.setattr(capabilities_module, "resolve_task_inline", inline)
    promote = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "promote_deferred_task_resolution",
        promote,
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        _resolver_state(),
    )

    assert observation["status"] == "failed"
    assert observation["prompt_safe_summary"] == (
        "The bounded task could not be completed."
    )
    promote.assert_not_awaited()
