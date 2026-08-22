"""Tests for the future_speak background-work worker path."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    build_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)


def _cognition_state() -> dict[str, object]:
    """Build the minimal trusted state needed to materialize source scope."""

    state = {
        "storage_timestamp_utc": "2026-05-15T21:00:00+00:00",
        "decontextualized_input": (
            "The user asks the character to remind them tomorrow at 10:00 "
            "to drink water."
        ),
        "platform": "debug",
        "platform_channel_id": "debug:user:test-user",
        "channel_type": "private",
        "platform_message_id": "message-001",
        "platform_bot_id": "debug-bot-001",
        "global_user_id": "global-user-001",
        "platform_user_id": "debug-user-001",
        "user_name": "Test User",
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-global-001",
        },
        "cognitive_episode": {
            "episode_id": "episode-2026-05-15-001",
            "origin_metadata": {
                "platform_message_id": "message-001",
            },
        },
        "conversation_progress": {},
    }
    return state


def _authority_proposal() -> dict[str, object]:
    """Build one closed planner authority proposal."""

    return {
        "schema_version": "scheduled_authority_proposal.v1",
        "temporal_alignment": "aligned",
        "authorized_content_summary": "在约定时间提醒用户喝水。",
        "authorized_detail_refs": [
            {
                "evidence_handle": "e1",
                "semantic_summary": "当前对话明确约定在约定时间提醒喝水。",
                "provenance_role": "current_event",
            }
        ],
    }


def _authority() -> dict[str, object]:
    """Build one validated immutable scheduled authority."""

    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-2026-05-15-001",
        source_message_id="message-001",
        source_action_attempt_id="action_attempt:future-speak-001",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-05-15T21:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-16 10:00",
        platform="debug",
        channel_type="private",
        audience_kind="private",
        semantic_objective="Remind the user to drink water.",
        authorized_content_summary="在约定时间提醒用户喝水。",
        authorized_detail_refs=_authority_proposal()[
            "authorized_detail_refs"
        ],
    )
    return dict(authority)


def test_l2d_materializes_future_speak_as_background_action() -> None:
    """future_speak should become a private background action spec."""

    requests = [
        {
            "capability": FUTURE_SPEAK_CAPABILITY,
            "decision": "2026-05-16 10:00",
            "detail": "Remind the user to drink water.",
            "reason": "The user asked for a delayed reminder.",
            "surface_role": "ordinary",
            "goal_continuation_ref": None,
            "scheduled_authority_proposal": _authority_proposal(),
        }
    ]

    action_specs = materialize_semantic_action_requests(
        requests,
        _cognition_state(),
    )

    assert len(action_specs) == 1
    action_spec = action_specs[0]
    assert action_spec["kind"] == FUTURE_SPEAK_CAPABILITY
    assert action_spec["visibility"] == "private"
    assert action_spec["target"]["owner"] == "background_work"
    assert action_spec["params"]["trigger_at"] == "2026-05-16 10:00"
    assert action_spec["params"]["continuation_objective"] == (
        "Remind the user to drink water."
    )
    eval_result = ActionSpecEvaluator().evaluate(action_spec)
    assert eval_result["ok"] is True
    assert eval_result["handler_owner"] == "background_work"


def test_future_speak_action_spec_declares_surface_metadata() -> None:
    """The worker's direct action producer must satisfy the metadata contract."""

    from kazusa_ai_chatbot.background_work.subagent import future_speak

    action_spec = future_speak._future_cognition_action_spec(
        {
            "source_platform": "debug",
            "source_channel_id": "debug:user:test-user",
            "source_channel_type": "private",
            "source_platform_bot_id": "debug-bot-001",
            "source_character_name": "Test Character",
            "requester_global_user_id": "global-user-001",
            "source_message_id": "message-001",
        },
        trigger_at="2026-05-16 10:00",
        continuation_objective="Remind the user to drink water.",
        scheduled_authority=None,
    )

    validated = validate_action_spec(action_spec)

    assert validated["surface_role"] == "ordinary"
    assert validated["goal_continuation_ref"] is None


def test_l2d_rejects_missing_surface_metadata() -> None:
    """Semantic action producers must declare both lifecycle fields."""

    with pytest.raises(ActionValidationError, match="surface_role"):
        materialize_semantic_action_requests(
            [{
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": _authority_proposal(),
            }],
            _cognition_state(),
        )

    with pytest.raises(ActionValidationError, match="goal_continuation_ref"):
        materialize_semantic_action_requests(
            [{
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "surface_role": "ordinary",
                "scheduled_authority_proposal": _authority_proposal(),
            }],
            _cognition_state(),
        )


def test_l2d_materializes_status_check_without_handler_params() -> None:
    """Status lookup provenance stays outside the empty handler params."""

    action_specs = materialize_semantic_action_requests(
        [
            {
                "capability": ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
                "decision": "check",
                "detail": "读取当前作用域中的任务状态。",
                "reason": "用户询问已经接纳的任务状态。",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
            }
        ],
        _cognition_state(),
    )

    assert len(action_specs) == 1
    action_spec = action_specs[0]
    assert action_spec["params"] == {}
    assert action_spec["cognition_provenance"]["evidence_handles"] == []
    eval_result = ActionSpecEvaluator().evaluate(action_spec)
    assert eval_result["ok"] is True


def test_scheduled_self_cognition_cannot_reschedule_future_speak() -> None:
    """A due future-speak cognition cycle must not schedule another copy."""

    state = _cognition_state()
    state["conversation_progress"] = {
        "source": "scheduled_future_cognition",
    }
    requests = [
        {
            "capability": FUTURE_SPEAK_CAPABILITY,
            "decision": "2026-05-16 10:00",
            "detail": "Remind the user to drink water.",
            "reason": "The due future reminder is now running.",
        }
    ]

    action_specs = materialize_semantic_action_requests(requests, state)

    assert action_specs == []


@pytest.mark.asyncio
async def test_future_speak_execution_enqueues_requested_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action execution should enqueue a real background job for future_speak."""

    from kazusa_ai_chatbot.action_spec import execution as execution_module
    from kazusa_ai_chatbot.accepted_task import lifecycle

    action_spec = materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": _authority_proposal(),
            }
        ],
        _cognition_state(),
    )[0]
    queued_requests: list[dict[str, object]] = []
    materialization_events: list[str] = []
    accepted_task = {
        "accepted_task_id": "task-future-speak-001",
        "task_identity_key": "accepted-task-identity-001",
        "accepted_task_summary": "Remind the user to drink water.",
    }

    async def create_accepted_task(request: dict[str, object]) -> dict:
        assert request["task_kind"] == FUTURE_SPEAK_CAPABILITY
        assert request["semantic_objective"] == (
            "Remind the user to drink water."
        )
        assert request["accepted_task_summary"] == (
            "Remind the user to drink water."
        )
        return {
            "status": "created",
            "task": accepted_task,
        }

    async def mark_pending(
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ) -> dict:
        assert accepted_task_id == "task-future-speak-001"
        assert executor_ref == "job-future-speak-001"
        assert updated_at == "2026-05-15T21:00:00+00:00"
        materialization_events.append("accepted_task_pending")
        return {
            **accepted_task,
            "state": "pending",
        }

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        assert materialization_events == ["accepted_task_pending"]
        materialization_events.append("job_inserted")
        assert request["job_id"] == "job-future-speak-001"
        queued_requests.append(request)
        return {
            "status": "pending",
            "queue_state": "queued",
            "job_id": "job-future-speak-001",
            "job_ref": "background_work_job:job-future-speak-001",
            "task_summary": request["semantic_objective"],
            "result_summary": "Background work job queued.",
            "operational_owner": "background_work_job",
            "acknowledgement_constraint": "promise_allowed",
            "evidence_ref": {
                "schema_version": "evidence_ref.v1",
                "evidence_kind": "system_event",
                "evidence_id": "background_work_job:job-future-speak-001",
                "owner": "background_work_job",
                "excerpt": "queued background work request",
                "observed_at": "2026-05-15T21:00:00+00:00",
            },
        }

    monkeypatch.setattr(
        lifecycle,
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        lifecycle,
        "mark_accepted_task_pending",
        mark_pending,
    )

    results = await execution_module.execute_action_specs_for_trace(
        [action_spec],
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        enqueue_background_work_func=enqueue_background_work,
    )

    assert results[0]["status"] == "pending"
    assert results[0]["action_kind"] == FUTURE_SPEAK_CAPABILITY
    assert results[0]["accepted_task_state"] == "scheduled"
    assert materialization_events == ["accepted_task_pending", "job_inserted"]
    assert results[0]["accepted_task_summary"] == (
        "Remind the user to drink water."
    )
    assert len(queued_requests) == 1
    queued = queued_requests[0]
    assert queued["requested_worker"] == "future_speak"
    assert queued["worker_payload"] == {
        "trigger_at": "2026-05-16 10:00",
        "continuation_objective": "Remind the user to drink water.",
    }


@pytest.mark.asyncio
async def test_worker_tick_dispatches_requested_future_speak_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested-worker jobs should run through the worker loop without rerouting."""

    from kazusa_ai_chatbot.background_work import worker as worker_module

    fake_job = {
        "schema_version": "background_work_job.v2",
        "job_id": "job-future-speak-001",
        "accepted_task_id": "task-future-speak-001",
        "semantic_objective": "Schedule a future reminder.",
        "max_output_chars": 3000,
        "requested_worker": "future_speak",
        "worker_payload": {
            "trigger_at": "2026-05-16 10:00",
            "continuation_objective": "Remind the user to drink water.",
        },
        "source_action_attempt_id": "action_attempt:future-speak-001",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "source_message_id": "message-001",
        "created_at": "2026-05-15T21:00:00+00:00",
        "updated_at": "2026-05-15T21:00:00+00:00",
    }
    execute_mock = AsyncMock(return_value={
        "artifact_text": "Future speak scheduled.",
        "result_summary": "Future speak scheduled.",
    })
    complete_mock = AsyncMock(return_value={"status": "completed"})
    mark_running_mock = AsyncMock(return_value={"state": "running"})
    mark_delivered_mock = AsyncMock(return_value={"state": "delivered"})

    monkeypatch.setattr(
        worker_module,
        "claim_background_work_job",
        AsyncMock(side_effect=[fake_job, None]),
    )
    monkeypatch.setattr(worker_module, "_execute_future_speak_job", execute_mock)
    monkeypatch.setattr(
        worker_module,
        "mark_accepted_task_running",
        mark_running_mock,
    )
    monkeypatch.setattr(
        worker_module,
        "mark_future_speak_accepted_task_delivered",
        mark_delivered_mock,
    )
    monkeypatch.setattr(
        worker_module,
        "complete_background_work_job",
        complete_mock,
    )

    result = await worker_module.run_background_work_worker_tick(
        claim_limit=1,
        lease_seconds=60,
        max_attempts=3,
        worker_id="worker-test",
    )

    assert result["processed_count"] == 1
    assert result["succeeded_count"] == 1
    execute_mock.assert_awaited_once_with(fake_job)
    complete_kwargs = complete_mock.await_args.kwargs
    assert complete_kwargs["task_resolution_result"] is None
    assert complete_kwargs["skip_result_delivery"] is True
    mark_running_mock.assert_awaited_once()
    assert mark_running_mock.await_args.kwargs["accepted_task_id"] == (
        "task-future-speak-001"
    )
    assert mark_running_mock.await_args.kwargs["started_at"]
    mark_delivered_mock.assert_awaited_once()
    assert mark_delivered_mock.await_args.kwargs["accepted_task_id"] == (
        "task-future-speak-001"
    )


@pytest.mark.asyncio
async def test_future_speak_worker_schedules_calendar_future_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The future_speak worker should create a future-cognition schedule."""

    from kazusa_ai_chatbot.action_spec.handlers import future_cognition
    from kazusa_ai_chatbot.background_work.subagent import future_speak

    execute_future_cognition = AsyncMock(return_value={
        "status": "scheduled",
        "calendar_trigger_kind": "future_cognition",
        "calendar_schedule_id": "calendar_schedule_001",
        "calendar_run_id": "calendar_run_001",
        "scheduled_count": 1,
        "episode_type": "self_cognition",
        "trigger_at": "2026-05-15T22:00:00+00:00",
        "reason": "The user asked for a delayed reminder.",
    })
    monkeypatch.setattr(
        future_cognition,
        "execute_future_cognition_action",
        execute_future_cognition,
    )

    result = await future_speak.execute_future_speak_job({
        "source_action_attempt_id": "action_attempt:future-speak-001",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "source_message_id": "message-001",
        "created_at": "2026-05-15T21:00:00+00:00",
        "updated_at": "2026-05-15T21:00:00+00:00",
        "worker_payload": {
            "trigger_at": "2026-05-16 10:00",
            "continuation_objective": "Remind the user to drink water.",
        },
    })

    assert result["artifact_text"] == (
        "Future speak scheduled for 2026-05-15T22:00:00+00:00."
    )
    assert result["result_summary"] == "Future speak scheduled."
    scheduled_spec = execute_future_cognition.await_args.args[0]
    assert scheduled_spec["kind"] == "trigger_future_cognition"
    assert scheduled_spec["params"]["trigger_at"] == "2026-05-16 10:00"
    assert scheduled_spec["params"]["continuation_objective"] == (
        "Remind the user to drink water."
    )


@pytest.mark.asyncio
async def test_future_speak_rejects_invalid_authority_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid authority can never create durable accepted-task work."""

    from kazusa_ai_chatbot.action_spec.handlers import background_work

    proposal = _authority_proposal()
    proposal["temporal_alignment"] = "relative_date_mismatch"
    action_spec = materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": proposal,
            }
        ],
        _cognition_state(),
    )[0]

    create_calls: list[dict[str, object]] = []

    async def create_accepted_task(request: dict[str, object]) -> dict:
        create_calls.append(request)
        return {
            "status": "created",
            "task": {
                "accepted_task_id": "task-future-speak-001",
                "task_identity_key": "identity-001",
                "accepted_task_summary": "Remind the user to drink water.",
            },
        }

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        del request
        raise AssertionError("enqueue must not run for invalid authority")

    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle."
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )

    with pytest.raises(ActionValidationError, match="temporal_alignment"):
        await background_work.enqueue_future_speak_action(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-speak-001",
            enqueue_background_work_func=enqueue_background_work,
        )

    assert create_calls == []


@pytest.mark.asyncio
async def test_future_speak_copies_immutable_authority_to_carriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task, job, and schedule carriers share one byte-equivalent authority."""

    from kazusa_ai_chatbot.action_spec.handlers import background_work
    from kazusa_ai_chatbot.background_work import jobs

    authority = _authority()
    action_spec = materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": _authority_proposal(),
            }
        ],
        _cognition_state(),
    )[0]
    accepted_task = {
        "accepted_task_id": "task-future-speak-001",
        "task_identity_key": "accepted-task-identity-001",
        "accepted_task_summary": "Remind the user to drink water.",
    }
    carriers: dict[str, dict[str, object]] = {}

    async def create_accepted_task(request: dict[str, object]) -> dict:
        carriers["accepted_task_request"] = dict(request)
        return {
            "status": "created",
            "task": dict(accepted_task),
        }

    async def mark_pending(
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ) -> dict:
        del accepted_task_id, executor_ref, updated_at
        return {**accepted_task, "state": "pending"}

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        carriers["job_request"] = dict(request)
        return {
            "status": "pending",
            "job_id": request["job_id"],
            "job_ref": f"background_work_job:{request['job_id']}",
            "result_summary": "Background work job queued.",
        }

    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle."
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle."
        "mark_accepted_task_pending",
        mark_pending,
    )

    result = await background_work.enqueue_future_speak_action(
        action_spec,
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
        action_attempt_id="action_attempt:future-speak-001",
        source_llm_trace_id="llmtrace_source-1",
        enqueue_background_work_func=enqueue_background_work,
    )

    assert result["status"] == "pending"
    task_request_authority = carriers["accepted_task_request"][
        "scheduled_future_speech_authority"
    ]
    job_request_authority = carriers["job_request"][
        "scheduled_future_speech_authority"
    ]
    assert task_request_authority == authority
    assert job_request_authority == authority

    job_document = jobs._build_job_document(
        carriers["job_request"],
        job_id=str(carriers["job_request"]["job_id"]),
        storage_timestamp_utc="2026-05-15T21:00:00+00:00",
    )
    assert job_document["scheduled_future_speech_authority"] == authority

    mutated_request = dict(carriers["job_request"])
    mutated_request["accepted_task_id"] = "task-mutated"
    mutated_request["idempotency_key"] = "background_work:task-mutated"
    assert mutated_request["scheduled_future_speech_authority"] == authority


@pytest.mark.asyncio
async def test_future_speak_subagent_does_not_author_dialog_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subagent only schedules; it never authors visible speech."""

    from kazusa_ai_chatbot.action_spec.handlers import future_cognition
    from kazusa_ai_chatbot.background_work.subagent import future_speak

    authority = _authority()
    captured: dict[str, object] = {}

    async def execute_future_cognition_action(
        action_spec: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        captured["action_spec"] = action_spec
        return {
            "status": "scheduled",
            "calendar_trigger_kind": "future_cognition",
            "calendar_schedule_id": "calendar_schedule_001",
            "calendar_run_id": "calendar_run_001",
            "scheduled_count": 1,
            "episode_type": "self_cognition",
            "trigger_at": authority["trigger"]["utc"],
            "reason": "The user asked for a delayed reminder.",
        }

    monkeypatch.setattr(
        future_cognition,
        "execute_future_cognition_action",
        execute_future_cognition_action,
    )

    result = await future_speak.execute_future_speak_job({
        "source_action_attempt_id": "action_attempt:future-speak-001",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "source_message_id": "message-001",
        "created_at": "2026-05-15T21:00:00+00:00",
        "updated_at": "2026-05-15T21:00:00+00:00",
        "worker_payload": {
            "trigger_at": "2026-05-16 10:00",
            "continuation_objective": "Remind the user to drink water.",
        },
        "scheduled_future_speech_authority": authority,
    })

    assert "final_dialog" not in result
    assert "dialog" not in result
    assert "text" not in result
    scheduled_spec = captured["action_spec"]
    assert scheduled_spec["kind"] == "trigger_future_cognition"
    assert (
        scheduled_spec["params"]["scheduled_future_speech_authority"]
        == authority
    )


@pytest.mark.asyncio
async def test_future_speak_active_duplicate_rejects_authority_mismatch_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed trigger cannot silently reuse an active task identity."""

    from kazusa_ai_chatbot.action_spec.handlers import background_work
    from kazusa_ai_chatbot.accepted_task import lifecycle

    stored_authority = _authority()
    action_spec = materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 11:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": _authority_proposal(),
            }
        ],
        _cognition_state(),
    )[0]

    async def create_accepted_task(request: dict[str, object]) -> dict:
        del request
        return {
            "status": "already_active",
            "task": {
                "accepted_task_id": "task-future-speak-001",
                "task_identity_key": "accepted-task-identity-001",
                "accepted_task_summary": "Remind the user to drink water.",
                "state": "pending",
                "executor_ref": "job-future-speak-001",
                "scheduled_future_speech_authority": stored_authority,
            },
        }

    async def mark_pending(**kwargs: object) -> dict:
        del kwargs
        raise AssertionError(
            "active duplicate mismatch must fail before task mutation"
        )

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        del request
        raise AssertionError(
            "active duplicate mismatch must fail before job enqueue"
        )

    monkeypatch.setattr(
        lifecycle,
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        lifecycle,
        "mark_accepted_task_pending",
        mark_pending,
    )

    with pytest.raises(
        ActionValidationError,
        match="different scheduled authority",
    ):
        await background_work.enqueue_future_speak_action(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-speak-001",
            enqueue_background_work_func=enqueue_background_work,
        )


def _future_speak_action_spec() -> dict[str, object]:
    """Build one materialized future-speak action spec for handler tests."""

    return materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": _authority_proposal(),
            }
        ],
        _cognition_state(),
    )[0]


@pytest.mark.asyncio
async def test_future_speak_authority_uses_action_accepted_at_not_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The authority is built from the accepted event instant, not storage time."""

    from kazusa_ai_chatbot.action_spec.handlers import background_work

    action_spec = _future_speak_action_spec()
    action_spec["params"]["accepted_at_utc"] = "2026-05-15T20:30:00+00:00"
    captured: dict[str, object] = {}

    async def create_accepted_task(request: dict[str, object]) -> dict:
        captured["accepted_task_request"] = dict(request)
        return {
            "status": "created",
            "task": {
                "accepted_task_id": "task-future-speak-001",
                "task_identity_key": "identity-001",
                "accepted_task_summary": "Remind the user to drink water.",
            },
        }

    async def mark_pending(**kwargs: object) -> dict:
        del kwargs
        return {
            "accepted_task_id": "task-future-speak-001",
            "state": "pending",
        }

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        captured["job_request"] = dict(request)
        return {
            "status": "pending",
            "job_id": request["job_id"],
            "job_ref": f"background_work_job:{request['job_id']}",
            "result_summary": "Background work job queued.",
        }

    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle."
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle."
        "mark_accepted_task_pending",
        mark_pending,
    )

    await background_work.enqueue_future_speak_action(
        action_spec,
        storage_timestamp_utc="2026-05-15T22:00:00+00:00",
        action_attempt_id="action_attempt:future-speak-001",
        enqueue_background_work_func=enqueue_background_work,
    )

    authority = captured["accepted_task_request"][
        "scheduled_future_speech_authority"
    ]
    assert authority["accepted_at"]["utc"] == "2026-05-15T20:30:00Z"
    assert authority["accepted_at"]["utc"] != "2026-05-15T22:00:00Z"
    job_authority = captured["job_request"][
        "scheduled_future_speech_authority"
    ]
    assert job_authority == authority


@pytest.mark.asyncio
async def test_future_speak_source_message_param_matches_trusted_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A params source message contradicting the trusted scope fails closed."""

    from kazusa_ai_chatbot.action_spec.handlers import background_work

    action_spec = _future_speak_action_spec()
    action_spec["params"]["source_message_id"] = "tampered-message-999"
    create_calls: list[dict[str, object]] = []

    async def create_accepted_task(request: dict[str, object]) -> dict:
        create_calls.append(request)
        return {
            "status": "created",
            "task": {
                "accepted_task_id": "task-future-speak-001",
                "task_identity_key": "identity-001",
                "accepted_task_summary": "Remind the user to drink water.",
            },
        }

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        del request
        raise AssertionError(
            "scope mismatch must fail before accepted-task creation or enqueue"
        )

    monkeypatch.setattr(
        "kazusa_ai_chatbot.accepted_task.lifecycle."
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )

    with pytest.raises(ActionValidationError, match="trusted scope mismatch"):
        await background_work.enqueue_future_speak_action(
            action_spec,
            storage_timestamp_utc="2026-05-15T21:00:00+00:00",
            action_attempt_id="action_attempt:future-speak-001",
            enqueue_background_work_func=enqueue_background_work,
        )

    assert create_calls == []
