"""Tests for the future_speak background-work worker path."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
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
        "conversation_progress": {},
    }
    return state


def test_l2d_materializes_future_speak_as_background_action() -> None:
    """future_speak should become a private background action spec."""

    requests = [
        {
            "capability": FUTURE_SPEAK_CAPABILITY,
            "decision": "2026-05-16 10:00",
            "detail": "Remind the user to drink water.",
            "reason": "The user asked for a delayed reminder.",
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


def test_l2d_materializes_status_check_without_handler_params() -> None:
    """Status lookup provenance stays outside the empty handler params."""

    action_specs = materialize_semantic_action_requests(
        [
            {
                "capability": ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
                "decision": "check",
                "detail": "读取当前作用域中的任务状态。",
                "reason": "用户询问已经接纳的任务状态。",
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
    from kazusa_ai_chatbot.action_spec.handlers import (
        background_work as background_work_handler,
    )

    action_spec = materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-16 10:00",
                "detail": "Remind the user to drink water.",
                "reason": "The user asked for a delayed reminder.",
            }
        ],
        _cognition_state(),
    )[0]
    queued_requests: list[dict[str, object]] = []
    accepted_task = {
        "accepted_task_id": "task-future-speak-001",
        "task_identity_key": "accepted-task-identity-001",
        "accepted_task_summary": "Remind the user to drink water.",
    }

    async def create_accepted_task(request: dict[str, object]) -> dict:
        assert request["action_kind"] == FUTURE_SPEAK_CAPABILITY
        assert request["accepted_task_seed"] == (
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
        return {
            **accepted_task,
            "state": "pending",
        }

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        queued_requests.append(request)
        return {
            "status": "pending",
            "queue_state": "queued",
            "job_id": "job-future-speak-001",
            "job_ref": "background_work_job:job-future-speak-001",
            "task_summary": request["task_brief"],
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
        background_work_handler,
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        background_work_handler,
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
        "job_id": "job-future-speak-001",
        "accepted_task_id": "task-future-speak-001",
        "task_brief": "Schedule a future reminder.",
        "max_output_chars": 3000,
        "source_context": "The user asked for a delayed reminder.",
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
    }
    route_mock = AsyncMock(side_effect=AssertionError("router should not run"))
    dispatch_mock = AsyncMock(return_value={
        "status": "succeeded",
        "worker": "future_speak",
        "artifact_text": "Future speak scheduled.",
        "failure_summary": "",
        "result_summary": "Future speak scheduled.",
        "worker_metadata": {
            "trigger_at": "2026-05-15T22:00:00+00:00",
            "skip_result_delivery": True,
        },
    })
    complete_mock = AsyncMock()
    mark_running_mock = AsyncMock()
    mark_result_ready_mock = AsyncMock()
    mark_delivered_mock = AsyncMock()

    monkeypatch.setattr(
        worker_module,
        "claim_background_work_job",
        AsyncMock(side_effect=[fake_job, None]),
    )
    monkeypatch.setattr(worker_module, "route_background_work", route_mock)
    monkeypatch.setattr(worker_module, "dispatch_background_work", dispatch_mock)
    monkeypatch.setattr(
        worker_module,
        "mark_accepted_task_running",
        mark_running_mock,
    )
    monkeypatch.setattr(
        worker_module,
        "mark_tool_result_ready",
        mark_result_ready_mock,
    )
    monkeypatch.setattr(
        worker_module,
        "mark_accepted_task_delivered",
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
    route_mock.assert_not_called()
    dispatch_decision = dispatch_mock.await_args.args[0]
    assert dispatch_decision["worker"] == "future_speak"
    assert dispatch_decision["worker_payload"]["trigger_at"] == (
        "2026-05-16 10:00"
    )
    complete_kwargs = complete_mock.await_args.kwargs
    assert complete_kwargs["worker"] == "future_speak"
    assert complete_kwargs["skip_result_delivery"] is True
    mark_running_mock.assert_awaited_once()
    assert mark_running_mock.await_args.kwargs["accepted_task_id"] == (
        "task-future-speak-001"
    )
    assert mark_running_mock.await_args.kwargs["started_at"]
    mark_result_ready_mock.assert_not_called()
    mark_delivered_mock.assert_awaited_once()
    assert mark_delivered_mock.await_args.kwargs["accepted_task_id"] == (
        "task-future-speak-001"
    )


@pytest.mark.asyncio
async def test_future_speak_worker_schedules_calendar_future_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The future_speak worker should create a future-cognition schedule."""

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
        future_speak,
        "execute_future_cognition_action",
        execute_future_cognition,
    )

    result = await future_speak.execute(
        {
            "action": "execute",
            "worker": "future_speak",
            "reason": "The user asked for a delayed reminder.",
            "source_summary": "Schedule a future reminder.",
            "worker_payload": {
                "trigger_at": "2026-05-16 10:00",
                "continuation_objective": "Remind the user to drink water.",
                "source_action_attempt_id": (
                    "action_attempt:future-speak-001"
                ),
                "source_scope": {
                    "source_platform": "debug",
                    "source_channel_id": "debug:user:test-user",
                    "source_channel_type": "private",
                    "source_user_id": "global-user-001",
                    "source_platform_bot_id": "debug-bot-001",
                    "source_character_name": "Test Character",
                },
            },
        },
        max_output_chars=3000,
    )

    assert result["status"] == "succeeded"
    assert result["worker"] == "future_speak"
    assert result["worker_metadata"]["calendar_run_id"] == "calendar_run_001"
    assert result["worker_metadata"]["skip_result_delivery"] is True
    scheduled_spec = execute_future_cognition.await_args.args[0]
    assert scheduled_spec["kind"] == "trigger_future_cognition"
    assert scheduled_spec["params"]["trigger_at"] == "2026-05-16 10:00"
    assert scheduled_spec["params"]["continuation_objective"] == (
        "Remind the user to drink water."
    )
