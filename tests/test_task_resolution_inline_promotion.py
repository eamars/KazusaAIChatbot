"""Tests for inline task resolution and its durable-promotion boundary."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.task_resolution import TaskResolutionContractError
from tests.test_task_resolution_orchestrator import (
    _context,
    _goal_continuation_ref,
)


def _request() -> dict[str, object]:
    return {
        "capability": "task_resolution_request",
        "semantic_goal": "Resolve one bounded public question.",
        "reason": "The current response lacks required evidence.",
        "evidence_handles": ["e1"],
        "goal_continuation_ref": _goal_continuation_ref(),
    }


def _deferred_result() -> dict[str, object]:
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    context = _context()
    checkpoint = state.create_task_resolution_checkpoint(_request(), context)
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": checkpoint["semantic_objective"],
        "status": "deferred",
        "scene_context": context["scene_context"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "evidence_state": "pending",
        "evidence_excerpts": [],
        "evidence_handles": [],
        "prompt_safe_summary": "Durable continuation is required.",
        "evidence": [],
        "completed_subgoals": [],
        "remaining_needs": list(checkpoint["remaining_needs"]),
        "checkpoint": checkpoint,
        "coding_run_context": {},
    }


def test_inline_budget_default_is_thirty_seconds() -> None:
    """Configuration and service validation share the approved default bound."""

    config = importlib.import_module("kazusa_ai_chatbot.config")
    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")

    assert config.TASK_RESOLUTION_INLINE_BUDGET_SECONDS == 30.0
    assert service.MINIMUM_INLINE_BUDGET_SECONDS == 1.0
    assert service.MAXIMUM_INLINE_BUDGET_SECONDS == 120.0


@pytest.mark.asyncio
async def test_background_start_creates_initial_checkpoint_and_promotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct background start reuses deferred promotion with a fresh session."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    promoted: list[dict[str, object]] = []

    async def promote(
        result: dict[str, object],
        context: dict[str, object],
        *,
        source_trigger_source: str,
        source_platform_bot_id: str,
        requester_display_name: str,
        source_llm_trace_id: str = "",
    ) -> dict[str, object]:
        assert context == _context()
        assert source_trigger_source == "user_message"
        assert source_platform_bot_id == "debug-bot"
        assert requester_display_name == "Test User"
        assert source_llm_trace_id == ""
        promoted.append(result)
        return {
            "status": "pending",
            "job_id": "job-001",
            "job_ref": "background_work_job:job-001",
            "accepted_task_id": "task-001",
            "task_identity_key": "accepted_task:v2:abc",
            "accepted_task_summary": _request()["semantic_goal"],
            "acknowledgement_constraint": "promise_allowed",
            "wait_guidance": "non_numeric_wait",
            "result_summary": "Accepted task continuation is durable.",
        }

    monkeypatch.setattr(
        service,
        "promote_deferred_task_resolution",
        promote,
    )

    result = await service.start_task_resolution_in_background(
        _request(),
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="debug-bot",
        requester_display_name="Test User",
    )

    assert result["status"] == "deferred"
    assert result["evidence"] == []
    checkpoint = result["checkpoint"]
    assert checkpoint["dispatch_count"] == 0
    assert checkpoint["orchestrator_call_count"] == 0
    assert checkpoint["semantic_objective"] == _request()["semantic_goal"]
    assert len(promoted) == 1


@pytest.mark.asyncio
async def test_background_start_materializes_one_pending_idempotent_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct background start creates one accepted task and one queue job."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    task = {
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "accepted_task_summary": _request()["semantic_goal"],
        "state": "enqueueing",
        "goal_continuation_ref": _context()["goal_continuation_ref"],
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
    enqueue = AsyncMock(return_value={
        "status": "pending",
        "job_id": "job-001",
        "job_ref": "background_work_job:job-001",
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "accepted_task_summary": _request()["semantic_goal"],
        "acknowledgement_constraint": "promise_allowed",
        "wait_guidance": "non_numeric_wait",
        "result_summary": "Accepted task continuation is durable.",
    })
    monkeypatch.setattr(service, "enqueue_background_work_request", enqueue)

    await service.start_task_resolution_in_background(
        _request(),
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="debug-bot",
        requester_display_name="Test User",
    )

    queued = enqueue.await_args.args[0]
    assert queued["job_id"] == "job-001"
    assert queued["accepted_task_id"] == "task-001"
    assert queued["idempotency_key"] == "background_work:task-001"
    assert queued["requested_worker"] == "task_orchestrator"
    assert queued["worker_payload"]["operation"] == "resume_task_resolution"
    assert queued["worker_payload"]["checkpoint"]["dispatch_count"] == 0
    assert enqueue.await_count == 1


@pytest.mark.asyncio
async def test_background_start_enqueue_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue failure surfaces a contract error instead of an acceptance."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")

    async def promote(
        result: dict[str, object],
        context: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del result, context, kwargs
        raise TaskResolutionContractError(
            "task-resolution durable promotion failed"
        )

    monkeypatch.setattr(
        service,
        "promote_deferred_task_resolution",
        promote,
    )

    with pytest.raises(TaskResolutionContractError, match="promotion failed"):
        await service.start_task_resolution_in_background(
            _request(),
            _context(),
            source_trigger_source="user_message",
            source_platform_bot_id="debug-bot",
            requester_display_name="Test User",
        )


@pytest.mark.asyncio
async def test_inline_service_hands_one_new_checkpoint_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline execution creates semantic state without materializing a job."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    captured: list[dict[str, object]] = []

    async def run_task_orchestrator(
        checkpoint: dict[str, object],
        execution_context: dict[str, object],
        *,
        inline_deadline: float | None,
    ) -> dict[str, object]:
        assert execution_context == _context()
        assert inline_deadline is not None
        captured.append(checkpoint)
        return {
            "schema_version": "task_resolution_result.v1",
            "status": "deferred",
            "prompt_safe_summary": "Durable continuation is required.",
            "evidence": [],
            "completed_subgoals": [],
            "remaining_needs": list(checkpoint["remaining_needs"]),
            "checkpoint": checkpoint,
            "coding_run_context": {},
        }

    monkeypatch.setattr(
        service,
        "run_task_orchestrator",
        run_task_orchestrator,
    )

    result = await service.resolve_task_inline(
        _request(),
        _context(),
        inline_budget_seconds=30.0,
    )

    assert result["status"] == "deferred"
    assert len(captured) == 1
    assert captured[0]["dispatch_count"] == 0
    assert captured[0]["semantic_objective"] == _request()["semantic_goal"]


@pytest.mark.asyncio
async def test_deferred_promotion_marks_pending_before_job_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker job becomes claimable only after accepted state is pending."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    task = {
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "accepted_task_summary": _request()["semantic_goal"],
        "state": "enqueueing",
    }
    create = AsyncMock(return_value={"status": "created", "task": task})
    events: list[str] = []
    queued_requests: list[dict[str, object]] = []

    async def mark_pending(**kwargs: object) -> dict[str, object]:
        assert kwargs["accepted_task_id"] == "task-001"
        assert kwargs["executor_ref"] == "job-001"
        events.append("accepted_task_pending")
        return {**task, "state": "pending"}

    async def enqueue(request: dict[str, object]) -> dict[str, object]:
        assert events == ["accepted_task_pending"]
        events.append("job_inserted")
        queued_requests.append(request)
        return {
            "status": "pending",
            "job_id": "job-001",
            "job_ref": "background_work_job:job-001",
            "accepted_task_id": "task-001",
            "task_identity_key": "accepted_task:v2:abc",
            "accepted_task_summary": _request()["semantic_goal"],
            "acknowledgement_constraint": "promise_allowed",
            "wait_guidance": "non_numeric_wait",
            "result_summary": "Accepted task continuation is durable.",
        }

    monkeypatch.setattr(
        service,
        "create_or_return_active_accepted_task",
        create,
    )
    monkeypatch.setattr(service, "mark_accepted_task_pending", mark_pending)
    monkeypatch.setattr(service, "enqueue_background_work_request", enqueue)

    result = await service.promote_deferred_task_resolution(
        _deferred_result(),
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="debug-bot",
        requester_display_name="Test User",
    )

    assert result["job_id"] == "job-001"
    assert events == ["accepted_task_pending", "job_inserted"]
    create.assert_awaited_once()
    queued = queued_requests[0]
    assert queued["job_id"] == "job-001"
    assert queued["requested_worker"] == "task_orchestrator"
    assert queued["semantic_objective"] == _request()["semantic_goal"]
    assert queued["worker_payload"]["checkpoint"]["dispatch_count"] == 0
    assert "source_context" not in queued


@pytest.mark.asyncio
async def test_interrupted_enqueueing_promotion_reuses_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry converges the same enqueueing task without duplicate identity."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    task = {
        "accepted_task_id": "task-existing",
        "task_identity_key": "accepted_task:v2:existing",
        "accepted_task_summary": _request()["semantic_goal"],
        "state": "enqueueing",
        "goal_continuation_ref": _context()["goal_continuation_ref"],
    }
    monkeypatch.setattr(
        service,
        "create_or_return_active_accepted_task",
        AsyncMock(return_value={"status": "already_active", "task": task}),
    )
    enqueue = AsyncMock(return_value={
        "status": "pending",
        "job_id": "job-existing",
        "job_ref": "background_work_job:job-existing",
        "accepted_task_id": "task-existing",
        "task_identity_key": "accepted_task:v2:existing",
        "accepted_task_summary": _request()["semantic_goal"],
        "acknowledgement_constraint": "promise_allowed",
        "wait_guidance": "non_numeric_wait",
        "result_summary": "Accepted task continuation is durable.",
    })
    monkeypatch.setattr(service, "enqueue_background_work_request", enqueue)
    monkeypatch.setattr(
        service,
        "mark_accepted_task_pending",
        AsyncMock(return_value={**task, "state": "pending"}),
    )

    await service.promote_deferred_task_resolution(
        _deferred_result(),
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="debug-bot",
        requester_display_name="Test User",
    )

    queued = enqueue.await_args.args[0]
    assert queued["job_id"] == "job-existing"
    assert queued["accepted_task_id"] == "task-existing"
    assert queued["idempotency_key"] == "background_work:task-existing"


@pytest.mark.asyncio
async def test_pending_promotion_repairs_missing_job_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after pending transition can insert the same reserved job."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    task = {
        "accepted_task_id": "task-existing",
        "task_identity_key": "accepted_task:v2:existing",
        "accepted_task_summary": _request()["semantic_goal"],
        "state": "pending",
        "executor_ref": "job-existing",
        "goal_continuation_ref": _context()["goal_continuation_ref"],
    }
    monkeypatch.setattr(
        service,
        "create_or_return_active_accepted_task",
        AsyncMock(return_value={"status": "already_active", "task": task}),
    )
    enqueue = AsyncMock(return_value={
        "status": "pending",
        "job_id": "job-existing",
        "job_ref": "background_work_job:job-existing",
        "accepted_task_id": "task-existing",
        "task_identity_key": "accepted_task:v2:existing",
        "accepted_task_summary": _request()["semantic_goal"],
        "acknowledgement_constraint": "promise_allowed",
        "wait_guidance": "non_numeric_wait",
        "result_summary": "Accepted task continuation is durable.",
    })
    mark_pending = AsyncMock()
    monkeypatch.setattr(service, "enqueue_background_work_request", enqueue)
    monkeypatch.setattr(service, "mark_accepted_task_pending", mark_pending)

    await service.promote_deferred_task_resolution(
        _deferred_result(),
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="debug-bot",
        requester_display_name="Test User",
    )

    queued = enqueue.await_args.args[0]
    assert queued["job_id"] == "job-existing"
    mark_pending.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", (0.9, 120.1, 30, "30"))
async def test_inline_service_rejects_invalid_budget_before_execution(
    budget: object,
) -> None:
    """The foreground wall-clock bound is a deterministic validated float."""

    package = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    with pytest.raises(package.TaskResolutionContractError, match="budget"):
        await package.resolve_task_inline(
            _request(),
            _context(),
            inline_budget_seconds=budget,
        )
