"""Executable lifecycle tests for the DSH task-resolution service."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable
from typing import Any

import pytest

from agentic_resolver.errors import RpcTransportError
from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from tests.task_resolution_test_helpers import (
    InMemoryAcceptedTaskStore,
    InMemoryBackgroundQueue,
    InMemoryDshBindingStore,
    _goal_continuation_ref,
    _resolution_ref,
)


def _scene() -> dict[str, object]:
    return {
        "channel_scope": "private",
        "character_role": "Test Character",
        "current_user_role": "Test User",
        "semantic_scene": "A bounded service race.",
        "public_group_scene": "",
        "conversation_continuity": "The same task remains active.",
        "semantic_temporal_context": "The test turn is current.",
    }


def _context() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "channel-1",
        "channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "trace-1",
        "brain_conversation_ref": "episode-task-001",
        "scene_context": _scene(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "local_time_context": {},
        "prompt_message_context": {"text": "Continue the task."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A test persona.",
        "conversation_summary": "A test conversation.",
        "current_timestamp_utc": "2026-08-30T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": [],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }


def _request() -> dict[str, object]:
    return {
        "capability": "task_resolution_request",
        "semantic_goal": "Continue the task.",
        "reason": "The user supplied a semantic continuation.",
        "evidence_handles": [],
        "start_in_background": False,
        "goal_continuation_ref": _goal_continuation_ref(),
    }


def _terminal_result() -> dict[str, object]:
    context = _context()
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Continue the task.",
        "status": "resolved",
        "scene_context": context["scene_context"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "evidence_state": "complete",
        "evidence_excerpts": [],
        "evidence_handles": [],
        "prompt_safe_summary": "The task is complete.",
        "evidence": [],
        "completed_subgoals": ["task"],
        "remaining_needs": [],
        "checkpoint": {},
        "coding_run_context": {},
    }


class _ControlledRuntime:
    """Small runtime double whose open operation can outlive the caller."""

    def __init__(self) -> None:
        self.open_started = asyncio.Event()
        self.checkpoint_requested = asyncio.Event()
        self.cancel_calls = 0
        self.reasoning_cancelled = False
        self._reasoning = asyncio.create_task(self._wait_forever())

    async def _wait_forever(self) -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.reasoning_cancelled = True
            raise

    async def open(self, **kwargs: object) -> asyncio.Task[None]:
        del kwargs
        self.open_started.set()
        return self._reasoning

    async def request_checkpoint(self, **kwargs: object) -> dict[str, object]:
        self.checkpoint_requested.set()
        session_id = str(kwargs["task_session_id"])
        return {
            "disposition": "checkpointed",
            **_resolution_ref(
                session_id=session_id,
                thread_id="thread-1",
                segment_id="segment-1",
                activation_id="activation-1",
            ),
        }

    async def cancel(self, **kwargs: object) -> None:
        del kwargs
        self.cancel_calls += 1

    async def close(self) -> None:
        self._reasoning.cancel()
        await asyncio.gather(self._reasoning, return_exceptions=True)


class _AuthorityBroker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def mint(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return "fresh-authority-token"


class _FailingRuntime:
    """Runtime double that fails after the binding admission callback."""

    async def open(self, **kwargs: object) -> None:
        before_resolve = kwargs["before_resolve"]
        await before_resolve(_resolution_ref(
            session_id=str(kwargs["task_session_id"]),
            thread_id="thread-failed",
            segment_id="segment-failed",
            activation_id="activation-failed",
        ))
        raise RpcTransportError("sidecar unavailable")


async def _await(value: object) -> object:
    if isinstance(value, Awaitable):
        return await value
    return value


def _service_module() -> Any:
    try:
        return importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned task-resolution service is unavailable: {exc}")


def test_task_session_identity_is_stable_across_objective_paraphrases() -> None:
    """Trusted scope and continuation identity ignore model wording."""

    service = _service_module()
    first_context = _context()
    later_context = {
        **first_context,
        "source_message_id": "message-2",
        "source_platform_bot_id": "bot-2",
        "source_llm_trace_id": "trace-2",
        "brain_conversation_ref": "episode-task-002",
        "active_turn_platform_message_ids": ["message-2"],
        "current_timestamp_utc": "2026-08-30T22:01:00+00:00",
    }
    first = service._task_session_id(
        _request(),
        first_context,
    )
    second = service._task_session_id(
        {
            **_request(),
            "semantic_goal": "Continue the bounded task using the same goal.",
        },
        later_context,
    )

    assert first == second

    different_continuation = build_goal_continuation_ref(
        source_episode_id="episode-task-002",
        source_message_id="message-2",
        branch_id="b1",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal-002",
        },
    )
    assert first != service._task_session_id(
        _request(),
        {
            **first_context,
            "goal_continuation_ref": different_continuation,
        },
    )
    assert first != service._task_session_id(
        _request(),
        {**first_context, "requester_global_user_id": "user-2"},
    )
    assert first != service._task_session_id(
        _request(),
        {**first_context, "channel_id": "channel-2"},
    )


@pytest.mark.asyncio
async def test_inline_admission_failure_terminally_faults_binding() -> None:
    """A failed runtime admission must leave no queued or opening binding."""

    service = _service_module()
    bindings = InMemoryDshBindingStore()

    with pytest.raises(RpcTransportError, match="sidecar unavailable"):
        await service.resolve_task_inline(
            _request(),
            _context(),
            inline_budget_seconds=1.0,
            runtime=_FailingRuntime(),
            binding_store=bindings,
        )

    binding = next(iter(bindings.bindings.values()))
    assert binding["state"] == "faulted"
    assert binding["resolution_thread_id"] == "thread-failed"


@pytest.mark.asyncio
async def test_inline_checkpoint_promotes_same_bound_dsh_session_without_canceling_reasoning() -> None:
    """Budget expiry checkpoints cooperatively while shielded work continues."""

    service = _service_module()
    resolve = getattr(service, "resolve_task_inline", None)
    if not callable(resolve):
        pytest.fail("task-resolution service lacks resolve_task_inline")
    runtime = _ControlledRuntime()
    bindings = InMemoryDshBindingStore()
    reasoning_cancelled_before_cleanup = False
    try:
        result = await asyncio.wait_for(_await(resolve(
            _request(),
            _context(),
            inline_budget_seconds=0.01,
            runtime=runtime,
            binding_store=bindings,
        )), timeout=0.5)
        reasoning_cancelled_before_cleanup = runtime.reasoning_cancelled
    finally:
        await runtime.close()

    assert runtime.open_started.is_set()
    assert runtime.checkpoint_requested.is_set()
    assert runtime.cancel_calls == 0
    assert reasoning_cancelled_before_cleanup is False
    assert result["status"] == "deferred"
    assert result["checkpoint"]["resolution_thread_id"] == "thread-1"


@pytest.mark.asyncio
async def test_background_start_mints_authority_only_when_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued payloads carry identity, while a claimed job obtains authority."""

    service = _service_module()
    start = getattr(service, "start_task_resolution_in_background", None)
    if not callable(start):
        pytest.fail("task-resolution service lacks start_task_resolution_in_background")
    bindings = InMemoryDshBindingStore()
    accepted = InMemoryAcceptedTaskStore()
    queue = InMemoryBackgroundQueue()
    authority = _AuthorityBroker()
    monkeypatch.setattr(service, "_TASK_RESOLUTION_RUNTIME", object())
    result = await _await(start(
        {**_request(), "start_in_background": True},
        _context(),
        source_trigger_source="user_message",
        source_platform_bot_id="bot-1",
        requester_display_name="Test User",
        source_llm_trace_id="trace-1",
        binding_store=bindings,
        accepted_task_store=accepted,
        background_queue=queue,
        authority_broker=authority,
    ))

    assert authority.calls == []
    payload = queue.requests[0]["worker_payload"]
    assert payload["operation"] == "open_dsh_resolution"
    assert payload["control"] is None
    assert "token" not in payload
    claimed = await authority.mint(
        task_session_id=payload["task_session_id"],
        operation_generation=payload["operation_generation"],
    )
    assert claimed == "fresh-authority-token"
    assert result["schema_version"] == "task_resolution_admission.v1"
    assert result["accepted_task_id"] == "task-1"
    assert result["background_work_job_id"] == "dsh-task:task-1:generation:0"
    assert result["task_session_id"] == payload["task_session_id"]
    assert "checkpoint" not in result


@pytest.mark.asyncio
async def test_interaction_terminal_before_promotion_reconciles_exactly_once() -> None:
    """Pre-promotion terminal completion is stored and replayed idempotently."""

    service = _service_module()
    reconcile = getattr(service, "reconcile_task_resolution_result", None)
    if not callable(reconcile):
        pytest.fail("task-resolution service lacks reconcile_task_resolution_result")
    bindings = InMemoryDshBindingStore()
    bindings.bindings["session-1"] = {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": "session-1",
        "semantic_objective": "Continue the task.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "source_scope": {},
        "state": "terminal",
        "start_spec": service._build_start_spec(_request(), _context()),
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "resolution_ref": _resolution_ref(
            session_id="session-1",
            thread_id="thread-1",
            segment_id="segment-1",
            activation_id="activation-1",
        ),
        "operation_generation": 0,
        "current_accepted_task_id": None,
        "current_background_work_job_id": None,
        "latest_task_resolution_result": None,
        "revision": 0,
        "created_at": "2026-08-30T22:00:00+00:00",
        "updated_at": "2026-08-30T22:00:00+00:00",
    }
    accepted = InMemoryAcceptedTaskStore()
    result = _terminal_result()
    first = await _await(reconcile(
        task_session_id="session-1",
        operation_generation=0,
        task_resolution_result=result,
        binding_store=bindings,
        accepted_task_store=accepted,
        promoted=False,
    ))
    second = await _await(reconcile(
        task_session_id="session-1",
        operation_generation=0,
        task_resolution_result=result,
        binding_store=bindings,
        accepted_task_store=accepted,
        promoted=True,
    ))

    assert first["disposition"] == "stored_before_promotion"
    assert second["disposition"] in {"reconciled", "already_reconciled"}
    sink_calls = [name for name, _ in bindings.calls if "result" in name]
    assert len(sink_calls) == 1


@pytest.mark.asyncio
async def test_delivered_followup_continues_same_thread_under_next_generation() -> None:
    """A claimed control reuses the DSH thread and advances one generation."""

    service = _service_module()
    continue_task = getattr(service, "continue_delivered_task", None)
    if not callable(continue_task):
        pytest.fail("task-resolution service lacks continue_delivered_task")
    bindings = InMemoryDshBindingStore()
    accepted = InMemoryAcceptedTaskStore()
    source_task = {
        "schema_version": "accepted_task.v2",
        "accepted_task_id": "task-1",
        "task_kind": "task_resolution",
        "task_identity_key": "identity-1",
        "active_identity_key": "identity-1",
        "accepted_task_summary": "Continue the task.",
        "semantic_objective": "Continue the task.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_trigger_source": "user_message",
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "storage_timestamp_utc": "2026-08-30T22:00:00+00:00",
        "state": "delivered",
        "executor_ref": "job-0",
        "revision": 0,
        "dsh_task_session_id": "session-1",
        "dsh_operation_generation": 0,
        "dsh_followup_open": True,
        "dsh_followup_claim_action_attempt_id": None,
    }
    accepted.tasks["task-1"] = source_task
    bindings.bindings["session-1"] = {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": "session-1",
        "semantic_objective": "Continue the task.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "source_scope": {},
        "state": "terminal",
        "start_spec": service._build_start_spec(_request(), _context()),
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "resolution_ref": _resolution_ref(
            session_id="session-1",
            thread_id="thread-1",
            segment_id="segment-1",
            activation_id="activation-1",
        ),
        "operation_generation": 0,
        "current_accepted_task_id": "task-1",
        "current_background_work_job_id": "job-0",
        "latest_task_resolution_result": None,
        "revision": 0,
        "created_at": "2026-08-30T22:00:00+00:00",
        "updated_at": "2026-08-30T22:00:00+00:00",
    }
    queue = InMemoryBackgroundQueue()
    authority = _AuthorityBroker()
    control = {
        "schema_version": "accepted_task_control.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "operation": "continue",
        "instruction": "Continue with the new semantic instruction.",
    }
    first = await _await(continue_task(
        control,
        action_attempt_id="attempt-1",
        binding={
            "task_session_id": "session-1",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "operation_generation": 0,
            "state": "terminal",
        },
        binding_store=bindings,
        accepted_task_store=accepted,
        background_queue=queue,
        authority_broker=authority,
    ))
    retry = await _await(continue_task(
        control,
        action_attempt_id="attempt-1",
        binding={
            "task_session_id": "session-1",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "operation_generation": 0,
            "state": "terminal",
        },
        binding_store=bindings,
        accepted_task_store=accepted,
        background_queue=queue,
        authority_broker=authority,
    ))

    assert first["task_session_id"] == retry["task_session_id"] == "session-1"
    assert first["resolution_thread_id"] == retry["resolution_thread_id"] == "thread-1"
    assert first["segment_id"] == retry["segment_id"] == "segment-1"
    assert first["operation_generation"] == retry["operation_generation"] == 1
    assert first["accepted_task_id"] == retry["accepted_task_id"]
    assert queue.requests[0]["worker_payload"]["operation"] == "continue_dsh_resolution"
    assert queue.requests[0]["worker_payload"]["operation_generation"] == 1
