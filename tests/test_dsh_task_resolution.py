"""Executable cross-boundary tests for DSH task resolution."""

from __future__ import annotations

import asyncio

import pytest

from tests.task_resolution_test_helpers import (
    InMemoryDshBindingStore,
    _goal_continuation_ref,
    _resolution_ref,
)


def _scene() -> dict[str, object]:
    return {
        "channel_scope": "private",
        "character_role": "Test Character",
        "current_user_role": "Test User",
        "semantic_scene": "A bounded cross-boundary test.",
        "public_group_scene": "",
        "conversation_continuity": "One task goal continues.",
        "semantic_temporal_context": "The test turn is current.",
    }


def _state() -> dict[str, object]:
    return {
        "storage_timestamp_utc": "2026-08-30T00:00:00Z",
        "cognitive_episode": {"trigger_source": "user_message"},
    }


def _request(*, priority: str = "now") -> dict[str, object]:
    return {
        "schema_version": "resolver_capability_request.v1",
        "capability_kind": "task_resolution_request",
        "objective": "Resolve this cross-boundary goal.",
        "reason": "The goal requires a DSH task.",
        "priority": priority,
        "goal_continuation_ref": _goal_continuation_ref(),
    }


def _result(status: str) -> dict[str, object]:
    terminal = status in {"resolved", "partial"}
    evidence = [{
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "evidence-1",
        "task_node_id": "node-1",
        "specialist": "dsh",
        "summary": "semantic-ref-1",
        "provenance_refs": ["receipt-1", "sha256:content"],
        "limitations": [],
    }] if terminal else []
    evidence_state = (
        "complete" if status == "resolved" else
        "partial" if status == "partial" else
        "pending" if status in {"needs_user_input", "approval_required"} else
        "missing" if status == "unavailable" else
        "blocked"
    )
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve this cross-boundary goal.",
        "status": status,
        "scene_context": _scene(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "evidence_state": evidence_state,
        "evidence_excerpts": ['{"answer":"bounded"}'] if terminal else [],
        "evidence_handles": ["semantic-ref-1"] if terminal else [],
        "prompt_safe_summary": "A bounded DSH result.",
        "evidence": evidence,
        "completed_subgoals": ["bounded evidence"] if terminal else [],
        "remaining_needs": [] if terminal else ["one typed next step"],
        "checkpoint": {},
        "coding_run_context": {},
    }


def _admission() -> dict[str, object]:
    """Build the transient identity returned before a DSH claim."""

    return {
        "schema_version": "task_resolution_admission.v1",
        "accepted_task_id": "task-1",
        "background_work_job_id": "job-1",
        "task_session_id": "session-1",
    }


@pytest.mark.asyncio
async def test_inline_resolved_partial_and_terminal_blockers_recur_through_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capability caller carries every DSH result state into cognition."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    context = {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "debug:channel-1",
        "channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "source_message_id": "message-1",
        "source_platform_bot_id": "debug-bot-1",
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "trace-1",
        "brain_conversation_ref": "episode-task-001",
        "scene_context": _scene(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "local_time_context": {"local_time": "2026-08-30 10:00"},
        "prompt_message_context": {"text": "Resolve this bounded goal."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A bounded test persona.",
        "conversation_summary": "A bounded test conversation.",
        "current_timestamp_utc": "2026-08-30T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": ["row-1"],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }
    calls: list[dict[str, object]] = []

    async def resolve(request: object, execution_context: object, **kwargs: object):
        calls.append({
            "request": request,
            "context": execution_context,
            **kwargs,
        })
        status = calls[-1].get("status", "resolved")
        if status == "partial":
            return {
                **_result(status),
                "remaining_needs": ["one bounded follow-up"],
            }
        return _result(status)

    def project_context(*_args: object, **_kwargs: object) -> dict[str, object]:
        return context

    monkeypatch.setattr(owner, "_task_resolution_execution_context_from_state", project_context)

    for status in (
        "resolved",
        "partial",
        "needs_user_input",
        "approval_required",
        "unavailable",
        "failed",
    ):
        async def resolve_one(
            request: object,
            execution_context: object,
            *,
            _status: str = status,
            **kwargs: object,
        ) -> dict[str, object]:
            calls.append({
                "request": request,
                "context": execution_context,
                "status": _status,
                **kwargs,
            })
            result = _result(_status)
            if _status == "partial":
                result["remaining_needs"] = ["one bounded follow-up"]
            return result

        monkeypatch.setattr(owner, "resolve_task_inline", resolve_one)
        observation = await owner.execute_resolver_capability_request(
            _request(),
            _state(),
        )
        expected_observation_status = "succeeded" if status in {"resolved", "partial"} else (
            "blocked" if status in {"needs_user_input", "approval_required"} else "failed"
        )
        assert observation["status"] == expected_observation_status
        assert observation["goal_continuation_ref"] == _goal_continuation_ref()

    assert len(calls) == 6
    assert all(row["context"] == context for row in calls)


@pytest.mark.asyncio
async def test_inline_budget_checkpoints_without_cancelling_and_promotes_same_session() -> None:
    """The real task service uses a shielded cooperative checkpoint path."""

    from kazusa_ai_chatbot.task_resolution import service as owner

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.completed = asyncio.Event()

        async def open(self, *_args: object, **_kwargs: object) -> object:
            self.calls.append(("open", dict(_kwargs)))
            await _kwargs["before_resolve"](reference)
            await asyncio.sleep(0.01)
            self.completed.set()
            return {
                "kind": "checkpointed",
                "resolution_thread_id": "thread-1",
                "segment_id": "segment-1",
                "dsh_session_id": "session-1",
                "activation_id": "activation-1",
                "lease_epoch": 1,
                "document_revision": 2,
                "last_committed_seq": 3,
            }

        async def request_checkpoint(self, *_args: object, **_kwargs: object) -> object:
            self.calls.append(("request_checkpoint", dict(_kwargs)))
            return {"kind": "checkpointed", "dsh_resolution_ref": reference}

    request = {
        "capability": "task_resolution_request",
        "semantic_goal": "Resolve this cross-boundary goal.",
        "reason": "The goal requires a DSH task.",
        "evidence_handles": [],
        "start_in_background": False,
        "goal_continuation_ref": _goal_continuation_ref(),
    }
    context = {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "debug:channel-1",
        "channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "source_message_id": "message-1",
        "source_platform_bot_id": "debug-bot-1",
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "trace-1",
        "brain_conversation_ref": "episode-task-001",
        "scene_context": _scene(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "local_time_context": {"local_time": "2026-08-30 10:00"},
        "prompt_message_context": {"text": "Resolve this bounded goal."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A bounded test persona.",
        "conversation_summary": "A bounded test conversation.",
        "current_timestamp_utc": "2026-08-30T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": ["row-1"],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }
    runtime = Runtime()
    service_context = owner._context_for_service(context)
    session_id = owner._task_session_id(request, service_context)
    reference = _resolution_ref(
        session_id=session_id,
        thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
    )
    bindings = InMemoryDshBindingStore(preassigned_ref=reference)
    result = await owner.resolve_task_inline(
        request,
        context,
        runtime=runtime,
        binding_store=bindings,
        inline_budget_seconds=0.001,
    )

    assert result["status"] == "deferred"
    resolution_ref = result["checkpoint"]
    assert resolution_ref == reference
    assert [name for name, _ in runtime.calls] == [
        "open",
        "request_checkpoint",
    ]
    await asyncio.wait_for(runtime.completed.wait(), timeout=0.5)
    assert bindings.bindings[session_id]["state"] == "checkpointed"


@pytest.mark.asyncio
async def test_direct_background_claim_opens_dsh_and_terminal_result_uses_normal_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Background admission carries a V2 request and returns a cognition result."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        owner,
        "_task_resolution_execution_context_from_state",
        lambda *_args, **_kwargs: {
            "schema_version": "task_resolution_execution_context.v2",
            "scene_context": _scene(),
            "goal_continuation_ref": _goal_continuation_ref(),
        },
    )

    async def start(request: object, context: object, **kwargs: object) -> dict[str, object]:
        captured.append({"request": request, "context": context, **kwargs})
        return _admission()

    monkeypatch.setattr(owner, "start_task_resolution_in_background", start)
    observation = await owner.execute_resolver_capability_request(
        _request(priority="background"),
        {
            **_state(),
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
            "platform_bot_id": "bot-1",
            "user_name": "Test User",
        },
    )

    assert len(captured) == 1
    assert captured[0]["request"]["capability"] == "task_resolution_request"
    assert captured[0]["request"]["start_in_background"] is True
    assert captured[0]["request"]["semantic_goal"] == _request(priority="background")["objective"]
    assert "authority_token" not in captured[0]["request"]
    assert observation["status"] == "succeeded"


@pytest.mark.asyncio
async def test_delivered_accepted_task_controls_continue_summarize_status_and_cancel_same_session() -> None:
    """Typed controls preserve one session while changing delivery generation."""

    from kazusa_ai_chatbot.action_spec import execution as owner

    handler = getattr(owner, "execute_accepted_task_control", None)
    assert callable(handler)
    calls: list[dict[str, object]] = []

    class Lifecycle:
        async def apply_control(self, **fields: object) -> dict[str, object]:
            calls.append(dict(fields))
            return {
                "dsh_task_session_id": "session-1",
                "delivery_generation": 1,
                "accepted_task_id": f"task-{len(calls)}",
                "operation": fields["operation"],
            }

    for operation, instruction in (
        ("continue", "Continue with the new evidence."),
        ("summarize", None),
        ("cancel", None),
    ):
        control = {
            "schema_version": "accepted_task_control.v1",
            "accepted_task_ref": "accepted_task:task-terminal",
            "operation": operation,
            "instruction": instruction,
        }
        output = await handler(
            control=control,
            lifecycle=Lifecycle(),
            action_attempt_id=f"attempt-{operation}",
            advertised_refs={"accepted_task:task-terminal"},
        )
        assert output["dsh_task_session_id"] == "session-1"

    assert [call["operation"] for call in calls] == ["continue", "summarize", "cancel"]


@pytest.mark.asyncio
async def test_sidecar_fault_and_restart_recover_bound_session() -> None:
    """Restart recovery resumes the durable DSH binding and its fence."""

    from agentic_resolver import controller

    recover = getattr(controller.ResolutionController, "recover_after_runtime_fault", None)
    assert callable(recover)
    output = await recover(
        {
            "schema_version": "dsh_runtime_fault.v1",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "dsh_session_id": "session-1",
            "document_revision": 9,
            "last_committed_seq": 12,
            "fault_code": "SIDECAR_UNAVAILABLE",
        },
    )
    assert output["resolution_thread_id"] == "thread-1"
    assert output["segment_id"] == "segment-1"
    assert output["dsh_session_id"] == "session-1"
