"""Executable tests for the closed Plan 3 task-resolution contracts."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tests.task_resolution_test_helpers import _goal_continuation_ref


def _scene() -> dict[str, object]:
    return {
        "channel_scope": "private",
        "character_role": "Test Character",
        "current_user_role": "Test User",
        "semantic_scene": "A bounded DSH contract test.",
        "public_group_scene": "",
        "conversation_continuity": "The current turn continues one goal.",
        "semantic_temporal_context": "The test turn is active.",
    }


def _execution_context() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "debug:channel-1",
        "channel_type": "private",
        "requester_global_user_id": "global-user-1",
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


def _resolution_ref() -> dict[str, object]:
    return {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "dsh_session_id": "session-1",
        "activation_id": "activation-1",
        "lease_epoch": 2,
        "document_revision": 7,
        "last_committed_seq": 11,
    }


def _accepted_task_control() -> dict[str, object]:
    return {
        "schema_version": "accepted_task_control.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "operation": "continue",
        "instruction": "Use the newly supplied evidence and continue.",
    }


def _admission() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_admission.v1",
        "accepted_task_id": "task-1",
        "background_work_job_id": "job-1",
        "task_session_id": "session-1",
    }


def _result(context: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve this bounded goal.",
        "status": "resolved",
        "scene_context": context["scene_context"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "evidence_state": "complete",
        "evidence_excerpts": ['{"answer":"bounded"}'],
        "evidence_handles": ["semantic-ref-1"],
        "prompt_safe_summary": "The bounded goal is resolved.",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-1",
            "task_node_id": "node-1",
            "specialist": "dsh",
            "summary": "semantic-ref-1",
            "provenance_refs": ["receipt-1", "sha256:content"],
            "limitations": [],
        }],
        "completed_subgoals": ["bounded evidence"],
        "remaining_needs": [],
        "checkpoint": {},
        "coding_run_context": {},
    }


def _annotations(contract: object) -> set[str]:
    annotations = getattr(contract, "__annotations__", None)
    if not isinstance(annotations, dict):
        pytest.fail(f"contract has no executable field declaration: {contract!r}")
    return set(annotations)


def _validator(module: object, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        pytest.fail(f"task-resolution validator is unavailable: {name}")
    return value


def test_dsh_start_binding_reference_and_v1_result_contracts_are_exact() -> None:
    """V2 carriers validate exact fields and reject authority leakage."""

    try:
        module = importlib.import_module("kazusa_ai_chatbot.task_resolution.contracts")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned task-resolution contract owner is unavailable: {exc}")

    assert _annotations(module.TaskResolutionExecutionContextV2) == {
        "schema_version",
        "character_name",
        "platform",
        "channel_id",
        "channel_type",
        "requester_global_user_id",
        "requester_platform_user_id",
        "requester_display_name",
        "source_message_id",
        "source_platform_bot_id",
        "source_trigger_source",
        "source_llm_trace_id",
        "brain_conversation_ref",
        "scene_context",
        "goal_continuation_ref",
        "local_time_context",
        "prompt_message_context",
        "chat_history_recent",
        "chat_history_wide",
        "conversation_progress",
        "persona_summary",
        "conversation_summary",
        "current_timestamp_utc",
        "active_turn_platform_message_ids",
        "active_turn_conversation_row_ids",
        "session_media_refs",
        "max_output_chars",
    }
    assert _annotations(module.DshResolutionRefV1) == {
        "schema_version",
        "resolution_thread_id",
        "segment_id",
        "dsh_session_id",
        "activation_id",
        "lease_epoch",
        "document_revision",
        "last_committed_seq",
    }
    assert _annotations(module.AcceptedTaskControlV1) == {
        "schema_version",
        "accepted_task_ref",
        "operation",
        "instruction",
    }
    assert _annotations(module.TaskResolutionAdmissionV1) == {
        "schema_version",
        "accepted_task_id",
        "background_work_job_id",
        "task_session_id",
    }

    context = _validator(module, "validate_task_resolution_execution_context")(
        _execution_context(),
    )
    assert context["brain_conversation_ref"] == "episode-task-001"
    assert context["requester_display_name"] == "Test User"
    assert "coding_workspace_root" not in context

    reference = _validator(module, "validate_dsh_resolution_ref")(_resolution_ref())
    assert reference["document_revision"] == 7
    control = _validator(module, "validate_accepted_task_control")(
        _accepted_task_control(),
    )
    assert control["operation"] == "continue"
    admission = _validator(module, "validate_task_resolution_admission")(
        _admission(),
    )
    assert admission["background_work_job_id"] == "job-1"
    with pytest.raises(ValueError):
        _validator(module, "validate_task_resolution_admission")({
            **_admission(),
            "operation_generation": 0,
        })

    result = _validator(module, "validate_task_resolution_result")(_result(context))
    assert result["evidence"][0]["specialist"] == "dsh"
    assert result["coding_run_context"] == {}

    with pytest.raises(ValueError):
        _validator(module, "validate_task_resolution_execution_context")({
            **_execution_context(),
            "semantic_authority_token": "must-stay-model-hidden",
        })


def test_task_resolution_public_exports_are_dsh_only() -> None:
    """The public boundary exposes the closed DSH vocabulary only."""

    try:
        module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned task-resolution public boundary is unavailable: {exc}")

    public_names = set(getattr(module, "__all__", ()))
    assert public_names
    assert {
        "TaskResolutionExecutionContextV2",
        "DshResolutionRefV1",
        "TaskResolutionAdmissionV1",
        "AcceptedTaskControlV1",
        "TaskResolutionResultV1",
    } <= public_names
    assert not public_names & {
        "TaskResolutionExecutionContextV1",
        "TaskSpecialistRequestV1",
        "TaskSpecialistResultV1",
        "TaskResolutionCheckpointV1",
    }
