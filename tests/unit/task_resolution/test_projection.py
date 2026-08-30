"""Executable tests for DSH facts and exhaust projection."""

from __future__ import annotations

import importlib

import pytest

from agentic_resolver.contracts import (
    DSHResolutionExhaustV2,
    EvidenceReceiptV2,
    SubmitResolutionV2,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json, content_digest
from tests.task_resolution_test_helpers import _goal_continuation_ref


def _scene() -> dict[str, object]:
    return {
        "channel_scope": "private",
        "character_role": "Test Character",
        "current_user_role": "Test User",
        "semantic_scene": "A deterministic projection scene.",
        "public_group_scene": "",
        "conversation_continuity": "The current goal is still active.",
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
        "local_time_context": {"local_time": "2026-08-30 10:00"},
        "prompt_message_context": {"text": "Find bounded evidence."},
        "chat_history_recent": [{"role": "user", "text": "Prior turn"}],
        "chat_history_wide": [{"role": "assistant", "text": "Earlier"}],
        "conversation_progress": {"goal": "evidence"},
        "persona_summary": "A deterministic persona.",
        "conversation_summary": "A deterministic conversation.",
        "current_timestamp_utc": "2026-08-30T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": ["row-1"],
        "session_media_refs": [{"cache_ref": "media-1"}],
        "max_output_chars": 3000,
    }


def _start_spec(context: dict[str, object]) -> dict[str, object]:
    model_facts = [
        "character_and_scene=" + canonical_json({
            "character_name": context["character_name"],
            "scene_context": context["scene_context"],
        }).decode("utf-8"),
        "local_time=" + canonical_json(context["local_time_context"]).decode("utf-8"),
        "current_message_context=" + canonical_json(
            context["prompt_message_context"],
        ).decode("utf-8"),
        "recent_conversation=" + canonical_json(
            context["chat_history_recent"],
        ).decode("utf-8"),
        "wide_conversation=" + canonical_json(
            context["chat_history_wide"],
        ).decode("utf-8"),
        "conversation_progress=" + canonical_json(
            context["conversation_progress"],
        ).decode("utf-8"),
        "persona_summary=" + canonical_json(context["persona_summary"]).decode("utf-8"),
        "conversation_summary=" + canonical_json(
            context["conversation_summary"],
        ).decode("utf-8"),
        "active_turn_lineage=" + canonical_json({
            "conversation_row_ids": context["active_turn_conversation_row_ids"],
            "platform_message_ids": context["active_turn_platform_message_ids"],
        }).decode("utf-8"),
        "attached_media_refs=" + canonical_json(
            context["session_media_refs"],
        ).decode("utf-8"),
    ]
    return {
        "schema_version": "dsh_task_start_spec.v1",
        "resolver_request": {
            "capability": "task_resolution_request",
            "semantic_goal": "Find bounded evidence.",
            "reason": "The goal requires a source-backed answer.",
            "evidence_handles": [],
            "start_in_background": False,
            "goal_continuation_ref": context["goal_continuation_ref"],
        },
        "execution_context": context,
        "model_facts": model_facts,
        "model_facts_digest": content_digest(model_facts),
        "objective_ref": content_digest(context["goal_continuation_ref"]),
    }


def _exhaust(
    kind: str,
    *,
    terminal_status: str = "resolved",
    include_evidence: bool = True,
) -> DSHResolutionExhaustV2:
    identity = {
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "policy_epoch": "dsh-standard-policy-v2",
    }
    if kind == "terminal":
        terminal_mapping = {
            "status": terminal_status,
            "summary": "The source supplied the requested fact.",
            "findings": [{"answer": "bounded"}],
            "completed_subgoals": ["source lookup"],
            "remaining_needs": [],
            "clarification_request": None,
            "approval_request": None,
            "artifact_refs": ["artifact-1", "semantic-ref-1"],
            "warnings": ["source is bounded"],
        }
        if terminal_status == "needs_user_input":
            terminal_mapping["clarification_request"] = {
                "question": "Which source should be preferred?",
            }
        if terminal_status == "approval_required":
            terminal_mapping["approval_request"] = {
                "question": "Approve the bounded source lookup.",
            }
        terminal = SubmitResolutionV2.from_mapping(terminal_mapping)
        evidence = [{
            "schema_version": "evidence_receipt.v2",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
            "policy_epoch": "dsh-standard-policy-v2",
            "evidence_id": "receipt-1",
            "source_kind": "semantic",
            "semantic_ref": "semantic-ref-1",
            "content_digest": "sha256:content",
            "provenance": {"tool_name": "web_search"},
        }]
        evidence_rows = (
            tuple(
                EvidenceReceiptV2.from_mapping(item)
                for item in evidence
            )
            if include_evidence
            else ()
        )
        return DSHResolutionExhaustV2.from_terminal(
            **identity,
            terminal=terminal,
            evidence=evidence_rows,
            last_committed_seq=12,
        )
    if kind == "checkpointed":
        return DSHResolutionExhaustV2.from_mapping({
            "kind": "checkpointed",
            "checkpoint": {
                "schema_version": "dsh_resolution_ref.v1",
                "resolution_thread_id": "thread-1",
                "segment_id": "segment-1",
                "dsh_session_id": "session-1",
                "activation_id": "activation-1",
                "lease_epoch": 2,
                "document_revision": 4,
                "last_committed_seq": 12,
            },
            "identity": identity,
            "last_committed_seq": 12,
        })
    return DSHResolutionExhaustV2.from_mapping({
        "kind": kind,
        "identity": identity,
        "last_committed_seq": 12,
        **({"fault": {"code": "RPC_UNAVAILABLE", "detail": "private detail"}}
           if kind == "runtime_fault" else {}),
    })


def test_dsh_exhaust_maps_to_task_result_without_semantic_reclassification() -> None:
    """The real projection preserves typed DSH status and trusted lineage."""

    try:
        module = importlib.import_module("kazusa_ai_chatbot.task_resolution.projection")
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned task-resolution projection is unavailable: {exc}")

    build_facts = getattr(module, "build_model_facts", None)
    project = getattr(module, "project_dsh_exhaust", None)
    if not callable(build_facts) or not callable(project):
        pytest.fail("projection owner must expose build_model_facts and project_dsh_exhaust")

    context = _context()
    start_spec = _start_spec(context)
    assert build_facts(context) == start_spec["model_facts"]
    assert all(isinstance(fact, str) for fact in build_facts(context))

    expected_terminal = {
        "resolved": ("resolved", "complete"),
        "partial": ("partial", "partial"),
        "needs_user_input": ("needs_user_input", "pending"),
        "approval_required": ("approval_required", "pending"),
        "unavailable": ("unavailable", "missing"),
        "failed": ("failed", "blocked"),
    }
    for terminal_status, (status, evidence_state) in expected_terminal.items():
        result = project(
            _exhaust("terminal", terminal_status=terminal_status),
            start_spec,
        )
        assert result["status"] == status
        assert result["evidence_state"] == evidence_state
        assert result["semantic_objective"] == start_spec["resolver_request"]["semantic_goal"]
        assert result["goal_continuation_ref"] == context["goal_continuation_ref"]
        assert result["coding_run_context"] == {}
        assert result["checkpoint"] == {}
        assert result["evidence_handles"] == ["semantic-ref-1", "artifact-1"]
        assert result["evidence"][0]["specialist"] == "dsh"
        assert "The source supplied the requested fact." in result[
            "prompt_safe_summary"
        ]
        assert "source is bounded" in result["prompt_safe_summary"]

    terminal_without_receipt = project(
        _exhaust("terminal", include_evidence=False),
        start_spec,
    )
    assert terminal_without_receipt["evidence_handles"] == [
        "artifact-1",
        "semantic-ref-1",
    ]
    assert terminal_without_receipt["evidence"] == [{
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "artifact-1",
        "task_node_id": "dsh",
        "specialist": "dsh",
        "summary": "artifact-1",
        "provenance_refs": ["artifact-1"],
        "limitations": [],
    }, {
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "semantic-ref-1",
        "task_node_id": "dsh",
        "specialist": "dsh",
        "summary": "semantic-ref-1",
        "provenance_refs": ["semantic-ref-1"],
        "limitations": [],
    }]

    expected_non_terminal = {
        "checkpointed": ("deferred", "pending"),
        "runtime_fault": ("unavailable", "missing"),
        "canceled": ("failed", "blocked"),
    }
    for kind, (status, evidence_state) in expected_non_terminal.items():
        result = project(_exhaust(kind), start_spec)
        assert result["status"] == status
        assert result["evidence_state"] == evidence_state
        assert result["evidence"] == []
        if kind == "checkpointed":
            assert result["checkpoint"]["resolution_thread_id"] == "thread-1"


def test_projection_rejects_unvalidated_or_unknown_dsh_exhaust() -> None:
    """Malformed/unknown DSH data fails closed at the structural boundary."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution.projection")
    project = module.project_dsh_exhaust
    context = _context()
    start_spec = _start_spec(context)
    with pytest.raises(ValueError, match="DSH exhaust"):
        project({"kind": "unknown"}, start_spec)
    with pytest.raises(ValueError, match="DSH exhaust"):
        project(DSHResolutionExhaustV2(kind="unknown"), start_spec)
    with pytest.raises(ValueError, match="DSH exhaust"):
        project(None, start_spec)


def test_checkpoint_projection_validates_runtime_identity_when_payload_is_empty() -> None:
    """The runtime identity is the checkpoint reference for a relay pause."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution.projection")
    project = module.project_dsh_exhaust
    context = _context()
    start_spec = _start_spec(context)
    result = project(
        DSHResolutionExhaustV2.from_mapping({
            "kind": "checkpointed",
            "checkpoint": {},
            "identity": {
                "resolution_thread_id": "thread-1",
                "segment_id": "segment-1",
                "dsh_session_id": "session-1",
                "activation_id": "activation-1",
                "lease_epoch": 2,
                "document_revision": 4,
                "last_committed_seq": 12,
            },
            "last_committed_seq": 12,
        }),
        start_spec,
    )

    assert result["status"] == "deferred"
    assert result["checkpoint"] == {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "dsh_session_id": "session-1",
        "activation_id": "activation-1",
        "lease_epoch": 2,
        "document_revision": 4,
        "last_committed_seq": 12,
    }
