"""Executable tests for the closed DSH task-resolution boundary."""

from __future__ import annotations

import importlib
import inspect

import pytest

from tests.task_resolution_test_helpers import (
    _context,
    _goal_continuation_ref,
    _scene_context,
)


def _evidence(*, evidence_id: str = "evidence-1") -> dict[str, object]:
    """Build one DSH-owned prompt-safe evidence receipt."""

    return {
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": evidence_id,
        "task_node_id": "dsh-node-1",
        "specialist": "dsh",
        "summary": "memory:item-1",
        "provenance_refs": ["memory:item-1"],
        "limitations": [],
    }


def _result(
    *,
    status: str,
    evidence: list[dict[str, object]],
    checkpoint: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one result carrier with the status-aligned evidence state."""

    evidence_state = {
        "resolved": "complete",
        "partial": "partial",
        "needs_user_input": "pending",
        "approval_required": "pending",
        "unavailable": "missing",
        "failed": "blocked",
        "deferred": "pending",
    }[status]
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve the user's bounded task.",
        "status": status,
        "scene_context": _scene_context(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "evidence_state": evidence_state,
        "evidence_excerpts": [
            "A relevant prior commitment was found." for _item in evidence
        ],
        "evidence_handles": [str(item["summary"]) for item in evidence],
        "prompt_safe_summary": "One relevant fact was recovered.",
        "evidence": evidence,
        "completed_subgoals": [],
        "remaining_needs": (
            [] if status == "resolved" else ["Confirm the current constraint."]
        ),
        "checkpoint": checkpoint or {},
        "coding_run_context": {},
    }


def _resolution_ref() -> dict[str, object]:
    """Build a complete durable DSH reference for deferred results."""

    return {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": "thread-task-001",
        "segment_id": "segment-task-001",
        "dsh_session_id": "session-task-001",
        "activation_id": "activation-task-001",
        "lease_epoch": 1,
        "document_revision": 0,
        "last_committed_seq": 0,
    }


def test_task_resolution_public_boundary_exposes_current_contracts() -> None:
    """The package exposes only the canonical V2 context and DSH carriers."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    for name in (
        "AcceptedTaskControlV1",
        "DshResolutionRefV1",
        "TaskResolutionContractError",
        "TaskResolutionExecutionContextV2",
        "TaskResolutionResultV1",
        "validate_accepted_task_control",
        "validate_dsh_resolution_ref",
        "validate_task_resolution_execution_context",
        "validate_task_resolution_result",
        "resolve_task_inline",
        "resume_task_resolution",
    ):
        assert hasattr(module, name), name



def test_task_resolution_entrypoints_accept_typed_runtime_injection() -> None:
    """The public entrypoints keep runtime collaborators keyword-only."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    inline = inspect.signature(module.resolve_task_inline)
    resume = inspect.signature(module.resume_task_resolution)

    assert tuple(inline.parameters) == (
        "request",
        "execution_context",
        "inline_budget_seconds",
        "runtime",
        "binding_store",
    )
    assert tuple(resume.parameters) == (
        "checkpoint",
        "execution_context",
        "runtime",
    )
    assert inline.parameters["inline_budget_seconds"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert inline.parameters["runtime"].kind is inspect.Parameter.KEYWORD_ONLY
    assert resume.parameters["runtime"].kind is inspect.Parameter.KEYWORD_ONLY


def test_execution_context_and_resolution_ref_validate_as_exact_carriers() -> None:
    """The context and opaque DSH reference use their exact schemas."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    context = module.validate_task_resolution_execution_context(_context())
    reference = module.validate_dsh_resolution_ref(_resolution_ref())

    assert context["schema_version"] == "task_resolution_execution_context.v2"
    assert context["source_message_id"] == "message-1"
    assert reference["resolution_thread_id"] == "thread-task-001"
    assert reference["last_committed_seq"] == 0


def test_accepted_task_control_requires_instruction_only_for_continue() -> None:
    """Typed controls keep operation-specific instruction ownership."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    continue_control = module.validate_accepted_task_control({
        "schema_version": "accepted_task_control.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "operation": "continue",
        "instruction": "Verify the remaining constraint.",
    })
    cancel_control = module.validate_accepted_task_control({
        "schema_version": "accepted_task_control.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "operation": "cancel",
        "instruction": None,
    })

    assert continue_control["operation"] == "continue"
    assert cancel_control["instruction"] is None
    with pytest.raises(module.TaskResolutionContractError, match="instruction"):
        module.validate_accepted_task_control({
            "schema_version": "accepted_task_control.v1",
            "accepted_task_ref": "accepted_task:task-1",
            "operation": "continue",
            "instruction": None,
        })


def test_evidence_bearing_partial_is_valid() -> None:
    """Partial is accepted only when a DSH evidence receipt is present."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    validated = module.validate_task_resolution_result(
        _result(status="partial", evidence=[_evidence()]),
    )

    assert validated["status"] == "partial"
    assert validated["evidence"][0]["specialist"] == "dsh"


def test_zero_evidence_partial_is_rejected() -> None:
    """An ungrounded partial result cannot enter the cognition boundary."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    with pytest.raises(module.TaskResolutionContractError, match="partial"):
        module.validate_task_resolution_result(
            _result(status="partial", evidence=[]),
        )


def test_foreign_evidence_owner_is_rejected() -> None:
    """Only DSH may populate the post-cutover task evidence carrier."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    foreign_evidence = _evidence()
    foreign_evidence["specialist"] = "public_research"

    with pytest.raises(module.TaskResolutionContractError, match="specialist"):
        module.validate_task_resolution_result(
            _result(status="partial", evidence=[foreign_evidence]),
        )


def test_deferred_result_requires_a_complete_dsh_reference() -> None:
    """A deferred result carries only the validated opaque DSH reference."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    validated = module.validate_task_resolution_result(
        _result(
            status="deferred",
            evidence=[],
            checkpoint=_resolution_ref(),
        ),
    )

    assert validated["checkpoint"]["schema_version"] == "dsh_resolution_ref.v1"
    assert validated["coding_run_context"] == {}


def test_nonempty_coding_context_is_rejected() -> None:
    """The predecessor field remains present but cannot carry legacy state."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    result = _result(status="resolved", evidence=[_evidence()])
    result["coding_run_context"] = {"legacy": "state"}

    with pytest.raises(module.TaskResolutionContractError, match="coding"):
        module.validate_task_resolution_result(result)
