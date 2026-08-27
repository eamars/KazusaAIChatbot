"""Deterministic tests for the canonical standalone resolution contracts."""

from __future__ import annotations

import inspect

import pytest

import agentic_resolver
from agentic_resolver.contracts import (
    DSHResolutionExhaustV1,
    DSHResolutionIntakeV1,
    DSHResolutionRuntimeV1,
    ResolutionThreadRecordV1,
    SubmitResolutionV1,
)
from agentic_resolver.errors import RuntimeFaultCode


def _runtime() -> dict[str, object]:
    """Return one valid runtime-only authority object."""

    return {
        "request_id": "rrq_1",
        "operation_id": "op_1",
        "operation_payload_digest": "sha256:payload",
        "resolution_thread_id": "res_1",
        "segment_id": "seg_1",
        "priority": "now",
        "soft_deadline_at": "2026-08-28T00:00:10Z",
        "hard_deadline_at": "2026-08-28T00:00:30Z",
        "max_model_steps": 4,
        "max_tool_calls": 4,
        "max_tool_bytes": 10000,
        "capability_token": "opaque-token",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "resolver_profile_version": "kazusa-resolver-v1",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-v1",
        "model_route": "resolver-model",
        "tool_catalog_digest": "sha256:catalog",
        "policy_epoch": "2026-08-28.1",
    }


def _intake() -> dict[str, object]:
    """Return one valid strict intake object."""

    return {
        "schema_version": "dsh_resolution_intake.v1",
        "mode": "start",
        "runtime": _runtime(),
        "model_input": {
            "objective": "Determine whether the bounded task is complete.",
            "constraints": ["Use only the registered resolver actions."],
            "success_criteria": ["Return a source-grounded terminal product."],
            "known_facts": ["The caller supplied one objective."],
            "uncertainty": [],
            "literal_inputs": [],
            "continuation_delta": None,
            "prior_resolution_refs": [],
            "requested_evidence_quality": "normal",
            "notes": [],
        },
    }


def _submit(status: str = "resolved") -> dict[str, object]:
    """Return a valid status-specific terminal payload."""

    return {
        "status": status,
        "summary": "The bounded objective is complete.",
        "findings": [],
        "completed_subgoals": ["Inspect the objective."],
        "remaining_needs": [],
        "clarification_request": None,
        "approval_request": None,
        "artifact_refs": [],
        "warnings": [],
    }


def test_intake_v1_rejects_unknown_fields_and_keeps_runtime_out_of_model_input() -> None:
    """Strict intake parsing separates deterministic authority from model text."""

    value = _intake()
    intake = DSHResolutionIntakeV1.from_mapping(value)
    assert intake.to_dict() == value
    assert "capability_token" not in intake.model_input.to_dict()

    value["runtime"] = {**_runtime(), "unknown": True}
    with pytest.raises(ValueError, match="unknown"):
        DSHResolutionIntakeV1.from_mapping(value)


def test_thread_segment_operation_activation_lease_and_store_epoch_validate() -> None:
    """Thread records require identity, compatibility, and monotonic lease data."""

    record = ResolutionThreadRecordV1.from_mapping({
        "schema_version": "resolution_thread_store.v1",
        "resolution_thread_id": "res_1",
        "brain_conversation_ref": "conv_1",
        "root_goal_ref": "goal_1",
        "current_segment_id": "seg_1",
        "state": "active",
        "priority": "now",
        "audience_fingerprint": "sha256:audience",
        "scope_fingerprint": "sha256:scope",
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z",
        "last_terminal_status": None,
        "continuation_eligible_until": "2026-08-29T00:00:00Z",
        "document_revision": 2,
        "lease_epoch": 3,
        "current_lease": {
            "activation_id": "act_1",
            "lease_epoch": 3,
            "owner_id": "worker_1",
            "expires_at": "2026-08-28T00:01:00Z",
        },
        "segments": [{
            "schema_version": "resolver_session_segment.v1",
            "segment_id": "seg_1",
            "resolution_thread_id": "res_1",
            "dsh_session_id": "dsh_1",
            "resolver_profile_version": "kazusa-resolver-v1",
            "dsh_release": "0.1.1-rc.2",
            "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-v1",
            "tool_catalog_digest": "sha256:catalog",
            "policy_epoch": "2026-08-28.1",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
            "model_route": "resolver-model",
            "state": "live",
            "last_committed_seq": 4,
            "parent_segment_id": None,
            "rotation_reason": None,
            "created_at": "2026-08-28T00:00:00Z",
            "last_used_at": "2026-08-28T00:00:00Z",
        }],
        "operations": [],
    })
    assert record.lease_epoch == 3
    assert record.segments[0]["session_store_epoch"] == (
        "dsh-sqlite-0.1.1-rc.2-v1"
    )


def test_submit_resolution_requires_status_specific_fields() -> None:
    """Clarification and approval statuses require their request objects."""

    resolved = SubmitResolutionV1.from_mapping(_submit())
    assert resolved.status == "resolved"

    needs_input = _submit("needs_user_input")
    with pytest.raises(ValueError, match="clarification_request"):
        SubmitResolutionV1.from_mapping(needs_input)
    needs_input["clarification_request"] = {
        "question_id": "q_1",
        "reason": "A required fact is missing.",
        "question": "Which time should be used?",
        "answer_type": "date_time",
        "choices": [],
        "required": True,
    }
    assert SubmitResolutionV1.from_mapping(needs_input).status == (
        "needs_user_input"
    )

    approval = _submit("approval_required")
    with pytest.raises(ValueError, match="approval_request"):
        SubmitResolutionV1.from_mapping(approval)


def test_public_contracts_expose_no_dsh_event_or_receipt_types() -> None:
    """Python exposes versioned DTOs without importing DSH event internals."""

    contracts_module = __import__("agentic_resolver.contracts", fromlist=["*"])
    names = set(vars(contracts_module))
    assert not any("Receipt" in name or "Event" in name for name in names)
    assert "runtime" not in DSHResolutionIntakeV1.from_mapping(
        _intake()
    ).model_input.to_dict()
    assert DSHResolutionExhaustV1.KINDS == {
        "terminal",
        "checkpointed",
        "runtime_fault",
    }


def test_typed_operation_fence_and_runtime_fault_codes_are_closed() -> None:
    """Runtime faults and live-control fences use closed machine codes."""

    assert RuntimeFaultCode.OPERATION_OUTCOME_UNCERTAIN.value == (
        "OPERATION_OUTCOME_UNCERTAIN"
    )
    assert RuntimeFaultCode.STALE_ACTIVATION_OR_LEASE.value == (
        "STALE_ACTIVATION_OR_LEASE"
    )
    assert set(DSHResolutionRuntimeV1.PRIORITIES) == {"now", "background"}
    assert inspect.isclass(DSHResolutionExhaustV1)


def test_public_api_exposes_standalone_runtime_only() -> None:
    """The package facade does not expose Brain or workflow contracts."""

    assert "AgenticResolverRuntime" in agentic_resolver.__all__
    assert "brain_service" not in agentic_resolver.__dict__
    assert "cognition" not in agentic_resolver.__dict__
