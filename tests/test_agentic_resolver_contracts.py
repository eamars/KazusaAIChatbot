"""Deterministic tests for the canonical standalone V2 resolver contracts."""

from __future__ import annotations

import inspect

import pytest

import agentic_resolver
from agentic_resolver.contracts import (
    DSHResolutionExhaustV2,
    DSHResolutionIntakeV2,
    DSHResolutionRuntimeV2,
    ResolutionThreadRecordV2,
    SubmitResolutionV2,
)
from agentic_resolver.errors import InteractionFaultCode, RuntimeFaultCode


def _intake() -> dict[str, object]:
    return {
        "schema_version": "dsh_resolution_intake.v2",
        "mode": "start",
        "request_id": "request-1",
        "operation_id": "operation-1",
        "operation_payload_digest": "sha256:payload",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "brain_conversation_ref": "chat:debug:one",
        "workspace_root": "C:/workspace/project",
        "route_digest": "sha256:route",
        "model_input": {"objective": "inspect the project", "facts": []},
        "semantic_tool_authority": {
            "catalog_digest": "sha256:catalog",
            "token": "opaque",
        },
        "interaction_authority": {
            "issuer": "dsh-sidecar",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
        },
    }


def _submit(status: str = "resolved") -> dict[str, object]:
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


def _segment() -> dict[str, object]:
    return {
        "schema_version": "resolver_session_segment.v2",
        "segment_id": "segment-1",
        "resolution_thread_id": "thread-1",
        "dsh_session_id": "session-1",
        "brain_conversation_ref": "chat:debug:one",
        "workspace_root": "C:/workspace/project",
        "workspace_fingerprint": "sha256:workspace",
        "route_digest": "sha256:route",
        "resolver_profile_version": "kazusa-resolver-standard-v2",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
        "standard_catalog_digest": "sha256:catalog",
        "semantic_catalog_digest": "sha256:catalog",
        "policy_epoch": "dsh-standard-policy-v2",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "interaction_id": "request-1",
        "state": "live",
        "last_committed_seq": 4,
        "parent_segment_id": None,
        "rotation_reason": None,
        "created_at": "2026-08-28T00:00:00Z",
        "last_used_at": "2026-08-28T00:00:00Z",
    }


def _thread() -> dict[str, object]:
    return {
        "schema_version": "resolution_thread_store.v2",
        "resolution_thread_id": "thread-1",
        "brain_conversation_ref": "chat:debug:one",
        "root_goal_ref": "inspect the project",
        "current_segment_id": "segment-1",
        "state": "active",
        "priority": "now",
        "workspace_root": "C:/workspace/project",
        "workspace_fingerprint": "sha256:workspace",
        "route_digest": "sha256:route",
        "profile_version": "kazusa-resolver-standard-v2",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
        "standard_catalog_digest": "sha256:catalog",
        "semantic_catalog_digest": "sha256:catalog",
        "policy_epoch": "dsh-standard-policy-v2",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "interaction_id": "request-1",
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z",
        "last_terminal_status": None,
        "continuation_eligible_until": "2026-08-29T00:00:00Z",
        "document_revision": 2,
        "lease_epoch": 3,
        "current_lease": {
            "activation_id": "activation-1",
            "lease_epoch": 3,
            "owner_id": "worker-1",
            "expires_at": "2026-08-28T00:01:00Z",
        },
        "segments": [_segment()],
        "operations": [],
    }


def test_intake_v2_rejects_unknown_fields_and_keeps_authority_out_of_model_input() -> None:
    intake = DSHResolutionIntakeV2.from_mapping(_intake())
    assert intake.to_dict() == _intake()
    assert intake.model_visible_input == {
        "objective": "inspect the project",
        "facts": [],
    }
    assert "workspace_root" not in intake.model_visible_input
    assert "semantic_tool_authority" not in intake.model_visible_input
    invalid = {**_intake(), "unknown": True}
    with pytest.raises(ValueError, match="unknown"):
        DSHResolutionIntakeV2.from_mapping(invalid)


def test_v2_intake_requires_canonical_absolute_workspace_and_all_authority_fields() -> None:
    for field in ("brain_conversation_ref", "workspace_root", "route_digest"):
        invalid = _intake()
        invalid[field] = ""
        with pytest.raises(ValueError):
            DSHResolutionIntakeV2.from_mapping(invalid)
    invalid = {**_intake(), "workspace_root": "relative/project"}
    with pytest.raises(ValueError, match="absolute"):
        DSHResolutionIntakeV2.from_mapping(invalid)
    invalid = {**_intake(), "workspace_root": "C:/workspace/../project"}
    with pytest.raises(ValueError, match="canonical"):
        DSHResolutionIntakeV2.from_mapping(invalid)
    missing_audience = _intake()
    del missing_audience["interaction_authority"]["audience_fingerprint"]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing"):
        DSHResolutionIntakeV2.from_mapping(missing_audience)


def test_thread_segment_operation_activation_lease_and_store_epoch_validate() -> None:
    record = ResolutionThreadRecordV2.from_mapping(_thread())
    assert record.lease_epoch == 3
    assert record.segments[0]["session_store_epoch"] == (
        "dsh-sqlite-0.1.1-rc.2-standard-v2"
    )


def test_submit_resolution_requires_status_specific_fields() -> None:
    resolved = SubmitResolutionV2.from_mapping(_submit())
    assert resolved.status == "resolved"
    needs_input = _submit("needs_user_input")
    with pytest.raises(ValueError, match="clarification_request"):
        SubmitResolutionV2.from_mapping(needs_input)
    needs_input["clarification_request"] = {
        "question_id": "q_1",
        "question": "Which time should be used?",
    }
    assert SubmitResolutionV2.from_mapping(needs_input).status == (
        "needs_user_input"
    )
    approval = _submit("approval_required")
    with pytest.raises(ValueError, match="approval_request"):
        SubmitResolutionV2.from_mapping(approval)


def test_public_contracts_expose_no_dsh_event_or_receipt_types() -> None:
    contracts_module = __import__("agentic_resolver.contracts", fromlist=["*"])
    names = set(vars(contracts_module))
    assert not any(name.endswith("Event") for name in names)
    assert "runtime" not in DSHResolutionIntakeV2.from_mapping(
        _intake()
    ).model_visible_input
    assert DSHResolutionExhaustV2.KINDS == {
        "terminal", "checkpointed", "runtime_fault", "canceled"
    }


def test_typed_operation_fence_and_runtime_fault_codes_are_closed() -> None:
    assert RuntimeFaultCode.OPERATION_OUTCOME_UNCERTAIN.value == (
        "OPERATION_OUTCOME_UNCERTAIN"
    )
    assert RuntimeFaultCode.STALE_ACTIVATION_OR_LEASE.value == (
        "STALE_ACTIVATION_OR_LEASE"
    )
    assert inspect.isclass(DSHResolutionExhaustV2)
    assert inspect.isclass(DSHResolutionRuntimeV2)


def test_public_api_exposes_standalone_runtime_only() -> None:
    assert "AgenticResolverRuntime" in agentic_resolver.__all__
    assert "brain_service" not in agentic_resolver.__dict__
    assert "cognition" not in agentic_resolver.__dict__


def test_public_resolver_exports_only_v2_product_contracts() -> None:
    assert "DSHResolutionIntakeV2" in agentic_resolver.__all__
    assert "DSHResolutionRuntimeV2" in agentic_resolver.__all__
    assert "DSHResolutionExhaustV2" in agentic_resolver.__all__
    assert "SubmitResolutionV2" in agentic_resolver.__all__
    assert not any(name.endswith("V1") for name in agentic_resolver.__all__)


def test_v2_intake_separates_model_input_from_workspace_tool_and_brain_authority() -> None:
    intake = DSHResolutionIntakeV2.from_mapping(_intake())
    assert intake.schema_version == "dsh_resolution_intake.v2"
    assert intake.model_input == {"objective": "inspect the project", "facts": []}
    assert intake.workspace_root == "C:/workspace/project"
    assert intake.semantic_tool_authority["token"] == "opaque"
    assert intake.interaction_authority["issuer"] == "dsh-sidecar"
    assert "workspace_root" not in intake.model_visible_input
    assert "semantic_tool_authority" not in intake.model_visible_input


def test_v2_runtime_and_interaction_fault_codes_are_closed() -> None:
    assert RuntimeFaultCode.RPC_CONTRACT_ERROR.value == "RPC_CONTRACT_ERROR"
    assert RuntimeFaultCode.OPERATION_OUTCOME_UNCERTAIN.value == (
        "OPERATION_OUTCOME_UNCERTAIN"
    )
    assert InteractionFaultCode.AUTHENTICATION_FAILED.value == (
        "BRAIN_INTERACTION_AUTHENTICATION_FAILED"
    )
    assert InteractionFaultCode.REPLAY.value == "BRAIN_INTERACTION_REPLAY"
    assert set(InteractionFaultCode) == {
        InteractionFaultCode.AUTHENTICATION_FAILED,
        InteractionFaultCode.REPLAY,
        InteractionFaultCode.EXPIRED,
        InteractionFaultCode.UNAVAILABLE,
        InteractionFaultCode.IDENTITY_INVALID,
    }
