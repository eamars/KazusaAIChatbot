"""Evidence and exhaust boundary tests for the standalone V2 resolver."""

from __future__ import annotations

import pytest

from agentic_resolver.contracts import (
    DSHResolutionExhaustV2,
    EvidenceReceiptV2,
    SubmitResolutionV2,
)


def _evidence(
    evidence_id: str = "ev_1",
    source_kind: str = "semantic",
    segment_id: str = "segment-v2",
) -> dict[str, object]:
    return {
        "schema_version": "evidence_receipt.v2",
        "evidence_id": evidence_id,
        "resolution_thread_id": "thread-v2",
        "segment_id": segment_id,
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "policy_epoch": "dsh-standard-policy-v2",
        "source_kind": source_kind,
        "semantic_ref": f"semantic-{evidence_id}",
        "content_digest": f"sha256:{evidence_id}",
        "provenance": {"tool_name": source_kind},
    }


def _terminal() -> SubmitResolutionV2:
    return SubmitResolutionV2.from_mapping({
        "status": "resolved",
        "summary": "grounded",
        "findings": [],
        "completed_subgoals": [],
        "remaining_needs": [],
        "clarification_request": None,
        "approval_request": None,
        "artifact_refs": [],
        "warnings": [],
    })


def test_standard_tool_result_contract_binds_thread_segment_scope_and_audience() -> None:
    reference = EvidenceReceiptV2.from_mapping(_evidence())
    assert reference.resolution_thread_id == "thread-v2"
    assert reference.segment_id == "segment-v2"
    assert reference.evidence_id == "ev_1"


def test_public_exhaust_preserves_validated_evidence_references_after_sidecar_restart() -> None:
    references = (EvidenceReceiptV2.from_mapping(_evidence()),)
    exhaust = DSHResolutionExhaustV2.from_terminal(
        resolution_thread_id="thread-v2",
        segment_id="segment-v2",
        scope_fingerprint="sha256:scope",
        audience_fingerprint="sha256:audience",
        policy_epoch="dsh-standard-policy-v2",
        terminal=_terminal(),
        evidence=references,
        operation_id="operation-v2",
        operation_payload_digest="sha256:payload",
        request_id="request-v2",
        activation_id="activation-v2",
        lease_epoch=1,
        brain_conversation_ref="chat:debug:one",
        workspace_root="C:/workspace/project",
        route_digest="sha256:route",
        catalog_digest="sha256:catalog",
        interaction_issuer="dsh-sidecar",
    )
    restored = DSHResolutionExhaustV2.from_mapping(exhaust.to_dict())
    assert restored.to_dict() == exhaust.to_dict()
    assert restored.evidence[0].evidence_id == "ev_1"


def test_production_profile_uses_no_custom_session_event_kind() -> None:
    contracts_module = __import__("agentic_resolver.contracts", fromlist=["*"])
    assert "SUPPORTED_DSH_EVENT_KIND" not in vars(contracts_module)


def test_exhaust_rejects_evidence_with_mismatched_thread_segment_scope_or_audience() -> None:
    bad = EvidenceReceiptV2.from_mapping(_evidence(segment_id="other"))
    with pytest.raises(ValueError, match="evidence"):
        DSHResolutionExhaustV2.check_evidence_bindings(
            (bad,),
            resolution_thread_id="thread-v2",
            segment_id="segment-v2",
            scope_fingerprint="sha256:scope",
            audience_fingerprint="sha256:audience",
            policy_epoch="dsh-standard-policy-v2",
        )


def test_v2_evidence_receipts_bind_native_semantic_and_artifact_provenance() -> None:
    receipts = tuple(
        EvidenceReceiptV2.from_mapping(
            _evidence(f"{source_kind}-1", source_kind)
        )
        for source_kind in ("native", "semantic", "artifact")
    )
    terminal = SubmitResolutionV2.from_mapping({
        **_terminal().to_dict(),
        "artifact_refs": ["artifact-1"],
    })
    exhaust = DSHResolutionExhaustV2.from_terminal(
        resolution_thread_id="thread-v2",
        segment_id="segment-v2",
        scope_fingerprint="sha256:scope",
        audience_fingerprint="sha256:audience",
        policy_epoch="dsh-standard-policy-v2",
        terminal=terminal,
        evidence=receipts,
    )
    assert {item.source_kind for item in exhaust.evidence} == {
        "native", "semantic", "artifact",
    }
    assert exhaust.identity["segment_id"] == "segment-v2"
    with pytest.raises(ValueError, match="evidence authority"):
        DSHResolutionExhaustV2.from_terminal(
            resolution_thread_id="thread-v2",
            segment_id="foreign-segment",
            scope_fingerprint="sha256:scope",
            audience_fingerprint="sha256:audience",
            policy_epoch="dsh-standard-policy-v2",
            terminal=terminal,
            evidence=receipts,
        )
