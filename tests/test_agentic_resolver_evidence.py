"""Evidence and exhaust boundary tests."""

from __future__ import annotations

import pytest

from agentic_resolver.contracts import (
    DSHResolutionExhaustV1,
    EvidenceReferenceV1,
    SubmitResolutionV1,
)


def _evidence() -> dict[str, object]:
    return {
        "schema_version": "evidence_reference.v1",
        "evidence_id": "ev_1",
        "resolution_thread_id": "res_1",
        "segment_id": "seg_1",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "policy_epoch": "2026-08-28.1",
        "tool_name": "fixture_evidence",
        "source_kind": "fixture",
        "source_id": "source_1",
        "content_digest": "sha256:content",
    }


def test_standard_tool_result_contract_binds_thread_segment_scope_and_audience() -> None:
    reference = EvidenceReferenceV1.from_mapping(_evidence())
    assert reference.resolution_thread_id == "res_1"
    assert reference.segment_id == "seg_1"
    assert reference.evidence_id == "ev_1"


def test_public_exhaust_preserves_validated_evidence_references_after_sidecar_restart() -> None:
    reference = EvidenceReferenceV1.from_mapping(_evidence())
    exhaust = DSHResolutionExhaustV1.from_terminal(
        operation_id="op_1",
        operation_payload_digest="sha256:payload",
        request_id="rrq_1",
        resolution_thread_id="res_1",
        segment_id="seg_1",
        activation_id="act_1",
        lease_epoch=1,
        scope_fingerprint="sha256:scope",
        audience_fingerprint="sha256:audience",
        resolver_profile_version="kazusa-resolver-v1",
        dsh_release="0.1.1-rc.2",
        session_store_epoch="dsh-sqlite-0.1.1-rc.2-v1",
        model_route="resolver-model",
        tool_catalog_digest="sha256:catalog",
        policy_epoch="2026-08-28.1",
        terminal=SubmitResolutionV1.from_mapping({
            "status": "resolved",
            "summary": "done",
            "findings": [],
            "completed_subgoals": [],
            "remaining_needs": [],
            "clarification_request": None,
            "approval_request": None,
            "artifact_refs": [],
            "warnings": [],
        }),
        evidence=(reference,),
        last_committed_seq=3,
    )
    restored = DSHResolutionExhaustV1.from_mapping(exhaust.to_dict())
    assert restored.to_dict() == exhaust.to_dict()
    assert restored.evidence[0].evidence_id == "ev_1"


def test_production_profile_uses_no_custom_session_event_kind() -> None:
    contracts_module = __import__("agentic_resolver.contracts", fromlist=["*"])
    assert "SUPPORTED_DSH_EVENT_KIND" not in vars(contracts_module)


def test_exhaust_rejects_evidence_with_mismatched_thread_segment_scope_or_audience() -> None:
    bad = _evidence()
    bad["resolution_thread_id"] = "other"
    reference = EvidenceReferenceV1.from_mapping(bad)
    with pytest.raises(ValueError, match="evidence"):
        DSHResolutionExhaustV1.check_evidence_bindings(
            (reference,),
            resolution_thread_id="res_1",
            segment_id="seg_1",
            scope_fingerprint="sha256:scope",
            audience_fingerprint="sha256:audience",
            policy_epoch="2026-08-28.1",
        )
