"""Brain interaction V2 contract gates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json


def _request_mapping(**overrides: object) -> dict[str, object]:
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

    now = datetime.now(UTC)
    stamp = now.isoformat().replace("+00:00", "Z")
    expiry = (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    scope = {
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
    }
    value: dict[str, object] = {
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": "interaction-1",
        "kind": "question",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "dsh_call_id": "call-1",
        "tool_name": "read_file",
        "operation_id": "operation-1",
        "operation_payload_digest": "sha256:operation",
        "arguments_digest": "sha256:args",
        "transient_detail": '{"questions":[{"id":"choice","question":"Choose."}]}',
        "brain_conversation_ref": "chat:debug:one",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
        "scope_fingerprint": content_digest(scope),
        "audience_fingerprint": content_digest({"audience": "dsh"}),
        "profile_version": "kazusa-resolver-standard-v2",
        "catalog_digest": "sha256:catalog",
        "model_route_digest": "sha256:route",
        "workspace_fingerprint": content_digest({
            "workspace_root": "C:/workspace/kazusa_ai_chatbot",
        }),
        "policy_epoch": "dsh-standard-policy-v2",
        "issued_reference_digest": "sha256:issued-refs",
        "nonce": "nonce-1",
        "issued_at": stamp,
        "expires_at": expiry,
        "issuer": "dsh-sidecar",
        "mac": "unsigned-test-mac",
    }
    value.update(overrides)
    return value


def test_public_contract_exports_v2_dtos() -> None:
    """The public interaction package exposes the canonical V2 DTO names."""

    module = __import__(
        "kazusa_ai_chatbot.dsh_interaction",
        fromlist=["__all__"],
    )
    exports = set(module.__all__)
    assert {
        "DshBrainInteractionRequestV2",
        "DshBrainInteractionDecisionV2",
        "DshOneShotGrantV2",
    } <= exports


def test_request_decision_and_grant_contracts_are_exact_and_kind_specific() -> None:
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionDecisionV2,
        DshBrainInteractionRequestV2,
        DshOneShotGrantV2,
    )

    request = DshBrainInteractionRequestV2.from_mapping(_request_mapping())
    decision = DshBrainInteractionDecisionV2.from_mapping({
        "schema_version": request.schema_version,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": "question",
        "decision": "answer",
        "answer": "Use the native workspace tool.",
        "reason": "context supports an answer",
    })
    assert decision.decision == "answer"
    assert "mac" not in request.unsigned_dict()
    expected_digest = (
        f"sha256:{sha256(canonical_json(request.unsigned_dict())).hexdigest()}"
    )
    assert request.request_digest == expected_digest
    with pytest.raises(ValueError):
        DshBrainInteractionDecisionV2.from_mapping({
            **decision.to_dict(),
            "kind": "approval",
            "decision": "answer",
        })
    with pytest.raises(ValueError, match="unknown fields"):
        DshBrainInteractionDecisionV2.from_mapping({
            **decision.to_dict(),
            "relay_mode": "question",
        })
    grant_mapping = {
        "schema_version": request.schema_version,
        "interaction_id": request.interaction_id,
        "resolution_thread_id": request.resolution_thread_id,
        "segment_id": request.segment_id,
        "activation_id": request.activation_id,
        "lease_epoch": request.lease_epoch,
        "tool_name": "pwsh",
        "arguments_digest": request.arguments_digest,
        "workspace_fingerprint": request.workspace_fingerprint,
        "scope_fingerprint": request.scope_fingerprint,
        "policy_epoch": request.policy_epoch,
        "grant_status": "available",
        "issued_at": request.issued_at,
        "expires_at": request.expires_at,
    }
    grant = DshOneShotGrantV2.from_mapping(grant_mapping)
    assert grant.activation_id == request.activation_id
    assert grant.lease_epoch == request.lease_epoch
    with pytest.raises(ValueError, match="missing fields"):
        DshOneShotGrantV2.from_mapping({
            key: value
            for key, value in grant_mapping.items()
            if key != "activation_id"
        })
