"""Authority and reference-lineage tests for semantic calls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


def _authority():
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        SemanticActivationAuthorityV1,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

    now = datetime.now(UTC)
    service_scope = {
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
    }
    workspace_root = "C:/workspace/project"
    return SemanticActivationAuthorityV1(
        activation_id="act_1",
        lease_epoch=3,
        resolution_thread_id="thread_1",
        segment_id="segment_1",
        brain_conversation_ref="chat:debug:one",
        service_scope=service_scope,
        scope_fingerprint=content_digest(service_scope),
        audience_fingerprint="sha256:audience",
        workspace_root=workspace_root,
        route_digest="sha256:route",
        catalog_digest="sha256:catalog",
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest="sha256:route",
        workspace_fingerprint=content_digest({"workspace_root": workspace_root}),
        issued_reference_digest="sha256:issued-references",
        policy_epoch="kazusa-resolver-standard-v2",
        interaction_issuer="dsh-sidecar-test",
        issued_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        token_id="tok_1",
        nonce="nonce_1",
    )


def test_activation_authenticates_complete_catalog_scope_fence_reference_lineage_and_replay() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        InMemorySemanticAuthorityReplayOwner,
        issue_semantic_call,
        verify_semantic_call,
    )

    authority = _authority()
    secret = b"semantic-secret"
    call = issue_semantic_call(
        authority,
        operation="kazusa_search_memories",
        arguments={"query": "hello", "max_results": 2},
        secret=secret,
        call_id="call_1",
        now=authority.issued_at,
    )
    replay = InMemorySemanticAuthorityReplayOwner()
    verified = verify_semantic_call(call, secret=secret, replay_owner=replay, now=authority.issued_at)
    assert verified.operation == "kazusa_search_memories"
    with pytest.raises(ValueError):
        verify_semantic_call(call, secret=secret, replay_owner=replay, now=authority.issued_at)
    restarted_replay = InMemorySemanticAuthorityReplayOwner(replay.snapshot())
    with pytest.raises(ValueError):
        verify_semantic_call(
            call,
            secret=secret,
            replay_owner=restarted_replay,
            now=authority.issued_at,
        )
    with pytest.raises(ValueError):
        issue_semantic_call(
            authority,
            operation="kazusa_unknown",
            arguments={},
            secret=secret,
            call_id="call_2",
            now=authority.issued_at,
        )


def test_activation_token_is_canonical_hmac_bound_and_survives_codec_recreation() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        issue_activation_token,
        verify_activation_token,
    )

    authority = _authority()
    secret = b"semantic-secret"
    token = issue_activation_token(authority, secret=secret, now=authority.issued_at)
    decoded = verify_activation_token(
        token,
        secret=secret,
        expected={
            "resolution_thread_id": authority.resolution_thread_id,
            "segment_id": authority.segment_id,
            "service_scope": authority.service_scope,
            "workspace_root": authority.workspace_root,
            "route_digest": authority.route_digest,
        },
        now=authority.issued_at,
    )
    assert decoded.to_dict() == authority.to_dict()
    with pytest.raises(ValueError):
        verify_activation_token(
            token,
            secret=b"wrong-secret",
            now=authority.issued_at,
        )
    with pytest.raises(ValueError):
        verify_activation_token(
            token,
            secret=secret,
            expected={"segment_id": "other-segment"},
            now=authority.issued_at,
        )


def test_mutation_idempotency_excludes_transport_call_id() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import issue_semantic_call

    authority = _authority()
    first = issue_semantic_call(
        authority,
        operation="kazusa_remember_information",
        arguments={
            "subject": "current_user",
            "information": "A semantic fact.",
            "memory_kind": "profile_fact",
            "reason": "test",
            "provenance": {"current_task": "authority-test"},
        },
        secret=b"semantic-secret",
        call_id="transport-1",
        now=authority.issued_at,
    )
    second = issue_semantic_call(
        authority,
        operation="kazusa_remember_information",
        arguments=first.arguments,
        secret=b"semantic-secret",
        call_id="transport-2",
        now=authority.issued_at,
    )
    assert first.idempotency_key == second.idempotency_key


def test_durable_replay_owner_rejects_duplicate_complete_identity(tmp_path) -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.worker import SQLiteSemanticOutcomeOwner

    owner = SQLiteSemanticOutcomeOwner(tmp_path / "semantic-outcomes.sqlite")
    authority = _authority()
    owner.consume(authority, "call-replay")
    with pytest.raises(ValueError, match="replayed"):
        owner.consume(authority, "call-replay")
