"""Durable interaction repository boundary tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_interaction_store_indexes_reply_lookup_and_atomic_one_shot_grant_consumption() -> None:
    from kazusa_ai_chatbot.db.dsh_interactions import (
        INTERACTION_INDEXES,
        InMemoryInteractionStore,
    )

    assert len(INTERACTION_INDEXES) == 4
    grant_index = next(
        index
        for index in INTERACTION_INDEXES
        if index["name"] == "dsh_interaction_grant_lookup_v2"
    )
    assert ("activation_id", 1) in grant_index["keys"]
    assert ("lease_epoch", 1) in grant_index["keys"]
    store = InMemoryInteractionStore()
    await store.create({
        "schema_version": "dsh_interaction_pending.v1",
        "interaction_id": "i1",
        "request_digest": "sha256:request",
        "status": "pending",
        "grant_status": "available",
        "issuer": "dsh-sidecar",
        "nonce": "nonce-1",
        "operation_id": "operation-1",
        "operation_payload_digest": "sha256:operation",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "dsh_call_id": "call-1",
        "tool_name": "read_file",
        "arguments_digest": "sha256:args",
        "workspace_fingerprint": "sha256:workspace",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
        "brain_conversation_ref": "chat:debug:one",
        "policy_epoch": "policy",
        "issued_at": "2026-08-28T00:00:00Z",
        "expires_at": "2099-01-01T00:00:00Z",
    })
    assert not await store.consume_grant(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="stale-activation",
        lease_epoch=1,
        tool_name="read_file",
        arguments_digest="sha256:args",
        workspace_fingerprint="sha256:workspace",
        scope_fingerprint="sha256:scope",
        policy_epoch="policy",
        now="2026-08-28T00:00:00Z",
    )
    available = await store.get("i1")
    assert available is not None
    assert available["grant_status"] == "available"
    assert not await store.consume_grant(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=2,
        tool_name="read_file",
        arguments_digest="sha256:args",
        workspace_fingerprint="sha256:workspace",
        scope_fingerprint="sha256:scope",
        policy_epoch="policy",
        now="2026-08-28T00:00:00Z",
    )
    available = await store.get("i1")
    assert available is not None
    assert available["grant_status"] == "available"
    assert await store.consume_grant(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=1,
        tool_name="read_file",
        arguments_digest="sha256:args",
        workspace_fingerprint="sha256:workspace",
        scope_fingerprint="sha256:scope",
        policy_epoch="policy",
        now="2026-08-28T00:00:00Z",
    )
    assert not await store.consume_grant(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=1,
        tool_name="read_file",
        arguments_digest="sha256:args",
        workspace_fingerprint="sha256:workspace",
        scope_fingerprint="sha256:scope",
        policy_epoch="policy",
        now="2026-08-28T00:00:00Z",
    )
