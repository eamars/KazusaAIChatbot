"""Durable V2 interaction audit and grant repository tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_v2_audit_and_one_shot_grant_are_idempotent_without_reply_lookup() -> None:
    from kazusa_ai_chatbot.db.dsh_interactions import (
        DSH_INTERACTION_SCHEMA_VERSION,
        INTERACTION_INDEXES,
        InMemoryInteractionStore,
    )

    assert DSH_INTERACTION_SCHEMA_VERSION == "dsh_brain_interaction.v2"
    assert len(INTERACTION_INDEXES) == 3
    assert all(
        index["name"] != "dsh_interaction_reply_lookup"
        for index in INTERACTION_INDEXES
    )
    grant_index = next(
        index
        for index in INTERACTION_INDEXES
        if index["name"] == "dsh_interaction_grant_lookup_v2"
    )
    assert ("activation_id", 1) in grant_index["keys"]
    assert ("lease_epoch", 1) in grant_index["keys"]
    store = InMemoryInteractionStore()
    await store.create({
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": "i1",
        "request_digest": "sha256:request",
        "status": "processing",
        "grant_status": "available",
        "issuer": "dsh-sidecar",
        "nonce": "nonce-1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "tool_name": "read_file",
        "arguments_digest": "sha256:args",
        "workspace_fingerprint": "sha256:workspace",
        "scope_fingerprint": "sha256:scope",
        "policy_epoch": "policy",
        "grant": {
            "schema_version": "dsh_brain_interaction.v2",
            "interaction_id": "i1",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "activation_id": "activation-1",
            "lease_epoch": 1,
            "tool_name": "read_file",
            "arguments_digest": "sha256:args",
            "workspace_fingerprint": "sha256:workspace",
            "scope_fingerprint": "sha256:scope",
            "policy_epoch": "policy",
            "grant_status": "available",
            "issued_at": "2026-08-28T00:00:00Z",
            "expires_at": "2099-01-01T00:00:00Z",
        },
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
    consumed = await store.consume_grant(
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
    assert consumed is not None
    assert consumed["grant_status"] == "consumed"
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
