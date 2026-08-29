"""Same-thread relay resume tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_user_resolution_schedules_one_same_thread_continuation_and_one_shot_matching_grant() -> None:
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.resume import InteractionResumer

    calls = []

    async def continue_resolution(**kwargs):
        calls.append(kwargs)
        return {"status": "continued"}

    async def issue_continuation_authority(**kwargs):
        assert kwargs["resolution_thread_id"] == "thread-1"
        assert kwargs["segment_id"] == "segment-1"
        assert kwargs["reply_decision"]["decision"] == "allow_once"
        return "ksa1-canonical-continuation-token"

    store = InMemoryInteractionStore()
    resumer = InteractionResumer(
        interaction_store=store,
        continue_resolution=continue_resolution,
        issue_continuation_authority=issue_continuation_authority,
    )
    grant = resumer.issue_grant(
        interaction_id="i1",
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=1,
        tool_name="read_file",
        arguments_digest="sha256:args",
        workspace_fingerprint="sha256:workspace",
        scope_fingerprint="sha256:scope",
        policy_epoch="policy",
    )
    await store.create({
        "schema_version": "dsh_interaction_pending.v1",
        "interaction_id": grant.interaction_id,
        "request_digest": "sha256:request",
        "grant_status": grant.grant_status,
        "resolution_thread_id": grant.resolution_thread_id,
        "segment_id": grant.segment_id,
        "activation_id": grant.activation_id,
        "lease_epoch": grant.lease_epoch,
        "tool_name": grant.tool_name,
        "arguments_digest": grant.arguments_digest,
        "workspace_fingerprint": grant.workspace_fingerprint,
        "scope_fingerprint": grant.scope_fingerprint,
        "policy_epoch": grant.policy_epoch,
        "expires_at": grant.expires_at,
    })
    result = await resumer.resume(
        grant=grant,
        reply_decision={"decision": "allow_once"},
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=1,
    )
    assert result["status"] == "continued"
    assert calls[0]["resolution_thread_id"] == "thread-1"
    assert calls[0]["continuation_authority_token"] == (
        "ksa1-canonical-continuation-token"
    )
    with pytest.raises(ValueError):
        await resumer.resume(
            grant=grant,
            reply_decision={"decision": "allow_once"},
            resolution_thread_id="thread-1",
            segment_id="segment-1",
            activation_id="activation-1",
            lease_epoch=1,
        )
