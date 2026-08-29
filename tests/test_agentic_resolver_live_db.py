"""Explicit real-Mongo resolution lifecycle test."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from agentic_resolver.persistence import MongoResolutionThreadRepository
from kazusa_ai_chatbot.db import resolution_threads


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_resolution_thread_store_enforces_operation_idempotency_lease_fencing_rotation_and_cold_resume() -> None:
    thread_id = f"live-dsh-{uuid4().hex}"
    segment_id = f"seg-{uuid4().hex}"
    session_id = f"session-{uuid4().hex}"
    workspace_root = str(Path.cwd().resolve())
    brain_conversation_ref = f"chat:live:{uuid4().hex}"
    now = "2026-08-28T00:00:00Z"
    segment = {
        "schema_version": "resolver_session_segment.v2",
        "segment_id": segment_id,
        "resolution_thread_id": thread_id,
        "dsh_session_id": session_id,
        "brain_conversation_ref": brain_conversation_ref,
        "workspace_root": workspace_root,
        "workspace_fingerprint": "sha256:workspace-live",
        "route_digest": "sha256:route-live",
        "resolver_profile_version": "kazusa-resolver-standard-v2",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
        "standard_catalog_digest": "sha256:standard-live",
        "semantic_catalog_digest": "sha256:semantic-live",
        "policy_epoch": "dsh-standard-policy-v2",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "interaction_id": f"interaction-{uuid4().hex}",
        "state": "live",
        "last_committed_seq": 0,
        "parent_segment_id": None,
        "rotation_reason": None,
        "created_at": now,
        "last_used_at": now,
    }
    await resolution_threads.ensure_indexes()
    try:
        await MongoResolutionThreadRepository().create_thread_v2(
            resolution_thread_id=thread_id,
            brain_conversation_ref=brain_conversation_ref,
            root_goal_ref="goal-live",
            priority="now",
            workspace_root=workspace_root,
            workspace_fingerprint=segment["workspace_fingerprint"],
            route_digest=segment["route_digest"],
            profile_version=segment["resolver_profile_version"],
            standard_catalog_digest=segment["standard_catalog_digest"],
            semantic_catalog_digest=segment["semantic_catalog_digest"],
            scope_fingerprint=segment["scope_fingerprint"],
            audience_fingerprint=segment["audience_fingerprint"],
            policy_epoch=segment["policy_epoch"],
            interaction_id=segment["interaction_id"],
            segment=segment,
            now=now,
        )
        first = await resolution_threads.prepare_operation(
            thread_id, "op-live", "sha256:payload", "resolution.open",
            segment["segment_id"], None, None,
        )
        second = await resolution_threads.prepare_operation(
            thread_id, "op-live", "sha256:payload", "resolution.open",
            segment["segment_id"], None, None,
        )
        assert first == second
        lease = await resolution_threads.acquire_lease(
            thread_id, "act-live", "owner-live",
            "2026-08-28T00:05:00Z", "2026-08-28T00:00:00Z",
        )
        assert lease["lease_epoch"] >= 1
        renewed = await resolution_threads.renew_lease(
            thread_id,
            "act-live",
            lease["lease_epoch"],
            "2026-08-28T00:06:00Z",
        )
        assert renewed["expires_at"] == "2026-08-28T00:06:00Z"
        updated_operation = await resolution_threads.update_operation(
            thread_id,
            "op-live",
            disposition="terminal",
            dsh_message_source_id="kazusa-operation:op-live",
            last_committed_seq=7,
            outcome_digest="sha256:outcome",
        )
        assert updated_operation["last_committed_seq"] == 7
        await resolution_threads.update_segment(
            thread_id,
            segment["segment_id"],
            state="checkpointed",
            last_committed_seq=7,
        )
        await resolution_threads.validate_fence(
            thread_id, "act-live", lease["lease_epoch"]
        )
        await resolution_threads.release_lease(
            thread_id, "act-live", lease["lease_epoch"]
        )
        rotated_segment = {
            **segment,
            "segment_id": f"seg-{uuid4().hex}",
            "dsh_session_id": f"session-{uuid4().hex}",
        }
        rotated = await resolution_threads.rotate_segment(
            thread_id, rotated_segment, reason="policy_epoch_mismatch"
        )
        assert rotated["current_segment_id"] == rotated_segment["segment_id"]
        restored = await resolution_threads.get_thread(thread_id)
        assert restored is not None
        assert restored["segments"][0]["dsh_session_id"] == segment["dsh_session_id"]
        assert restored["operations"][0]["outcome_digest"] == "sha256:outcome"
    finally:
        await resolution_threads.delete_thread(thread_id)


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_v2_gateway_abstracts_storage_and_preserves_idempotent_memory_mutations() -> None:
    """Mongo-backed interaction persistence retains one semantic mutation result."""

    from kazusa_ai_chatbot.db import dsh_interactions
    from kazusa_ai_chatbot.db._client import get_db

    interaction_id = f"live-semantic-{uuid4().hex}"
    document = {
        "schema_version": dsh_interactions.DSH_INTERACTION_SCHEMA_VERSION,
        "interaction_id": interaction_id,
        "issuer": "dsh-sidecar",
        "nonce": f"nonce-{uuid4().hex}",
        "request_digest": "sha256:request",
        "decision": {"decision": "allow_once"},
        "grant_status": "available",
        "resolution_thread_id": "thread-live",
        "segment_id": "segment-live",
        "activation_id": "activation-live",
        "lease_epoch": 1,
        "tool_name": "kazusa_remember_information",
        "arguments_digest": "sha256:args",
        "workspace_fingerprint": "sha256:workspace",
        "scope_fingerprint": "sha256:scope",
        "policy_epoch": "dsh-standard-policy-v2",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    await dsh_interactions.ensure_indexes()
    collection = (await get_db())[dsh_interactions.DSH_INTERACTIONS_COLLECTION]
    try:
        first = await dsh_interactions.create_interaction(document)
        second = await dsh_interactions.create_interaction(document)
        assert first["interaction_id"] == second["interaction_id"]
        assert first["request_digest"] == "sha256:request"
        consumed = await dsh_interactions.consume_one_shot_grant(
            resolution_thread_id="thread-live",
            segment_id="segment-live",
            activation_id="activation-live",
            lease_epoch=1,
            tool_name="kazusa_remember_information",
            arguments_digest="sha256:args",
            workspace_fingerprint="sha256:workspace",
            scope_fingerprint="sha256:scope",
            policy_epoch="dsh-standard-policy-v2",
            now="2026-08-28T00:00:00Z",
        )
        assert consumed is not None
        replay = await dsh_interactions.consume_one_shot_grant(
            resolution_thread_id="thread-live",
            segment_id="segment-live",
            activation_id="activation-live",
            lease_epoch=1,
            tool_name="kazusa_remember_information",
            arguments_digest="sha256:args",
            workspace_fingerprint="sha256:workspace",
            scope_fingerprint="sha256:scope",
            policy_epoch="dsh-standard-policy-v2",
            now="2026-08-28T00:00:00Z",
        )
        assert replay is None
    finally:
        await collection.delete_one({"interaction_id": interaction_id})


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_v2_brain_pending_and_one_shot_grant_survive_service_restart() -> None:
    """A re-created service reads pending and grant state from Mongo."""

    from kazusa_ai_chatbot.db import dsh_interactions
    from kazusa_ai_chatbot.db._client import get_db

    interaction_id = f"live-restart-{uuid4().hex}"
    document = {
        "schema_version": dsh_interactions.DSH_INTERACTION_SCHEMA_VERSION,
        "interaction_id": interaction_id,
        "issuer": "dsh-sidecar",
        "nonce": f"nonce-{uuid4().hex}",
        "request_digest": "sha256:restart-request",
        "platform": "debug",
        "platform_channel_id": "channel-live",
        "global_user_id": "user-live",
        "delivered_platform_message_id": "message-live",
        "status": "delivered",
        "grant_status": "available",
        "resolution_thread_id": "thread-restart",
        "segment_id": "segment-restart",
        "activation_id": "activation-restart",
        "lease_epoch": 1,
        "tool_name": "pwsh",
        "arguments_digest": "sha256:restart-args",
        "workspace_fingerprint": "sha256:restart-workspace",
        "scope_fingerprint": "sha256:restart-scope",
        "policy_epoch": "dsh-standard-policy-v2",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    await dsh_interactions.ensure_indexes()
    collection = (await get_db())[dsh_interactions.DSH_INTERACTIONS_COLLECTION]
    try:
        await dsh_interactions.create_interaction(document)
        restored = await dsh_interactions.get_interaction(interaction_id)
        assert restored is not None
        assert restored["delivered_platform_message_id"] == "message-live"
        consumed = await dsh_interactions.consume_one_shot_grant(
            resolution_thread_id="thread-restart",
            segment_id="segment-restart",
            activation_id="activation-restart",
            lease_epoch=1,
            tool_name="pwsh",
            arguments_digest="sha256:restart-args",
            workspace_fingerprint="sha256:restart-workspace",
            scope_fingerprint="sha256:restart-scope",
            policy_epoch="dsh-standard-policy-v2",
            now="2026-08-28T00:00:00Z",
        )
        assert consumed is not None
    finally:
        await collection.delete_one({"interaction_id": interaction_id})
