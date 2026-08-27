"""Explicit real-Mongo resolution lifecycle test."""

from __future__ import annotations

from uuid import uuid4

import pytest

from kazusa_ai_chatbot.db import resolution_threads


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_resolution_thread_store_enforces_operation_idempotency_lease_fencing_rotation_and_cold_resume() -> None:
    thread_id = f"live-dsh-{uuid4().hex}"
    segment = {
        "schema_version": "resolver_session_segment.v1",
        "segment_id": f"seg-{uuid4().hex}",
        "resolution_thread_id": thread_id,
        "dsh_session_id": f"session-{uuid4().hex}",
        "resolver_profile_version": "kazusa-resolver-v1",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-v1",
        "tool_catalog_digest": "sha256:catalog",
        "policy_epoch": "live-test",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "model_route": "resolver-model",
        "state": "live",
        "last_committed_seq": 0,
        "parent_segment_id": None,
        "rotation_reason": None,
        "created_at": "2026-08-28T00:00:00Z",
        "last_used_at": "2026-08-28T00:00:00Z",
    }
    await resolution_threads.ensure_indexes()
    try:
        await resolution_threads.create_thread(
            thread_id, "conv-live", "goal-live", "now",
            "sha256:scope", "sha256:audience", segment,
            "2026-08-28T00:00:00Z",
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
