"""Durable V2 resolution-thread repository behavior tests."""

from __future__ import annotations

import pytest

from agentic_resolver.errors import (
    OperationIdReuseMismatchError,
    ResolutionPersistenceError,
)
from agentic_resolver.persistence import InMemoryResolutionThreadRepository


def _repository() -> InMemoryResolutionThreadRepository:
    repository = InMemoryResolutionThreadRepository()
    repository.create_thread_v2(
        resolution_thread_id="thread-v2",
        brain_conversation_ref="chat:debug:one",
        root_goal_ref="goal-v2",
        priority="now",
        workspace_root="C:/workspace/project",
        workspace_fingerprint="sha256:workspace",
        route_digest="sha256:route",
        profile_version="kazusa-resolver-standard-v2",
        standard_catalog_digest="sha256:standard",
        semantic_catalog_digest="sha256:semantic",
        scope_fingerprint="sha256:scope",
        audience_fingerprint="sha256:audience",
        policy_epoch="dsh-standard-policy-v2",
        interaction_id="interaction-v2",
        segment={
            "schema_version": "resolver_session_segment.v2",
            "segment_id": "segment-v2",
            "resolution_thread_id": "thread-v2",
            "dsh_session_id": "session-v2",
            "brain_conversation_ref": "chat:debug:one",
            "workspace_root": "C:/workspace/project",
            "workspace_fingerprint": "sha256:workspace",
            "route_digest": "sha256:route",
            "resolver_profile_version": "kazusa-resolver-standard-v2",
            "dsh_release": "0.1.1-rc.2",
            "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
            "standard_catalog_digest": "sha256:standard",
            "semantic_catalog_digest": "sha256:semantic",
            "policy_epoch": "dsh-standard-policy-v2",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
            "interaction_id": "interaction-v2",
            "state": "live",
            "last_committed_seq": 0,
            "parent_segment_id": None,
            "rotation_reason": None,
            "created_at": "2026-08-28T00:00:00Z",
            "last_used_at": "2026-08-28T00:00:00Z",
        },
        now="2026-08-28T00:00:00Z",
    )
    return repository


def test_thread_segment_operation_lease_epoch_and_store_epoch_round_trip() -> None:
    repository = _repository()
    lease = repository.acquire_lease(
        "thread-v2",
        activation_id="activation-v2",
        owner_id="worker-v2",
        expires_at="2026-08-28T00:01:00Z",
        now="2026-08-28T00:00:00Z",
    )
    repository.prepare_operation(
        "thread-v2",
        "operation-v2",
        "sha256:payload",
        "resolution.open",
        "segment-v2",
        activation_id="activation-v2",
        lease_epoch=lease["lease_epoch"],
    )
    record = repository.get_thread("thread-v2")
    assert record is not None
    assert record.lease_epoch == 1
    assert record.segments[0]["session_store_epoch"] == (
        "dsh-sqlite-0.1.1-rc.2-standard-v2"
    )
    assert record.brain_conversation_ref == "chat:debug:one"


def test_operation_admission_is_idempotent_only_for_matching_digest() -> None:
    repository = _repository()
    first = repository.prepare_operation(
        "thread-v2", "operation-v2", "sha256:payload", "resolution.open", "segment-v2"
    )
    second = repository.prepare_operation(
        "thread-v2", "operation-v2", "sha256:payload", "resolution.open", "segment-v2"
    )
    assert first == second
    with pytest.raises(OperationIdReuseMismatchError):
        repository.prepare_operation(
            "thread-v2", "operation-v2", "sha256:other", "resolution.open", "segment-v2"
        )


def test_cold_resume_uses_persisted_session_reference_and_revision() -> None:
    repository = _repository()
    before = repository.get_thread("thread-v2")
    repository.update_segment(
        "thread-v2", "segment-v2", dsh_session_id="session-persisted", last_committed_seq=7,
    )
    after = repository.get_thread("thread-v2")
    assert before is not None and after is not None
    assert after.document_revision > before.document_revision
    assert after.segments[0]["dsh_session_id"] == "session-persisted"
    assert after.segments[0]["last_committed_seq"] == 7


def test_expired_or_corrupt_segment_fails_closed_or_rotates() -> None:
    repository = _repository()
    with pytest.raises(ResolutionPersistenceError):
        repository.update_segment("thread-v2", "missing", state="faulted")
    current = repository.get_thread("thread-v2")
    assert current is not None
    next_segment = dict(current.segments[0])
    next_segment.update({
        "segment_id": "segment-v2b",
        "dsh_session_id": "session-v2b",
    })
    rotated = repository.rotate_segment(
        "thread-v2", next_segment, reason="route_digest_mismatch"
    )
    assert rotated.current_segment_id == "segment-v2b"
    assert rotated.segments[-1]["rotation_reason"] == "route_digest_mismatch"


def test_v2_thread_persists_brain_workspace_route_and_interaction_identity() -> None:
    repository = _repository()
    record = repository.get_thread("thread-v2")
    assert record is not None
    assert record.schema_version == "resolution_thread_store.v2"
    assert record.brain_conversation_ref == "chat:debug:one"
    assert record.workspace_root == "C:/workspace/project"
    assert record.route_digest == "sha256:route"
    assert record.interaction_id == "interaction-v2"


def test_v1_rows_are_historical_and_never_resumed_as_v2() -> None:
    repository = InMemoryResolutionThreadRepository()
    repository.seed_historical({
        "schema_version": "resolution_thread_store.v1",
        "resolution_thread_id": "historical-v1",
        "state": "checkpointed",
    })
    assert repository.get_thread("historical-v1") is None
    with pytest.raises(ResolutionPersistenceError, match="historical"):
        repository.resume_v2("historical-v1")
