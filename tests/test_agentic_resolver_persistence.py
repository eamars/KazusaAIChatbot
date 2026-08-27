"""Resolution thread repository behavior tests."""

from __future__ import annotations

import pytest

from agentic_resolver.errors import (
    OperationIdReuseMismatchError,
    ResolutionPersistenceError,
)
from agentic_resolver.persistence import InMemoryResolutionThreadRepository


def _segment(segment_id: str = "seg_1") -> dict[str, object]:
    return {
        "schema_version": "resolver_session_segment.v1",
        "segment_id": segment_id,
        "resolution_thread_id": "res_1",
        "dsh_session_id": "dsh_1",
        "resolver_profile_version": "kazusa-resolver-v1",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-v1",
        "tool_catalog_digest": "sha256:catalog",
        "policy_epoch": "2026-08-28.1",
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


def _repository() -> InMemoryResolutionThreadRepository:
    repository = InMemoryResolutionThreadRepository()
    repository.create_thread(
        resolution_thread_id="res_1",
        brain_conversation_ref="conv_1",
        root_goal_ref="goal_1",
        priority="now",
        scope_fingerprint="sha256:scope",
        audience_fingerprint="sha256:audience",
        segment=_segment(),
        now="2026-08-28T00:00:00Z",
    )
    return repository


def test_thread_segment_operation_lease_epoch_and_store_epoch_round_trip() -> None:
    repository = _repository()
    lease = repository.acquire_lease(
        "res_1", activation_id="act_1", owner_id="worker_1",
        expires_at="2026-08-28T00:01:00Z", now="2026-08-28T00:00:00Z",
    )
    repository.prepare_operation(
        "res_1", "op_1", "sha256:payload", "resolution.open",
        "seg_1", activation_id="act_1", lease_epoch=lease["lease_epoch"],
    )
    record = repository.get_thread("res_1")
    assert record is not None
    assert record.lease_epoch == 1
    assert record.segments[0]["session_store_epoch"] == "dsh-sqlite-0.1.1-rc.2-v1"
    assert record.operations[0]["operation_id"] == "op_1"


def test_operation_admission_is_idempotent_only_for_matching_digest() -> None:
    repository = _repository()
    first = repository.prepare_operation(
        "res_1", "op_1", "sha256:payload", "resolution.open", "seg_1"
    )
    second = repository.prepare_operation(
        "res_1", "op_1", "sha256:payload", "resolution.open", "seg_1"
    )
    assert first == second
    with pytest.raises(OperationIdReuseMismatchError):
        repository.prepare_operation(
            "res_1", "op_1", "sha256:other", "resolution.open", "seg_1"
        )


def test_cold_resume_uses_persisted_session_reference_and_revision() -> None:
    repository = _repository()
    before = repository.get_thread("res_1")
    repository.update_segment(
        "res_1", "seg_1", dsh_session_id="dsh_persisted", last_committed_seq=7,
    )
    after = repository.get_thread("res_1")
    assert after.document_revision > before.document_revision
    assert after.segments[0]["dsh_session_id"] == "dsh_persisted"
    assert after.segments[0]["last_committed_seq"] == 7


def test_expired_or_corrupt_segment_fails_closed_or_rotates() -> None:
    repository = _repository()
    with pytest.raises(ResolutionPersistenceError):
        repository.update_segment("res_1", "missing", state="faulted")
    rotated = repository.rotate_segment(
        "res_1", _segment("seg_2"), reason="session_store_epoch_mismatch"
    )
    assert rotated.current_segment_id == "seg_2"
    assert rotated.segments[-1]["rotation_reason"] == "session_store_epoch_mismatch"
