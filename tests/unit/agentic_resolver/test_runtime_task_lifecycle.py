"""Executable gates for the generic DSH runtime and controller lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pymongo.errors import AutoReconnect

from agentic_resolver.errors import ResolutionPersistenceError
from agentic_resolver.fingerprints import workspace_fingerprint
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest


class _TransientRepositoryOwner:
    """Database owner double for persistence taxonomy and retry tests."""

    def __init__(self) -> None:
        self.index_calls = 0

    async def ensure_indexes(self) -> None:
        self.index_calls += 1
        if self.index_calls < 3:
            try:
                raise AutoReconnect("connection closed")
            except AutoReconnect as exc:
                raise DatabaseOperationError("index failure") from exc

    async def get_operation(self, *args: object) -> None:
        del args
        raise DatabaseOperationError("read failure")


class _RuntimeController:
    """Record the public lifecycle calls made by the runtime facade."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def request_checkpoint(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(("request_checkpoint", args, kwargs))
        return {
            "disposition": "checkpointed",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "lease_epoch": 7,
        }

    async def continue_after_terminal(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        self.calls.append(("continue_after_terminal", args, kwargs))
        return {
            "disposition": "terminal",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "lease_epoch": 8,
            "fresh_authority": True,
        }


@pytest.mark.asyncio
async def test_mongo_repository_retries_transient_index_creation_once() -> None:
    """Transient index connection loss is bounded and cached after recovery."""

    from agentic_resolver.persistence import MongoResolutionThreadRepository

    owner = _TransientRepositoryOwner()
    repository = MongoResolutionThreadRepository()
    repository._db = owner

    await repository.ensure_indexes()
    await repository.ensure_indexes()

    assert owner.index_calls == 3


@pytest.mark.asyncio
async def test_mongo_repository_translates_database_owner_errors() -> None:
    """Raw database errors cannot escape the resolver repository boundary."""

    from agentic_resolver.persistence import MongoResolutionThreadRepository

    repository = MongoResolutionThreadRepository()
    repository._db = _TransientRepositoryOwner()

    with pytest.raises(
        ResolutionPersistenceError,
        match="get_operation",
    ):
        await repository.get_operation("thread-1", "operation-1")


@pytest.mark.asyncio
async def test_checkpoint_and_terminal_continuation_issue_fresh_authority_and_preserve_thread_segment() -> None:
    """Runtime wrappers delegate fenced operations without opening a thread."""

    from agentic_resolver.runtime import AgenticResolverRuntime

    controller = _RuntimeController()
    runtime = AgenticResolverRuntime(controller)
    checkpoint = await runtime.request_checkpoint(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=7,
    )
    continuation = await runtime.continue_after_terminal(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=7,
        continuation_delta={"instruction": "continue the same goal"},
    )

    assert checkpoint["disposition"] == "checkpointed"
    assert continuation["disposition"] == "terminal"
    assert continuation["resolution_thread_id"] == "thread-1"
    assert continuation["segment_id"] == "segment-1"
    assert continuation["lease_epoch"] == 8
    assert continuation["fresh_authority"] is True
    assert [call[0] for call in controller.calls] == [
        "request_checkpoint",
        "continue_after_terminal",
    ]
    assert controller.calls[1][2]["continuation_delta"] == {
        "instruction": "continue the same goal"
    }


class _ControllerRepository:
    """Focused repository fake for controller fencing and inspect projection."""

    def __init__(self) -> None:
        workspace = "C:/workspace/project"
        scope = {
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
        }
        segment = {
            "segment_id": "segment-1",
            "brain_conversation_ref": "chat:debug:one",
            "workspace_root": workspace,
            "workspace_fingerprint": workspace_fingerprint(workspace),
            "route_digest": "sha256:route",
            "resolver_profile_version": "kazusa-resolver-standard-v2",
            "dsh_release": "0.1.1-rc.2",
            "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
            "standard_catalog_digest": "sha256:native",
            "semantic_catalog_digest": "sha256:catalog",
            "policy_epoch": "dsh-standard-policy-v2",
            "scope_fingerprint": content_digest(scope),
            "audience_fingerprint": "sha256:audience",
            "interaction_id": "interaction-1",
        }
        self.record = SimpleNamespace(
            current_segment_id="segment-1",
            state="live",
            lease_epoch=4,
            document_revision=9,
            current_lease=None,
            segments=[segment],
            brain_conversation_ref="chat:debug:one",
            root_goal_ref="Resolve one goal.",
            workspace_root=workspace,
            workspace_fingerprint=workspace_fingerprint(workspace),
            route_digest="sha256:route",
            profile_version="kazusa-resolver-standard-v2",
            dsh_release="0.1.1-rc.2",
            session_store_epoch="dsh-sqlite-0.1.1-rc.2-standard-v2",
            standard_catalog_digest="sha256:native",
            semantic_catalog_digest="sha256:catalog",
            policy_epoch="dsh-standard-policy-v2",
            scope_fingerprint=content_digest(scope),
            audience_fingerprint="sha256:audience",
            interaction_id="interaction-1",
        )
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def validate_fence(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("validate_fence", args, kwargs))

    async def get_thread(self, *args: object, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("get_thread", args, kwargs))
        return self.record

    async def get_operation(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("get_operation", args, kwargs))

    async def prepare_operation(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(("prepare_operation", args, kwargs))
        return {"operation_payload_digest": args[2]}

    async def acquire_lease(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(("acquire_lease", args, kwargs))
        lease = {
            "activation_id": args[1],
            "lease_epoch": self.record.lease_epoch + 1,
            "owner_id": "controller-test",
            "expires_at": "2099-01-01T00:00:00Z",
        }
        self.record.lease_epoch = int(lease["lease_epoch"])
        self.record.current_lease = lease
        return lease

    async def update_operation(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("update_operation", args, kwargs))

    async def update_segment(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("update_segment", args, kwargs))

    async def release_lease(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("release_lease", args, kwargs))
        self.record.current_lease = None


class _ControllerRpc:
    """Deterministic sidecar collaborator for controller operations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(
        self, method: str, params: dict[str, object], **_: object
    ) -> dict[str, object]:
        self.calls.append((method, params))
        if method == "resolution.request_checkpoint":
            return {"disposition": "checkpointed", "last_committed_seq": 12}
        if method == "system.health":
            return {
                "protocol_version": "kazusa.dsh-resolution-rpc.v2",
                "status": "ready",
                "profile_version": "kazusa-resolver-standard-v2",
                "dsh_release": "0.1.1-rc.2",
                "store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
                "route": {"digest": "sha256:route"},
                "catalog": {
                    "native_catalog_digest": "sha256:native",
                    "semantic_catalog_digest": "sha256:catalog",
                    "published_catalog_digest": "sha256:published",
                },
                "policy": {"epoch": "dsh-standard-policy-v2"},
                "workspace": {"root": "C:/workspace/project"},
            }
        if method == "resolution.dispose_activation":
            return {"disposition": "canceled"}
        if method == "resolution.continue":
            return {"disposition": "terminal", "session_id": "session-1"}
        raise AssertionError(f"unexpected controller RPC: {method}")


@pytest.mark.asyncio
async def test_controller_checkpoint_terminal_control_and_replay_are_fenced_and_idempotent() -> None:
    """Controller operations carry the active fence and expose exact inspect state."""

    from agentic_resolver.controller import ResolutionController

    repository = _ControllerRepository()
    rpc = _ControllerRpc()
    controller = ResolutionController(
        repository,
        rpc,
        owner_id="controller-test",
        semantic_authority_secret=b"controller-secret",
    )

    checkpoint = await controller.request_checkpoint(
        "thread-1", "activation-1", 4
    )
    inspected = await controller.inspect("thread-1")
    continuation = await controller.continue_after_terminal(
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        activation_id="activation-1",
        lease_epoch=4,
        continuation_delta={"instruction": "same semantic goal"},
        execution_context={
            "platform": "debug",
            "channel_id": "channel-1",
            "requester_global_user_id": "user-1",
        },
    )

    assert checkpoint["disposition"] == "checkpointed"
    assert inspected == {
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "state": "checkpointed",
        "lease_epoch": 4,
        "document_revision": 9,
    }
    assert continuation["disposition"] == "terminal"
    assert rpc.calls[0][0] == "resolution.request_checkpoint"
    assert rpc.calls[0][1]["resolution_thread_id"] == "thread-1"
    assert rpc.calls[0][1]["segment_id"] == "segment-1"
    assert rpc.calls[0][1]["activation_id"] == "activation-1"
    assert rpc.calls[0][1]["lease_epoch"] == 4
    continuation_call = next(
        params for method, params in rpc.calls
        if method == "resolution.continue"
    )
    assert set(continuation_call) == {
        "operation_id",
        "operation_payload_digest",
        "activation_id",
        "lease_epoch",
        "intake",
    }
    assert continuation_call["lease_epoch"] == 5
    assert continuation_call["intake"]["resolution_thread_id"] == "thread-1"
    assert continuation_call["intake"]["segment_id"] == "segment-1"
    assert continuation_call["intake"]["model_input"]["objective"] == (
        "same semantic goal"
    )
    assert continuation_call["intake"]["semantic_tool_authority"]["token"]
    assert any(call[0] == "validate_fence" for call in repository.calls)
    assert any(call[0] == "release_lease" for call in repository.calls)


@pytest.mark.asyncio
async def test_semantic_catalog_digest_change_rotates_segment_and_rejects_old_authority(
    monkeypatch,
) -> None:
    """A catalog digest change rotates a compatible thread through the controller."""

    from agentic_resolver.controller import ResolutionController
    from agentic_resolver.fingerprints import workspace_fingerprint

    previous_digest = "sha256:previous-semantic-catalog"
    current_digest = "sha256:current-semantic-catalog"

    class Repository:
        def __init__(self) -> None:
            self.rotations: list[tuple[object, object, object]] = []
            self.record = SimpleNamespace(
                current_segment_id="segment-old",
                lease_epoch=2,
                segments=[{
                    "segment_id": "segment-old",
                    "brain_conversation_ref": "chat:debug:one",
                    "workspace_root": "C:/workspace/project",
                    "workspace_fingerprint": workspace_fingerprint(
                        "C:/workspace/project"
                    ),
                    "route_digest": "sha256:route",
                    "resolver_profile_version": "kazusa-resolver-standard-v2",
                    "dsh_release": "0.1.1-rc.2",
                    "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
                    "standard_catalog_digest": "sha256:native",
                    "semantic_catalog_digest": previous_digest,
                    "policy_epoch": "dsh-standard-policy-v2",
                    "scope_fingerprint": "sha256:scope",
                    "audience_fingerprint": "sha256:audience",
                    "interaction_id": "request-1",
                }],
            )

        async def get_thread(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return self.record

        async def rotate_segment(
            self, *args: object, **kwargs: object
        ) -> None:
            self.rotations.append((args, kwargs, self.record.current_segment_id))

    repository = Repository()
    controller = ResolutionController(
        repository,
        SimpleNamespace(),
        owner_id="controller-test",
        semantic_authority_secret=b"controller-secret",
    )

    async def health() -> dict[str, str]:
        return {
            "route_digest": "sha256:route",
            "native_catalog_digest": "sha256:native",
            "semantic_catalog_digest": current_digest,
            "published_catalog_digest": "sha256:published",
            "profile_version": "kazusa-resolver-standard-v2",
            "dsh_release": "0.1.1-rc.2",
            "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
            "policy_epoch": "dsh-standard-policy-v2",
            "workspace_root": "C:/workspace/project",
        }

    async def activated(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"disposition": "admitted_active"}

    monkeypatch.setattr(controller, "_health_identity", health)
    monkeypatch.setattr(
        controller,
        "_verify_intake_authority",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(controller, "_activate", activated)

    result = await controller.continue_resolution({
        "schema_version": "dsh_resolution_intake.v2",
        "mode": "continue",
        "request_id": "request-1",
        "operation_id": "operation-1",
        "operation_payload_digest": "sha256:payload",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-new",
        "brain_conversation_ref": "chat:debug:one",
        "workspace_root": "C:/workspace/project",
        "route_digest": "sha256:route",
        "model_input": {"objective": "finish", "facts": []},
        "semantic_tool_authority": {
            "catalog_digest": current_digest,
            "token": "opaque",
        },
        "interaction_authority": {
            "issuer": "brain",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
        },
    })

    assert result == {"disposition": "admitted_active"}
    assert len(repository.rotations) == 1
    rotation_args, rotation_kwargs, _ = repository.rotations[0]
    assert rotation_args[0] == "thread-1"
    rotated_segment = rotation_args[1]
    assert rotated_segment["segment_id"] == "segment-new"
    assert rotated_segment["semantic_catalog_digest"] == current_digest
    assert rotation_kwargs == {"reason": "semantic_catalog_digest_mismatch"}
