"""Controller lifecycle and lease-fencing tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agentic_resolver.controller import ResolutionController
from agentic_resolver.errors import (
    DuplicateActivationError,
    StaleActivationOrLeaseError,
)
from agentic_resolver.persistence import InMemoryResolutionThreadRepository
from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
    SemanticActivationAuthorityV1,
    activation_id_for,
    issue_activation_token,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

_SEMANTIC_SECRET = b"controller-semantic-secret"


def _intake(*, mode: str = "start", **changes: object) -> dict[str, object]:
    scope_variant = str(changes.pop("scope_variant", "default"))
    token_lease_epoch = int(changes.pop("token_lease_epoch", 1))
    service_scope = {
        "platform": "debug",
        "platform_channel_id": (
            "channel-controller"
            if scope_variant == "default"
            else f"channel-{scope_variant}"
        ),
        "global_user_id": "user-controller",
    }
    issued = datetime.now(UTC).replace(microsecond=0)
    expires = issued + timedelta(minutes=5)
    thread_id = str(changes.get("resolution_thread_id", "res_controller"))
    segment_id = str(changes.get("segment_id", "seg_controller"))
    interaction = dict(changes.get("interaction_authority", {}))
    interaction.setdefault("issuer", "dsh-sidecar")
    interaction.setdefault("scope_fingerprint", content_digest(service_scope))
    interaction.setdefault(
        "audience_fingerprint",
        content_digest({"audience": "controller-test"}),
    )
    workspace_root = str(changes.get("workspace_root", "C:/workspace/project"))
    route_digest = str(changes.get("route_digest", "sha256:route"))
    catalog_digest = str(
        dict(changes.get("semantic_tool_authority", {})).get(
            "catalog_digest", "sha256:catalog"
        )
    )
    brain_ref = str(
        changes.get("brain_conversation_ref", "chat:debug:controller")
    )
    authority = SemanticActivationAuthorityV1(
        activation_id=activation_id_for(thread_id, segment_id, token_lease_epoch),
        lease_epoch=token_lease_epoch,
        resolution_thread_id=thread_id,
        segment_id=segment_id,
        brain_conversation_ref=brain_ref,
        service_scope=service_scope,
        scope_fingerprint=str(interaction["scope_fingerprint"]),
        audience_fingerprint=str(interaction["audience_fingerprint"]),
        workspace_root=workspace_root,
        route_digest=route_digest,
        catalog_digest=catalog_digest,
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest=route_digest,
        workspace_fingerprint=content_digest({"workspace_root": workspace_root}),
        issued_reference_digest="sha256:controller-issued",
        policy_epoch="dsh-standard-policy-v2",
        interaction_issuer=str(interaction["issuer"]),
        issued_at=issued.isoformat().replace("+00:00", "Z"),
        expires_at=expires.isoformat().replace("+00:00", "Z"),
        token_id=f"tok_{thread_id}_{segment_id}_{token_lease_epoch}",
        nonce=f"nonce_{thread_id}_{segment_id}_{token_lease_epoch}",
    )
    intake = {
        "schema_version": "dsh_resolution_intake.v2",
        "mode": mode,
        "request_id": "rrq_controller",
        "operation_id": "op_controller",
        "operation_payload_digest": "sha256:payload",
        "resolution_thread_id": "res_controller",
        "segment_id": "seg_controller",
        "brain_conversation_ref": "chat:debug:controller",
        "workspace_root": "C:/workspace/project",
        "route_digest": "sha256:route",
        "model_input": {"objective": "finish", "facts": []},
        "semantic_tool_authority": {
            "catalog_digest": catalog_digest,
            "token": issue_activation_token(
                authority,
                secret=_SEMANTIC_SECRET,
                now=authority.issued_at,
            ),
        },
        "interaction_authority": interaction,
    }
    intake.update(changes)
    return intake


class FakeRpc:
    """Small semantic RPC fixture with deterministic dispositions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.disposition = "admitted_active"
        self.inspect_result: dict[str, object] | None = None
        self.activation_result: dict[str, object] | None = None

    async def call(self, method: str, params: dict[str, object], **_: object) -> dict[str, object]:
        self.calls.append((method, params))
        if method == "system.health":
            return {
                "protocol_version": "kazusa.dsh-resolution-rpc.v2",
                "status": "ready",
                "profile_version": "kazusa-resolver-standard-v2",
                "dsh_release": "0.1.1-rc.2",
                "store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
                "route": {"digest": "sha256:route"},
                "catalog": {
                    "native_catalog_digest": "sha256:native-catalog",
                    "semantic_catalog_digest": "sha256:catalog",
                    "published_catalog_digest": "sha256:published-catalog",
                },
                "policy": {"epoch": "dsh-standard-policy-v2"},
                "workspace": {"root": "C:/workspace/project"},
            }
        if method == "resolution.inspect":
            if self.inspect_result is not None:
                return self.inspect_result
            return {"disposition": self.disposition, "protocol_version": "kazusa.dsh-resolution-rpc.v2"}
        if method == "resolution.request_checkpoint":
            return {"disposition": "checkpointed", "exhaust": {"kind": "checkpointed"}}
        if method == "resolution.cancel":
            return {"disposition": "canceled", "exhaust": {"kind": "runtime_fault"}}
        if method == "resolution.dispose_activation":
            return {"disposition": "canceled"}
        if (
            method in {"resolution.open", "resolution.continue"}
            and self.activation_result is not None
        ):
            return self.activation_result
        return {
            "disposition": self.disposition,
            "exhaust": {
                "kind": "terminal",
                "operation_id": params.get("operation_id", "op_controller"),
            },
        }


def _controller() -> tuple[ResolutionController, InMemoryResolutionThreadRepository, FakeRpc]:
    repository = InMemoryResolutionThreadRepository()
    rpc = FakeRpc()
    return ResolutionController(
        repository,
        rpc,
        owner_id="controller-test",
        semantic_authority_secret=_SEMANTIC_SECRET,
    ), repository, rpc


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _continuation_token() -> str:
    issued = datetime.now(UTC).replace(microsecond=0)
    authority = SemanticActivationAuthorityV1(
        activation_id="activation-v2",
        lease_epoch=3,
        resolution_thread_id="thread-v2",
        segment_id="segment-v2",
        brain_conversation_ref="chat:debug:controller-v2",
        service_scope={
            "platform": "debug",
            "platform_channel_id": "channel-v2",
            "global_user_id": "user-v2",
        },
        scope_fingerprint=content_digest({
            "platform": "debug",
            "platform_channel_id": "channel-v2",
            "global_user_id": "user-v2",
        }),
        audience_fingerprint=content_digest({"audience": "controller-v2"}),
        workspace_root="C:/workspace/project",
        route_digest="sha256:route",
        catalog_digest="sha256:catalog",
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest="sha256:route",
        workspace_fingerprint=content_digest({
            "workspace_root": "C:/workspace/project",
        }),
        issued_reference_digest="sha256:controller-v2-issued",
        policy_epoch="dsh-standard-policy-v2",
        interaction_issuer="dsh-brain-v2",
        issued_at=issued.isoformat().replace("+00:00", "Z"),
        expires_at=(issued + timedelta(minutes=5)).isoformat().replace(
            "+00:00", "Z"
        ),
        token_id="tok-controller-v2",
        nonce="nonce-controller-v2",
    )
    return issue_activation_token(
        authority,
        secret=_SEMANTIC_SECRET,
        now=authority.issued_at,
    )


def test_open_creates_one_thread_segment_activation_and_lease_epoch() -> None:
    controller, repository, _ = _controller()
    result = _run(controller.open(_intake()))
    assert result["activation_id"]
    record = repository.get_thread("res_controller")
    assert record is not None
    assert len(record.segments) == 1
    assert record.lease_epoch == 1
    assert record.standard_catalog_digest == "sha256:native-catalog"
    assert record.semantic_catalog_digest == "sha256:catalog"
    assert record.standard_catalog_digest != record.semantic_catalog_digest
    assert record.audience_fingerprint == _intake()["interaction_authority"][
        "audience_fingerprint"
    ]


def test_continue_reuses_segment_only_for_same_goal_scope_audience_and_epoch() -> None:
    controller, repository, _ = _controller()
    opened = _run(controller.open(_intake()))
    _run(controller.dispose_activation(
        "res_controller", opened["activation_id"], opened["lease_epoch"]
    ))
    continued = _run(controller.continue_resolution(_intake(
        mode="continue",
        operation_id="op_continue",
        operation_payload_digest="sha256:continue",
        token_lease_epoch=2,
    )))
    assert continued["segment_id"] == opened["segment_id"]
    assert len(repository.get_thread("res_controller").segments) == 1


def test_incompatible_scope_audience_profile_release_store_model_catalog_or_policy_rotates_segment() -> None:
    controller, repository, _ = _controller()
    opened = _run(controller.open(_intake()))
    continued = _run(controller.continue_resolution(_intake(
        mode="continue",
        operation_id="op_rotated",
        operation_payload_digest="sha256:rotated",
        segment_id="seg_rotated",
        scope_variant="rotated",
        token_lease_epoch=2,
        interaction_authority={
            "issuer": "dsh-sidecar",
            "scope_fingerprint": content_digest({
                "platform": "debug",
                "platform_channel_id": "channel-rotated",
                "global_user_id": "user-controller",
            }),
            "audience_fingerprint": content_digest({"audience": "controller-test"}),
        },
    )))
    assert continued["segment_id"] != opened["segment_id"]
    assert len(repository.get_thread("res_controller").segments) == 2


def test_duplicate_execution_lease_fails_closed() -> None:
    controller, _, _ = _controller()
    _run(controller.open(_intake()))
    with pytest.raises(DuplicateActivationError):
        _run(controller.open(_intake(operation_id="op_other")))


def test_long_activation_renews_lease_and_expired_takeover_increments_epoch() -> None:
    controller, repository, _ = _controller()
    opened = _run(controller.open(_intake()))
    renewed = _run(controller.renew_lease(
        "res_controller", opened["activation_id"], opened["lease_epoch"]
    ))
    assert renewed["lease_epoch"] == 1
    renewed_expires_at = renewed["expires_at"]
    assert isinstance(renewed_expires_at, str)
    assert renewed_expires_at
    taken = _run(controller.takeover_expired(
        "res_controller", now=renewed_expires_at
    ))
    assert taken["lease_epoch"] == 2
    assert repository.get_thread("res_controller").lease_epoch == 2


def test_stale_activation_or_lease_epoch_rejects_every_live_control() -> None:
    controller, _, rpc = _controller()
    opened = _run(controller.open(_intake()))
    with pytest.raises(StaleActivationOrLeaseError):
        _run(controller.request_checkpoint(
            "res_controller", "stale", opened["lease_epoch"]
        ))
    with pytest.raises(StaleActivationOrLeaseError):
        _run(controller.cancel("res_controller", opened["activation_id"], 99))
    with pytest.raises(StaleActivationOrLeaseError):
        _run(controller.dispose_activation(
            "res_controller", opened["activation_id"], 99
        ))
    assert not any(method == "resolution.request_checkpoint" for method, _ in rpc.calls)


def test_amend_steers_in_flight_work_and_followup_queues_next_turn() -> None:
    controller, _, rpc = _controller()
    opened = _run(controller.open(_intake()))
    _run(controller.amend(
        "res_controller", opened["activation_id"], opened["lease_epoch"],
        {"objective": "new objective"},
    ))
    methods = [method for method, _ in rpc.calls]
    assert "resolution.amend" in methods


def test_checkpoint_cancel_inspect_and_dispose_preserve_durable_lineage() -> None:
    controller, repository, _ = _controller()
    opened = _run(controller.open(_intake()))
    checkpoint = _run(controller.request_checkpoint(
        "res_controller", opened["activation_id"], opened["lease_epoch"]
    ))
    assert checkpoint["disposition"] == "checkpointed"
    inspected = _run(controller.inspect("res_controller"))
    assert inspected["resolution_thread_id"] == "res_controller"
    continued = _run(controller.continue_resolution(_intake(
        mode="continue",
        operation_id="op_after_checkpoint",
        operation_payload_digest="sha256:after-checkpoint",
        token_lease_epoch=2,
    )))
    canceled = _run(controller.cancel(
        "res_controller",
        continued["activation_id"],
        continued["lease_epoch"],
    ))
    assert canceled["disposition"] == "canceled"
    continued_again = _run(controller.continue_resolution(_intake(
        mode="continue",
        operation_id="op_after_cancel",
        operation_payload_digest="sha256:after-cancel",
        token_lease_epoch=3,
    )))
    _run(controller.dispose_activation(
        "res_controller",
        continued_again["activation_id"],
        continued_again["lease_epoch"],
    ))
    assert repository.get_thread("res_controller") is not None


def test_terminal_submit_maps_to_typed_terminal_exhaust() -> None:
    controller, _, rpc = _controller()
    rpc.disposition = "terminal"
    result = _run(controller.open(_intake()))
    assert result["exhaust"]["kind"] == "terminal"


def test_controller_restart_reconciles_terminal_projection_and_lease() -> None:
    controller, repository, rpc = _controller()
    _run(controller.open(_intake()))
    rpc.inspect_result = {
        "disposition": "terminal",
        "session_id": "kazusa-resolution-reconciled",
        "dsh_message_source_id": "dsh-message-reconciled",
        "last_committed_seq": 12,
        "exhaust": {"kind": "terminal"},
    }

    restarted = ResolutionController(
        repository,
        rpc,
        owner_id="restarted-controller-test",
        semantic_authority_secret=_SEMANTIC_SECRET,
    )
    result = _run(restarted.open(_intake()))

    assert result["exhaust"]["kind"] == "terminal"
    operation = repository.get_operation("res_controller", "op_controller")
    assert operation is not None
    assert operation["disposition"] == "terminal"
    assert operation["dsh_message_source_id"] == "dsh-message-reconciled"
    assert operation["last_committed_seq"] == 12
    record = repository.get_thread("res_controller")
    assert record is not None
    segment = record.segments[0]
    assert segment["state"] == "terminal"
    assert segment["dsh_session_id"] == "kazusa-resolution-reconciled"
    assert segment["last_committed_seq"] == 12
    assert record.current_lease is None
    assert any(
        method == "resolution.dispose_activation"
        for method, _params in rpc.calls
    )


def test_controller_restart_reattaches_admitted_operation_with_same_fence() -> None:
    controller, repository, rpc = _controller()
    opened = _run(controller.open(_intake()))
    rpc.inspect_result = {
        "disposition": "admitted_active",
        "dsh_message_source_id": "dsh-message-active",
    }
    rpc.activation_result = {
        "disposition": "terminal",
        "session_id": "kazusa-resolution-attached",
        "dsh_message_source_id": "dsh-message-active",
        "last_committed_seq": 14,
        "exhaust": {"kind": "terminal"},
    }

    restarted = ResolutionController(
        repository,
        rpc,
        owner_id="restarted-controller-test",
        semantic_authority_secret=_SEMANTIC_SECRET,
    )
    result = _run(restarted.open(_intake()))

    assert result["exhaust"]["kind"] == "terminal"
    open_calls = [
        params for method, params in rpc.calls
        if method == "resolution.open"
    ]
    assert len(open_calls) == 2
    assert open_calls[1]["operation_id"] == "op_controller"
    assert open_calls[1]["activation_id"] == opened["activation_id"]
    assert open_calls[1]["lease_epoch"] == opened["lease_epoch"]
    operation = repository.get_operation("res_controller", "op_controller")
    assert operation is not None
    assert operation["disposition"] == "terminal"
    assert operation["last_committed_seq"] == 14


def test_checkpoint_maps_to_runtime_owned_checkpointed_exhaust() -> None:
    controller, _, _ = _controller()
    opened = _run(controller.open(_intake()))
    result = _run(controller.request_checkpoint(
        "res_controller", opened["activation_id"], opened["lease_epoch"]
    ))
    assert result["exhaust"]["kind"] == "checkpointed"


@pytest.mark.asyncio
async def test_interaction_checkpoint_and_resume_preserve_exact_thread_segment_and_fence() -> None:
    """A Brain interaction checkpoint resumes only the fenced V2 segment."""

    from agentic_resolver.controller import ResolutionController

    calls: list[tuple[str, dict[str, object]]] = []

    class Rpc:
        async def call(self, method: str, params: dict[str, object], **kwargs):
            del kwargs
            calls.append((method, params))
            if method == "resolution.request_checkpoint":
                return {
                    "disposition": "checkpointed",
                    "resolution_thread_id": params["resolution_thread_id"],
                    "segment_id": params["segment_id"],
                    "activation_id": params["activation_id"],
                    "lease_epoch": params["lease_epoch"],
                }
            return {
                "disposition": "terminal",
                "resolution_thread_id": params["resolution_thread_id"],
                "segment_id": params["segment_id"],
            }

    controller = ResolutionController(
        None,
        Rpc(),
        owner_id="controller-v2",
        semantic_authority_secret=_SEMANTIC_SECRET,
    )
    checkpoint = await controller.interaction_checkpoint(
        resolution_thread_id="thread-v2",
        segment_id="segment-v2",
        activation_id="activation-v2",
        lease_epoch=3,
        interaction_id="interaction-v2",
    )
    assert checkpoint["disposition"] == "checkpointed"
    resumed = await controller.resume_after_interaction(
        resolution_thread_id="thread-v2",
        segment_id="segment-v2",
        activation_id="activation-v2",
        lease_epoch=3,
        interaction_id="interaction-v2",
        continuation_delta={"answer": "typed-answer"},
        continuation_authority_token=_continuation_token(),
    )
    assert resumed["resolution_thread_id"] == "thread-v2"
    assert resumed["segment_id"] == "segment-v2"
    assert calls[0][0] == "resolution.request_checkpoint"
    assert calls[1][1]["resolution_thread_id"] == "thread-v2"
    assert calls[1][1]["segment_id"] == "segment-v2"
    assert calls[1][1]["lease_epoch"] == 3
    assert calls[1][1]["intake"]["model_input"] == {
        "objective": "typed-answer",
        "facts": [],
    }


@pytest.mark.asyncio
async def test_allow_once_resume_projects_bounded_retry_fact_without_reply_text() -> None:
    """Explain one-shot retry semantics without exposing reply or authority data."""

    controller, _, rpc = _controller()
    opened = await controller.open(_intake())
    continuation_delta = {
        "interaction_id": "interaction-allow-once",
        "kind": "approval",
        "decision": "allow_once",
        "answer": None,
        "response_goal": None,
        "relay_mode": None,
        "reason": "Brain approved one exact retry.",
    }
    await controller.resume_after_interaction(
        resolution_thread_id="res_controller",
        segment_id="seg_controller",
        activation_id=opened["activation_id"],
        lease_epoch=opened["lease_epoch"],
        interaction_id="interaction-allow-once",
        continuation_delta=continuation_delta,
        continuation_authority_token=str(
            _intake()["semantic_tool_authority"]["token"]
        ),
    )

    continuation_call = next(
        params
        for method, params in reversed(rpc.calls)
        if method == "resolution.continue"
    )
    model_input = continuation_call["intake"]["model_input"]
    assert model_input["objective"] == "finish"
    facts = model_input["facts"]
    assert isinstance(facts, list)
    assert len(facts) == 1
    fact = facts[0]
    assert isinstance(fact, str)
    assert "earlier native approval cancellation" in fact
    assert "one-shot approval" in fact
    assert "semantically identical executable arguments" in fact
    assert "fresh call id" in fact
    assert "atomically consume" in fact
    assert "reply_text" not in fact
    assert "typed-answer" not in fact
    assert "continuation_authority_token" not in fact


@pytest.mark.asyncio
async def test_resume_requires_fresh_brain_authority_token() -> None:
    controller = ResolutionController(
        None,
        type("Rpc", (), {"call": lambda self, *args, **kwargs: None})(),
        owner_id="controller-v2",
        semantic_authority_secret=_SEMANTIC_SECRET,
    )
    with pytest.raises(ValueError, match="authority_token"):
        await controller.resume_after_interaction(
            resolution_thread_id="thread-v2",
            segment_id="segment-v2",
            activation_id="activation-v2",
            lease_epoch=3,
            interaction_id="interaction-v2",
            continuation_delta={"answer": "typed-answer"},
            continuation_authority_token="",
        )


@pytest.mark.asyncio
async def test_resume_rejects_raw_user_reply_text() -> None:
    controller = ResolutionController(
        None,
        type("Rpc", (), {"call": lambda self, *args, **kwargs: None})(),
        owner_id="controller-v2",
        semantic_authority_secret=_SEMANTIC_SECRET,
    )
    with pytest.raises(ValueError, match="reply_text"):
        await controller.resume_after_interaction(
            resolution_thread_id="thread-v2",
            segment_id="segment-v2",
            activation_id="activation-v2",
            lease_epoch=3,
            interaction_id="interaction-v2",
            continuation_delta={"reply_text": "yes"},
            continuation_authority_token="",
        )
