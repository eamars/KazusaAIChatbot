"""Controller lifecycle and lease-fencing tests."""

from __future__ import annotations

import asyncio

import pytest

from agentic_resolver.controller import ResolutionController
from agentic_resolver.errors import (
    DuplicateActivationError,
    StaleActivationOrLeaseError,
)
from agentic_resolver.persistence import InMemoryResolutionThreadRepository


def _intake(*, mode: str = "start", **changes: object) -> dict[str, object]:
    runtime = {
        "request_id": "rrq_controller",
        "operation_id": "op_controller",
        "operation_payload_digest": "sha256:payload",
        "resolution_thread_id": "res_controller",
        "segment_id": "seg_controller",
        "priority": "now",
        "soft_deadline_at": "2026-08-28T00:00:10Z",
        "hard_deadline_at": "2026-08-28T00:00:30Z",
        "max_model_steps": 4,
        "max_tool_calls": 4,
        "max_tool_bytes": 4096,
        "capability_token": "opaque",
        "scope_fingerprint": "sha256:scope",
        "audience_fingerprint": "sha256:audience",
        "resolver_profile_version": "kazusa-resolver-v1",
        "dsh_release": "0.1.1-rc.2",
        "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-v1",
        "model_route": "resolver-model",
        "tool_catalog_digest": "sha256:catalog",
        "policy_epoch": "2026-08-28.1",
    }
    runtime.update(changes)
    return {
        "schema_version": "dsh_resolution_intake.v1",
        "mode": mode,
        "runtime": runtime,
        "model_input": {
            "objective": "finish",
            "constraints": [],
            "success_criteria": [],
            "known_facts": [],
            "uncertainty": [],
            "literal_inputs": [],
            "continuation_delta": None,
            "prior_resolution_refs": [],
            "requested_evidence_quality": "normal",
            "notes": [],
        },
    }


class FakeRpc:
    """Small semantic RPC fixture with deterministic dispositions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.disposition = "admitted_active"
        self.inspect_result: dict[str, object] | None = None
        self.activation_result: dict[str, object] | None = None

    async def call(self, method: str, params: dict[str, object], **_: object) -> dict[str, object]:
        self.calls.append((method, params))
        if method == "resolution.inspect":
            if self.inspect_result is not None:
                return self.inspect_result
            return {"disposition": self.disposition, "protocol_version": "kazusa.dsh-resolution-rpc.v1"}
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
    return ResolutionController(repository, rpc, owner_id="controller-test"), repository, rpc


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_open_creates_one_thread_segment_activation_and_lease_epoch() -> None:
    controller, repository, _ = _controller()
    result = _run(controller.open(_intake()))
    assert result["activation_id"]
    record = repository.get_thread("res_controller")
    assert record is not None
    assert len(record.segments) == 1
    assert record.lease_epoch == 1


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
        scope_fingerprint="sha256:new",
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
    taken = _run(controller.takeover_expired("res_controller", now="2026-08-29T00:00:00Z"))
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
