"""Authenticated JSON-RPC and semantic operation reconciliation tests."""

from __future__ import annotations

import pytest

from agentic_resolver.contracts import (
    RPC_PROTOCOL_VERSION,
    DSHResolutionExhaustV1,
    DSHResolutionIntakeV1,
)
from agentic_resolver.errors import (
    OperationIdReuseMismatchError,
    OperationOutcomeUncertainError,
    RpcAuthenticationError,
    RpcContractError,
)
from agentic_resolver.rpc import (
    DSHRpcClient,
    DSHRpcServer,
    InMemoryRpcTransport,
)


def _intake() -> dict[str, object]:
    return {
        "schema_version": "dsh_resolution_intake.v1",
        "mode": "start",
        "runtime": {
            "request_id": "rrq_rpc",
            "operation_id": "op_rpc",
            "operation_payload_digest": "sha256:payload",
            "resolution_thread_id": "res_rpc",
            "segment_id": "seg_rpc",
            "priority": "now",
            "soft_deadline_at": "2026-08-28T00:00:10Z",
            "hard_deadline_at": "2026-08-28T00:00:30Z",
            "max_model_steps": 2,
            "max_tool_calls": 2,
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
        },
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


def _server() -> DSHRpcServer:
    return DSHRpcServer(token="rpc-secret")


def test_versioned_authenticated_rpc_rejects_bad_version_token_and_method() -> None:
    server = _server()
    with pytest.raises(RpcContractError):
        server.dispatch({"jsonrpc": "1.0"}, authorization="Bearer rpc-secret")
    with pytest.raises(RpcAuthenticationError):
        server.dispatch({"jsonrpc": "2.0"}, authorization="Bearer wrong")
    with pytest.raises(RpcContractError, match="method"):
        server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "unknown",
                "params": {"protocol_version": RPC_PROTOCOL_VERSION},
            },
            authorization="Bearer rpc-secret",
        )


def test_rpc_round_trip_preserves_typed_intake_and_exhaust() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    client = DSHRpcClient(
        "http://127.0.0.1:8081/rpc",
        "rpc-secret",
        transport=transport,
    )
    value = client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    assert isinstance(value, dict)
    assert value["protocol_version"] == RPC_PROTOCOL_VERSION
    DSHResolutionIntakeV1.from_mapping(value["intake"])
    assert "exhaust" in value
    DSHResolutionExhaustV1.from_mapping(value["exhaust"])


def test_same_operation_id_and_digest_reconciles_one_admission() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    first = client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    second = client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    assert first == second
    assert server.operations.admission_count("op_rpc") == 1


def test_operation_id_reuse_with_different_digest_fails_closed() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    with pytest.raises(OperationIdReuseMismatchError):
        client.call_sync(
            "resolution.open",
            {"intake": _intake()},
            operation_id="op_rpc",
            operation_payload_digest="sha256:other",
        )


def test_disconnect_before_admission_inspects_then_replays_same_operation_once() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    transport.fail_next_before_dispatch = True
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    result = client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    assert result["protocol_version"] == RPC_PROTOCOL_VERSION
    assert server.operations.admission_count("op_rpc") == 1


def test_disconnect_after_admission_attaches_to_active_operation() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    transport.fail_next_after_admission = True
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    result = client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    assert result["disposition"] in {"admitted_active", "terminal"}
    assert server.operations.admission_count("op_rpc") == 1


def test_disconnect_after_terminal_commit_reconciles_exact_exhaust_without_model_call() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    transport.fail_next_after_commit = True
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    first = client.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    second = client.call_sync(
        "resolution.inspect",
        {"operation_id": "op_rpc"},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    assert first["exhaust"] == second["exhaust"]
    assert server.operations.execution_count("op_rpc") == 1


def test_controller_restart_reconciles_admitted_operation_without_duplicate_model_entry() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    first = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    first.call_sync(
        "resolution.open",
        {"intake": _intake()},
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    restarted = DSHRpcClient(
        "http://127.0.0.1:8081/rpc",
        "rpc-secret",
        transport=transport,
    )
    result = restarted.reconcile_sync("op_rpc", "sha256:payload")
    assert result["protocol_version"] == RPC_PROTOCOL_VERSION
    assert server.operations.execution_count("op_rpc") == 1


def test_unknown_operation_outcome_returns_uncertain_fault_without_new_admission() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    transport.return_unknown_inspection = True
    with pytest.raises(OperationOutcomeUncertainError):
        client.call_sync(
            "resolution.open",
            {"intake": _intake()},
            operation_id="op_unknown",
            operation_payload_digest="sha256:unknown",
        )
    assert server.operations.admission_count("op_unknown") == 0


@pytest.mark.asyncio
async def test_rpc_client_async_call_is_available_for_controller() -> None:
    server = _server()
    client = DSHRpcClient(
        "http://127.0.0.1:8081/rpc",
        "rpc-secret",
        transport=InMemoryRpcTransport(server),
    )
    result = await client.call(
        "system.health",
        {},
    )
    assert result["protocol_version"] == RPC_PROTOCOL_VERSION
