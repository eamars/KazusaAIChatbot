"""Authenticated JSON-RPC and semantic operation reconciliation tests."""

from __future__ import annotations

import json
from typing import Self

import pytest

from agentic_resolver.contracts import (
    RPC_PROTOCOL_VERSION,
    DSHResolutionExhaustV2,
    DSHResolutionIntakeV2,
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
        "schema_version": "dsh_resolution_intake.v2",
        "mode": "start",
        "request_id": "rrq_rpc",
        "operation_id": "op_rpc",
        "operation_payload_digest": "sha256:payload",
        "resolution_thread_id": "res_rpc",
        "segment_id": "seg_rpc",
        "brain_conversation_ref": "chat:debug:rpc",
        "workspace_root": "C:/workspace/project",
        "route_digest": "sha256:route",
        "semantic_tool_authority": {
            "catalog_digest": "sha256:catalog",
            "token": "opaque",
        },
        "interaction_authority": {
            "issuer": "dsh-sidecar",
            "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience",
        },
        "model_input": {"objective": "finish", "facts": []},
    }


def _open_params(
    operation_id: str = "op_rpc",
    payload_digest: str = "sha256:payload",
) -> dict[str, object]:
    intake = _intake()
    intake["operation_id"] = operation_id
    intake["operation_payload_digest"] = payload_digest
    return {
        "intake": intake,
        "activation_id": f"activation-{operation_id}",
        "lease_epoch": 1,
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
        _open_params(),
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    assert isinstance(value, dict)
    assert value["protocol_version"] == RPC_PROTOCOL_VERSION
    DSHResolutionIntakeV2.from_mapping(value["intake"])
    assert "exhaust" in value
    DSHResolutionExhaustV2.from_mapping(value["exhaust"])


def test_same_operation_id_and_digest_reconciles_one_admission() -> None:
    server = _server()
    transport = InMemoryRpcTransport(server)
    client = DSHRpcClient("http://127.0.0.1:8081/rpc", "rpc-secret", transport=transport)
    first = client.call_sync(
        "resolution.open",
        _open_params(),
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    second = client.call_sync(
        "resolution.open",
        _open_params(),
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
        _open_params(),
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    with pytest.raises(OperationIdReuseMismatchError):
        client.call_sync(
            "resolution.open",
            _open_params(payload_digest="sha256:other"),
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
        _open_params(),
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
        _open_params(),
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
        _open_params(),
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
        _open_params(),
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
            _open_params(
                operation_id="op_unknown",
                payload_digest="sha256:unknown",
            ),
            operation_id="op_unknown",
            operation_payload_digest="sha256:unknown",
        )
    assert server.operations.admission_count("op_unknown") == 0


def test_http_transport_has_no_dsh_execution_deadline_and_bounds_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long model execution has no Kazusa deadline; control RPC remains bounded."""

    observed_timeouts: list[float | None] = []

    class Response:
        def __init__(self, value: dict[str, object]) -> None:
            self._value = value

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self._value).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float | None) -> Response:
        observed_timeouts.append(timeout)
        body = json.loads(request.data)
        return Response({
            "jsonrpc": "2.0",
            "id": body["id"],
            "protocol_version": RPC_PROTOCOL_VERSION,
            "result": {
                "protocol_version": RPC_PROTOCOL_VERSION,
                "disposition": "terminal",
            },
        })

    monkeypatch.setattr("agentic_resolver.rpc.urlopen", fake_urlopen)
    client = DSHRpcClient(
        "http://127.0.0.1:8081/rpc",
        "rpc-secret",
        timeout=17,
    )
    client.call_sync(
        "resolution.open",
        _open_params(),
        operation_id="op_rpc",
        operation_payload_digest="sha256:payload",
    )
    client.call_sync("system.health", {})

    assert observed_timeouts == [None, 17]


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


def test_v2_rpc_reconciles_committed_interaction_checkpoint_without_duplicate_execution() -> None:
    """A committed interaction checkpoint is replayed exactly after transport loss."""

    from agentic_resolver.contracts import RPC_PROTOCOL_VERSION_V2
    from agentic_resolver.rpc import DSHRpcClient, DSHRpcServer, InMemoryRpcTransport

    server = DSHRpcServer(token="rpc-secret")
    transport = InMemoryRpcTransport(server)
    client = DSHRpcClient(
        "http://127.0.0.1:8081/rpc",
        "rpc-secret",
        transport=transport,
    )
    params = {
        "resolution_thread_id": "thread-v2",
        "segment_id": "segment-v2",
        "activation_id": "activation-v2",
        "lease_epoch": 2,
    }
    first = client.call_sync(
        "resolution.request_checkpoint",
        params,
        operation_id="interaction-v2",
        operation_payload_digest="sha256:pending",
    )
    second = client.call_sync(
        "resolution.request_checkpoint",
        params,
        operation_id="interaction-v2",
        operation_payload_digest="sha256:pending",
    )
    assert first == second
    assert first["protocol_version"] == RPC_PROTOCOL_VERSION_V2
    assert first["disposition"] == "checkpointed"
    assert server.operations.execution_count("interaction-v2") == 1
