"""Deterministic DSH transport and runtime-readiness owner tests."""

from __future__ import annotations

import socket
from collections.abc import Mapping
from typing import Any

import pytest

from agentic_resolver.errors import RpcTransportError
from agentic_resolver.rpc import _HttpRpcTransport
from agentic_resolver.runtime import AgenticResolverRuntime


class _ReadinessController:
    """Expose one authenticated sidecar identity through the runtime seam."""

    def __init__(self) -> None:
        self.calls = 0

    async def resolve(self, intake: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError(f"resolve was not expected: {intake}")

    async def readiness(self) -> Mapping[str, str]:
        self.calls += 1
        return {
            "status": "ready",
            "route_digest": "route-digest",
            "semantic_catalog_digest": "catalog-digest",
        }


def _unused_loopback_port() -> int:
    """Reserve and release one loopback port for a refused connection."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    return port


@pytest.mark.asyncio
async def test_transport_error_preserves_low_level_cause() -> None:
    """A loopback failure should retain its concrete OS/network explanation."""

    port = _unused_loopback_port()
    transport = _HttpRpcTransport(f"http://127.0.0.1:{port}/rpc", 0.25)

    with pytest.raises(RpcTransportError) as exc_info:
        await transport.send(
            {
                "jsonrpc": "2.0",
                "id": "health-test",
                "method": "system.health",
                "params": {
                    "protocol_version": "kazusa.dsh-resolution-rpc.v2",
                },
            },
            "Bearer test-token",
        )

    cause = exc_info.value.__cause__
    assert cause is not None
    assert str(cause)
    assert str(cause) in str(exc_info.value)


@pytest.mark.asyncio
async def test_runtime_readiness_delegates_to_authenticated_sidecar_health() -> None:
    """Brain readiness should consume the controller's real sidecar probe."""

    controller = _ReadinessController()
    runtime = AgenticResolverRuntime(controller)

    readiness = await runtime.readiness()

    assert readiness == {
        "status": "ready",
        "route_digest": "route-digest",
        "semantic_catalog_digest": "catalog-digest",
    }
    assert controller.calls == 1

