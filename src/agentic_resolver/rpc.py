"""Authenticated loopback JSON-RPC client and operation reconciliation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from secrets import token_hex
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from agentic_resolver.contracts import RPC_PROTOCOL_VERSION_V2
from agentic_resolver.errors import (
    OperationIdReuseMismatchError,
    OperationOutcomeUncertainError,
    RpcAuthenticationError,
    RpcContractError,
    RpcTransportError,
)

_METHODS = frozenset({
    "system.health", "resolution.open", "resolution.continue",
    "resolution.amend", "resolution.request_checkpoint", "resolution.cancel",
    "resolution.inspect", "resolution.dispose_activation",
})
_MUTATING = _METHODS - {"system.health", "resolution.inspect"}
_LONG_RUNNING = frozenset({"resolution.open", "resolution.continue"})
_COMMITTED = frozenset({"checkpointed", "terminal", "canceled", "faulted"})


class RpcTransport(Protocol):
    async def send(
        self, frame: Mapping[str, Any], authorization: str
    ) -> Mapping[str, Any]:
        """Send one JSON-RPC frame and return one response frame."""


class _HttpRpcTransport:
    """Own one cancellable HTTP request and its connection lifetime."""

    def __init__(self, endpoint: str, timeout: float) -> None:
        self._endpoint = endpoint
        self._control_timeout = timeout

    async def send(
        self, frame: Mapping[str, Any], authorization: str
    ) -> Mapping[str, Any]:
        timeout = (
            None
            if frame.get("method") in _LONG_RUNNING
            else self._control_timeout
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self._endpoint,
                    content=json.dumps(frame, separators=(",", ":")),
                    headers={
                        "Authorization": authorization,
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                value = response.json()
        except httpx.HTTPError as exc:
            detail = str(exc) or type(exc).__name__
            raise RpcTransportError(
                f"sidecar loopback transport failed: {detail}",
            ) from exc
        except json.JSONDecodeError as exc:
            raise RpcContractError(
                f"sidecar returned invalid JSON: {exc}",
            ) from exc
        if not isinstance(value, Mapping):
            raise RpcContractError("RPC response must be an object")
        response_frame = dict(value)
        return response_frame


class DSHRpcClient:
    """Reusable strict JSON-RPC client with semantic outcome reconciliation."""

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        timeout: float = 30.0,
        transport: RpcTransport | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.path != "/rpc"
        ):
            raise RpcContractError("RPC endpoint must be loopback /rpc")
        if not token:
            raise RpcAuthenticationError("RPC token must be non-empty")
        self._token = token
        self._transport = transport or _HttpRpcTransport(endpoint, timeout)

    async def call(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        operation_payload_digest: str | None = None,
    ) -> dict[str, Any]:
        if method not in _METHODS:
            raise RpcContractError("RPC method is unsupported")
        payload = dict(params)
        payload["protocol_version"] = RPC_PROTOCOL_VERSION_V2
        if operation_id is not None:
            payload["operation_id"] = operation_id
        if operation_payload_digest is not None:
            payload["operation_payload_digest"] = operation_payload_digest
        if method in _MUTATING and (
            operation_id is None or operation_payload_digest is None
        ):
            raise RpcContractError("mutating RPC requires semantic identity")
        frame = {
            "jsonrpc": "2.0",
            "id": f"rpc_{token_hex(12)}",
            "method": method,
            "params": payload,
        }
        try:
            response = await self._transport.send(
                frame, f"Bearer {self._token}",
            )
        except RpcTransportError:
            if method not in _MUTATING:
                raise
            reconciled = await self._reconcile_after_disconnect(
                frame,
                operation_id,
                operation_payload_digest,
            )
            return reconciled
        result = self._validate_response(response, frame["id"])
        return result

    async def inspect_operation(
        self, operation_id: str, payload_digest: str
    ) -> dict[str, Any]:
        inspected = await self.call(
            "resolution.inspect",
            {
                "operation_id": operation_id,
                "operation_payload_digest": payload_digest,
            },
        )
        return inspected

    async def reconcile(
        self, operation_id: str, payload_digest: str
    ) -> dict[str, Any]:
        inspected = await self.inspect_operation(operation_id, payload_digest)
        disposition = inspected.get("disposition")
        if disposition == "unknown":
            raise OperationOutcomeUncertainError(
                "operation outcome remains unknown",
            )
        if disposition == "not_admitted":
            raise OperationOutcomeUncertainError("operation was not admitted")
        if disposition in _COMMITTED or disposition == "admitted_active":
            return inspected
        raise RpcContractError("inspection disposition is unsupported")

    async def _reconcile_after_disconnect(
        self,
        original_frame: Mapping[str, Any],
        operation_id: str | None,
        payload_digest: str | None,
    ) -> dict[str, Any]:
        if operation_id is None or payload_digest is None:
            raise RpcContractError("mutating RPC requires semantic identity")
        inspected = await self.inspect_operation(operation_id, payload_digest)
        disposition = inspected.get("disposition")
        if disposition == "not_admitted":
            response = await self._transport.send(
                original_frame,
                f"Bearer {self._token}",
            )
            retried = self._validate_response(response, original_frame["id"])
            return retried
        if disposition == "unknown":
            raise OperationOutcomeUncertainError(
                "operation outcome remains unknown",
            )
        if disposition in _COMMITTED or disposition == "admitted_active":
            return inspected
        raise RpcContractError("inspection disposition is unsupported")

    @staticmethod
    def _validate_response(
        response: Mapping[str, Any], request_id: object
    ) -> dict[str, Any]:
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise RpcContractError("RPC response identity mismatch")
        if response.get("protocol_version") != RPC_PROTOCOL_VERSION_V2:
            raise RpcContractError("RPC response protocol mismatch")
        if "error" in response:
            error = response["error"]
            if isinstance(error, Mapping) and error.get("code") == (
                "OPERATION_ID_REUSE_MISMATCH"
            ):
                raise OperationIdReuseMismatchError(
                    "operation id reuse mismatch",
                )
            raise RpcContractError("sidecar returned a bounded RPC error")
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise RpcContractError("RPC result must be an object")
        validated_result = dict(result)
        return validated_result
