"""Authenticated loopback JSON-RPC client and deterministic test server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from secrets import token_hex
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from agentic_resolver.contracts import RPC_PROTOCOL_VERSION
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
_COMMITTED = frozenset({"checkpointed", "terminal", "canceled", "faulted"})


class RpcTransport(Protocol):
    async def send(
        self, frame: Mapping[str, Any], authorization: str
    ) -> Mapping[str, Any]:
        """Send one JSON-RPC frame and return one response frame."""


@dataclass(slots=True)
class _Operation:
    operation_id: str
    payload_digest: str
    method: str
    disposition: str = "admitted_active"
    result: dict[str, Any] | None = None
    admissions: int = 1
    executions: int = 0


class _OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, _Operation] = {}

    def admit(self, operation_id: str, digest: str, method: str) -> _Operation:
        existing = self._operations.get(operation_id)
        if existing is not None:
            if existing.payload_digest != digest:
                raise OperationIdReuseMismatchError(
                    "OPERATION_ID_REUSE_MISMATCH"
                )
            return existing
        operation = _Operation(operation_id, digest, method)
        self._operations[operation_id] = operation
        return operation

    def inspect(self, operation_id: str) -> dict[str, Any]:
        operation = self._operations.get(operation_id)
        if operation is None:
            return {"disposition": "not_admitted"}
        result: dict[str, Any] = {"disposition": operation.disposition}
        if operation.result is not None:
            result.update(operation.result)
        return result

    def admission_count(self, operation_id: str) -> int:
        operation = self._operations.get(operation_id)
        return operation.admissions if operation else 0

    def execution_count(self, operation_id: str) -> int:
        operation = self._operations.get(operation_id)
        return operation.executions if operation else 0


class DSHRpcServer:
    """In-memory strict dispatcher used for transport unit tests."""

    def __init__(
        self,
        *,
        token: str,
        handlers: Mapping[
            str, Callable[[dict[str, Any]], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
        ] | None = None,
    ) -> None:
        if not token:
            raise RpcAuthenticationError("RPC token must be non-empty")
        self._token = token
        self._handlers = dict(handlers or {})
        self.operations = _OperationRegistry()

    def dispatch(
        self, frame: object, *, authorization: str
    ) -> dict[str, Any]:
        if authorization != f"Bearer {self._token}":
            raise RpcAuthenticationError("RPC authentication failed")
        if not isinstance(frame, Mapping):
            raise RpcContractError("RPC frame must be an object")
        if frame.get("jsonrpc") != "2.0":
            raise RpcContractError("jsonrpc must be 2.0")
        if set(frame) != {"jsonrpc", "id", "method", "params"}:
            raise RpcContractError("RPC request has unknown or missing fields")
        method = frame.get("method")
        params = frame.get("params")
        if not isinstance(method, str) or method not in _METHODS:
            raise RpcContractError("RPC method is unsupported")
        if not isinstance(params, Mapping):
            raise RpcContractError("RPC params must be an object")
        if params.get("protocol_version") != RPC_PROTOCOL_VERSION:
            raise RpcContractError("RPC protocol version is unsupported")
        result = self._dispatch_method(method, dict(params))
        return {
            "jsonrpc": "2.0",
            "id": frame.get("id"),
            "protocol_version": RPC_PROTOCOL_VERSION,
            "result": result,
        }

    def _dispatch_method(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        if method == "system.health":
            return {"protocol_version": RPC_PROTOCOL_VERSION, "status": "ok"}
        if method == "resolution.inspect":
            operation_id = params.get("operation_id")
            if not isinstance(operation_id, str):
                raise RpcContractError("operation_id is required")
            return {
                "protocol_version": RPC_PROTOCOL_VERSION,
                **self.operations.inspect(operation_id),
            }
        operation_id = params.get("operation_id")
        digest = params.get("operation_payload_digest")
        if not isinstance(operation_id, str) or not isinstance(digest, str):
            raise RpcContractError("semantic operation identity is required")
        operation = self.operations.admit(operation_id, digest, method)
        if operation.result is not None:
            return operation.result
        operation.executions += 1
        handler = self._handlers.get(method)
        if handler is not None:
            candidate = handler(params)
            if isinstance(candidate, Awaitable):
                raise RpcContractError("sync test server cannot await handler")
            result = dict(candidate)
        elif method in {"resolution.open", "resolution.continue"}:
            intake = params.get("intake")
            if not isinstance(intake, Mapping):
                raise RpcContractError("intake is required")
            runtime = intake.get("runtime", {})
            terminal = {
                "status": "resolved",
                "summary": "resolved by deterministic RPC fixture",
                "findings": [],
                "completed_subgoals": [],
                "remaining_needs": [],
                "clarification_request": None,
                "approval_request": None,
                "artifact_refs": [],
                "warnings": [],
            }
            result = {
                "protocol_version": RPC_PROTOCOL_VERSION,
                "disposition": "terminal",
                "intake": dict(intake),
                "exhaust": {
                    "kind": "terminal",
                    "terminal": terminal,
                    "evidence": [],
                    "identity": {
                        key: runtime[key]
                        for key in (
                            "operation_id", "operation_payload_digest",
                            "request_id", "resolution_thread_id", "segment_id",
                            "scope_fingerprint", "audience_fingerprint",
                            "resolver_profile_version", "dsh_release",
                            "session_store_epoch", "model_route",
                            "tool_catalog_digest", "policy_epoch",
                        )
                    } | {"activation_id": params.get("activation_id", "act_1"), "lease_epoch": params.get("lease_epoch", 1)},
                    "usage": {},
                    "last_committed_seq": 1,
                },
            }
        else:
            disposition = {
                "resolution.request_checkpoint": "checkpointed",
                "resolution.cancel": "canceled",
                "resolution.dispose_activation": "canceled",
            }.get(method, "admitted_active")
            result = {
                "protocol_version": RPC_PROTOCOL_VERSION,
                "disposition": disposition,
            }
        operation.disposition = str(result.get("disposition", "admitted_active"))
        operation.result = result
        return result


class InMemoryRpcTransport:
    """Fault-injectable transport for semantic reconciliation tests."""

    def __init__(self, server: DSHRpcServer) -> None:
        self._server = server
        self.fail_next_before_dispatch = False
        self.fail_next_after_admission = False
        self.fail_next_after_commit = False
        self.return_unknown_inspection = False

    async def send(
        self, frame: Mapping[str, Any], authorization: str
    ) -> Mapping[str, Any]:
        method = frame.get("method")
        if self.return_unknown_inspection:
            if method == "resolution.inspect":
                return {
                    "jsonrpc": "2.0",
                    "id": frame.get("id"),
                    "protocol_version": RPC_PROTOCOL_VERSION,
                    "result": {
                        "protocol_version": RPC_PROTOCOL_VERSION,
                        "disposition": "unknown",
                    },
                }
            raise RpcTransportError("injected ambiguous transport failure")
        if self.fail_next_before_dispatch:
            self.fail_next_before_dispatch = False
            raise RpcTransportError("injected pre-admission transport failure")
        response = self._server.dispatch(frame, authorization=authorization)
        if self.fail_next_after_admission:
            self.fail_next_after_admission = False
            raise RpcTransportError("injected post-admission transport failure")
        if self.fail_next_after_commit:
            self.fail_next_after_commit = False
            raise RpcTransportError("injected post-commit transport failure")
        return response


class _HttpRpcTransport:
    def __init__(self, endpoint: str, timeout: float) -> None:
        self._endpoint = endpoint
        self._timeout = timeout

    async def send(
        self, frame: Mapping[str, Any], authorization: str
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(self._send_sync, frame, authorization)

    def _send_sync(
        self, frame: Mapping[str, Any], authorization: str
    ) -> Mapping[str, Any]:
        request = Request(
            self._endpoint,
            data=json.dumps(frame, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                value = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RpcTransportError("sidecar loopback transport failed") from exc
        if not isinstance(value, Mapping):
            raise RpcContractError("RPC response must be an object")
        return value


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
        payload["protocol_version"] = RPC_PROTOCOL_VERSION
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
                frame, f"Bearer {self._token}"
            )
        except RpcTransportError:
            if method not in _MUTATING:
                raise
            return await self._reconcile_after_disconnect(
                frame, operation_id, operation_payload_digest
            )
        return self._validate_response(response, frame["id"])

    def call_sync(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return asyncio.run(self.call(*args, **kwargs))

    async def inspect_operation(
        self, operation_id: str, payload_digest: str
    ) -> dict[str, Any]:
        return await self.call(
            "resolution.inspect",
            {
                "operation_id": operation_id,
                "operation_payload_digest": payload_digest,
            },
        )

    async def reconcile(
        self, operation_id: str, payload_digest: str
    ) -> dict[str, Any]:
        inspected = await self.inspect_operation(operation_id, payload_digest)
        disposition = inspected.get("disposition")
        if disposition == "unknown":
            raise OperationOutcomeUncertainError(
                "operation outcome remains unknown"
            )
        if disposition == "not_admitted":
            raise OperationOutcomeUncertainError("operation was not admitted")
        if disposition in _COMMITTED or disposition == "admitted_active":
            return inspected
        raise RpcContractError("inspection disposition is unsupported")

    def reconcile_sync(
        self, operation_id: str, payload_digest: str
    ) -> dict[str, Any]:
        return asyncio.run(self.reconcile(operation_id, payload_digest))

    async def _reconcile_after_disconnect(
        self,
        original_frame: Mapping[str, Any],
        operation_id: str,
        payload_digest: str,
    ) -> dict[str, Any]:
        inspected = await self.inspect_operation(operation_id, payload_digest)
        disposition = inspected.get("disposition")
        if disposition == "not_admitted":
            response = await self._transport.send(
                original_frame, f"Bearer {self._token}"
            )
            return self._validate_response(response, original_frame["id"])
        if disposition == "unknown":
            raise OperationOutcomeUncertainError(
                "operation outcome remains unknown"
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
        if response.get("protocol_version") != RPC_PROTOCOL_VERSION:
            raise RpcContractError("RPC response protocol mismatch")
        if "error" in response:
            error = response["error"]
            if isinstance(error, Mapping) and error.get("code") == (
                "OPERATION_ID_REUSE_MISMATCH"
            ):
                raise OperationIdReuseMismatchError(
                    "operation id reuse mismatch"
                )
            raise RpcContractError("sidecar returned a bounded RPC error")
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise RpcContractError("RPC result must be an object")
        return dict(result)
