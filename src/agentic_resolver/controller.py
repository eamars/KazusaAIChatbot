"""Standalone resolution lifecycle controller."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from typing import Any, Protocol
from uuid import uuid4

from agentic_resolver.contracts import (
    SEGMENT_SCHEMA_VERSION,
    DSHResolutionIntakeV1,
    DSHResolutionRuntimeV1,
)
from agentic_resolver.errors import (
    OperationIdReuseMismatchError,
    OperationOutcomeUncertainError,
)
from agentic_resolver.persistence import ResolutionThreadRepository


class ResolutionRpc(Protocol):
    async def call(
        self,
        method: str,
        params: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call one sidecar lifecycle method."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _expires() -> str:
    return (datetime.now(UTC) + timedelta(seconds=30)).isoformat().replace(
        "+00:00", "Z"
    )


class ResolutionController:
    """Own thread compatibility, semantic operations, and lease fencing."""

    _COMPATIBILITY_FIELDS = (
        "scope_fingerprint", "audience_fingerprint",
        "resolver_profile_version", "dsh_release", "session_store_epoch",
        "model_route", "tool_catalog_digest", "policy_epoch",
    )
    _LEASE_RENEWAL_SECONDS = 10.0
    _COMMITTED_DISPOSITIONS = frozenset({
        "terminal", "checkpointed", "canceled", "faulted",
    })

    def __init__(
        self,
        repository: ResolutionThreadRepository,
        rpc: ResolutionRpc,
        *,
        owner_id: str,
    ) -> None:
        self._repository = repository
        self._rpc = rpc
        self._owner_id = owner_id

    async def resolve(self, value: Mapping[str, Any]) -> dict[str, Any]:
        intake = DSHResolutionIntakeV1.from_mapping(value)
        if intake.mode == "start":
            return await self.open(intake.to_dict())
        return await self.continue_resolution(intake.to_dict())

    async def open(self, value: Mapping[str, Any]) -> dict[str, Any]:
        intake = DSHResolutionIntakeV1.from_mapping(value)
        runtime = intake.runtime
        record = await self._repository_call(
            "get_thread", runtime.resolution_thread_id
        )
        if record is None:
            await self._repository_call(
                "create_thread",
                runtime.resolution_thread_id,
                runtime.resolution_thread_id,
                intake.model_input.objective,
                runtime.priority,
                runtime.scope_fingerprint,
                runtime.audience_fingerprint,
                self._segment(runtime),
                _now(),
            )
        return await self._activate("resolution.open", intake)

    async def continue_resolution(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        intake = DSHResolutionIntakeV1.from_mapping(value)
        runtime = intake.runtime
        record = await self._repository_call(
            "get_thread", runtime.resolution_thread_id
        )
        if record is None:
            return await self.open({**intake.to_dict(), "mode": "start"})
        current = next(
            segment
            for segment in record.segments
            if segment["segment_id"] == record.current_segment_id
        )
        mismatch = next(
            (
                field
                for field in self._COMPATIBILITY_FIELDS
                if current[field] != getattr(runtime, field)
            ),
            None,
        )
        if mismatch is not None:
            rotated_runtime = DSHResolutionRuntimeV1.from_mapping({
                **runtime.to_dict(),
                "segment_id": f"seg_{uuid4().hex}",
            })
            await self._repository_call(
                "rotate_segment",
                runtime.resolution_thread_id,
                self._segment(rotated_runtime),
                reason=f"{mismatch}_mismatch",
            )
            intake = DSHResolutionIntakeV1(
                schema_version=intake.schema_version,
                mode=intake.mode,
                runtime=rotated_runtime,
                model_input=intake.model_input,
            )
        return await self._activate("resolution.continue", intake)

    async def amend(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
        continuation_delta: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = await self._fence(
            resolution_thread_id, activation_id, lease_epoch
        )
        operation_id = f"op_{uuid4().hex}"
        digest = f"sha256:{operation_id}"
        await self._repository_call(
            "prepare_operation",
            resolution_thread_id,
            operation_id,
            digest,
            "resolution.amend",
            record.current_segment_id,
            activation_id,
            lease_epoch,
        )
        result = await self._rpc.call(
            "resolution.amend",
            {
                "operation_id": operation_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": record.current_segment_id,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
                "amendment": dict(continuation_delta),
            },
            operation_id=operation_id,
            operation_payload_digest=digest,
        )
        await self._repository_call(
            "update_operation",
            resolution_thread_id,
            operation_id,
            disposition=str(result.get("disposition", "faulted")),
            fault_code=self._fault_code(result),
        )
        return result

    async def request_checkpoint(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        return await self._control(
            "resolution.request_checkpoint",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )

    async def cancel(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        return await self._control(
            "resolution.cancel",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )

    async def inspect(self, resolution_thread_id: str) -> dict[str, Any]:
        record = await self._repository_call("get_thread", resolution_thread_id)
        if record is None:
            return {
                "resolution_thread_id": resolution_thread_id,
                "disposition": "not_admitted",
            }
        return {
            "resolution_thread_id": resolution_thread_id,
            "segment_id": record.current_segment_id,
            "state": record.state,
            "lease_epoch": record.lease_epoch,
            "document_revision": record.document_revision,
        }

    async def dispose_activation(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        result = await self._control(
            "resolution.dispose_activation",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )
        await self._repository_call(
            "release_lease",
            resolution_thread_id, activation_id, lease_epoch
        )
        return result

    async def renew_lease(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        return await self._repository_call(
            "renew_lease",
            resolution_thread_id,
            activation_id,
            lease_epoch,
            _expires(),
        )

    async def takeover_expired(
        self, resolution_thread_id: str, *, now: str
    ) -> dict[str, Any]:
        return await self._repository_call(
            "acquire_lease",
            resolution_thread_id,
            f"act_{uuid4().hex}",
            self._owner_id,
            _expires(),
            now,
        )

    async def _activate(
        self, method: str, intake: DSHResolutionIntakeV1
    ) -> dict[str, Any]:
        runtime = intake.runtime
        record = await self._repository_call(
            "get_thread", runtime.resolution_thread_id
        )
        if record is None:
            raise RuntimeError("created thread disappeared")
        segment_id = record.current_segment_id
        existing = await self._repository_call(
            "get_operation",
            runtime.resolution_thread_id,
            runtime.operation_id,
        )
        if existing is not None:
            if (
                existing["operation_payload_digest"]
                != runtime.operation_payload_digest
                or existing["method"] != method
            ):
                raise OperationIdReuseMismatchError(
                    "operation id was reused with a different method or digest"
                )
            inspected = await self._rpc.call(
                "resolution.inspect",
                {
                    "operation_id": runtime.operation_id,
                    "operation_payload_digest": (
                        runtime.operation_payload_digest
                    ),
                },
            )
            inspected_disposition = inspected.get("disposition")
            if inspected_disposition == "unknown":
                raise OperationOutcomeUncertainError(
                    "operation outcome remains unknown"
                )
            has_current_fence = self._operation_has_current_fence(
                record, existing
            )
            if inspected_disposition in {"not_admitted", "admitted_active"}:
                if has_current_fence:
                    result = await self._call_existing_activation(
                        method,
                        intake,
                        existing,
                    )
                    return await self._complete_activation(
                        runtime.resolution_thread_id,
                        str(existing["segment_id"]),
                        runtime.operation_id,
                        str(existing["activation_id"]),
                        int(existing["lease_epoch"]),
                        result,
                    )
                if inspected_disposition == "admitted_active":
                    raise OperationOutcomeUncertainError(
                        "admitted operation has no matching controller fence"
                    )
            elif inspected_disposition in self._COMMITTED_DISPOSITIONS:
                activation_id = existing.get("activation_id")
                lease_epoch = existing.get("lease_epoch")
                return await self._complete_activation(
                    runtime.resolution_thread_id,
                    str(existing["segment_id"]),
                    runtime.operation_id,
                    activation_id if isinstance(activation_id, str) else None,
                    lease_epoch if isinstance(lease_epoch, int) else None,
                    inspected,
                )
            else:
                raise OperationOutcomeUncertainError(
                    "operation inspection returned an unsupported disposition"
                )

        activation_id = f"act_{uuid4().hex}"
        lease = await self._repository_call(
            "acquire_lease",
            runtime.resolution_thread_id,
            activation_id,
            self._owner_id,
            _expires(),
            _now(),
        )
        prepared = await self._repository_call(
            "prepare_operation",
            runtime.resolution_thread_id,
            runtime.operation_id,
            runtime.operation_payload_digest,
            method,
            segment_id,
            activation_id,
            lease["lease_epoch"],
        )
        if (
            prepared["operation_payload_digest"]
            != runtime.operation_payload_digest
        ):
            raise OperationIdReuseMismatchError(
                "operation id was reused with a different digest"
            )
        result = await self._call_with_lease_renewal(
            method,
            {
                "operation_id": runtime.operation_id,
                "operation_payload_digest": (
                    runtime.operation_payload_digest
                ),
                "activation_id": activation_id,
                "lease_epoch": lease["lease_epoch"],
                "intake": {
                    **intake.to_dict(),
                    "runtime": {
                        **runtime.to_dict(),
                        "segment_id": segment_id,
                    },
                },
            },
            runtime.resolution_thread_id,
            activation_id,
            lease["lease_epoch"],
            runtime.operation_id,
            runtime.operation_payload_digest,
        )
        return await self._complete_activation(
            runtime.resolution_thread_id,
            segment_id,
            runtime.operation_id,
            activation_id,
            lease["lease_epoch"],
            result,
        )

    async def _call_existing_activation(
        self,
        method: str,
        intake: DSHResolutionIntakeV1,
        operation: Mapping[str, Any],
    ) -> dict[str, Any]:
        activation_id = operation["activation_id"]
        lease_epoch = operation["lease_epoch"]
        segment_id = operation["segment_id"]
        if (
            not isinstance(activation_id, str)
            or not isinstance(lease_epoch, int)
            or not isinstance(segment_id, str)
        ):
            raise OperationOutcomeUncertainError(
                "prepared operation has no reusable activation fence"
            )
        runtime = intake.runtime
        return await self._call_with_lease_renewal(
            method,
            {
                "operation_id": runtime.operation_id,
                "operation_payload_digest": (
                    runtime.operation_payload_digest
                ),
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
                "intake": {
                    **intake.to_dict(),
                    "runtime": {
                        **runtime.to_dict(),
                        "segment_id": segment_id,
                    },
                },
            },
            runtime.resolution_thread_id,
            activation_id,
            lease_epoch,
            runtime.operation_id,
            runtime.operation_payload_digest,
        )

    async def _complete_activation(
        self,
        resolution_thread_id: str,
        segment_id: str,
        operation_id: str,
        activation_id: str | None,
        lease_epoch: int | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        disposition = str(result.get("disposition", "faulted"))
        last_committed_seq = result.get("last_committed_seq")
        if not isinstance(last_committed_seq, int):
            exhaust = result.get("exhaust")
            if isinstance(exhaust, Mapping):
                candidate = exhaust.get("last_committed_seq")
                if isinstance(candidate, int):
                    last_committed_seq = candidate
        message_source_id = result.get("dsh_message_source_id")
        if not isinstance(message_source_id, str):
            message_source_id = None
        await self._repository_call(
            "update_operation",
            resolution_thread_id,
            operation_id,
            disposition=disposition,
            dsh_message_source_id=message_source_id,
            last_committed_seq=last_committed_seq,
            fault_code=self._fault_code(result),
        )
        session_id = result.get("session_id")
        if not isinstance(session_id, str):
            session_id = self._dsh_session_id(
                resolution_thread_id, segment_id
            )
        segment_state = {
            "admitted_active": "live",
            "terminal": "terminal",
            "checkpointed": "checkpointed",
            "canceled": "canceled",
            "faulted": "faulted",
        }.get(disposition, "faulted")
        segment_changes: dict[str, Any] = {
            "dsh_session_id": session_id,
            "state": segment_state,
            "last_used_at": _now(),
        }
        if isinstance(last_committed_seq, int):
            segment_changes["last_committed_seq"] = last_committed_seq
        await self._repository_call(
            "update_segment",
            resolution_thread_id,
            segment_id,
            **segment_changes,
        )
        if (
            disposition in self._COMMITTED_DISPOSITIONS
            and activation_id is not None
            and lease_epoch is not None
        ):
            await self._dispose_and_release_if_current(
                resolution_thread_id,
                segment_id,
                activation_id,
                lease_epoch,
            )
        return self._activation_result(
            result,
            resolution_thread_id,
            segment_id,
            activation_id,
            lease_epoch,
        )

    async def _dispose_and_release_if_current(
        self,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> None:
        record = await self._repository_call(
            "get_thread", resolution_thread_id
        )
        if record is None:
            raise RuntimeError("resolution thread disappeared during disposal")
        current_lease = record.current_lease
        if (
            current_lease is None
            or current_lease.get("activation_id") != activation_id
            or current_lease.get("lease_epoch") != lease_epoch
        ):
            return
        await self._dispose_sidecar_activation(
            resolution_thread_id,
            segment_id,
            activation_id,
            lease_epoch,
        )
        await self._repository_call(
            "release_lease",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )

    @staticmethod
    def _operation_has_current_fence(
        record: Any, operation: Mapping[str, Any]
    ) -> bool:
        current_lease = record.current_lease
        return (
            current_lease is not None
            and isinstance(operation.get("activation_id"), str)
            and isinstance(operation.get("lease_epoch"), int)
            and current_lease.get("activation_id")
            == operation.get("activation_id")
            and current_lease.get("lease_epoch")
            == operation.get("lease_epoch")
        )

    async def _call_with_lease_renewal(
        self,
        method: str,
        params: Mapping[str, Any],
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
        operation_id: str,
        payload_digest: str,
    ) -> dict[str, Any]:
        stopped = asyncio.Event()

        async def renew_until_stopped() -> None:
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(
                        stopped.wait(),
                        timeout=self._LEASE_RENEWAL_SECONDS,
                    )
                except TimeoutError:
                    await self._repository_call(
                        "renew_lease",
                        resolution_thread_id,
                        activation_id,
                        lease_epoch,
                        _expires(),
                    )

        renewal = asyncio.create_task(renew_until_stopped())
        try:
            result = await self._rpc.call(
                method,
                params,
                operation_id=operation_id,
                operation_payload_digest=payload_digest,
            )
        finally:
            stopped.set()
            await renewal
        return result

    async def _dispose_sidecar_activation(
        self,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> None:
        operation_id = f"op_{uuid4().hex}"
        digest = f"sha256:{operation_id}"
        await self._rpc.call(
            "resolution.dispose_activation",
            {
                "operation_id": operation_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": segment_id,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
            },
            operation_id=operation_id,
            operation_payload_digest=digest,
        )

    @staticmethod
    def _fault_code(result: Mapping[str, Any]) -> str | None:
        exhaust = result.get("exhaust")
        if not isinstance(exhaust, Mapping):
            return None
        fault = exhaust.get("fault")
        if not isinstance(fault, Mapping):
            return None
        code = fault.get("code")
        return code if isinstance(code, str) else None

    @staticmethod
    def _activation_result(
        result: Mapping[str, Any],
        resolution_thread_id: str,
        segment_id: str,
        activation_id: object,
        lease_epoch: object,
    ) -> dict[str, Any]:
        return {
            **result,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
        }

    async def _control(
        self,
        method: str,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        record = await self._fence(
            resolution_thread_id, activation_id, lease_epoch
        )
        operation_id = f"op_{uuid4().hex}"
        digest = f"sha256:{operation_id}"
        await self._repository_call(
            "prepare_operation",
            resolution_thread_id,
            operation_id,
            digest,
            method,
            record.current_segment_id,
            activation_id,
            lease_epoch,
        )
        result = await self._rpc.call(
            method,
            {
                "operation_id": operation_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": record.current_segment_id,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
            },
            operation_id=operation_id,
            operation_payload_digest=digest,
        )
        disposition = str(result.get("disposition", "faulted"))
        last_committed_seq = result.get("last_committed_seq")
        if not isinstance(last_committed_seq, int):
            last_committed_seq = None
        message_source_id = result.get("dsh_message_source_id")
        if not isinstance(message_source_id, str):
            message_source_id = None
        await self._repository_call(
            "update_operation",
            resolution_thread_id,
            operation_id,
            disposition=disposition,
            dsh_message_source_id=message_source_id,
            last_committed_seq=last_committed_seq,
            fault_code=self._fault_code(result),
        )
        if method in {"resolution.request_checkpoint", "resolution.cancel"}:
            state = (
                "checkpointed"
                if method == "resolution.request_checkpoint"
                else "canceled"
            )
            changes: dict[str, Any] = {
                "state": state,
                "last_used_at": _now(),
            }
            if isinstance(last_committed_seq, int):
                changes["last_committed_seq"] = last_committed_seq
            await self._repository_call(
                "update_segment",
                resolution_thread_id,
                record.current_segment_id,
                **changes,
            )
            await self._dispose_sidecar_activation(
                resolution_thread_id,
                record.current_segment_id,
                activation_id,
                lease_epoch,
            )
            await self._repository_call(
                "release_lease",
                resolution_thread_id,
                activation_id,
                lease_epoch,
            )
        return result

    async def _fence(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ):
        await self._repository_call(
            "validate_fence",
            resolution_thread_id, activation_id, lease_epoch
        )
        record = await self._repository_call("get_thread", resolution_thread_id)
        if record is None:
            raise RuntimeError("validated thread disappeared")
        return record

    async def _repository_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._repository, name)
        value = method(*args, **kwargs)
        return await value if isawaitable(value) else value

    @staticmethod
    def _segment(runtime: DSHResolutionRuntimeV1) -> dict[str, Any]:
        now = _now()
        return {
            "schema_version": SEGMENT_SCHEMA_VERSION,
            "segment_id": runtime.segment_id,
            "resolution_thread_id": runtime.resolution_thread_id,
            "dsh_session_id": ResolutionController._dsh_session_id(
                runtime.resolution_thread_id, runtime.segment_id
            ),
            "resolver_profile_version": runtime.resolver_profile_version,
            "dsh_release": runtime.dsh_release,
            "session_store_epoch": runtime.session_store_epoch,
            "tool_catalog_digest": runtime.tool_catalog_digest,
            "policy_epoch": runtime.policy_epoch,
            "scope_fingerprint": runtime.scope_fingerprint,
            "audience_fingerprint": runtime.audience_fingerprint,
            "model_route": runtime.model_route,
            "state": "live",
            "last_committed_seq": 0,
            "parent_segment_id": None,
            "rotation_reason": None,
            "created_at": now,
            "last_used_at": now,
        }

    @staticmethod
    def _dsh_session_id(
        resolution_thread_id: str, segment_id: str
    ) -> str:
        identity = f"{resolution_thread_id}\0{segment_id}".encode()
        suffix = hashlib.sha256(identity).hexdigest()[:32]
        return f"kazusa-resolution-{suffix}"
