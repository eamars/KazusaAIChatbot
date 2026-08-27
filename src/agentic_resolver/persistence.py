"""Controller-facing resolution thread repository boundary."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from agentic_resolver.contracts import (
    THREAD_SCHEMA_VERSION,
    ResolutionThreadRecordV1,
)
from agentic_resolver.errors import (
    DuplicateActivationError,
    OperationIdReuseMismatchError,
    ResolutionPersistenceError,
    StaleActivationOrLeaseError,
)
from kazusa_ai_chatbot.db import resolution_threads


def _future(now: str, days: int = 1) -> str:
    parsed = datetime.fromisoformat(now.replace("Z", "+00:00"))
    return (parsed + timedelta(days=days)).astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    )


class ResolutionThreadRepository(Protocol):
    """Lifecycle metadata operations required by the controller."""

    def get_thread(self, resolution_thread_id: str) -> ResolutionThreadRecordV1 | None:
        """Return one validated thread record."""

    def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        """Return one durable semantic operation when present."""


class InMemoryResolutionThreadRepository:
    """Deterministic repository used by unit tests and local composition."""

    def __init__(self) -> None:
        self._threads: dict[str, dict[str, Any]] = {}

    def create_thread(
        self,
        resolution_thread_id: str,
        brain_conversation_ref: str,
        root_goal_ref: str,
        priority: str,
        scope_fingerprint: str,
        audience_fingerprint: str,
        segment: dict[str, Any],
        now: str,
    ) -> ResolutionThreadRecordV1:
        if resolution_thread_id in self._threads:
            raise ResolutionPersistenceError("resolution thread already exists")
        document = {
            "schema_version": THREAD_SCHEMA_VERSION,
            "resolution_thread_id": resolution_thread_id,
            "brain_conversation_ref": brain_conversation_ref,
            "root_goal_ref": root_goal_ref,
            "current_segment_id": segment["segment_id"],
            "state": "active",
            "priority": priority,
            "audience_fingerprint": audience_fingerprint,
            "scope_fingerprint": scope_fingerprint,
            "created_at": now,
            "updated_at": now,
            "last_terminal_status": None,
            "continuation_eligible_until": _future(now),
            "document_revision": 1,
            "lease_epoch": 0,
            "current_lease": None,
            "segments": [deepcopy(segment)],
            "operations": [],
        }
        validated = ResolutionThreadRecordV1.from_mapping(document)
        self._threads[resolution_thread_id] = validated.to_dict()
        return validated

    def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV1 | None:
        document = self._threads.get(resolution_thread_id)
        if document is None:
            return None
        return ResolutionThreadRecordV1.from_mapping(deepcopy(document))

    def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        document = self._threads.get(resolution_thread_id)
        if document is None:
            return None
        for operation in document["operations"]:
            if operation["operation_id"] == operation_id:
                return deepcopy(operation)
        return None

    def prepare_operation(
        self,
        resolution_thread_id: str,
        operation_id: str,
        payload_digest: str,
        method: str,
        segment_id: str,
        activation_id: str | None = None,
        lease_epoch: int | None = None,
    ) -> dict[str, Any]:
        document = self._required(resolution_thread_id)
        for operation in document["operations"]:
            if operation["operation_id"] != operation_id:
                continue
            if operation["operation_payload_digest"] != payload_digest:
                raise OperationIdReuseMismatchError(
                    "operation id was reused with a different digest"
                )
            return deepcopy(operation)
        operation = {
            "operation_id": operation_id,
            "operation_payload_digest": payload_digest,
            "method": method,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "dsh_message_source_id": None,
            "disposition": "prepared",
            "last_committed_seq": None,
            "outcome_digest": None,
            "fault_code": None,
        }
        document["operations"].append(operation)
        self._touch(document)
        return deepcopy(operation)

    def update_operation(
        self,
        resolution_thread_id: str,
        operation_id: str,
        *,
        disposition: str,
        dsh_message_source_id: str | None = None,
        last_committed_seq: int | None = None,
        outcome_digest: str | None = None,
        fault_code: str | None = None,
    ) -> dict[str, Any]:
        document = self._required(resolution_thread_id)
        operation = self._operation(document, operation_id)
        operation.update({
            "disposition": disposition,
            "dsh_message_source_id": dsh_message_source_id,
            "last_committed_seq": last_committed_seq,
            "outcome_digest": outcome_digest,
            "fault_code": fault_code,
        })
        self._touch(document)
        return deepcopy(operation)

    def acquire_lease(
        self,
        resolution_thread_id: str,
        activation_id: str,
        owner_id: str,
        expires_at: str,
        now: str,
    ) -> dict[str, Any]:
        document = self._required(resolution_thread_id)
        current = document["current_lease"]
        if current is not None and current["expires_at"] > now:
            if current["activation_id"] != activation_id:
                raise DuplicateActivationError("segment already has a live lease")
            return deepcopy(current)
        document["lease_epoch"] += 1
        lease = {
            "activation_id": activation_id,
            "lease_epoch": document["lease_epoch"],
            "owner_id": owner_id,
            "expires_at": expires_at,
        }
        document["current_lease"] = lease
        self._touch(document, now)
        return deepcopy(lease)

    def renew_lease(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
        expires_at: str,
    ) -> dict[str, Any]:
        document = self._required(resolution_thread_id)
        lease = self._fenced_lease(document, activation_id, lease_epoch)
        lease["expires_at"] = expires_at
        self._touch(document)
        return deepcopy(lease)

    def release_lease(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> None:
        document = self._required(resolution_thread_id)
        self._fenced_lease(document, activation_id, lease_epoch)
        document["current_lease"] = None
        self._touch(document)

    def rotate_segment(
        self,
        resolution_thread_id: str,
        segment: dict[str, Any],
        *,
        reason: str,
    ) -> ResolutionThreadRecordV1:
        document = self._required(resolution_thread_id)
        new_segment = deepcopy(segment)
        new_segment["rotation_reason"] = reason
        new_segment["parent_segment_id"] = document["current_segment_id"]
        document["segments"].append(new_segment)
        document["current_segment_id"] = new_segment["segment_id"]
        document["current_lease"] = None
        self._touch(document)
        return ResolutionThreadRecordV1.from_mapping(deepcopy(document))

    def update_segment(
        self,
        resolution_thread_id: str,
        segment_id: str,
        **changes: Any,
    ) -> ResolutionThreadRecordV1:
        document = self._required(resolution_thread_id)
        segment = next(
            (
                item
                for item in document["segments"]
                if item["segment_id"] == segment_id
            ),
            None,
        )
        if segment is None:
            raise ResolutionPersistenceError("segment does not exist")
        allowed = {"dsh_session_id", "last_committed_seq", "state", "last_used_at"}
        if set(changes) - allowed:
            raise ResolutionPersistenceError("unsupported segment update")
        segment.update(changes)
        self._touch(document)
        return ResolutionThreadRecordV1.from_mapping(deepcopy(document))

    def validate_fence(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        document = self._required(resolution_thread_id)
        return deepcopy(
            self._fenced_lease(document, activation_id, lease_epoch)
        )

    def _required(self, resolution_thread_id: str) -> dict[str, Any]:
        document = self._threads.get(resolution_thread_id)
        if document is None:
            raise ResolutionPersistenceError("resolution thread does not exist")
        return document

    @staticmethod
    def _operation(document: dict[str, Any], operation_id: str) -> dict[str, Any]:
        for operation in document["operations"]:
            if operation["operation_id"] == operation_id:
                return operation
        raise ResolutionPersistenceError("operation does not exist")

    @staticmethod
    def _fenced_lease(
        document: dict[str, Any], activation_id: str, lease_epoch: int
    ) -> dict[str, Any]:
        lease = document["current_lease"]
        if (
            lease is None
            or lease["activation_id"] != activation_id
            or lease["lease_epoch"] != lease_epoch
        ):
            raise StaleActivationOrLeaseError(
                "activation id or lease epoch is stale"
            )
        return lease

    @staticmethod
    def _touch(document: dict[str, Any], now: str | None = None) -> None:
        document["document_revision"] += 1
        if now is not None:
            document["updated_at"] = now


class MongoResolutionThreadRepository:
    """Async adapter that delegates all raw selectors to the DB owner module."""

    def __init__(self) -> None:
        self._db = resolution_threads

    async def ensure_indexes(self) -> None:
        await self._db.ensure_indexes()

    async def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV1 | None:
        value = await self._db.get_thread(resolution_thread_id)
        if value is None:
            return None
        return ResolutionThreadRecordV1.from_mapping(value)

    async def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        return await self._db.get_operation(
            resolution_thread_id, operation_id
        )

    async def create_thread(self, *args: Any, **kwargs: Any) -> ResolutionThreadRecordV1:
        await self.ensure_indexes()
        return ResolutionThreadRecordV1.from_mapping(
            await self._db.create_thread(*args, **kwargs)
        )

    async def prepare_operation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._db.prepare_operation(*args, **kwargs)

    async def update_operation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._db.update_operation(*args, **kwargs)

    async def acquire_lease(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._db.acquire_lease(*args, **kwargs)

    async def renew_lease(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._db.renew_lease(*args, **kwargs)

    async def release_lease(self, *args: Any, **kwargs: Any) -> None:
        await self._db.release_lease(*args, **kwargs)

    async def rotate_segment(self, *args: Any, **kwargs: Any) -> ResolutionThreadRecordV1:
        return ResolutionThreadRecordV1.from_mapping(
            await self._db.rotate_segment(*args, **kwargs)
        )

    async def update_segment(self, *args: Any, **kwargs: Any) -> ResolutionThreadRecordV1:
        return ResolutionThreadRecordV1.from_mapping(
            await self._db.update_segment(*args, **kwargs)
        )

    async def validate_fence(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._db.validate_fence(*args, **kwargs)
