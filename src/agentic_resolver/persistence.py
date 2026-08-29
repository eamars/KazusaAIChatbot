"""Controller-facing V2 resolution-thread repository boundary."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from agentic_resolver.contracts import (
    DSH_RELEASE,
    SESSION_STORE_EPOCH,
    THREAD_SCHEMA_VERSION,
    ResolutionThreadRecordV2,
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
    """Durable lifecycle operations required by the V2 controller."""

    def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV2 | None:
        """Return one validated V2 thread record."""

    def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        """Return one durable operation when present."""


class InMemoryResolutionThreadRepository:
    """Deterministic injected repository used by resolver unit tests."""

    def __init__(self) -> None:
        self._threads: dict[str, dict[str, Any]] = {}

    def create_thread_v2(
        self,
        *,
        resolution_thread_id: str,
        brain_conversation_ref: str,
        root_goal_ref: str,
        priority: str,
        workspace_root: str,
        workspace_fingerprint: str,
        route_digest: str,
        profile_version: str,
        standard_catalog_digest: str,
        semantic_catalog_digest: str,
        scope_fingerprint: str,
        audience_fingerprint: str,
        policy_epoch: str,
        interaction_id: str,
        segment: Mapping[str, Any],
        now: str,
    ) -> ResolutionThreadRecordV2:
        if resolution_thread_id in self._threads:
            existing = self._threads[resolution_thread_id]
            if existing.get("schema_version") != THREAD_SCHEMA_VERSION:
                raise ResolutionPersistenceError(
                    "historical thread rows cannot be resumed as V2"
                )
            raise ResolutionPersistenceError("resolution thread already exists")
        persisted_segment = deepcopy(dict(segment))
        segment_id = str(persisted_segment.get("segment_id", ""))
        document = {
            "schema_version": THREAD_SCHEMA_VERSION,
            "resolution_thread_id": resolution_thread_id,
            "brain_conversation_ref": brain_conversation_ref,
            "root_goal_ref": root_goal_ref,
            "current_segment_id": segment_id,
            "state": "active",
            "priority": priority,
            "workspace_root": workspace_root,
            "workspace_fingerprint": workspace_fingerprint,
            "route_digest": route_digest,
            "profile_version": profile_version,
            "dsh_release": DSH_RELEASE,
            "session_store_epoch": SESSION_STORE_EPOCH,
            "standard_catalog_digest": standard_catalog_digest,
            "semantic_catalog_digest": semantic_catalog_digest,
            "policy_epoch": policy_epoch,
            "scope_fingerprint": scope_fingerprint,
            "audience_fingerprint": audience_fingerprint,
            "interaction_id": interaction_id,
            "created_at": now,
            "updated_at": now,
            "last_terminal_status": None,
            "continuation_eligible_until": _future(now),
            "document_revision": 1,
            "lease_epoch": 0,
            "current_lease": None,
            "segments": [persisted_segment],
            "operations": [],
        }
        validated = ResolutionThreadRecordV2.from_mapping(document)
        self._threads[resolution_thread_id] = validated.to_dict()
        return validated

    def seed_historical(self, document: dict[str, Any]) -> None:
        """Seed an explicitly historical row for rejection tests."""

        resolution_thread_id = document.get("resolution_thread_id")
        if not isinstance(resolution_thread_id, str) or not resolution_thread_id:
            raise ResolutionPersistenceError("historical row requires a thread id")
        self._threads[resolution_thread_id] = deepcopy(document)

    def resume_v2(self, resolution_thread_id: str) -> ResolutionThreadRecordV2:
        document = self._threads.get(resolution_thread_id)
        if document is None:
            raise ResolutionPersistenceError("resolution thread does not exist")
        if document.get("schema_version") != THREAD_SCHEMA_VERSION:
            raise ResolutionPersistenceError(
                "historical thread rows cannot be resumed as V2"
            )
        return ResolutionThreadRecordV2.from_mapping(deepcopy(document))

    def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV2 | None:
        document = self._threads.get(resolution_thread_id)
        if document is None or document.get("schema_version") != THREAD_SCHEMA_VERSION:
            return None
        return ResolutionThreadRecordV2.from_mapping(deepcopy(document))

    def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        document = self._threads.get(resolution_thread_id)
        if document is None or document.get("schema_version") != THREAD_SCHEMA_VERSION:
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
    ) -> ResolutionThreadRecordV2:
        document = self._required(resolution_thread_id)
        new_segment = deepcopy(segment)
        new_segment["rotation_reason"] = reason
        new_segment["parent_segment_id"] = document["current_segment_id"]
        document["segments"].append(new_segment)
        document["current_segment_id"] = new_segment["segment_id"]
        document["current_lease"] = None
        self._touch(document)
        return ResolutionThreadRecordV2.from_mapping(deepcopy(document))

    def update_segment(
        self,
        resolution_thread_id: str,
        segment_id: str,
        **changes: Any,
    ) -> ResolutionThreadRecordV2:
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
        return ResolutionThreadRecordV2.from_mapping(deepcopy(document))

    def validate_fence(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        document = self._required(resolution_thread_id)
        return deepcopy(self._fenced_lease(document, activation_id, lease_epoch))

    def _required(self, resolution_thread_id: str) -> dict[str, Any]:
        document = self._threads.get(resolution_thread_id)
        if document is None:
            raise ResolutionPersistenceError("resolution thread does not exist")
        if document.get("schema_version") != THREAD_SCHEMA_VERSION:
            raise ResolutionPersistenceError(
                "historical thread rows cannot be resumed as V2"
            )
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
    """Async adapter delegating raw selectors to the V2 DB owner module."""

    def __init__(self) -> None:
        self._db = resolution_threads

    async def ensure_indexes(self) -> None:
        await self._db.ensure_indexes()

    async def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV2 | None:
        value = await self._db.get_thread(resolution_thread_id)
        if value is None:
            return None
        return ResolutionThreadRecordV2.from_mapping(value)

    async def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        return await self._db.get_operation(resolution_thread_id, operation_id)

    async def create_thread_v2(self, *args: Any, **kwargs: Any) -> ResolutionThreadRecordV2:
        await self.ensure_indexes()
        return ResolutionThreadRecordV2.from_mapping(
            await self._db.create_thread_v2(*args, **kwargs)
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

    async def rotate_segment(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        return ResolutionThreadRecordV2.from_mapping(
            await self._db.rotate_segment(*args, **kwargs)
        )

    async def update_segment(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        return ResolutionThreadRecordV2.from_mapping(
            await self._db.update_segment(*args, **kwargs)
        )

    async def validate_fence(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._db.validate_fence(*args, **kwargs)
