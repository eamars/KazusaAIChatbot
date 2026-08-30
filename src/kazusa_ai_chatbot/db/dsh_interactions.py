"""Raw Mongo owner for durable DSH Brain interaction audit rows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError

DSH_INTERACTIONS_COLLECTION = "dsh_interaction_store"
DSH_INTERACTION_SCHEMA_VERSION = "dsh_brain_interaction.v2"
INTERACTION_INDEXES = (
    {
        "keys": [("interaction_id", 1)],
        "unique": True,
        "name": "dsh_interaction_id_unique",
    },
    {
        "keys": [("issuer", 1), ("nonce", 1)],
        "unique": True,
        "name": "dsh_interaction_issuer_nonce_unique",
    },
    {
        "keys": [
            ("grant_status", 1),
            ("resolution_thread_id", 1),
            ("segment_id", 1),
            ("activation_id", 1),
            ("lease_epoch", 1),
            ("tool_name", 1),
            ("arguments_digest", 1),
            ("workspace_fingerprint", 1),
            ("scope_fingerprint", 1),
            ("policy_epoch", 1),
            ("expires_at", 1),
        ],
        "name": "dsh_interaction_grant_lookup_v2",
    },
)


class InteractionRepository(Protocol):
    """Durable owner interface for interaction state and replay lineage."""

    async def create(self, document: dict[str, Any]) -> dict[str, Any]:
        """Create or return one immutable interaction row."""

    async def get(self, interaction_id: str) -> dict[str, Any] | None:
        """Load one interaction row."""

    async def update(
        self,
        interaction_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Update mutable interaction state."""

    async def consume_nonce(self, issuer: str, nonce: str) -> None:
        """Claim one issuer/nonce pair durably."""

    async def consume_grant(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        tool_name: str,
        arguments_digest: str,
        workspace_fingerprint: str,
        scope_fingerprint: str,
        policy_epoch: str,
        now: str,
    ) -> dict[str, Any] | None:
        """Atomically consume one exact grant."""


async def _collection() -> Any:
    """Return the dedicated interaction collection."""

    db = await get_db()
    return db[DSH_INTERACTIONS_COLLECTION]


async def ensure_indexes() -> None:
    """Create the exact durable interaction indexes."""

    collection = await _collection()
    try:
        for index in INTERACTION_INDEXES:
            await collection.create_index(
                index["keys"],
                unique=bool(index.get("unique", False)),
                name=str(index["name"]),
            )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to ensure DSH interaction indexes") from exc


async def create_interaction(document: dict[str, Any]) -> dict[str, Any]:
    """Insert one immutable signed interaction row idempotently."""

    collection = await _collection()
    value = deepcopy(document)
    if value.get("schema_version") != DSH_INTERACTION_SCHEMA_VERSION:
        raise ValueError("DSH interaction schema_version is required")
    interaction_id = value.get("interaction_id")
    request_digest = value.get("request_digest")
    if not isinstance(interaction_id, str) or not interaction_id.strip():
        raise ValueError("DSH interaction interaction_id is required")
    if not isinstance(request_digest, str) or not request_digest.strip():
        raise ValueError("DSH interaction request_digest is required")
    try:
        await collection.insert_one(value)
    except DuplicateKeyError:
        existing = await get_interaction(interaction_id)
        if existing is None:
            raise DatabaseOperationError("duplicate DSH interaction could not be loaded")
        if existing.get("request_digest") != request_digest:
            raise ValueError("interaction id was reused with a different request")
        return existing
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to create DSH interaction") from exc
    return value


async def get_interaction(interaction_id: str) -> dict[str, Any] | None:
    """Load one interaction audit row without the Mongo object id."""

    collection = await _collection()
    try:
        row = await collection.find_one(
            {"interaction_id": interaction_id},
            {"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to load DSH interaction") from exc
    return dict(row) if row is not None else None


async def update_interaction(
    interaction_id: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Update mutable decision or grant state."""

    collection = await _collection()
    try:
        row = await collection.find_one_and_update(
            {"interaction_id": interaction_id},
            {"$set": dict(fields)},
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )
    except PyMongoError as exc:
        raise DatabaseOperationError("failed to update DSH interaction") from exc
    if row is None:
        raise DatabaseOperationError("DSH interaction does not exist")
    return dict(row)


async def consume_one_shot_grant(
    *,
    resolution_thread_id: str,
    segment_id: str,
    activation_id: str,
    lease_epoch: int,
    tool_name: str,
    arguments_digest: str,
    workspace_fingerprint: str,
    scope_fingerprint: str,
    policy_epoch: str,
    now: str,
) -> dict[str, Any] | None:
    """Atomically consume one exact unexpired approval grant."""

    collection = await _collection()
    row = await collection.find_one_and_update(
        {
            "grant_status": "available",
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "tool_name": tool_name,
            "arguments_digest": arguments_digest,
            "workspace_fingerprint": workspace_fingerprint,
            "scope_fingerprint": scope_fingerprint,
            "policy_epoch": policy_epoch,
            "expires_at": {"$gt": now},
        },
        {
            "$set": {
                "grant_status": "consumed",
                "grant_consumed_at": now,
                "grant.grant_status": "consumed",
            }
        },
        return_document=ReturnDocument.AFTER,
        projection={"_id": 0},
    )
    return dict(row) if row is not None else None


class MongoInteractionStore:
    """Repository adapter backed exclusively by the interaction collection."""

    async def create(self, document: dict[str, Any]) -> dict[str, Any]:
        """Create or return one durable interaction row."""

        return await create_interaction(document)

    async def get(self, interaction_id: str) -> dict[str, Any] | None:
        """Load one durable interaction row."""

        return await get_interaction(interaction_id)

    async def update(
        self,
        interaction_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Update one durable interaction row."""

        return await update_interaction(interaction_id, fields)

    async def consume_nonce(self, issuer: str, nonce: str) -> None:
        """Claim an existing request nonce with a durable conditional update."""

        collection = await _collection()
        try:
            row = await collection.find_one_and_update(
                {
                    "issuer": issuer,
                    "nonce": nonce,
                    "nonce_status": {"$exists": False},
                },
                {"$set": {"nonce_status": "claimed"}},
                return_document=ReturnDocument.AFTER,
                projection={"_id": 0},
            )
        except PyMongoError as exc:
            raise DatabaseOperationError("failed to claim DSH interaction nonce") from exc
        if row is None:
            raise ValueError("interaction nonce was replayed")

    async def consume_grant(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        tool_name: str,
        arguments_digest: str,
        workspace_fingerprint: str,
        scope_fingerprint: str,
        policy_epoch: str,
        now: str,
    ) -> dict[str, Any] | None:
        """Consume one exact grant with one atomic Mongo update."""

        return await consume_one_shot_grant(
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            tool_name=tool_name,
            arguments_digest=arguments_digest,
            workspace_fingerprint=workspace_fingerprint,
            scope_fingerprint=scope_fingerprint,
            policy_epoch=policy_epoch,
            now=now,
        )


class InMemoryInteractionStore:
    """Atomic deterministic interaction store for tests and local composition."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._nonces: set[tuple[str, str]] = set()
        self._lock = Lock()

    async def create(self, document: dict[str, Any]) -> dict[str, Any]:
        """Insert or return an interaction row by id."""

        interaction_id = document.get("interaction_id")
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError("interaction_id is required")
        with self._lock:
            existing = self._rows.get(interaction_id)
            if existing is not None:
                if existing.get("request_digest") != document.get("request_digest"):
                    raise ValueError("interaction id was reused with a different request")
                return deepcopy(existing)
            value = deepcopy(document)
            if value.get("schema_version") != DSH_INTERACTION_SCHEMA_VERSION:
                raise ValueError("DSH interaction schema_version is required")
            self._rows[interaction_id] = value
            return deepcopy(value)

    async def consume_nonce(self, issuer: str, nonce: str) -> None:
        """Claim one nonce in the injected deterministic owner."""

        identity = (issuer, nonce)
        with self._lock:
            if identity in self._nonces:
                raise ValueError("interaction nonce was replayed")
            self._nonces.add(identity)

    async def get(self, interaction_id: str) -> dict[str, Any] | None:
        """Return one interaction row."""

        with self._lock:
            row = self._rows.get(interaction_id)
            return deepcopy(row) if row is not None else None

    async def update(self, interaction_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Update mutable interaction fields."""

        with self._lock:
            row = self._rows.get(interaction_id)
            if row is None:
                raise ValueError("interaction does not exist")
            row.update(deepcopy(fields))
            return deepcopy(row)

    async def consume_grant(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        tool_name: str,
        arguments_digest: str,
        workspace_fingerprint: str,
        scope_fingerprint: str,
        policy_epoch: str,
        now: str,
    ) -> dict[str, Any] | None:
        """Consume one exact available unexpired grant atomically."""

        current = _parse(now)
        with self._lock:
            for row in self._rows.values():
                if (
                    row.get("grant_status") == "available"
                    and row.get("resolution_thread_id") == resolution_thread_id
                    and row.get("segment_id") == segment_id
                    and row.get("activation_id") == activation_id
                    and row.get("lease_epoch") == lease_epoch
                    and row.get("tool_name") == tool_name
                    and row.get("arguments_digest") == arguments_digest
                    and row.get("workspace_fingerprint") == workspace_fingerprint
                    and row.get("scope_fingerprint") == scope_fingerprint
                    and row.get("policy_epoch") == policy_epoch
                    and _parse(str(row.get("expires_at"))) > current
                ):
                    row["grant_status"] = "consumed"
                    row["grant_consumed_at"] = now
                    grant = row.get("grant")
                    if isinstance(grant, Mapping):
                        grant["grant_status"] = "consumed"
                    return deepcopy(row)
        return None


def _parse(value: str) -> datetime:
    """Parse one timezone-aware timestamp."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("interaction timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("interaction timestamp requires timezone")
    return parsed.astimezone(UTC)
