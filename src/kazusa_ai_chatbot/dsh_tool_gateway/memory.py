"""Semantic memory search and idempotent mutation services."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from datetime import UTC, datetime
from typing import Any

from kazusa_ai_chatbot.db.memory import search_memory
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
    OpaqueReferenceCodec,
    SemanticMutationOutcomeV1,
    SemanticPageV1,
    content_digest,
    new_evidence_receipt,
)
from kazusa_ai_chatbot.memory_evolution.repository import (
    insert_memory_unit,
    read_memory_unit,
    supersede_memory_unit,
    transition_memory_lifecycle,
)

_MAX_RESULTS = 50
_SUBJECTS = frozenset({"current_user", "active_character", "shared_world"})
_MEMORY_KINDS = frozenset({
    "profile_fact",
    "relationship",
    "commitment",
    "experience",
    "world_knowledge",
})
_LIFECYCLE_STATUS = {
    "activate": "active",
    "complete": "fulfilled",
    "cancel": "rejected",
    "archive": "expired",
}


def _limit(value: object, default: int = 10) -> int:
    """Clamp a semantic result limit."""

    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(1, min(value, _MAX_RESULTS))


def _mapping(value: object, field: str) -> dict[str, Any]:
    """Convert a memory service result to a mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an object") from exc


def _source_id(row: Mapping[str, Any]) -> str | None:
    """Read the internal source identity only for reference creation."""

    value = row.get("memory_unit_id")
    return value if isinstance(value, str) and value else None


def _semantic_memory(row: Mapping[str, Any], reference: str) -> dict[str, Any]:
    """Project a memory row into a prompt-safe semantic entity."""

    entity: dict[str, Any] = {"memory_ref": reference}
    for source, target in (
        ("memory_name", "name"),
        ("content", "information"),
        ("memory_type", "kind"),
        ("status", "lifecycle"),
        ("timestamp", "remembered_at"),
        ("updated_at", "updated_at"),
        ("confidence_note", "confidence"),
    ):
        value = row.get(source)
        if isinstance(value, str) and value:
            entity[target] = value
    return entity


class MemorySemanticService:
    """Expose semantic memory operations behind opaque references."""

    def __init__(
        self,
        *,
        codec: OpaqueReferenceCodec,
        source_global_user_id: str | None,
        search: Callable[..., Awaitable[list[tuple[float, Any]]]] = search_memory,
        active_search: Callable[..., Awaitable[list[tuple[float, Any]]]] | None = None,
        read: Callable[..., Awaitable[Mapping[str, Any] | None]] = read_memory_unit,
        insert: Callable[..., Awaitable[Mapping[str, Any]]] = insert_memory_unit,
        revise: Callable[..., Awaitable[Mapping[str, Any]]] = supersede_memory_unit,
        update_lifecycle: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        if source_global_user_id is not None and not source_global_user_id.strip():
            raise ValueError("source_global_user_id is required")
        self._codec = codec
        self._source_global_user_id = (
            source_global_user_id.strip()
            if source_global_user_id is not None
            else None
        )
        self._search = search
        self._active_search = active_search
        self._read = read
        self._insert = insert
        self._revise = revise
        self._update_lifecycle = update_lifecycle

    def _page_offset(self, reference: str | None) -> int | None:
        """Resolve a call-bound continuation offset."""

        if reference is None:
            return 0
        try:
            payload = self._codec.resolve(reference, "memory-page")
            offset = payload.get("offset")
        except ValueError:
            return None
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return None
        return offset

    def _page(self, offset: int, has_more: bool):
        """Issue a call-bound continuation page."""

        return SemanticPageV1(
            has_more=has_more,
            next_page_ref=(
                self._codec.issue("memory-page", {"offset": offset})
                if has_more
                else None
            ),
        )

    @staticmethod
    def _stable_id(prefix: str, idempotency_key: str) -> str:
        """Derive a restart-stable storage id from signed mutation lineage."""

        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        return f"{prefix}_{digest}"

    def with_authority(self, authority: Mapping[str, Any] | object) -> MemorySemanticService:
        """Return a call-local service bound to the signed authority."""

        bound = copy(self)
        bound._codec = self._codec.with_authority(authority)
        if hasattr(authority, "to_dict"):
            authority = authority.to_dict()  # type: ignore[union-attr]
        if not isinstance(authority, Mapping):
            raise ValueError("memory authority must be an object")
        scope = authority.get("service_scope")
        if not isinstance(scope, Mapping):
            raise ValueError("memory authority service scope is required")
        user_id = scope.get("global_user_id")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("memory authority user scope is required")
        bound._source_global_user_id = user_id.strip()
        return bound

    async def search_memories(
        self,
        *,
        query: str,
        subject_scope: str = "current_user",
        memory_kinds: Sequence[str] | None = None,
        max_results: int = 10,
        next_page_ref: str | None = None,
    ) -> KazusaSemanticCapabilityResultV1:
        """Search active semantic memories."""

        if not isinstance(query, str) or not query.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "QUERY_REQUIRED", "A semantic memory query is required."
            )
        if subject_scope not in _SUBJECTS | {"all"}:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "SUBJECT_SCOPE_INVALID", "The memory subject scope is unsupported."
            )
        if memory_kinds is not None and (
            not isinstance(memory_kinds, Sequence)
            or isinstance(memory_kinds, (str, bytes))
            or any(kind not in _MEMORY_KINDS for kind in memory_kinds)
        ):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_KINDS_INVALID", "The memory kind filter is unsupported."
            )
        offset = self._page_offset(next_page_ref)
        if offset is None:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "PAGE_REFERENCE_INVALID", "The continuation reference is invalid."
            )
        if subject_scope == "current_user" and not self._source_global_user_id:
            return KazusaSemanticCapabilityResultV1.failure(
                "denied",
                "MEMORY_SCOPE_UNBOUND",
                "The current-user memory scope is unavailable.",
            )
        user_id = (
            None
            if subject_scope in {"active_character", "shared_world", "all"}
            else self._source_global_user_id
        )
        memory_type = None
        if memory_kinds is not None and len(memory_kinds) == 1:
            memory_type = str(memory_kinds[0])
        rows = await self._search(
            query.strip(),
            limit=offset + _limit(max_results) + 1,
            source_global_user_id=user_id,
            memory_type=memory_type,
        )
        selected_rows = list(rows)[offset: offset + _limit(max_results) + 1]
        has_more = len(selected_rows) > _limit(max_results)
        selected_rows = selected_rows[: _limit(max_results)]
        entities: list[dict[str, Any]] = []
        evidence = []
        for index, pair in enumerate(selected_rows):
            if not isinstance(pair, Sequence) or len(pair) != 2:
                continue
            row = _mapping(pair[1], "memory result")
            source_id = _source_id(row)
            if source_id is None:
                continue
            reference = self._codec.issue("memory", {"source_id": source_id})
            entity = _semantic_memory(row, reference)
            score = pair[0]
            if isinstance(score, (int, float)):
                entity["relevance"] = float(score)
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=f"receipt-memory-{content_digest(reference + str(entity))}",
                source_kind="semantic_memory",
                semantic_ref=reference,
                value=entity,
            ))
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
            page=self._page(offset + _limit(max_results), has_more),
        )

    async def read_memories(
        self,
        *,
        memory_refs: Sequence[str],
    ) -> KazusaSemanticCapabilityResultV1:
        """Read complete semantic memory records by opaque reference."""

        if not isinstance(memory_refs, Sequence) or isinstance(memory_refs, (str, bytes)):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_REFS_REQUIRED", "Memory references are required."
            )
        entities: list[dict[str, Any]] = []
        evidence = []
        for index, reference in enumerate(memory_refs[:_MAX_RESULTS]):
            try:
                payload = self._codec.resolve(str(reference), "memory")
            except ValueError:
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "MEMORY_REFERENCE_INVALID", "A memory reference is invalid."
                )
            source_id = payload.get("source_id")
            if not isinstance(source_id, str):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "MEMORY_REFERENCE_INVALID", "A memory reference is invalid."
                )
            row = await self._read(source_id)
            if row is None:
                continue
            entity = _semantic_memory(_mapping(row, "memory result"), str(reference))
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=(
                    "receipt-memory-read-"
                    + content_digest(f"{reference}:{entity}")
                ),
                source_kind="semantic_memory",
                semantic_ref=str(reference),
                value=entity,
            ))
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
        )

    async def remember_information(
        self,
        *,
        subject: str,
        information: str,
        memory_kind: str,
        reason: str,
        provenance: Mapping[str, Any],
        idempotency_key: str,
    ) -> KazusaSemanticCapabilityResultV1:
        """Commit one new semantic memory idempotently."""

        if subject not in _SUBJECTS:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "SUBJECT_INVALID", "The memory subject is unsupported."
            )
        if memory_kind not in _MEMORY_KINDS:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_KIND_INVALID", "The memory kind is unsupported."
            )
        if (
            not isinstance(provenance, Mapping)
            or set(provenance) not in ({"conversation_entry_ref"}, {"current_task"})
            or not isinstance(next(iter(provenance.values())), str)
            or not next(iter(provenance.values())).strip()
        ):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "PROVENANCE_INVALID", "Memory provenance is required."
            )
        if not information.strip() or not reason.strip() or not idempotency_key.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MUTATION_FIELDS_REQUIRED", "Information, reason, and idempotency key are required."
            )
        if subject == "current_user" and not self._source_global_user_id:
            return KazusaSemanticCapabilityResultV1.failure(
                "denied",
                "MEMORY_SCOPE_UNBOUND",
                "The current-user memory scope is unavailable.",
            )
        source_id = self._stable_id("memory", idempotency_key)
        existing_row = await self._read(source_id)
        if existing_row is not None:
            reference = self._codec.issue("memory", {"source_id": source_id})
            existing_entity = _semantic_memory(_mapping(existing_row, "memory mutation"), reference)
            mutation = SemanticMutationOutcomeV1("already_committed", reference, idempotency_key)
            return KazusaSemanticCapabilityResultV1.success(
                entities=[existing_entity],
                evidence=[new_evidence_receipt(
                    receipt_id=f"receipt-memory-write-{content_digest(reference)}",
                    source_kind="semantic_memory",
                    semantic_ref=reference,
                    value=existing_entity,
                )],
                mutation=mutation,
            )
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        candidate = {
            "memory_unit_id": source_id,
            "lineage_id": self._stable_id("lineage", idempotency_key),
            "version": 1,
            "memory_name": information.strip()[:120],
            "content": information.strip(),
            "source_global_user_id": (
                self._source_global_user_id if subject == "current_user" else ""
            ),
            "memory_type": memory_kind,
            "source_kind": "conversation_extracted",
            "authority": "conversation_accepted",
            "status": "active",
            "supersedes_memory_unit_ids": [],
            "merged_from_memory_unit_ids": [],
            "evidence_refs": [{**dict(provenance), "captured_at": now}],
            "privacy_review": {
                "global_applicability": "global" if subject == "shared_world" else "scoped"
            },
            "confidence_note": reason.strip(),
            "timestamp": now,
            "updated_at": now,
            "expiry_timestamp": None,
        }
        persisted = await self._insert(document=candidate)
        row = _mapping(persisted, "memory mutation")
        source_id = _source_id(row) or source_id
        reference = self._codec.issue("memory", {"source_id": source_id})
        mutation = SemanticMutationOutcomeV1("committed", reference, idempotency_key)
        entity = _semantic_memory(row, reference)
        evidence = new_evidence_receipt(
            receipt_id=f"receipt-memory-write-{content_digest(reference)}",
            source_kind="semantic_memory",
            semantic_ref=reference,
            value=entity,
        )
        return KazusaSemanticCapabilityResultV1.success(
            entities=[entity], evidence=[evidence], mutation=mutation
        )

    async def revise_memory(
        self,
        *,
        memory_ref: str,
        revised_information: str,
        reason: str,
        idempotency_key: str,
    ) -> KazusaSemanticCapabilityResultV1:
        """Commit a revision under the original semantic memory identity."""

        if not revised_information.strip() or not reason.strip() or not idempotency_key.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MUTATION_FIELDS_REQUIRED", "Revision, reason, and idempotency key are required."
            )
        try:
            payload = self._codec.resolve(memory_ref, "memory")
        except ValueError:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_REFERENCE_INVALID", "A memory reference is invalid."
            )
        source_id = payload.get("source_id")
        if not isinstance(source_id, str):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_REFERENCE_INVALID", "A memory reference is invalid."
            )
        target = await self._read(source_id)
        if target is None:
            return KazusaSemanticCapabilityResultV1.failure(
                "empty", "MEMORY_NOT_FOUND", "The requested memory was not found."
            )
        row = _mapping(target, "memory target")
        replacement_id = self._stable_id("memory-revision", idempotency_key)
        existing_replacement = await self._read(replacement_id)
        if existing_replacement is not None:
            existing_ref = self._codec.issue(
                "memory", {"source_id": replacement_id}
            )
            existing_entity = _semantic_memory(
                _mapping(existing_replacement, "memory revision"),
                existing_ref,
            )
            return KazusaSemanticCapabilityResultV1.success(
                entities=[existing_entity],
                evidence=[new_evidence_receipt(
                    receipt_id=f"receipt-memory-revise-{content_digest(existing_ref)}",
                    source_kind="semantic_memory",
                    semantic_ref=existing_ref,
                    value=existing_entity,
                )],
                mutation=SemanticMutationOutcomeV1(
                    "already_committed", existing_ref, idempotency_key
                ),
            )
        replacement = dict(row)
        replacement.pop("embedding", None)
        replacement["memory_unit_id"] = replacement_id
        replacement["content"] = revised_information.strip()
        replacement["confidence_note"] = reason.strip()
        persisted = await self._revise(active_unit_id=source_id, replacement=replacement)
        result_row = _mapping(persisted, "memory revision")
        new_source_id = _source_id(result_row) or replacement["memory_unit_id"]
        new_ref = self._codec.issue("memory", {"source_id": new_source_id})
        mutation = SemanticMutationOutcomeV1("committed", new_ref, idempotency_key)
        entity = _semantic_memory(result_row, new_ref)
        return KazusaSemanticCapabilityResultV1.success(
            entities=[entity],
            evidence=[new_evidence_receipt(
                receipt_id=f"receipt-memory-revise-{content_digest(new_ref)}",
                source_kind="semantic_memory",
                semantic_ref=new_ref,
                value=entity,
            )],
            mutation=mutation,
        )

    async def change_memory_lifecycle(
        self,
        *,
        memory_ref: str,
        transition: str,
        reason: str,
        idempotency_key: str,
    ) -> KazusaSemanticCapabilityResultV1:
        """Apply one explicit memory lifecycle transition."""

        if transition not in _LIFECYCLE_STATUS:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "LIFECYCLE_TRANSITION_INVALID", "The lifecycle transition is unsupported."
            )
        if not reason.strip() or not idempotency_key.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MUTATION_FIELDS_REQUIRED", "Reason and idempotency key are required."
            )
        try:
            payload = self._codec.resolve(memory_ref, "memory")
        except ValueError:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_REFERENCE_INVALID", "A memory reference is invalid."
            )
        source_id = payload.get("source_id")
        if not isinstance(source_id, str):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MEMORY_REFERENCE_INVALID", "A memory reference is invalid."
            )
        target = await self._read(source_id)
        if target is None:
            return KazusaSemanticCapabilityResultV1.failure(
                "empty", "MEMORY_NOT_FOUND", "The requested memory was not found."
            )
        new_status = _LIFECYCLE_STATUS[transition]
        current_status = str(target.get("status", ""))
        if current_status == new_status:
            mutation = SemanticMutationOutcomeV1("already_committed", memory_ref, idempotency_key)
        else:
            if self._update_lifecycle is None:
                persisted = await transition_memory_lifecycle(
                    memory_unit_id=source_id,
                    transition=transition,
                    reason=reason,
                )
            else:
                await self._update_lifecycle(source_id, {"status": new_status})
                refreshed = await self._read(source_id)
                persisted = (
                    _mapping(refreshed, "memory lifecycle result")
                    if refreshed is not None
                    and str(refreshed.get("status", "")) == new_status
                    else {**_mapping(target, "memory target"), "status": new_status}
                )
            persisted_status = str(persisted.get("status", ""))
            if persisted_status != new_status:
                return KazusaSemanticCapabilityResultV1.failure(
                    "unavailable",
                    "MEMORY_LIFECYCLE_NOT_COMMITTED",
                    "The memory lifecycle transition was not committed.",
                )
            mutation = SemanticMutationOutcomeV1("committed", memory_ref, idempotency_key)
            target = persisted
        entity = _semantic_memory(
            {**_mapping(target, "memory target"), "status": new_status},
            memory_ref,
        )
        return KazusaSemanticCapabilityResultV1.success(
            entities=[entity],
            evidence=[new_evidence_receipt(
                receipt_id=f"receipt-memory-lifecycle-{content_digest(memory_ref)}",
                source_kind="semantic_memory",
                semantic_ref=memory_ref,
                value=entity,
            )],
            mutation=mutation,
        )
