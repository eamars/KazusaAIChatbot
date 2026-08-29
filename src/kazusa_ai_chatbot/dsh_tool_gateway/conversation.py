"""Semantic conversation services exposed through the DSH gateway."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from typing import Any

from kazusa_ai_chatbot.db.conversation import (
    aggregate_conversation_by_user,
    list_conversation_rows_by_row_ids,
    search_conversation_history,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    EvidenceReceiptV2,
    KazusaSemanticCapabilityResultV1,
    OpaqueReferenceCodec,
    SemanticPageV1,
    content_digest,
    new_evidence_receipt,
)

_MAX_RESULTS = 50


def _limit(value: object, default: int = 10) -> int:
    """Clamp a model-supplied result limit to the semantic service bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(1, min(value, _MAX_RESULTS))


def _row_mapping(value: object) -> dict[str, Any]:
    """Project a database row without retaining its storage-only identity."""

    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("conversation service returned an invalid row") from exc


def _row_id(row: Mapping[str, Any], index: int) -> str:
    """Select a stable source identity for an opaque reference."""

    candidate = row.get("message_id")
    if not isinstance(candidate, str) or not candidate:
        candidate = row.get("_id")
    if candidate is None:
        candidate = f"entry-{index}"
    return str(candidate)


def _entry_entity(
    row: Mapping[str, Any],
    *,
    reference: str,
    score: float | None,
) -> dict[str, Any]:
    """Build one prompt-safe semantic entry entity."""

    entity: dict[str, Any] = {"entry_ref": reference}
    for source, target in (
        ("body_text", "text"),
        ("timestamp", "occurred_at"),
        ("display_name", "speaker_name"),
        ("role", "speaker_role"),
        ("platform", "platform"),
    ):
        value = row.get(source)
        if isinstance(value, str) and value:
            entity[target] = value
    if score is not None:
        entity["relevance"] = float(score)
    return entity


class ConversationSemanticService:
    """Map conversation leaves to opaque semantic entities and evidence."""

    def __init__(
        self,
        *,
        codec: OpaqueReferenceCodec,
        search: Callable[..., Awaitable[list[tuple[float, Any]]]] = search_conversation_history,
        read: Callable[..., Awaitable[list[Any]]] = list_conversation_rows_by_row_ids,
        summarize: Callable[..., Awaitable[dict[str, Any]]] = aggregate_conversation_by_user,
    ) -> None:
        self._codec = codec
        self._search = search
        self._read = read
        self._summarize = summarize

    def with_authority(self, authority: Mapping[str, Any] | object) -> "ConversationSemanticService":
        """Return a call-local service bound to the signed authority."""

        bound = copy(self)
        bound._codec = self._codec.with_authority(authority)
        return bound

    async def search_conversation_history(
        self,
        *,
        query: str,
        max_results: int = 10,
        time_range: Mapping[str, Any] | None = None,
        platform: str | None = None,
        platform_channel_id: str | None = None,
        global_user_id: str | None = None,
        next_page_ref: str | None = None,
    ) -> KazusaSemanticCapabilityResultV1:
        """Search conversation text and return opaque entries."""

        if not isinstance(query, str) or not query.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid",
                "QUERY_REQUIRED",
                "A semantic conversation query is required.",
            )
        limit = _limit(max_results)
        offset = 0
        if next_page_ref is not None:
            try:
                page = self._codec.resolve(next_page_ref, "conversation-page")
                offset = int(page["offset"])
            except (KeyError, TypeError, ValueError):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid",
                    "PAGE_REFERENCE_INVALID",
                    "The continuation reference is invalid.",
                )
        bounds = _time_bounds(time_range)
        rows = await self._search(
            query,
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            limit=limit + 1,
            from_timestamp=bounds[0],
            to_timestamp=bounds[1],
        )
        selected = list(rows)[offset: offset + limit + 1]
        has_more = len(selected) > limit
        selected = selected[:limit]
        entities: list[dict[str, Any]] = []
        evidence: list[EvidenceReceiptV2] = []
        for index, pair in enumerate(selected):
            if not isinstance(pair, Sequence) or len(pair) != 2:
                continue
            score = pair[0] if isinstance(pair[0], (int, float)) else None
            row = _row_mapping(pair[1])
            source_id = _row_id(row, index + offset)
            reference = self._codec.issue(
                "conversation-entry",
                {"source_id": source_id},
            )
            entity = _entry_entity(row, reference=reference, score=score)
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=f"receipt-conversation-{index + offset}",
                source_kind="conversation_entry",
                semantic_ref=reference,
                value=entity,
                occurred_at=_timestamp(row),
            ))
        page_result = _next_page(
            self._codec,
            kind="conversation-page",
            offset=offset + limit,
            has_more=has_more,
        )
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
            page=page_result,
        )

    async def read_conversation_entries(
        self,
        *,
        conversation_entry_refs: Sequence[str],
    ) -> KazusaSemanticCapabilityResultV1:
        """Read entries by exact opaque references."""

        if not isinstance(conversation_entry_refs, Sequence) or isinstance(conversation_entry_refs, (str, bytes)):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "ENTRY_REFS_REQUIRED", "Entry references are required."
            )
        source_ids: list[str] = []
        references: list[str] = []
        for reference in conversation_entry_refs[:_MAX_RESULTS]:
            try:
                resolved = self._codec.resolve(str(reference), "conversation-entry")
                source_id = resolved.get("source_id")
            except ValueError:
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "ENTRY_REFERENCE_INVALID", "An entry reference is invalid."
                )
            if not isinstance(source_id, str) or not source_id:
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "ENTRY_REFERENCE_INVALID", "An entry reference is invalid."
                )
            source_ids.append(source_id)
            references.append(str(reference))
        try:
            loaded_rows = await self._read(source_ids, limit=len(source_ids))
        except TypeError:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "CONVERSATION_READ_UNAVAILABLE", "Conversation entry reads are unavailable."
            )
        by_id = {
            _row_id(_row_mapping(row), index): _row_mapping(row)
            for index, row in enumerate(loaded_rows)
        }
        entities: list[dict[str, Any]] = []
        evidence: list[EvidenceReceiptV2] = []
        for reference, source_id in zip(references, source_ids):
            row = by_id.get(source_id)
            if row is None:
                continue
            entity = _entry_entity(row, reference=str(reference), score=None)
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=f"receipt-conversation-read-{content_digest(reference)}",
                source_kind="conversation_entry",
                semantic_ref=str(reference),
                value=entity,
                occurred_at=_timestamp(row),
            ))
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
        )

    async def summarize_conversation_participants(
        self,
        *,
        time_range: Mapping[str, Any] | None = None,
        max_people: int = 10,
        next_page_ref: str | None = None,
    ) -> KazusaSemanticCapabilityResultV1:
        """Return semantic participant summaries for a bounded time range."""

        bounds = _time_bounds(time_range)
        limit = _limit(max_people)
        offset = 0
        if next_page_ref is not None:
            try:
                payload = self._codec.resolve(next_page_ref, "participant-page")
                offset = payload["offset"]
                if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "PAGE_REFERENCE_INVALID", "The continuation reference is invalid."
                )
        summary = await self._summarize(
            from_timestamp=bounds[0],
            to_timestamp=bounds[1],
            limit=offset + limit + 1,
        )
        rows = []
        if isinstance(summary, Mapping):
            rows = summary.get("participants", summary.get("rows", []))
        selected = list(rows)[offset: offset + limit + 1]
        has_more = len(selected) > limit
        selected = selected[:limit]
        entities: list[dict[str, Any]] = []
        evidence: list[EvidenceReceiptV2] = []
        for index, value in enumerate(selected):
            row = _row_mapping(value)
            source_id = _row_id(row, index)
            reference = self._codec.issue("person", {"source_id": source_id})
            entity = {
                "person_ref": reference,
                "name": row.get("display_name", ""),
                "message_count": row.get("message_count", 0),
            }
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=f"receipt-participant-{content_digest(reference)}",
                source_kind="conversation_participant",
                semantic_ref=reference,
                value=entity,
            ))
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
            page=SemanticPageV1(
                has_more=has_more,
                next_page_ref=(
                    self._codec.issue("participant-page", {"offset": offset + limit})
                    if has_more
                    else None
                ),
            ),
        )


def _time_bounds(value: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Extract bounded ISO time range fields."""

    if not isinstance(value, Mapping):
        return None, None
    start = value.get("start_at")
    end = value.get("end_at")
    return (
        start if isinstance(start, str) and start else None,
        end if isinstance(end, str) and end else None,
    )


def _timestamp(row: Mapping[str, Any]) -> str | None:
    """Read a row timestamp when it is already a string."""

    value = row.get("timestamp")
    return value if isinstance(value, str) and value else None


def _next_page(
    codec: OpaqueReferenceCodec,
    *,
    kind: str,
    offset: int,
    has_more: bool,
) -> SemanticPageV1:
    """Create a page result with a signed continuation token."""

    reference = codec.issue(kind, {"offset": offset}) if has_more else None
    return SemanticPageV1(has_more=has_more, next_page_ref=reference)
