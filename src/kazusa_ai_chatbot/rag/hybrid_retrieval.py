"""Shared deterministic fusion for semantic and keyword RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import text_or_empty

HybridSource = Literal["conversation", "persistent_memory"]


@dataclass(frozen=True)
class HybridCandidate:
    """One fused retrieval candidate with deterministic provenance."""

    row: dict[str, Any]
    identity: str
    source: HybridSource
    methods: tuple[str, ...]
    matched_anchors: tuple[str, ...]
    score: float = 0.0
    best_rank: int = 999999


def hybrid_row_identity(
    row: Mapping[str, Any],
    *,
    source: HybridSource,
) -> str:
    """Build a stable cross-method identity for one retrieval row."""

    platform_message_id = text_or_empty(row.get("platform_message_id"))
    if platform_message_id:
        return_value = f"message:{platform_message_id}"
        return return_value

    conversation_row_id = text_or_empty(row.get("conversation_row_id"))
    if conversation_row_id:
        return_value = f"conversation_row:{conversation_row_id}"
        return return_value

    mongo_id = text_or_empty(row.get("_id"))
    if mongo_id:
        prefix = "memory_id" if source == "persistent_memory" else "mongo_id"
        return_value = f"{prefix}:{mongo_id}"
        return return_value

    memory_name = text_or_empty(row.get("memory_name"))
    timestamp = text_or_empty(row.get("timestamp"))
    if source == "persistent_memory" and memory_name and timestamp:
        return_value = f"memory:{memory_name}:{timestamp}"
        return return_value

    fallback_text = candidate_prompt_text(
        row,
        source=source,
        text_limit=80,
    )
    fallback = fallback_text[:80]
    return_value = f"{source}:{timestamp}:{fallback}"
    return return_value


def merge_hybrid_candidates(
    semantic_rows: list[Mapping[str, Any]],
    keyword_rows: list[Mapping[str, Any]],
    neighbor_rows: list[Mapping[str, Any]] | None = None,
    *,
    semantic_only_floor: float,
    selected_limit: int,
    source: HybridSource = "conversation",
) -> list[HybridCandidate]:
    """Merge semantic, keyword, and neighbor rows into ranked candidates."""

    by_identity: dict[str, HybridCandidate] = {}
    for rank, row in enumerate(semantic_rows, start=1):
        _merge_row(
            by_identity,
            row,
            method="semantic",
            source=source,
            rank=rank,
        )

    for rank, row in enumerate(keyword_rows, start=1):
        keyword_method = _keyword_method(row)
        _merge_row(
            by_identity,
            row,
            method=keyword_method,
            source=source,
            rank=rank,
        )

    for rank, row in enumerate(neighbor_rows or [], start=1):
        _merge_row(
            by_identity,
            row,
            method="neighbor",
            source=source,
            rank=rank,
        )

    ranked = [
        candidate
        for candidate in by_identity.values()
        if _candidate_bucket(candidate, semantic_only_floor) is not None
    ]
    ranked.sort(
        key=lambda candidate: (
            _candidate_bucket(candidate, semantic_only_floor),
            -_keyword_support_count(candidate.methods),
            -candidate.score,
            candidate.best_rank,
            _candidate_timestamp(candidate.row),
            candidate.identity,
        )
    )
    return_value = ranked[:selected_limit]
    return return_value


def select_neighbor_seed_candidates(
    candidates: list[HybridCandidate],
    *,
    keyword_rows_present: bool,
    semantic_only_floor: float,
    seed_limit: int,
) -> list[HybridCandidate]:
    """Select candidates safe for neighboring context expansion."""

    seeds: list[HybridCandidate] = []
    for candidate in candidates:
        if keyword_rows_present and _keyword_support_count(candidate.methods) < 1:
            continue
        if (
            not keyword_rows_present
            and "semantic" in candidate.methods
            and candidate.score < semantic_only_floor
        ):
            continue
        seeds.append(candidate)
        if len(seeds) >= seed_limit:
            break

    return_value = seeds
    return return_value


def candidate_prompt_text(
    row: Mapping[str, Any],
    *,
    source: HybridSource,
    text_limit: int,
) -> str:
    """Project one candidate row into compact prompt-facing text."""

    if source == "persistent_memory":
        text = _persistent_memory_text(row)
    else:
        text = _conversation_text(row)

    if len(text) > text_limit:
        text = text[: max(0, text_limit - 3)].rstrip() + "..."
    return text


def _merge_row(
    by_identity: dict[str, HybridCandidate],
    row: Mapping[str, Any],
    *,
    method: str,
    source: HybridSource,
    rank: int,
) -> None:
    """Merge one retrieval row into a candidate accumulator."""

    row_dict = dict(row)
    identity = hybrid_row_identity(row_dict, source=source)
    matched_anchors = _matched_anchors(row_dict, method)
    score = _row_score(row_dict)

    existing = by_identity.get(identity)
    if existing is None:
        candidate = HybridCandidate(
            row=row_dict,
            identity=identity,
            source=source,
            methods=(method,),
            matched_anchors=matched_anchors,
            score=score,
            best_rank=rank,
        )
        by_identity[identity] = candidate
        return

    methods = _merge_tuple(existing.methods, (method,))
    anchors = _merge_tuple(existing.matched_anchors, matched_anchors)
    merged_row = dict(existing.row)
    merged_row.update(
        {
            key: value
            for key, value in row_dict.items()
            if value not in ("", None, [], {})
        }
    )
    candidate = HybridCandidate(
        row=merged_row,
        identity=identity,
        source=source,
        methods=methods,
        matched_anchors=anchors,
        score=max(existing.score, score),
        best_rank=min(existing.best_rank, rank),
    )
    by_identity[identity] = candidate


def _merge_tuple(
    existing: tuple[str, ...],
    incoming: tuple[str, ...],
) -> tuple[str, ...]:
    """Append new strings while preserving first-seen order."""

    merged = list(existing)
    for item in incoming:
        if item and item not in merged:
            merged.append(item)
    return_value = tuple(merged)
    return return_value


def _keyword_method(row: Mapping[str, Any]) -> str:
    """Return normalized keyword provenance for one row."""

    method = text_or_empty(row.get("method"))
    if method.startswith("keyword"):
        return method

    methods = row.get("methods")
    if isinstance(methods, list):
        for item in methods:
            text = text_or_empty(item)
            if text.startswith("keyword"):
                return text

    return_value = "keyword"
    return return_value


def _matched_anchors(
    row: Mapping[str, Any],
    method: str,
) -> tuple[str, ...]:
    """Extract literal anchors from a row or keyword method name."""

    anchors: list[str] = []
    matched_anchors = row.get("matched_anchors")
    if isinstance(matched_anchors, list):
        for item in matched_anchors:
            text = text_or_empty(item)
            if text:
                anchors.append(text)

    if method.startswith("keyword:"):
        _, _, anchor = method.partition(":")
        anchor_text = anchor.strip()
        if anchor_text:
            anchors.append(anchor_text)

    return_value = tuple(dict.fromkeys(anchors))
    return return_value


def _row_score(row: Mapping[str, Any]) -> float:
    """Read a numeric semantic score from a row."""

    for key in ("score", "cosine_similarity", "similarity"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return_value = float(value)
            return return_value

    return_value = 0.0
    return return_value


def _candidate_bucket(
    candidate: HybridCandidate,
    semantic_only_floor: float,
) -> int | None:
    """Return ranking bucket or ``None`` when a candidate should be rejected."""

    has_semantic = "semantic" in candidate.methods
    keyword_count = _keyword_support_count(candidate.methods)
    has_neighbor = "neighbor" in candidate.methods
    if has_semantic and keyword_count:
        return 0
    if keyword_count:
        return 1
    if has_neighbor:
        return 2
    if has_semantic and candidate.score >= semantic_only_floor:
        return 3
    return_value = None
    return return_value


def _keyword_support_count(methods: tuple[str, ...]) -> int:
    """Count keyword provenance markers in a methods tuple."""

    count = sum(1 for method in methods if method.startswith("keyword"))
    return count


def _candidate_timestamp(row: Mapping[str, Any]) -> str:
    """Return timestamp text for deterministic tie-breaking."""

    timestamp = text_or_empty(row.get("timestamp"))
    return timestamp


def _conversation_text(row: Mapping[str, Any]) -> str:
    """Build prompt text for a conversation row."""

    body = _first_text_field(row, ("body_text", "content", "summary", "text"))
    attachment_text = _attachment_description_text(row.get("attachments"))
    reply_text = _reply_context_text(row.get("reply_context"))

    parts = [body]
    if attachment_text:
        parts.append(f"attachment: {attachment_text}")
    if reply_text:
        parts.append(f"reply: {reply_text}")
    content = "\n".join(part for part in parts if part)

    display_name = text_or_empty(row.get("display_name"))
    timestamp = format_storage_utc_for_llm(text_or_empty(row.get("timestamp")))
    if display_name and timestamp and content:
        text = f"{display_name} at {timestamp}: {content}"
        return text
    if display_name and content:
        text = f"{display_name}: {content}"
        return text
    return content


def _persistent_memory_text(row: Mapping[str, Any]) -> str:
    """Build prompt text for a persistent-memory row."""

    content = _first_text_field(row, ("content", "description", "text", "summary"))
    memory_name = text_or_empty(row.get("memory_name"))
    timestamp = format_storage_utc_for_llm(text_or_empty(row.get("timestamp")))
    parts = []
    if memory_name:
        parts.append(memory_name)
    if timestamp:
        parts.append(timestamp)
    if content:
        parts.append(content)
    text = " | ".join(parts)
    return text


def _first_text_field(row: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    """Return the first non-empty text field from a row."""

    for field in fields:
        value = text_or_empty(row.get(field))
        if value:
            return value
    return_value = ""
    return return_value


def _attachment_description_text(attachments: object) -> str:
    """Join attachment descriptions from a typed conversation row."""

    if not isinstance(attachments, list):
        return_value = ""
        return return_value

    descriptions: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, Mapping):
            continue
        description = text_or_empty(attachment.get("description"))
        if description:
            descriptions.append(description)

    return_value = "\n".join(descriptions)
    return return_value


def _reply_context_text(reply_context: object) -> str:
    """Extract a compact reply excerpt from a typed reply context."""

    if not isinstance(reply_context, Mapping):
        return_value = ""
        return return_value

    text = _first_text_field(
        reply_context,
        (
            "reply_excerpt",
            "body_text",
            "content",
            "summary",
            "text",
        ),
    )
    return text
