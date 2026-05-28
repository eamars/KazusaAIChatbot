"""Memory worker result projection."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.config import (
    RAG_MEMORY_EVIDENCE_TEXT_LIMIT,
    RAG_SEARCH_SELECTED_SUMMARY_LIMIT,
)
from kazusa_ai_chatbot.rag.memory_evidence.contracts import _clip_text
from kazusa_ai_chatbot.utils import text_or_empty

def _rows_from_worker(worker_result: dict[str, Any]) -> list[object]:
    """Normalize a memory worker result into rows."""
    value = worker_result.get("result")
    if isinstance(value, dict):
        memory_rows = value.get("memory_rows")
        if isinstance(memory_rows, list):
            return memory_rows
    if isinstance(value, list):
        return value
    rows = [value]
    return rows

def _memory_rows(worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve raw memory rows while guaranteeing a content field."""
    rows: list[dict[str, Any]] = []
    for row in _rows_from_worker(worker_result):
        if isinstance(row, dict):
            rows.append(dict(row))
            continue
        text = text_or_empty(row)
        if text:
            rows.append({"content": text})
    return rows

def _nearby_memory_rows(worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve nearby scoped-memory rows for unresolved observations."""
    value = worker_result.get("result")
    if not isinstance(value, dict):
        return_value: list[dict[str, Any]] = []
        return return_value

    raw_rows = value.get("nearby_memory_rows")
    if not isinstance(raw_rows, list):
        return_value = []
        return return_value

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(dict(row))
    return rows

def _content_from_row(row: dict[str, Any]) -> str:
    """Extract the durable memory content from one row."""
    for field in ("content", "description", "text", "summary", "fact"):
        text = text_or_empty(row.get(field))
        if text:
            return text
    return_value = ""
    return return_value

def _summaries_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Build prompt-facing memory evidence summaries."""
    summaries = [
        summary
        for row in rows
        if (
            summary := _clip_text(
                _content_from_row(row),
                limit=RAG_MEMORY_EVIDENCE_TEXT_LIMIT,
            )
        )
    ]
    return summaries

def _refs_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build structured memory references for downstream slots."""
    refs: list[dict[str, Any]] = []
    for row in rows:
        memory_name = text_or_empty(row.get("memory_name"))
        source_kind = text_or_empty(row.get("source_kind"))
        if memory_name or source_kind:
            refs.append(
                {
                    "ref_type": "memory",
                    "memory_name": memory_name,
                    "source_kind": source_kind,
                }
            )
    return refs

def _memory_observation_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build source-scoped observation rows for unresolved memory retrieval."""

    candidates: list[dict[str, Any]] = []
    for row in rows[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT]:
        content = _clip_text(
            _content_from_row(row),
            limit=RAG_MEMORY_EVIDENCE_TEXT_LIMIT,
        )
        if not content:
            continue
        candidates.append(
            {
                "content": content,
                "source": _memory_row_source(row),
            }
        )
    return candidates

def _memory_row_source(row: dict[str, Any]) -> str:
    """Build a compact source label for one memory row."""

    memory_name = text_or_empty(row.get("memory_name"))
    if memory_name:
        source = f"memory:memory_name:{memory_name}"
        return source

    source_kind = text_or_empty(row.get("source_kind"))
    if source_kind:
        source = f"memory:source_kind:{source_kind}"
        return source

    mongo_id = text_or_empty(row.get("_id"))
    if mongo_id:
        source = f"memory:id:{mongo_id}"
        return source

    source = "memory:unknown"
    return source
