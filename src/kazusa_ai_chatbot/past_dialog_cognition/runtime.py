"""Runtime builder for L2a-only past-dialog cognition context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.db import DatabaseBackendError
from kazusa_ai_chatbot.db.conversation import list_conversation_rows_by_row_ids
from kazusa_ai_chatbot.db.llm_tracing import list_llm_trace_steps_for_trace_ids
from kazusa_ai_chatbot.past_dialog_cognition.models import (
    PastDialogCognitionCandidate,
    PastDialogCognitionLookupResult,
    PastDialogCognitionSource,
)
from kazusa_ai_chatbot.past_dialog_cognition.projection import (
    PAST_DIALOG_COGNITION_STAGE_NAMES,
    join_projected_contexts,
    project_candidate_trace_context,
)

_CONVERSATION_EVIDENCE_AGENTS = frozenset((
    "conversation_evidence_agent",
    "conversation_search_agent",
    "conversation_filter_agent",
    "conversation_keyword_agent",
    "conversation_aggregate_agent",
))


async def build_past_dialog_cognition_context(
    candidates: Sequence[PastDialogCognitionCandidate],
    *,
    character_global_user_id: str,
    max_dialogs: int = 3,
    context_char_limit: int = 1800,
) -> PastDialogCognitionLookupResult:
    """Build bounded private residual context for already-attached dialogs.

    Args:
        candidates: Structurally attached past-dialog candidates.
        character_global_user_id: Internal id of the active character.
        max_dialogs: Maximum valid candidates to inspect.
        context_char_limit: Maximum characters in the prompt-facing context.

    Returns:
        Lookup result containing only one prompt-facing field plus diagnostics.
    """

    candidate_list = list(candidates)
    if not candidate_list:
        result = _empty_result(
            candidate_count=0,
            status="no_candidates",
            diagnostics=[],
        )
        return result

    valid_candidates, diagnostics = _valid_candidates(
        candidate_list,
        character_global_user_id=character_global_user_id,
        max_dialogs=max_dialogs,
    )
    if not valid_candidates:
        result = _empty_result(
            candidate_count=len(candidate_list),
            status="no_valid_candidates",
            diagnostics=diagnostics,
        )
        return result

    trace_ids = _unique_trace_ids(valid_candidates)
    try:
        trace_steps = await list_llm_trace_steps_for_trace_ids(
            trace_ids,
            stage_names=PAST_DIALOG_COGNITION_STAGE_NAMES,
        )
    except DatabaseBackendError as exc:
        diagnostics.append({
            "status": "skipped",
            "reason": f"trace lookup failed: {exc}",
        })
        result = _empty_result(
            candidate_count=len(candidate_list),
            status="lookup_failed",
            diagnostics=diagnostics,
        )
        return result

    steps_by_trace_id = _steps_by_trace_id(trace_steps)
    projected_contexts: list[str] = []
    for candidate in valid_candidates:
        candidate_steps = steps_by_trace_id.get(candidate.llm_trace_id, [])
        projected_context = project_candidate_trace_context(
            candidate,
            candidate_steps,
        )
        if not projected_context:
            diagnostics.append({
                "status": "skipped",
                "reason": "no parsed residual fields",
                "source": candidate.source,
            })
            continue
        projected_contexts.append(projected_context)

    prompt_context = join_projected_contexts(
        projected_contexts,
        context_char_limit=context_char_limit,
    )
    if prompt_context:
        status = "ok"
    else:
        status = "no_projected_context"
    result: PastDialogCognitionLookupResult = {
        "past_dialog_cognition_context": prompt_context,
        "candidate_count": len(candidate_list),
        "selected_count": len(projected_contexts),
        "status": status,
        "diagnostics": diagnostics,
    }
    return result


async def build_past_dialog_cognition_context_from_rag_result(
    rag_result: Mapping[str, Any],
    *,
    character_global_user_id: str,
    max_dialogs: int = 3,
    context_char_limit: int = 1800,
) -> PastDialogCognitionLookupResult:
    """Build private residual context from RAG conversation source refs.

    Args:
        rag_result: RAG result already attached to cognition state.
        character_global_user_id: Internal id of the active character.
        max_dialogs: Maximum conversation rows to resolve and inspect.
        context_char_limit: Maximum characters in the prompt-facing context.

    Returns:
        Lookup result from row-id anchored conversation evidence. Public
        ``rag_result`` content is never mutated.
    """

    row_ids = conversation_row_ids_from_rag_result(
        rag_result,
        limit=max_dialogs,
    )
    if not row_ids:
        result = _empty_result(
            candidate_count=0,
            status="no_rag_row_ids",
            diagnostics=[],
        )
        return result

    try:
        rows = await list_conversation_rows_by_row_ids(
            row_ids,
            limit=max_dialogs,
        )
    except DatabaseBackendError as exc:
        result = _empty_result(
            candidate_count=0,
            status="row_lookup_failed",
            diagnostics=[{
                "status": "skipped",
                "reason": f"conversation row lookup failed: {exc}",
            }],
        )
        return result
    candidates = candidates_from_conversation_rows(
        rows,
        source="conversation_evidence",
    )
    result = await build_past_dialog_cognition_context(
        candidates,
        character_global_user_id=character_global_user_id,
        max_dialogs=max_dialogs,
        context_char_limit=context_char_limit,
    )
    return result


def candidate_from_conversation_row(
    row: Mapping[str, Any],
    *,
    source: PastDialogCognitionSource,
) -> PastDialogCognitionCandidate | None:
    """Project one conversation row into a residual candidate shape.

    Args:
        row: Conversation-history row loaded through a scoped structural path.
        source: Structural source that attached the row to current context.

    Returns:
        Candidate object, or ``None`` when the row lacks required fields.
    """

    visible_text = _row_text(row, "body_text")
    llm_trace_id = _row_text(row, "llm_trace_id")
    role = _row_text(row, "role")
    global_user_id = _row_text(row, "global_user_id")
    conversation_row_id = _conversation_row_id(row)
    if not visible_text or not llm_trace_id or not role or not global_user_id:
        candidate = None
        return candidate
    candidate = PastDialogCognitionCandidate(
        visible_text=visible_text,
        llm_trace_id=llm_trace_id,
        created_at=row.get("timestamp", ""),
        source=source,
        role=role,
        global_user_id=global_user_id,
        conversation_row_id=conversation_row_id,
        platform_message_id=_row_text(row, "platform_message_id"),
        platform=_row_text(row, "platform"),
        platform_channel_id=_row_text(row, "platform_channel_id"),
    )
    return candidate


def candidates_from_conversation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: PastDialogCognitionSource,
) -> list[PastDialogCognitionCandidate]:
    """Project conversation rows into residual candidates."""

    candidates: list[PastDialogCognitionCandidate] = []
    for row in rows:
        candidate = candidate_from_conversation_row(row, source=source)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def conversation_row_ids_from_rag_result(
    rag_result: Mapping[str, Any],
    *,
    limit: int = 3,
) -> list[str]:
    """Extract conversation row ids from RAG source refs only.

    Args:
        rag_result: RAG result already attached to cognition state.
        limit: Maximum row ids to return.

    Returns:
        Ordered row ids from ``conversation_row_id`` or ``_id`` refs. Unscoped
        platform message ids are intentionally ignored.
    """

    supervisor_trace = rag_result.get("supervisor_trace")
    if not isinstance(supervisor_trace, Mapping):
        row_ids: list[str] = []
        return row_ids

    dispatched = supervisor_trace.get("dispatched")
    if not isinstance(dispatched, list):
        row_ids = []
        return row_ids

    row_ids = []
    seen_row_ids: set[str] = set()
    for dispatched_entry in dispatched:
        if not isinstance(dispatched_entry, Mapping):
            continue
        agent = _mapping_text(dispatched_entry, "agent")
        if agent not in _CONVERSATION_EVIDENCE_AGENTS:
            continue
        source_refs = dispatched_entry.get("source_refs")
        if not isinstance(source_refs, list):
            continue
        for source_ref in source_refs:
            if not isinstance(source_ref, Mapping):
                continue
            row_id = _mapping_text(source_ref, "conversation_row_id")
            if not row_id:
                row_id = _mapping_text(source_ref, "_id")
            if not row_id or row_id in seen_row_ids:
                continue
            row_ids.append(row_id)
            seen_row_ids.add(row_id)
            if len(row_ids) >= limit:
                return row_ids
    return row_ids


def _valid_candidates(
    candidates: Sequence[PastDialogCognitionCandidate],
    *,
    character_global_user_id: str,
    max_dialogs: int,
) -> tuple[list[PastDialogCognitionCandidate], list[dict[str, str]]]:
    """Return candidates eligible for trace lookup and skip diagnostics."""

    valid_candidates: list[PastDialogCognitionCandidate] = []
    diagnostics: list[dict[str, str]] = []
    seen_trace_ids: set[str] = set()
    character_id = character_global_user_id.strip()
    if max_dialogs <= 0:
        diagnostics.append({
            "status": "skipped",
            "reason": "max dialogs is zero",
        })
        return valid_candidates, diagnostics

    for candidate in candidates:
        skip_reason = _candidate_skip_reason(
            candidate,
            character_global_user_id=character_id,
        )
        if skip_reason:
            diagnostics.append({
                "status": "skipped",
                "reason": skip_reason,
                "source": candidate.source,
            })
            continue
        trace_id = candidate.llm_trace_id.strip()
        if trace_id in seen_trace_ids:
            diagnostics.append({
                "status": "skipped",
                "reason": "duplicate llm_trace_id",
                "source": candidate.source,
            })
            continue
        seen_trace_ids.add(trace_id)
        valid_candidates.append(candidate)
        if len(valid_candidates) >= max_dialogs:
            break
    return valid_candidates, diagnostics


def _candidate_skip_reason(
    candidate: PastDialogCognitionCandidate,
    *,
    character_global_user_id: str,
) -> str:
    """Return the structural reason a candidate cannot be used."""

    if candidate.role.strip().lower() != "assistant":
        skip_reason = "non-assistant row"
        return skip_reason
    if candidate.global_user_id.strip() != character_global_user_id:
        skip_reason = "non-character row"
        return skip_reason
    if not candidate.visible_text.strip():
        skip_reason = "empty visible dialog"
        return skip_reason
    if not candidate.llm_trace_id.strip():
        skip_reason = "missing trace id"
        return skip_reason
    skip_reason = ""
    return skip_reason


def _unique_trace_ids(
    candidates: Sequence[PastDialogCognitionCandidate],
) -> list[str]:
    """Return trace ids in candidate order without duplicates."""

    trace_ids: list[str] = []
    seen_trace_ids: set[str] = set()
    for candidate in candidates:
        trace_id = candidate.llm_trace_id.strip()
        if trace_id in seen_trace_ids:
            continue
        trace_ids.append(trace_id)
        seen_trace_ids.add(trace_id)
    return trace_ids


def _steps_by_trace_id(
    trace_steps: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group trace steps by trace id."""

    grouped_steps: dict[str, list[Mapping[str, Any]]] = {}
    for step in trace_steps:
        trace_id = _mapping_text(step, "trace_id")
        if not trace_id:
            continue
        if trace_id not in grouped_steps:
            grouped_steps[trace_id] = []
        grouped_steps[trace_id].append(step)
    return grouped_steps


def _row_text(row: Mapping[str, Any], field_name: str) -> str:
    """Read one conversation row field as stripped text."""

    value = row.get(field_name)
    if value is None:
        text = ""
    else:
        text = str(value).strip()
    return text


def _mapping_text(row: Mapping[str, Any], field_name: str) -> str:
    """Read one mapping field as stripped text."""

    value = row.get(field_name)
    if value is None:
        text = ""
    else:
        text = str(value).strip()
    return text


def _conversation_row_id(row: Mapping[str, Any]) -> str:
    """Return the canonical row ref from conversation fields."""

    row_id = _row_text(row, "conversation_row_id")
    if row_id:
        return row_id
    raw_object_id = row.get("_id")
    if raw_object_id is None:
        row_id = ""
    else:
        row_id = str(raw_object_id).strip()
    return row_id


def _empty_result(
    *,
    candidate_count: int,
    status: str,
    diagnostics: list[dict[str, str]],
) -> PastDialogCognitionLookupResult:
    """Build an empty lookup result with status metadata."""

    result: PastDialogCognitionLookupResult = {
        "past_dialog_cognition_context": "",
        "candidate_count": candidate_count,
        "selected_count": 0,
        "status": status,
        "diagnostics": diagnostics,
    }
    return result
