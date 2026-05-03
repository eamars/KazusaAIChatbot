"""RAG helper agent for scoped current-user continuity evidence."""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import OpenAIError

from kazusa_ai_chatbot.db import (
    get_text_embedding,
    query_user_memory_units,
    search_user_memory_units_by_keyword,
    search_user_memory_units_by_vector,
)
from kazusa_ai_chatbot.db.schemas import UserMemoryUnitStatus
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

_AGENT_NAME = "user_memory_evidence_agent"
_UNCACHED_REASON = "scoped_user_memory_uncached"
_MAX_ROWS = 5
_MAX_LITERAL_TERMS = 8
_DEFAULT_TRUTH_STATUS = "character_lore_or_interaction_continuity"
_DEFAULT_ORIGIN = "consolidated_interaction"
_DEFAULT_AUTHORITY = "scoped_continuity"


def _cache_status() -> dict[str, Any]:
    """Build the fixed no-cache metadata block for scoped user memory evidence."""
    status = {
        "enabled": False,
        "hit": False,
        "reason": _UNCACHED_REASON,
    }
    return status


def _task_body(task: str) -> str:
    """Remove the top-level capability prefix when present."""
    if ":" not in task:
        stripped_task = task.strip()
        return stripped_task
    _, _, remainder = task.partition(":")
    stripped_task = remainder.strip()
    return stripped_task


def _extract_literal_terms(task: str) -> list[str]:
    """Extract bounded literal anchors for scoped lexical retrieval.

    Args:
        task: Memory-evidence slot text.

    Returns:
        Ordered literal search terms, strongest anchors first.
    """
    task_body = _task_body(task)
    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r"“([^”]+)”",
        r"‘([^’]+)’",
        r"「([^」]+)」",
        r"『([^』]+)』",
    ]
    literals: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, task_body):
            literal = text_or_empty(match).strip()
            if literal and literal not in literals:
                literals.append(literal)
            for cjk_run in re.findall(r"[\u3400-\u9fff]+", literal):
                if len(cjk_run) <= 2:
                    continue
                for index in range(len(cjk_run) - 1):
                    cjk_term = cjk_run[index:index + 2]
                    if cjk_term and cjk_term not in literals:
                        literals.append(cjk_term)
                    if len(literals) >= _MAX_LITERAL_TERMS:
                        return literals

    return literals


def _allows_recency_fallback(task: str) -> bool:
    """Return whether an unanchored request can use bounded recent continuity."""
    task_body = _task_body(task).lower()
    broad_markers = (
        "current user's continuity",
        "current user continuity",
        "private continuity",
        "user memory evidence",
        "当前用户之间已经形成的私有连续性",
    )
    specific_markers = (
        " about ",
        " around ",
        " relevant to ",
        "exact term",
        "关于",
    )
    has_broad_marker = any(marker in task_body for marker in broad_markers)
    has_specific_marker = any(marker in task_body for marker in specific_markers)
    return_value = has_broad_marker and not has_specific_marker
    return return_value


def _row_content(row: dict[str, Any]) -> str:
    """Choose the prompt-facing content field for one memory row."""
    for field in ("content", "fact", "subjective_appraisal", "relationship_signal"):
        text = text_or_empty(row.get(field))
        if text:
            return text
    return ""


def _project_row(row: dict[str, Any], global_user_id: str) -> dict[str, Any]:
    """Project one raw user-memory row into the public scoped evidence shape."""
    projected_row = dict(row)
    content = _row_content(projected_row)
    projected_row["content"] = content
    projected_row["source_system"] = "user_memory_units"
    projected_row["scope_type"] = "user_continuity"
    projected_row["scope_global_user_id"] = global_user_id

    authority = text_or_empty(projected_row.get("authority")) or _DEFAULT_AUTHORITY
    truth_status = (
        text_or_empty(projected_row.get("truth_status")) or _DEFAULT_TRUTH_STATUS
    )
    origin = text_or_empty(projected_row.get("origin")) or _DEFAULT_ORIGIN
    projected_row["authority"] = authority
    projected_row["truth_status"] = truth_status
    projected_row["origin"] = origin
    return projected_row


def _project_rows(rows: list[dict[str, Any]], global_user_id: str) -> list[dict[str, Any]]:
    """Project bounded raw rows into public scoped evidence rows."""
    projected_rows = [
        _project_row(row, global_user_id)
        for row in rows[:_MAX_ROWS]
        if isinstance(row, dict)
    ]
    return projected_rows


def _result_payload(
    *,
    rows: list[dict[str, Any]],
    global_user_id: str,
    missing_context: list[str],
) -> dict[str, Any]:
    """Build the public result payload for scoped user-memory evidence."""
    selected_summary = "\n".join(
        content
        for row in rows
        if (content := _row_content(row))
    )
    payload = {
        "selected_summary": selected_summary,
        "memory_rows": rows,
        "source_system": "user_memory_units",
        "scope_type": "user_continuity",
        "scope_global_user_id": global_user_id,
        "missing_context": missing_context,
    }
    return payload


def _agent_result(*, resolved: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap the scoped user-memory payload in the helper-agent result envelope."""
    result = {
        "resolved": resolved,
        "result": payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return result


class UserMemoryEvidenceAgent(BaseRAGHelperAgent):
    """Retrieve current-user continuity evidence from ``user_memory_units`` only."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached scoped user-memory evidence agent."""
        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Retrieve scoped user continuity for one memory-evidence slot.

        Args:
            task: The ``Memory-evidence:`` slot text.
            context: Runtime context containing the current user scope.
            max_attempts: Ignored. Present for interface parity.

        Returns:
            Standard helper-agent result with scoped user-memory evidence rows.
        """
        del max_attempts

        global_user_id = text_or_empty(context.get("global_user_id"))
        if not global_user_id:
            payload = _result_payload(
                rows=[],
                global_user_id="",
                missing_context=["global_user_id"],
            )
            result = _agent_result(resolved=False, payload=payload)
            return result

        rows: list[dict[str, Any]] = []
        task_body = _task_body(task)
        literal_terms = _extract_literal_terms(task)

        if literal_terms:
            for literal in literal_terms:
                keyword_rows = await search_user_memory_units_by_keyword(
                    global_user_id,
                    literal,
                    statuses=[UserMemoryUnitStatus.ACTIVE],
                    limit=_MAX_ROWS,
                )
                if keyword_rows:
                    rows = keyword_rows
                    break

        if not rows and not literal_terms:
            query_embedding: list[float] | None = None
            try:
                query_embedding = await get_text_embedding(task_body)
            except OpenAIError as exc:
                logger.info(
                    f"{_AGENT_NAME} embedding unavailable; "
                    f"falling back to scoped non-vector retrieval: {exc}"
                )

            if query_embedding:
                rows = await search_user_memory_units_by_vector(
                    global_user_id,
                    query_embedding,
                    statuses=[UserMemoryUnitStatus.ACTIVE],
                    limit=_MAX_ROWS,
                )

        if not rows and not literal_terms and _allows_recency_fallback(task):
            rows = await query_user_memory_units(
                global_user_id,
                statuses=[UserMemoryUnitStatus.ACTIVE],
                limit=_MAX_ROWS,
            )

        projected_rows = _project_rows(rows, global_user_id)
        resolved = bool(projected_rows)
        missing_context = [] if resolved else ["user_memory_evidence"]
        payload = _result_payload(
            rows=projected_rows,
            global_user_id=global_user_id,
            missing_context=missing_context,
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"scope_global_user_id={global_user_id} "
            f"row_count={len(projected_rows)} "
            f"missing_context={missing_context}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result
