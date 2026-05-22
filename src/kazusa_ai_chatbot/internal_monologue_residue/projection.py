"""Prompt-facing projection for selected internal monologue residue rows."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from kazusa_ai_chatbot.internal_monologue_residue.models import (
    InternalMonologueResidueRow,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


def project_residue_window(
    *,
    rows: Sequence[InternalMonologueResidueRow],
    current_timestamp_utc: str,
    context_char_limit: int,
) -> str:
    """Compile selected residue rows into one bounded prompt-facing string.

    Args:
        rows: Already selected and ordered residue rows.
        current_timestamp_utc: Storage UTC timestamp for age labels.
        context_char_limit: Maximum characters allowed in the returned string.

    Returns:
        One age-labeled string for L2a, or an empty string when no row exists.
    """

    if not rows or context_char_limit <= 0:
        return_value = ""
        return return_value

    current_time = parse_storage_utc_datetime(current_timestamp_utc)
    lines: list[str] = []
    newest_first = sorted(
        rows,
        key=lambda row: str(row.get("created_at") or ""),
        reverse=True,
    )
    for row in newest_first:
        residue_text = str(row.get("residue_text") or "").strip()
        if not residue_text:
            continue
        created_at = str(row.get("created_at") or current_timestamp_utc)
        age_label = _age_label(
            current_timestamp_utc=current_timestamp_utc,
            current_time=current_time,
            created_at=created_at,
        )
        line = f"- {age_label}: {residue_text}"
        candidate_lines = [*lines, line]
        candidate_projected = "\n".join(candidate_lines)
        if _budget_length(candidate_projected) <= context_char_limit:
            lines = candidate_lines
            continue
        if not lines:
            projected = _truncate_to_budget(
                text=line,
                context_char_limit=context_char_limit,
            )
            return_value = projected
            return return_value

    projected = "\n".join(lines)
    return_value = projected
    return return_value


def _age_label(
    *,
    current_timestamp_utc: str,
    current_time,
    created_at: str,
) -> str:
    """Return a coarse age label without exposing row identifiers."""

    try:
        created_time = parse_storage_utc_datetime(created_at)
    except ValueError:
        created_time = parse_storage_utc_datetime(current_timestamp_utc)

    age = current_time - created_time
    if age < timedelta():
        age = timedelta()

    total_minutes = int(age.total_seconds() // 60)
    if total_minutes < 1:
        label = "刚刚留下"
    elif total_minutes < 60:
        label = f"约{total_minutes}分钟前"
    else:
        total_hours = total_minutes // 60
        if total_hours < 24:
            label = f"约{total_hours}小时前"
        else:
            total_days = total_hours // 24
            label = f"约{total_days}天前"
    return label


def _budget_length(text: str) -> int:
    """Return the prompt budget length for rendered CJK-heavy context."""

    budget_length = len(text.encode("utf-8"))
    return budget_length


def _truncate_to_budget(*, text: str, context_char_limit: int) -> str:
    """Truncate one rendered line without splitting UTF-8 characters."""

    encoded_text = text.encode("utf-8")
    truncated = encoded_text[:context_char_limit]
    decoded_text = truncated.decode("utf-8", errors="ignore").rstrip()
    return decoded_text
