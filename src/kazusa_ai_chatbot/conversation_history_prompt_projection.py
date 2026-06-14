"""Universal model-facing conversation-history projection.

Every LLM payload that presents conversation-history rows must use
``project_conversation_history_for_llm`` as the sole formatter.  Internal
structured rows remain available to deterministic code before projection.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import project_text_with_image_blocks


def project_conversation_history_for_llm(
    rows: Iterable[Mapping[str, Any]],
    *,
    character_name: str = "",
    max_rows: int | None = None,
) -> list[str]:
    """Return model-facing logging-style conversation transcript lines.

    Args:
        rows: Conversation-history row mappings, already selected, filtered,
            and ordered by the caller.
        character_name: Active character display name, used as speaker fallback
            for assistant rows that have no ``display_name``.
        max_rows: When provided, keep only the last *max_rows* rows while
            preserving chronological order.

    Returns:
        Chronological transcript lines.  Each line follows the grammar::

            [<timestamp>] <speaker> reply_to <target>: <text>
            [<timestamp>] <speaker>: <text>
            <speaker> reply_to <target>: <text>
            <speaker>: <text>

        Timestamps appear only when available.  Reply targets appear only when
        reply metadata is present.
    """

    row_list = list(rows)
    if max_rows is not None and len(row_list) > max_rows:
        row_list = row_list[-max_rows:]

    projected_lines: list[str] = []
    for row in row_list:
        if not isinstance(row, Mapping):
            continue

        line = _project_single_row(row, character_name=character_name)
        projected_lines.append(line)

    return_value = projected_lines
    return return_value


def _project_single_row(
    row: Mapping[str, Any],
    *,
    character_name: str,
) -> str:
    """Render one conversation-history row as a transcript line."""

    speaker_name = _resolve_speaker(row, character_name=character_name)
    body_text = _resolve_body_text(row)
    timestamp = _resolve_timestamp(row)
    reply_target = _resolve_reply_target(row)

    speaker_segment = speaker_name
    if reply_target:
        speaker_segment = f"{speaker_name} reply_to {reply_target}"

    if timestamp:
        line_prefix = f"[{timestamp}] {speaker_segment}"
    else:
        line_prefix = speaker_segment

    line = f"{line_prefix}: {body_text}"
    return line


def _resolve_speaker(
    row: Mapping[str, Any],
    *,
    character_name: str,
) -> str:
    """Determine the visible speaker label for a row."""

    display_name = row.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = row.get("name")
    if isinstance(display_name, str) and display_name.strip():
        speaker_name = display_name.strip()
    elif row.get("role") == "assistant":
        speaker_name = character_name if character_name else "unknown"
    else:
        speaker_name = "unknown"
    return speaker_name


def _resolve_body_text(row: Mapping[str, Any]) -> str:
    """Resolve message text from the row, with attachment projection."""

    body_text = row.get("body_text")
    if body_text is None:
        body_text = row.get("content")
    if body_text is None:
        body_text = row.get("text")
    if not isinstance(body_text, str):
        body_text = ""

    projected_body_text = project_text_with_image_blocks(
        body_text.strip(),
        row.get("attachments"),
    )
    return projected_body_text


def _resolve_timestamp(row: Mapping[str, Any]) -> str:
    """Resolve timestamp for prompt display."""

    raw_timestamp = row.get("timestamp")
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        return ""

    formatted_timestamp = format_storage_utc_for_llm(raw_timestamp)
    if formatted_timestamp:
        timestamp = formatted_timestamp
    else:
        timestamp = raw_timestamp.strip()
    return timestamp


def _resolve_reply_target(row: Mapping[str, Any]) -> str:
    """Resolve the reply-to display name if present."""

    reply_context_value = row.get("reply_context")
    if isinstance(reply_context_value, dict):
        reply_context = reply_context_value
    else:
        reply_context = {}

    reply_target = reply_context.get("reply_to_display_name")
    if not isinstance(reply_target, str) or not reply_target.strip():
        reply_target = row.get("reply_to_display_name")

    if isinstance(reply_target, str) and reply_target.strip():
        clean_reply_target = reply_target.strip()
        return clean_reply_target
    return ""
