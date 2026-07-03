"""Outbound OneBot payload and mention rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence

from adapters.envelope_common import normalize_numeric_platform_user_id
from adapters.inline_mentions import InlineMention, inline_mention_parts


def outbound_message_payload(
    text: str,
    reply_to_msg_id: str | None,
    delivery_mentions: Sequence[dict] | None = None,
) -> str | list[dict[str, dict[str, str] | str]]:
    """Build a OneBot message payload for normal or reply-anchored sends.

    Args:
        text: Plain message text to send.
        reply_to_msg_id: Platform message id to quote with a native reply
            segment, if the caller requested reply anchoring.
        delivery_mentions: Optional adapter-owned mention requests.

    Returns:
        A plain text payload for ordinary sends, or a structured message segment
        array for reply or native mention sends.
    """

    message_segments = inline_user_mention_segments(text, delivery_mentions)
    has_native_mention = any(
        segment.get("type") == "at" for segment in message_segments
    )
    if not reply_to_msg_id and not has_native_mention:
        return_value = text
        return return_value

    return_value = []
    if reply_to_msg_id:
        return_value.append({
            "type": "reply",
            "data": {"id": str(reply_to_msg_id)},
        })
    return_value.extend(message_segments)
    return return_value


def inline_user_mention_segments(
    text: str,
    delivery_mentions: Sequence[dict] | None,
) -> list[dict[str, dict[str, str] | str]]:
    """Return OneBot segments with exact inline user tags rendered natively."""

    segments: list[dict[str, dict[str, str] | str]] = []
    for part in inline_mention_parts(text, delivery_mentions):
        if isinstance(part, InlineMention):
            normalized_user_id = normalize_numeric_platform_user_id(
                part.platform_user_id
            )
            if normalized_user_id:
                segments.append({
                    "type": "at",
                    "data": {"qq": normalized_user_id},
                })
            else:
                _append_text_segment(segments, part.token)
            continue
        _append_text_segment(segments, part)

    return segments


def _append_text_segment(
    segments: list[dict[str, dict[str, str] | str]],
    text: str,
) -> None:
    """Append non-empty text, merging adjacent text segments."""

    if not text:
        return
    if segments and segments[-1].get("type") == "text":
        data = segments[-1]["data"]
        if isinstance(data, dict):
            current_text = data["text"]
            data["text"] = f"{current_text}{text}"
        return
    segments.append({
        "type": "text",
        "data": {"text": text},
    })
