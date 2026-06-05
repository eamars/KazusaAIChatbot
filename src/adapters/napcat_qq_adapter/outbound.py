"""Outbound OneBot payload and mention rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence

from adapters.envelope_common import normalize_numeric_platform_user_id


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

    mention_segments = prefix_user_mention_segments(delivery_mentions)
    outbound_text = text
    if mention_segments and text and not text[0].isspace():
        outbound_text = f" {text}"
    if not reply_to_msg_id and not mention_segments:
        return_value = text
    else:
        return_value = []
        if reply_to_msg_id:
            return_value.append({
                "type": "reply",
                "data": {"id": str(reply_to_msg_id)},
            })
        return_value.extend(mention_segments)
        return_value.append({
            "type": "text",
            "data": {"text": outbound_text},
        })
    return return_value


def prefix_user_mention_segments(
    delivery_mentions: Sequence[dict] | None,
) -> list[dict[str, dict[str, str] | str]]:
    """Return QQ native prefix mention segments when metadata is renderable."""

    if not delivery_mentions:
        return_value: list[dict[str, dict[str, str] | str]] = []
        return return_value

    for mention in delivery_mentions:
        if not isinstance(mention, dict):
            continue
        if mention.get("entity_kind") != "user":
            continue
        if mention.get("placement") != "prefix":
            continue
        normalized_user_id = normalize_numeric_platform_user_id(
            mention.get("platform_user_id")
        )
        if not normalized_user_id:
            continue
        return_value = [
            {
                "type": "at",
                "data": {"qq": normalized_user_id},
            }
        ]
        return return_value

    return_value = []
    return return_value
