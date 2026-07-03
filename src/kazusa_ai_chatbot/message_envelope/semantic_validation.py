"""Storage-boundary validation for semantic message-envelope fields."""

from __future__ import annotations

import re
from collections.abc import Iterator

from kazusa_ai_chatbot.message_envelope.types import MessageEnvelope


_FORBIDDEN_FIELD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(@mentioned-(?:user|role|entity)-\d+|#mentioned-channel-\d+)"
        ),
        "legacy occurrence placeholder",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])[@#]?"
            r"(?:qq|discord|platform)-"
            r"(?:user|bot|role|channel|entity):[^\s]+"
        ),
        "platform-qualified semantic fallback label",
    ),
    (re.compile(r"\[CQ:"), "CQ transport marker"),
    (re.compile(r"<@!?\d+>"), "Discord user mention tag"),
    (re.compile(r"<@&\d+>"), "Discord role mention tag"),
    (re.compile(r"<#\d+>"), "Discord channel mention tag"),
    (
        re.compile(r"<a?:[A-Za-z0-9_]+:\d+>"),
        "Discord custom emoji tag",
    ),
)


def _semantic_text_fields(
    *,
    display_name: str,
    envelope: MessageEnvelope,
) -> Iterator[tuple[str, str]]:
    """Yield semantic fields that must be safe before storage."""

    yield ("display_name", display_name)
    yield ("body_text", envelope["body_text"])

    for mention_index, mention in enumerate(envelope["mentions"]):
        mention_display_name = mention.get("display_name", "")
        if isinstance(mention_display_name, str):
            yield (f"mentions[{mention_index}].display_name", mention_display_name)

    reply = envelope.get("reply")
    if reply is None:
        return

    reply_display_name = reply.get("display_name", "")
    if isinstance(reply_display_name, str):
        yield ("reply.display_name", reply_display_name)

    reply_excerpt = reply.get("excerpt", "")
    if isinstance(reply_excerpt, str):
        yield ("reply.excerpt", reply_excerpt)


def validate_semantic_storage_fields(
    *,
    platform: str,
    display_name: str,
    envelope: MessageEnvelope,
) -> None:
    """Reject transport syntax and legacy placeholders before persistence.

    Args:
        platform: Adapter platform key used only for safe error scope.
        display_name: Top-level sender display label supplied by the adapter.
        envelope: Adapter-normalized message envelope to validate.

    Raises:
        ValueError: If any semantic storage field contains raw platform syntax
            or occurrence-local legacy placeholders.
    """

    platform_label = str(platform or "").strip() or "unknown"
    for field_name, field_value in _semantic_text_fields(
        display_name=display_name,
        envelope=envelope,
    ):
        for pattern, reason in _FORBIDDEN_FIELD_PATTERNS:
            if pattern.search(field_value):
                raise ValueError(
                    "invalid semantic storage field: "
                    f"platform={platform_label} field={field_name} "
                    f"reason={reason}"
                )
