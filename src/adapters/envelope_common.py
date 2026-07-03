"""Shared adapter-owned helpers for typed message-envelope construction.

This module belongs to the adapter boundary. Inputs are request-like adapter
payloads, raw mention fragments, and the public message-envelope registries;
outputs are typed pieces used by concrete platform normalizers inside adapter
modules. Brain modules do not import this file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.message_envelope.protocols import (
    AttachmentHandlerRegistryProtocol,
    MentionResolver,
)
from kazusa_ai_chatbot.message_envelope.types import (
    AttachmentRef,
    Mention,
    RawMention,
    ReplyTarget,
)


def normalize_body_spacing(text: str) -> str:
    """Trim marker-removal whitespace without collapsing authored newlines.

    Args:
        text: Text after platform marker removal.

    Returns:
        Body text with surplus spaces and empty-line runs cleaned.
    """

    without_runs = re.sub(r"[ \t]+", " ", text)
    normalized_lines = [
        line.strip()
        for line in without_runs.splitlines()
    ]
    without_empty_runs = "\n".join(normalized_lines)
    return_value = re.sub(r"\n{3,}", "\n\n", without_empty_runs).strip()
    return return_value


def normalize_mention_display_label(value: object) -> str:
    """Normalize an adapter-provided display label for LLM-visible text.

    Args:
        value: External platform label value from an event or lookup response.

    Returns:
        Trimmed label with internal whitespace collapsed, or an empty string
        when no readable label is available.
    """

    if not isinstance(value, str):
        return_value = ""
        return return_value

    normalized = re.sub(r"\s+", " ", value).strip()
    normalized = normalized.lstrip("@#").strip()
    return normalized


def normalize_mention_display_map(value: object) -> dict[str, str]:
    """Normalize an adapter-provided id-to-display-label mapping.

    Args:
        value: External mapping supplied by an adapter event handler.

    Returns:
        Mapping with string ids and non-empty sanitized display labels.
    """

    if not isinstance(value, Mapping):
        return_value: dict[str, str] = {}
        return return_value

    display_names: dict[str, str] = {}
    for platform_user_id, display_name in value.items():
        label = normalize_mention_display_label(display_name)
        if not label:
            continue
        display_names[str(platform_user_id)] = label
    return display_names


def normalize_numeric_platform_user_id(value: object) -> str:
    """Return a native user id only when it is safe to interpolate.

    Args:
        value: Adapter-facing platform user id from delivery metadata.

    Returns:
        A stripped ASCII digit string, or an empty string when the value is not
        a concrete platform user id.
    """

    if not isinstance(value, str):
        return_value = ""
        return return_value

    normalized_id = value.strip()
    if not normalized_id:
        return_value = ""
        return return_value
    if not normalized_id.isascii() or not normalized_id.isdecimal():
        return_value = ""
        return return_value

    return_value = normalized_id
    return return_value


def semantic_entity_fallback_label(
    *,
    entity_kind: str,
    mention_context: bool,
) -> str:
    """Return a platform-agnostic fallback label for an unresolved entity.

    Args:
        entity_kind: Message-envelope mention kind.
        mention_context: Whether the label will appear as a visible mention
            token in body text.

    Returns:
        Prompt-facing semantic fallback text. Platform ids stay in typed
        metadata and raw replay fields, not in this label.
    """

    normalized_kind = str(entity_kind or "unknown").strip()
    if normalized_kind == "bot":
        base_label = "character"
        prefix = "@"
    elif normalized_kind == "user":
        base_label = "user"
        prefix = "@"
    elif normalized_kind == "platform_role":
        base_label = "role"
        prefix = "@"
    elif normalized_kind == "channel":
        base_label = "channel"
        prefix = "#"
    elif normalized_kind == "everyone":
        base_label = "everyone"
        prefix = "@"
    else:
        base_label = "entity"
        prefix = "@"

    if mention_context:
        return_value = f"{prefix}{base_label}"
    else:
        return_value = base_label
    return return_value


def readable_mention_token(
    *,
    entity_kind: str,
    display_name: str,
    raw_label: str = "",
) -> str:
    """Format a platform-neutral visible mention token for body text.

    Args:
        entity_kind: Typed mention kind such as user, bot, role, or channel.
        display_name: Adapter-resolved display label, if one is available.
        raw_label: Safe authored broadcast label for everyone/here/all tokens.

    Returns:
        A readable mention token that does not expose platform wire syntax.
    """

    label = normalize_mention_display_label(display_name)
    normalized_kind = entity_kind or "unknown"
    if normalized_kind == "channel":
        if label:
            return_value = f"#{label}"
        else:
            return_value = semantic_entity_fallback_label(
                entity_kind=normalized_kind,
                mention_context=True,
            )
        return return_value

    if normalized_kind == "everyone":
        broadcast_label = normalize_mention_display_label(raw_label) or "everyone"
        return_value = f"@{broadcast_label}"
        return return_value

    if label:
        return_value = f"@{label}"
    else:
        return_value = semantic_entity_fallback_label(
            entity_kind=normalized_kind,
            mention_context=True,
        )
    return return_value


def attachment_refs(
    raw_attachments: list[object],
    handlers: AttachmentHandlerRegistryProtocol,
) -> list[AttachmentRef]:
    """Build attachment refs using the registered handler for each payload.

    Args:
        raw_attachments: Adapter-owned attachment payloads.
        handlers: Registry-like object exposing `handler_for(media_type)`.

    Returns:
        Normalized attachment references, excluding droppable empty shells.
    """

    refs: list[AttachmentRef] = []
    for raw_attachment in raw_attachments:
        if isinstance(raw_attachment, Mapping):
            media_type = raw_attachment.get("media_type")
        else:
            media_type = getattr(raw_attachment, "media_type", None)
        if not isinstance(media_type, str):
            continue
        handler = handlers.handler_for(media_type)
        if handler is None:
            continue
        attachment = handler.build_ref(raw_attachment)
        if attachment.get("storage_shape") == "drop":
            continue
        refs.append(attachment)

    return refs


def resolve_mentions(
    raw_mentions: list[RawMention],
    resolver: MentionResolver,
) -> list[Mention]:
    """Resolve platform mentions and project them into stored mention shape.

    Args:
        raw_mentions: Mention fragments emitted by a platform normalizer.
        resolver: Resolver implementation for platform-to-global identity.

    Returns:
        Mention payloads suitable for an envelope or conversation row.
    """

    mentions: list[Mention] = []
    for raw_mention in raw_mentions:
        resolved = resolver.resolve(raw_mention)
        mention: Mention = {
            "platform_user_id": resolved.get("platform_user_id", ""),
            "global_user_id": resolved.get("global_user_id", ""),
            "display_name": resolved.get("display_name", ""),
            "entity_kind": resolved.get("entity_kind", "unknown"),
            "raw_text": resolved.get("raw_text", ""),
        }
        mentions.append(mention)
    return mentions


def addressed_to_from_envelope_parts(
    *,
    mentions: list[Mention],
    reply: ReplyTarget,
    channel_type: str,
) -> list[str]:
    """Derive inbound addressees from typed mentions, reply, and DM context.

    Args:
        mentions: Resolved mentions on the inbound message.
        reply: Typed reply target, if any.
        channel_type: Adapter-supplied channel type.

    Returns:
        Deduplicated list of global user ids addressed by the inbound row.
    """

    addressed_to: list[str] = []
    if channel_type == "private":
        addressed_to.append(CHARACTER_GLOBAL_USER_ID)

    for mention in mentions:
        if mention.get("entity_kind") not in ("bot", "user"):
            continue
        global_user_id = mention.get("global_user_id", "")
        if global_user_id:
            addressed_to.append(global_user_id)

    if reply.get("global_user_id"):
        global_user_id = reply["global_user_id"]
        addressed_to.append(global_user_id)

    deduplicated = list(dict.fromkeys(addressed_to))
    return deduplicated
