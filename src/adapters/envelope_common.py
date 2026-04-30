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
