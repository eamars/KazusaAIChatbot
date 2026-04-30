"""TypedDict payloads for the adapter-to-brain message envelope.

This module owns the dict-shaped data contracts exchanged between platform
adapters, the service intake layer, storage, RAG, cognition, and dialog. Inputs
are normalized platform message fragments; outputs are TypedDicts that keep
wire syntax separate from user-authored `body_text`. Concrete normalizers
extend the system by producing these shapes.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

ConversationAuthorRole = Literal["user", "assistant"]
MentionEntityKind = Literal[
    "bot",
    "user",
    "platform_role",
    "channel",
    "everyone",
    "unknown",
]
ReplyDerivation = Literal["platform_native", "leading_mention"]
AttachmentStorageShape = Literal["inline", "url_only", "drop"]


class RawMention(TypedDict, total=False):
    """Unresolved mention extracted by a platform normalizer."""

    platform: str
    platform_user_id: str
    raw_text: str
    entity_kind: MentionEntityKind


class ResolvedMention(TypedDict, total=False):
    """Mention after user/profile resolution."""

    platform: str
    platform_user_id: str
    global_user_id: str
    display_name: str
    entity_kind: MentionEntityKind
    raw_text: str


class Mention(TypedDict, total=False):
    """Typed mention stored on a message envelope and conversation row."""

    platform_user_id: str
    global_user_id: str
    display_name: str
    entity_kind: MentionEntityKind
    raw_text: str


class RawReply(TypedDict, total=False):
    """Unresolved platform reply target before profile resolution."""

    platform_message_id: str
    platform_user_id: str
    display_name: str
    excerpt: str
    derivation: ReplyDerivation


class ReplyTarget(TypedDict, total=False):
    """Typed reply target retained outside user-authored body text."""

    platform_message_id: str
    platform_user_id: str
    global_user_id: str
    display_name: str
    excerpt: str
    derivation: ReplyDerivation


class AttachmentRef(TypedDict, total=False):
    """Attachment reference stored without changing current text consumption."""

    media_type: str
    url: str
    base64_data: str
    description: str
    size_bytes: int
    storage_shape: AttachmentStorageShape


class NormalizedEnvelopeFragment(TypedDict, total=False):
    """Partial normalized payload returned by adapter-specific parsers."""

    body_text: str
    raw_wire_text: str
    raw_mentions: list[RawMention]
    raw_reply: RawReply
    attachments: list[AttachmentRef]
    broadcast: bool


class MessageEnvelope(TypedDict):
    """Complete message contract consumed by the brain graph."""

    body_text: str
    raw_wire_text: str
    mentions: list[Mention]
    reply: NotRequired[ReplyTarget]
    attachments: list[AttachmentRef]
    addressed_to_global_user_ids: list[str]
    broadcast: bool
