"""Prompt-safe projection for current-message envelope context.

This module owns the LLM-facing projection of a typed message envelope. Inputs
are the storage/audit `MessageEnvelope` and optional multimedia summaries;
outputs are bounded `PromptMessageContext` dictionaries that preserve semantic
metadata while excluding storage-only binary and wire fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.message_envelope.types import MentionEntityKind, MessageEnvelope

MAX_PROMPT_BODY_TEXT_CHARS = 2000
MAX_PROMPT_REPLY_EXCERPT_CHARS = 500
MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS = 800
MAX_PROMPT_ATTACHMENTS = 4
MAX_PROMPT_MESSAGE_CONTEXT_CHARS = 5000

_ELLIPSIS = "..."
_SENTENCE_BOUNDARIES = {".", "!", "?", "\n", "\u3002", "\uff01", "\uff1f"}
_TRAILING_SENTENCE_PUNCTUATION = ".!?\u3002\uff01\uff1f"

_PROMPT_MESSAGE_KEYS = {
    "body_text",
    "addressed_to_global_user_ids",
    "broadcast",
    "mentions",
    "attachments",
    "reply",
}
_PROMPT_ATTACHMENT_KEYS = {"media_kind", "description", "summary_status"}
_PROMPT_MENTION_KEYS = {
    "platform_user_id",
    "global_user_id",
    "display_name",
    "entity_kind",
}
_PROMPT_REPLY_KEYS = {
    "platform_message_id",
    "platform_user_id",
    "global_user_id",
    "display_name",
    "excerpt",
    "derivation",
}


class PromptAttachmentSummary(TypedDict):
    """Prompt-facing description of one current-turn attachment."""

    media_kind: str
    description: str
    summary_status: Literal["available", "unavailable"]


class PromptReplyContext(TypedDict, total=False):
    """Prompt-safe reply-target metadata."""

    platform_message_id: str
    platform_user_id: str
    global_user_id: str
    display_name: str
    excerpt: str
    derivation: str


class PromptMentionContext(TypedDict, total=False):
    """Prompt-safe mention metadata without raw platform syntax."""

    platform_user_id: str
    global_user_id: str
    display_name: str
    entity_kind: MentionEntityKind


class PromptMessageContext(TypedDict):
    """Bounded current-message structure safe to serialize into text prompts."""

    body_text: str
    addressed_to_global_user_ids: list[str]
    broadcast: bool
    mentions: list[PromptMentionContext]
    attachments: list[PromptAttachmentSummary]
    reply: NotRequired[PromptReplyContext]


class PromptContextTooLargeError(ValueError):
    """Raised when the prompt-safe projection cannot fit the global cap."""


def _json_size(payload: Mapping[str, object]) -> int:
    rendered = json.dumps(payload, ensure_ascii=False)
    return_value = len(rendered)
    return return_value


def _truncate_with_suffix(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= len(_ELLIPSIS):
        return_value = _ELLIPSIS[:max_chars]
        return return_value
    body_limit = max_chars - len(_ELLIPSIS)
    return_value = f"{value[:body_limit].rstrip()}{_ELLIPSIS}"
    return return_value


def _trim_description(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text

    hard_trimmed = text[: max_chars - len(_ELLIPSIS)].rstrip()
    boundary_index = -1
    for index, character in enumerate(hard_trimmed):
        if character in _SENTENCE_BOUNDARIES:
            boundary_index = index

    if boundary_index >= max_chars // 2:
        hard_trimmed = hard_trimmed[: boundary_index + 1].rstrip()

    hard_trimmed = hard_trimmed.rstrip(_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    return_value = f"{hard_trimmed}{_ELLIPSIS}"
    return return_value


def _media_kind(media_type: object) -> str:
    if not isinstance(media_type, str) or not media_type:
        return "file"
    prefix = media_type.split("/", 1)[0].casefold()
    if prefix in {"image", "audio", "video"}:
        return prefix
    return "file"


def _multimedia_image_descriptions(
    multimedia_input: list[Mapping[str, object]] | None,
) -> list[str]:
    descriptions: list[str] = []
    for item in multimedia_input or []:
        content_type = item.get("content_type")
        if not isinstance(content_type, str) or not content_type.startswith("image/"):
            continue

        description = item.get("description")
        if isinstance(description, str) and description.strip():
            descriptions.append(description.strip())
        else:
            descriptions.append("")
    return descriptions


def _project_attachments(
    *,
    message_envelope: MessageEnvelope,
    multimedia_input: list[Mapping[str, object]] | None,
    description_cap: int,
    attachment_limit: int,
) -> list[PromptAttachmentSummary]:
    summaries: list[PromptAttachmentSummary] = []
    image_descriptions = _multimedia_image_descriptions(multimedia_input)
    image_description_index = 0
    attachments = message_envelope["attachments"][:attachment_limit]
    for attachment in attachments:
        media_kind = _media_kind(attachment.get("media_type"))
        description = ""
        if (
            media_kind == "image"
            and image_description_index < len(image_descriptions)
        ):
            description = image_descriptions[image_description_index]
            image_description_index += 1
        if not description:
            stored_description = attachment.get("description")
            if isinstance(stored_description, str):
                description = stored_description.strip()

        summary_status: Literal["available", "unavailable"] = "unavailable"
        if description:
            summary_status = "available"

        summary: PromptAttachmentSummary = {
            "media_kind": media_kind,
            "description": _trim_description(description, description_cap),
            "summary_status": summary_status,
        }
        summaries.append(summary)
    return summaries


def _project_mentions(message_envelope: MessageEnvelope) -> list[PromptMentionContext]:
    mentions: list[PromptMentionContext] = []
    for mention in message_envelope["mentions"]:
        projected: PromptMentionContext = {}
        for key in (
            "platform_user_id",
            "global_user_id",
            "display_name",
            "entity_kind",
        ):
            value = mention.get(key)
            if isinstance(value, str) and value:
                projected[key] = value  # type: ignore[literal-required]
        if projected:
            mentions.append(projected)
    return mentions


def _project_reply(message_envelope: MessageEnvelope, excerpt_cap: int) -> PromptReplyContext | None:
    reply = message_envelope.get("reply")
    if not isinstance(reply, Mapping) or not reply:
        return None

    projected: PromptReplyContext = {}
    for key in (
        "platform_message_id",
        "platform_user_id",
        "global_user_id",
        "display_name",
        "derivation",
    ):
        value = reply.get(key)
        if isinstance(value, str) and value:
            projected[key] = value  # type: ignore[literal-required]

    excerpt = reply.get("excerpt")
    if isinstance(excerpt, str) and excerpt:
        projected["excerpt"] = _truncate_with_suffix(excerpt.strip(), excerpt_cap)

    if not projected:
        return None
    return projected


def _build_projection(
    *,
    message_envelope: MessageEnvelope,
    multimedia_input: list[Mapping[str, object]] | None,
    body_cap: int,
    reply_cap: int,
    description_cap: int,
    attachment_limit: int,
) -> PromptMessageContext:
    projection: PromptMessageContext = {
        "body_text": _truncate_with_suffix(
            message_envelope["body_text"].strip(),
            body_cap,
        ),
        "addressed_to_global_user_ids": [
            str(user_id).strip()
            for user_id in message_envelope["addressed_to_global_user_ids"]
            if str(user_id).strip()
        ],
        "broadcast": bool(message_envelope["broadcast"]),
        "mentions": _project_mentions(message_envelope),
        "attachments": _project_attachments(
            message_envelope=message_envelope,
            multimedia_input=multimedia_input,
            description_cap=description_cap,
            attachment_limit=attachment_limit,
        ),
    }
    reply = _project_reply(message_envelope, reply_cap)
    if reply is not None:
        projection["reply"] = reply
    return projection


def _reduced_projection(
    *,
    message_envelope: MessageEnvelope,
    multimedia_input: list[Mapping[str, object]] | None,
    body_cap: int,
    reply_cap: int,
    description_cap: int,
    attachment_limit: int,
) -> PromptMessageContext:
    projection = _build_projection(
        message_envelope=message_envelope,
        multimedia_input=multimedia_input,
        body_cap=body_cap,
        reply_cap=reply_cap,
        description_cap=description_cap,
        attachment_limit=attachment_limit,
    )
    if _json_size(projection) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS:
        return projection

    projection = _build_projection(
        message_envelope=message_envelope,
        multimedia_input=multimedia_input,
        body_cap=body_cap,
        reply_cap=reply_cap,
        description_cap=description_cap,
        attachment_limit=min(2, attachment_limit),
    )
    if _json_size(projection) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS:
        return projection

    projection = _build_projection(
        message_envelope=message_envelope,
        multimedia_input=multimedia_input,
        body_cap=body_cap,
        reply_cap=reply_cap,
        description_cap=max(1, description_cap // 2),
        attachment_limit=min(2, attachment_limit),
    )
    if _json_size(projection) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS:
        return projection

    projection = _build_projection(
        message_envelope=message_envelope,
        multimedia_input=multimedia_input,
        body_cap=max(1, body_cap // 2),
        reply_cap=reply_cap,
        description_cap=max(1, description_cap // 2),
        attachment_limit=min(2, attachment_limit),
    )
    if _json_size(projection) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS:
        return projection

    projection = _build_projection(
        message_envelope=message_envelope,
        multimedia_input=multimedia_input,
        body_cap=max(1, body_cap // 2),
        reply_cap=max(1, reply_cap // 2),
        description_cap=max(1, description_cap // 2),
        attachment_limit=min(2, attachment_limit),
    )
    if _json_size(projection) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS:
        return projection

    raise PromptContextTooLargeError(
        "Prompt message context exceeds cap after reducing attachments, "
        "attachment descriptions, body_text, and reply.excerpt"
    )


def project_prompt_message_context(
    *,
    message_envelope: MessageEnvelope,
    multimedia_input: list[Mapping[str, object]] | None = None,
) -> PromptMessageContext:
    """Build a bounded LLM-safe current-message context.

    Args:
        message_envelope: Full storage/audit envelope for the current turn.
        multimedia_input: Optional current-turn media summaries generated by
            the multimedia descriptor node.

    Returns:
        Prompt-safe context that preserves semantic structure and excludes
        binary/wire fields.
    """

    projection = _reduced_projection(
        message_envelope=message_envelope,
        multimedia_input=multimedia_input,
        body_cap=MAX_PROMPT_BODY_TEXT_CHARS,
        reply_cap=MAX_PROMPT_REPLY_EXCERPT_CHARS,
        description_cap=MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS,
        attachment_limit=MAX_PROMPT_ATTACHMENTS,
    )
    assert_prompt_message_context_safe(projection)
    return projection


def _check_mapping_keys(
    *,
    value: Mapping[str, object],
    allowed_keys: set[str],
    path: str,
) -> None:
    for key in value:
        if key not in allowed_keys:
            raise ValueError(f"Unexpected prompt context key at {path}.{key}")


def _reject_nested_mapping_keys(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key in value:
            raise ValueError(f"Unexpected prompt context key at {path}.{key}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nested_mapping_keys(item, f"{path}[{index}]")


def _assert_attachment_safe(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected attachment mapping at {path}")
    _check_mapping_keys(
        value=value,
        allowed_keys=_PROMPT_ATTACHMENT_KEYS,
        path=path,
    )
    for key, item in value.items():
        _reject_nested_mapping_keys(item, f"{path}.{key}")


def _assert_mention_safe(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mention mapping at {path}")
    _check_mapping_keys(value=value, allowed_keys=_PROMPT_MENTION_KEYS, path=path)
    for key, item in value.items():
        _reject_nested_mapping_keys(item, f"{path}.{key}")


def _assert_reply_safe(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected reply mapping at {path}")
    _check_mapping_keys(value=value, allowed_keys=_PROMPT_REPLY_KEYS, path=path)
    for key, item in value.items():
        _reject_nested_mapping_keys(item, f"{path}.{key}")


def assert_prompt_message_context_safe(payload: Mapping[str, object]) -> None:
    """Validate that a prompt context contains only whitelisted keys.

    Args:
        payload: Prompt message context to validate before prompt serialization.

    Returns:
        None.

    Raises:
        ValueError: If any mapping contains a key outside the prompt-safe
            schema.
    """

    _check_mapping_keys(
        value=payload,
        allowed_keys=_PROMPT_MESSAGE_KEYS,
        path="prompt_message_context",
    )

    for key in ("body_text", "addressed_to_global_user_ids", "broadcast"):
        _reject_nested_mapping_keys(payload.get(key), f"prompt_message_context.{key}")

    attachments = payload.get("attachments", [])
    if not isinstance(attachments, list):
        raise ValueError("Expected attachments list at prompt_message_context")
    for index, attachment in enumerate(attachments):
        _assert_attachment_safe(
            attachment,
            f"prompt_message_context.attachments[{index}]",
        )

    mentions = payload.get("mentions", [])
    if not isinstance(mentions, list):
        raise ValueError("Expected mentions list at prompt_message_context")
    for index, mention in enumerate(mentions):
        _assert_mention_safe(mention, f"prompt_message_context.mentions[{index}]")

    reply = payload.get("reply")
    if reply is not None:
        _assert_reply_safe(reply, "prompt_message_context.reply")
