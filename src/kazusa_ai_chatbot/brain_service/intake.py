"""Inbound chat request hydration helpers for the brain service."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from kazusa_ai_chatbot.chat_input_queue import QueuedChatItem
from kazusa_ai_chatbot.message_envelope import MessageEnvelope
from kazusa_ai_chatbot.state import ReplyContext

from .contracts import ChatRequest


ResolveGlobalUserId = Callable[..., Awaitable[str]]
GetUserProfile = Callable[[str], Awaitable[dict]]
SaveConversation = Callable[[dict], Awaitable[None]]
ResolveEnvelope = Callable[[ChatRequest], Awaitable[MessageEnvelope]]


def compact_reply_context(reply_context: ReplyContext) -> ReplyContext:
    """Remove empty reply-context fields before storing or graph input.

    Args:
        reply_context: Reply metadata projected from a typed envelope.

    Returns:
        Reply metadata without empty string or null values.
    """

    compacted: ReplyContext = {}
    for key, value in reply_context.items():
        if value in ("", None):
            continue
        compacted[key] = value
    return compacted


async def hydrate_reply_context(req: ChatRequest) -> ReplyContext:
    """Build service-facing reply context from the typed envelope only.

    Args:
        req: Incoming chat request from an adapter.

    Returns:
        Compact reply context projected from ``message_envelope.reply``.
    """

    envelope: MessageEnvelope = req.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    reply = envelope.get("reply") or {}
    reply_context: ReplyContext = {}

    if reply.get("platform_message_id"):
        reply_context["reply_to_message_id"] = str(reply["platform_message_id"])
    if reply.get("platform_user_id"):
        reply_context["reply_to_platform_user_id"] = str(reply["platform_user_id"])
    if reply.get("display_name"):
        reply_context["reply_to_display_name"] = str(reply["display_name"])
    if reply.get("excerpt"):
        reply_context["reply_excerpt"] = str(reply["excerpt"])

    compacted_context = compact_reply_context(reply_context)
    return compacted_context


async def resolve_message_envelope_identities(
    req: ChatRequest,
    *,
    character_global_user_id: str,
    resolve_global_user_id_func: ResolveGlobalUserId,
) -> MessageEnvelope:
    """Resolve typed envelope mentions and reply targets to global user ids.

    Args:
        req: Incoming chat request carrying an adapter-normalized envelope.
        character_global_user_id: Stable global id of the active character.
        resolve_global_user_id_func: Profile resolver supplied by the service.

    Returns:
        Message envelope with profile identities resolved and addressees
        recomputed from typed mentions, typed reply target, and DM defaults.
    """

    envelope: MessageEnvelope = req.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    addressed_to: list[str] = []
    if req.channel_type == "private":
        addressed_to.append(character_global_user_id)

    resolved_mentions = []
    for mention in envelope["mentions"]:
        resolved_mention = dict(mention)
        mention_entity_kind = str(
            resolved_mention.get("entity_kind", "unknown")
        ).strip()
        platform_user_id = str(
            resolved_mention.get("platform_user_id", "")
        ).strip()
        global_user_id = str(resolved_mention.get("global_user_id", "")).strip()

        if mention_entity_kind == "bot":
            global_user_id = character_global_user_id
        elif (
            mention_entity_kind == "user"
            and platform_user_id
            and not global_user_id
        ):
            display_name = str(resolved_mention.get("display_name", "")).strip()
            global_user_id = await resolve_global_user_id_func(
                platform=req.platform,
                platform_user_id=platform_user_id,
                display_name=display_name,
            )

        if global_user_id:
            resolved_mention["global_user_id"] = global_user_id
        if mention_entity_kind in ("bot", "user") and global_user_id:
            addressed_to.append(global_user_id)
        resolved_mentions.append(resolved_mention)
    envelope["mentions"] = resolved_mentions

    reply = envelope.get("reply")
    if reply is not None:
        resolved_reply = dict(reply)
        platform_user_id = str(resolved_reply.get("platform_user_id", "")).strip()
        global_user_id = str(resolved_reply.get("global_user_id", "")).strip()

        if platform_user_id and platform_user_id == req.platform_bot_id.strip():
            global_user_id = character_global_user_id
        elif platform_user_id and not global_user_id:
            display_name = str(resolved_reply.get("display_name", "")).strip()
            global_user_id = await resolve_global_user_id_func(
                platform=req.platform,
                platform_user_id=platform_user_id,
                display_name=display_name,
            )

        if global_user_id:
            resolved_reply["global_user_id"] = global_user_id
            addressed_to.append(global_user_id)
        envelope["reply"] = resolved_reply

    envelope["addressed_to_global_user_ids"] = list(dict.fromkeys(addressed_to))
    envelope["broadcast"] = False
    return envelope


def active_turn_platform_message_ids(item: QueuedChatItem) -> list[str]:
    """Build the platform message ID list answered by one graph turn.

    Args:
        item: Surviving queued item that will run through the chat graph.

    Returns:
        Non-empty platform message IDs from the survivor and collapsed
        follow-ups, deduplicated in arrival order.
    """

    seen: set[str] = set()
    active_ids: list[str] = []
    for queued_item in [item, *item.collapsed_items]:
        message_id = str(queued_item.request.platform_message_id or "").strip()
        if not message_id or message_id in seen:
            continue
        seen.add(message_id)
        active_ids.append(message_id)

    return active_ids


async def save_user_message_from_item(
    item: QueuedChatItem,
    *,
    global_user_id: str,
    reply_context: ReplyContext,
    save_conversation_func: SaveConversation,
    resolve_message_envelope_identities_func: ResolveEnvelope,
    message_envelope: MessageEnvelope | None = None,
    logger: logging.Logger,
) -> None:
    """Persist one queued user message.

    Args:
        item: Queued chat item containing the request and timestamp.
        global_user_id: Resolved global user identifier.
        reply_context: Adapter-supplied reply metadata after compacting.
        save_conversation_func: Service-level persistence function.
        resolve_message_envelope_identities_func: Envelope identity resolver.
        message_envelope: Already-resolved envelope, when available.
        logger: Logger used for compatibility with service logging.

    Returns:
        None.
    """

    req = item.request
    if message_envelope is None:
        message_envelope = await resolve_message_envelope_identities_func(req)
    attachment_docs = list(message_envelope["attachments"])

    try:
        await save_conversation_func({
            "platform": req.platform,
            "platform_channel_id": req.platform_channel_id,
            "role": "user",
            "platform_message_id": req.platform_message_id,
            "platform_user_id": req.platform_user_id,
            "global_user_id": global_user_id,
            "display_name": req.display_name,
            "channel_type": req.channel_type,
            "body_text": message_envelope["body_text"],
            "raw_wire_text": message_envelope["raw_wire_text"],
            "content_type": req.content_type,
            "addressed_to_global_user_ids": message_envelope[
                "addressed_to_global_user_ids"
            ],
            "mentions": message_envelope["mentions"],
            "broadcast": False,
            "attachments": attachment_docs,
            "reply_context": reply_context,
            "timestamp": item.timestamp,
        })
    except Exception as exc:
        logger.exception(f"Failed to save queued user message: {exc}")


async def resolve_queued_user(
    item: QueuedChatItem,
    *,
    resolve_global_user_id_func: ResolveGlobalUserId,
    get_user_profile_func: GetUserProfile,
) -> tuple[str, dict]:
    """Resolve the user identity and profile for a queued item.

    Args:
        item: Queued chat item.
        resolve_global_user_id_func: Profile identity resolver.
        get_user_profile_func: User profile loader.

    Returns:
        Pair of global user ID and user profile.
    """

    req = item.request
    global_user_id = await resolve_global_user_id_func(
        platform=req.platform,
        platform_user_id=req.platform_user_id,
        display_name=req.display_name,
    )
    user_profile = await get_user_profile_func(global_user_id)
    return_value = (global_user_id, user_profile)
    return return_value

