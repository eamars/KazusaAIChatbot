"""QQ/NapCat envelope normalizer for brain-safe typed messages."""

from __future__ import annotations

from adapters.envelope_common import (
    addressed_to_from_envelope_parts,
    attachment_refs,
    normalize_mention_display_map,
    resolve_mentions,
    semantic_entity_fallback_label,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.message_envelope import (
    AttachmentHandlerRegistryProtocol,
    MentionResolver,
)
from kazusa_ai_chatbot.message_envelope.types import (
    MessageEnvelope,
    RawMention,
    ReplyTarget,
)

from .cq_projection import (
    CQ_AT_PATTERN,
    CQ_REPLY_PATTERN,
    project_qq_semantic_text,
    qq_mention_entity_kind,
)


class QQEnvelopeNormalizer:
    """Adapter-local QQ/NapCat normalizer for brain-safe envelopes."""

    def normalize(
        self,
        request: object,
        mention_resolver: MentionResolver,
        attachment_handlers: AttachmentHandlerRegistryProtocol,
    ) -> MessageEnvelope:
        """Normalize one QQ request.

        Args:
            request: Request-like object with QQ wire content and metadata.
            mention_resolver: Resolver used to project raw mentions.
            attachment_handlers: Registry used to normalize attachments.

        Returns:
            Message envelope with QQ mention markers represented as readable
            tokens in `body_text`.
        """

        raw_wire_text = str(request.content or "")
        platform_bot_id = str(request.platform_bot_id)
        channel_type = str(request.channel_type)
        reply_context = dict(request.reply_context)
        mention_display_names = normalize_mention_display_map(
            getattr(request, "mention_display_names", {}),
        )

        raw_mentions = self._raw_mentions(
            raw_wire_text,
            platform_bot_id,
            mention_display_names,
        )
        reply = self._reply_target(
            raw_wire_text,
            reply_context,
            platform_bot_id,
            mention_display_names,
        )
        body_text = project_qq_semantic_text(
            raw_wire_text,
            platform_bot_id,
            mention_display_names,
        )

        mentions = resolve_mentions(raw_mentions, mention_resolver)
        addressed_to = addressed_to_from_envelope_parts(
            mentions=mentions,
            reply=reply,
            channel_type=channel_type,
        )
        envelope: MessageEnvelope = {
            "body_text": body_text,
            "raw_wire_text": raw_wire_text,
            "mentions": mentions,
            "attachments": attachment_refs(request.attachments, attachment_handlers),
            "addressed_to_global_user_ids": addressed_to,
            "broadcast": False,
        }
        if reply:
            envelope["reply"] = reply

        return envelope

    def _raw_mentions(
        self,
        raw_wire_text: str,
        platform_bot_id: str,
        display_names: dict[str, str],
    ) -> list[RawMention]:
        """Extract QQ wire mention markers for bot/user addressing."""

        raw_mentions: list[RawMention] = []
        for match in CQ_AT_PATTERN.finditer(raw_wire_text):
            platform_user_id = match.group(1)
            entity_kind = qq_mention_entity_kind(
                platform_user_id,
                platform_bot_id,
            )
            raw_mentions.append({
                "platform": "qq",
                "platform_user_id": platform_user_id,
                "entity_kind": entity_kind,
                "raw_text": match.group(0),
                "display_name": (
                    display_names.get(platform_user_id, "")
                    or semantic_entity_fallback_label(
                        entity_kind=entity_kind,
                        mention_context=False,
                    )
                ),
            })

        return raw_mentions

    def _reply_target(
        self,
        raw_wire_text: str,
        reply_context: dict,
        platform_bot_id: str,
        display_names: dict[str, str],
    ) -> ReplyTarget:
        """Extract the typed reply target from CQ text and adapter metadata."""

        reply: ReplyTarget = {}
        reply_match = CQ_REPLY_PATTERN.search(raw_wire_text)
        if reply_match is not None:
            reply["platform_message_id"] = reply_match.group(1)
            reply["derivation"] = "platform_native"

        if reply_context.get("reply_to_message_id"):
            reply["platform_message_id"] = str(reply_context["reply_to_message_id"])
            reply["derivation"] = "platform_native"
        if reply_context.get("reply_to_platform_user_id"):
            platform_user_id = str(reply_context["reply_to_platform_user_id"])
            reply["platform_user_id"] = platform_user_id
            if platform_user_id == platform_bot_id:
                reply["global_user_id"] = CHARACTER_GLOBAL_USER_ID
            display_name = str(reply_context.get("reply_to_display_name") or "")
            if not display_name:
                display_name = semantic_entity_fallback_label(
                    entity_kind="user",
                    mention_context=False,
                )
            if display_name:
                reply["display_name"] = display_name
        elif reply_context.get("reply_to_display_name"):
            reply["display_name"] = str(reply_context["reply_to_display_name"])
        if reply_context.get("reply_excerpt"):
            reply_excerpt = project_qq_semantic_text(
                str(reply_context["reply_excerpt"]),
                platform_bot_id,
                display_names,
            )
            if reply_excerpt:
                reply["excerpt"] = reply_excerpt
        return reply
