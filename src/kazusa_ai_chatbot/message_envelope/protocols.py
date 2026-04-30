"""Protocol interfaces for message-envelope extension points.

This module owns the abstract adapter and attachment contracts. Inputs are
platform request objects, raw mention fragments, and raw attachment payloads;
outputs are `MessageEnvelope` and related TypedDicts. New platform behavior is
added by implementing these Protocols and registering the implementation through
the public registries.
"""

from __future__ import annotations

from typing import Protocol

from kazusa_ai_chatbot.message_envelope.types import (
    AttachmentRef,
    AttachmentStorageShape,
    MessageEnvelope,
    RawMention,
    ResolvedMention,
)


class MentionResolver(Protocol):
    """Resolve platform mention fragments into internal user identities."""

    def resolve(self, raw_mention: RawMention) -> ResolvedMention:
        """Resolve one raw mention.

        Args:
            raw_mention: Platform mention fragment from an envelope normalizer.

        Returns:
            Resolved mention data, including `global_user_id` when known.
        """

        ...


class AttachmentHandler(Protocol):
    """Convert one platform attachment payload into a stored attachment ref."""

    def supports(self, media_type: str) -> bool:
        """Return whether this handler accepts the media type.

        Args:
            media_type: MIME type or platform-specific media type string.

        Returns:
            True when this handler can normalize the attachment.
        """

        ...

    def build_ref(self, raw_attachment: object) -> AttachmentRef:
        """Build the stored attachment reference.

        Args:
            raw_attachment: Adapter-provided attachment object.

        Returns:
            Attachment reference preserving description and storage payload.
        """

        ...

    def storage_shape(self, attachment: AttachmentRef) -> AttachmentStorageShape:
        """Choose inline or URL-only storage for an attachment reference.

        Args:
            attachment: Normalized attachment reference.

        Returns:
            Storage shape selected for persistence.
        """

        ...


class AttachmentHandlerRegistryProtocol(Protocol):
    """Lookup interface for adapter-owned attachment handler registries."""

    def handler_for(self, media_type: str) -> AttachmentHandler | None:
        """Return the handler registered for a media type.

        Args:
            media_type: MIME type or platform-specific media type string.

        Returns:
            Matching attachment handler, or None when the registry has no
            handler for the media type.
        """

        ...


class EnvelopeNormalizer(Protocol):
    """Normalize one platform request into a clean message envelope."""

    def normalize(
        self,
        request: object,
        mention_resolver: MentionResolver,
        attachment_handlers: AttachmentHandlerRegistryProtocol,
    ) -> MessageEnvelope:
        """Normalize platform wire data into brain-safe structure.

        Args:
            request: Adapter/service request object for one inbound message.
            mention_resolver: Resolver for converting mentions to user IDs.
            attachment_handlers: Registry used to resolve attachment handlers.

        Returns:
            Message envelope whose `body_text` has no platform wire syntax.
        """

        ...
