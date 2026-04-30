"""Registries for message-envelope normalizers and attachment handlers.

This module owns lookup tables that connect platform names and media-type
prefixes to Protocol implementations. Inputs are registered implementations;
outputs are Protocol-typed objects consumed by adapters. New platform
normalizers are registered by adapters; shared attachment modalities
extend the system by registering here.
"""

from __future__ import annotations

from kazusa_ai_chatbot.message_envelope.attachment_handlers.image import (
    ImageAttachmentHandler,
)
from kazusa_ai_chatbot.message_envelope.protocols import (
    AttachmentHandler,
    EnvelopeNormalizer,
)


class NormalizerRegistry:
    """Map platform names to envelope normalizer implementations."""

    def __init__(self) -> None:
        """Initialize an empty platform normalizer registry.

        Args:
            None.

        Returns:
            None.
        """

        self._normalizers: dict[str, EnvelopeNormalizer] = {}

    def register(self, platform: str, normalizer: EnvelopeNormalizer) -> None:
        """Register the normalizer for one platform.

        Args:
            platform: Platform key such as `qq` or `discord`.
            normalizer: Protocol implementation for that platform.

        Returns:
            None.
        """

        platform_key = platform.strip().lower()
        self._normalizers[platform_key] = normalizer

    def get(self, platform: str) -> EnvelopeNormalizer:
        """Return the normalizer registered for one platform.

        Args:
            platform: Platform key from the inbound chat request.

        Returns:
            Registered normalizer implementation.
        """

        platform_key = platform.strip().lower()
        normalizer = self._normalizers[platform_key]
        return normalizer


class AttachmentHandlerRegistry:
    """Map media-type prefixes to attachment handler implementations."""

    def __init__(self) -> None:
        """Initialize an empty attachment handler registry.

        Args:
            None.

        Returns:
            None.
        """

        self._handlers: dict[str, AttachmentHandler] = {}

    def register(self, media_type_prefix: str, handler: AttachmentHandler) -> None:
        """Register a handler for a media-type prefix.

        Args:
            media_type_prefix: Prefix such as `image/` or `audio/`.
            handler: Protocol implementation for matching attachments.

        Returns:
            None.
        """

        self._handlers[media_type_prefix] = handler

    def handler_for(self, media_type: str) -> AttachmentHandler | None:
        """Return the best matching handler for a media type.

        Args:
            media_type: MIME type or platform-specific media type string.

        Returns:
            Matching handler, or None when no handler is registered.
        """

        matching_prefixes = [
            prefix for prefix in self._handlers if media_type.startswith(prefix)
        ]
        if not matching_prefixes:
            return_value = None
            return return_value

        matching_prefixes.sort(key=len, reverse=True)
        handler = self._handlers[matching_prefixes[0]]
        return handler


def build_default_attachment_handler_registry() -> AttachmentHandlerRegistry:
    """Build the default attachment handler registry.

    Args:
        None.

    Returns:
        Registry populated with the image attachment handler.
    """

    registry = AttachmentHandlerRegistry()
    registry.register("image/", ImageAttachmentHandler())
    return registry
