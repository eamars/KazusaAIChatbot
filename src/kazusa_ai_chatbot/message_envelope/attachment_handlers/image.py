"""Image attachment handling for typed message envelopes.

This module owns image attachment normalization. Inputs are adapter-provided
attachment payloads with optional text descriptions and binary/url references;
outputs are `AttachmentRef` dictionaries. The extension slot is the
`AttachmentHandler` Protocol registered for the `image/` media-type prefix.
"""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.message_envelope.attachment_policy import (
    INLINE_ATTACHMENT_BYTE_LIMIT,
)
from kazusa_ai_chatbot.message_envelope.types import (
    AttachmentRef,
    AttachmentStorageShape,
)


class ImageAttachmentHandler:
    """Normalize image attachments while preserving existing descriptions."""

    def supports(self, media_type: str) -> bool:
        """Return whether this handler accepts the media type.

        Args:
            media_type: MIME type from the adapter payload.

        Returns:
            True when the payload is an image.
        """

        return_value = media_type.startswith("image/")
        return return_value

    def build_ref(self, raw_attachment: object) -> AttachmentRef:
        """Build a normalized image attachment reference.

        Args:
            raw_attachment: Adapter-provided image attachment payload.

        Returns:
            Attachment reference preserving description, media type, URL,
            inline base64 payload, and selected storage shape.
        """

        if isinstance(raw_attachment, Mapping):
            media_type_value = raw_attachment.get("media_type")
            url_value = raw_attachment.get("url")
            base64_value = raw_attachment.get("base64_data")
            description_value = raw_attachment.get("description")
            size_value = raw_attachment.get("size_bytes")
        else:
            media_type_value = getattr(raw_attachment, "media_type", None)
            url_value = getattr(raw_attachment, "url", None)
            base64_value = getattr(raw_attachment, "base64_data", None)
            description_value = getattr(raw_attachment, "description", None)
            size_value = getattr(raw_attachment, "size_bytes", None)

        attachment: AttachmentRef = {}
        if isinstance(media_type_value, str) and media_type_value:
            attachment["media_type"] = media_type_value
        if isinstance(url_value, str) and url_value:
            attachment["url"] = url_value
        if isinstance(base64_value, str) and base64_value:
            attachment["base64_data"] = base64_value
        if isinstance(description_value, str) and description_value:
            attachment["description"] = description_value
        if isinstance(size_value, int):
            attachment["size_bytes"] = size_value

        storage_shape = self.storage_shape(attachment)
        attachment["storage_shape"] = storage_shape
        return attachment

    def storage_shape(self, attachment: AttachmentRef) -> AttachmentStorageShape:
        """Choose how an image attachment should be stored.

        Args:
            attachment: Normalized image attachment reference.

        Returns:
            `inline` for bounded or size-unknown inline payloads, `url_only`
            for large URL-backed images, and `drop` for empty attachment shells.
        """

        has_base64 = bool(attachment.get("base64_data"))
        has_url = bool(attachment.get("url"))
        has_description = bool(attachment.get("description"))
        if not has_base64 and not has_url and not has_description:
            return_value: AttachmentStorageShape = "drop"
            return return_value

        size_bytes = attachment.get("size_bytes")
        if has_base64 and (
            not isinstance(size_bytes, int)
            or size_bytes <= INLINE_ATTACHMENT_BYTE_LIMIT
        ):
            return_value = "inline"
            return return_value

        if has_url:
            return_value = "url_only"
            return return_value

        if has_base64:
            return_value = "inline"
            return return_value

        return_value = "url_only"
        return return_value
