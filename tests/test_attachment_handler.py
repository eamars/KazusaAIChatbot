"""Deterministic tests for envelope attachment handlers."""

from __future__ import annotations

from kazusa_ai_chatbot.message_envelope.attachment_handlers.image import (
    INLINE_ATTACHMENT_BYTE_LIMIT,
    ImageAttachmentHandler,
)


def test_image_attachment_handler_preserves_description_bytes() -> None:
    """Image handler should preserve the adapter-provided description exactly."""

    handler = ImageAttachmentHandler()
    description = "portrait description: cat ears and warm light"

    attachment = handler.build_ref({
        "media_type": "image/png",
        "url": "https://example.test/image.png",
        "base64_data": "abc123",
        "description": description,
        "size_bytes": 123,
    })

    assert attachment["description"] == description
    assert attachment["description"].encode("utf-8") == description.encode("utf-8")


def test_image_attachment_handler_marks_small_inline_payloads() -> None:
    """Small inline images should keep their inline storage shape."""

    handler = ImageAttachmentHandler()

    attachment = handler.build_ref({
        "media_type": "image/jpeg",
        "url": "https://example.test/small.jpg",
        "base64_data": "small-bytes",
        "description": "small image",
        "size_bytes": INLINE_ATTACHMENT_BYTE_LIMIT,
    })

    assert attachment["storage_shape"] == "inline"
    assert attachment["base64_data"] == "small-bytes"
    assert attachment["media_type"] == "image/jpeg"
    assert attachment["url"] == "https://example.test/small.jpg"


def test_image_attachment_handler_marks_large_url_only_payloads() -> None:
    """Large URL-backed images should choose URL-only storage."""

    handler = ImageAttachmentHandler()

    attachment = handler.build_ref({
        "media_type": "image/jpeg",
        "url": "https://example.test/large.jpg",
        "base64_data": "large-bytes",
        "description": "large image",
        "size_bytes": INLINE_ATTACHMENT_BYTE_LIMIT + 1,
    })

    assert attachment["storage_shape"] == "url_only"
    assert attachment["url"] == "https://example.test/large.jpg"
    assert attachment["description"] == "large image"


def test_image_attachment_handler_drops_empty_shells() -> None:
    """Empty image shells should be marked as droppable."""

    handler = ImageAttachmentHandler()

    attachment = handler.build_ref({"media_type": "image/png"})

    assert attachment["storage_shape"] == "drop"
