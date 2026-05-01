"""Deterministic tests for prompt-safe current-message projection."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.message_envelope.prompt_projection import (
    MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS,
    MAX_PROMPT_MESSAGE_CONTEXT_CHARS,
    PromptContextTooLargeError,
    assert_prompt_message_context_safe,
    project_prompt_message_context,
)


def _image_envelope(*, base64_data: str = "image-bytes") -> dict:
    """Build a storage envelope fixture with binary media fields.

    Args:
        base64_data: Inline image payload used to prove prompt projection drops
            storage-only bytes.

    Returns:
        Message envelope shaped dictionary.
    """

    return_value = {
        "body_text": "",
        "raw_wire_text": "[CQ:image,url=https://example.test/image.jpg]",
        "mentions": [{
            "platform_user_id": "bot-1",
            "global_user_id": "character-global",
            "display_name": "Character",
            "entity_kind": "bot",
            "raw_text": "[CQ:at,qq=bot-1]",
        }],
        "attachments": [{
            "media_type": "image/jpeg",
            "url": "https://example.test/image.jpg",
            "base64_data": base64_data,
            "description": "",
            "storage_shape": "inline",
        }],
        "addressed_to_global_user_ids": ["character-global"],
        "broadcast": False,
        "reply": {
            "platform_message_id": "reply-1",
            "platform_user_id": "user-2",
            "global_user_id": "global-user-2",
            "display_name": "Other",
            "excerpt": "reply excerpt",
            "derivation": "platform_native",
        },
    }
    return return_value


def test_projection_uses_multimedia_description_and_omits_storage_fields() -> None:
    """Prompt projection should keep summaries while dropping binary/wire data."""

    base64_payload = "a" * (1024 * 1024 + 1)
    projection = project_prompt_message_context(
        message_envelope=_image_envelope(base64_data=base64_payload),
        multimedia_input=[{
            "content_type": "image/jpeg",
            "base64_data": base64_payload,
            "description": "image shows a desk and handwritten notes",
        }],
    )

    rendered = json.dumps(projection, ensure_ascii=False)

    assert projection["attachments"][0]["description"] == (
        "image shows a desk and handwritten notes"
    )
    assert projection["attachments"][0]["summary_status"] == "available"
    assert projection["mentions"][0]["entity_kind"] == "bot"
    assert "raw_text" not in projection["mentions"][0]
    assert "base64_data" not in rendered
    assert "raw_wire_text" not in rendered
    assert "https://example.test/image.jpg" not in rendered
    assert base64_payload not in rendered
    assert len(rendered) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS


def test_projection_maps_image_description_past_non_image_attachment() -> None:
    """Image summaries should not be assigned to preceding file attachments."""

    envelope = _image_envelope()
    envelope["attachments"] = [
        {
            "media_type": "application/pdf",
            "url": "https://example.test/file.pdf",
            "description": "",
            "storage_shape": "url_only",
        },
        {
            "media_type": "image/jpeg",
            "base64_data": "image-bytes",
            "description": "",
            "storage_shape": "inline",
        },
    ]

    projection = project_prompt_message_context(
        message_envelope=envelope,
        multimedia_input=[{
            "content_type": "image/jpeg",
            "base64_data": "image-bytes",
            "description": "image shows a whiteboard diagram",
        }],
    )

    assert projection["attachments"][0] == {
        "media_kind": "file",
        "description": "",
        "summary_status": "unavailable",
    }
    assert projection["attachments"][1] == {
        "media_kind": "image",
        "description": "image shows a whiteboard diagram",
        "summary_status": "available",
    }


def test_projection_treats_text_media_type_as_file() -> None:
    """The prompt media-kind contract keeps text/* under generic file."""

    envelope = _image_envelope()
    envelope["attachments"] = [{
        "media_type": "text/plain",
        "description": "plain text upload summary",
        "storage_shape": "url_only",
    }]

    projection = project_prompt_message_context(message_envelope=envelope)

    assert projection["attachments"][0] == {
        "media_kind": "file",
        "description": "plain text upload summary",
        "summary_status": "available",
    }


def test_projection_trimmed_description_does_not_stack_period_ellipsis() -> None:
    """Description trimming should avoid sentence-period plus ellipsis stacking."""

    description = (
        "x" * (MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS - 10)
        + ". "
        + "y" * 200
    )
    projection = project_prompt_message_context(
        message_envelope=_image_envelope(),
        multimedia_input=[{
            "content_type": "image/jpeg",
            "base64_data": "image-bytes",
            "description": description,
        }],
    )

    trimmed = projection["attachments"][0]["description"]

    assert trimmed.endswith("...")
    assert not trimmed.endswith("....")


def test_large_inline_image_prompt_payloads_remain_bounded() -> None:
    """Decontextualizer and RAG payload shapes should not carry image bytes."""

    base64_payload = "a" * (1024 * 1024 + 1)
    description = "image shows a desk and handwritten notes"
    projection = project_prompt_message_context(
        message_envelope=_image_envelope(base64_data=base64_payload),
        multimedia_input=[{
            "content_type": "image/jpeg",
            "base64_data": base64_payload,
            "description": description,
        }],
    )
    decontextualizer_payload = {
        "user_input": "",
        "platform_user_id": "user-1",
        "user_name": "User",
        "platform_bot_id": "bot-1",
        "prompt_message_context": projection,
        "chat_history": [],
        "channel_topic": "",
        "indirect_speech_context": "",
        "reply_context": {},
    }
    rag_initializer_payload = {
        "original_query": "",
        "context": {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "prompt_message_context": projection,
        },
    }

    projection_rendered = json.dumps(projection, ensure_ascii=False)
    decontextualizer_rendered = json.dumps(
        decontextualizer_payload,
        ensure_ascii=False,
    )
    rag_initializer_rendered = json.dumps(
        rag_initializer_payload,
        ensure_ascii=False,
    )

    assert len(projection_rendered) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS
    for rendered in (decontextualizer_rendered, rag_initializer_rendered):
        assert description in rendered
        assert "base64_data" not in rendered
        assert "raw_wire_text" not in rendered
        assert base64_payload not in rendered


def test_projection_marks_missing_legacy_description_unavailable() -> None:
    """Legacy media rows without descriptions should stay prompt-safe."""

    projection = project_prompt_message_context(
        message_envelope=_image_envelope(),
        multimedia_input=None,
    )

    rendered = json.dumps(projection, ensure_ascii=False)

    assert projection["attachments"][0]["summary_status"] == "unavailable"
    assert projection["attachments"][0]["description"] == ""
    assert "base64_data" not in rendered
    assert len(rendered) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS


def test_descriptor_crash_saved_row_projects_unavailable() -> None:
    """Rows saved before description persistence should remain prompt-safe."""

    saved_row = _image_envelope(base64_data="b" * (1024 * 1024 + 1))

    assert saved_row["attachments"][0]["description"] == ""

    projection = project_prompt_message_context(
        message_envelope=saved_row,
        multimedia_input=None,
    )
    rendered = json.dumps(projection, ensure_ascii=False)

    assert projection["attachments"][0]["summary_status"] == "unavailable"
    assert projection["attachments"][0]["description"] == ""
    assert "base64_data" not in rendered
    assert saved_row["attachments"][0]["base64_data"] not in rendered
    assert len(rendered) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS


def test_projection_applies_overflow_degradation_order() -> None:
    """Oversized projections should drop and trim low-priority fields."""

    envelope = {
        "body_text": "b" * 3000,
        "raw_wire_text": "wire",
        "mentions": [{
            "platform_user_id": "user-1",
            "global_user_id": "global-user-1",
            "display_name": "m" * 2400,
            "entity_kind": "user",
        }],
        "attachments": [
            {
                "media_type": "image/png",
                "base64_data": "payload",
                "description": f"attachment {index}. " + ("d" * 1200),
                "storage_shape": "inline",
            }
            for index in range(4)
        ],
        "addressed_to_global_user_ids": ["character-global"],
        "broadcast": False,
        "reply": {
            "platform_message_id": "reply-1",
            "excerpt": "r" * 1000,
            "derivation": "platform_native",
        },
    }

    projection = project_prompt_message_context(message_envelope=envelope)
    rendered = json.dumps(projection, ensure_ascii=False)

    assert len(projection["attachments"]) == 2
    assert len(projection["body_text"]) <= 1000
    assert len(projection["reply"]["excerpt"]) <= 250
    assert all(
        len(attachment["description"]) <= (
            MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS // 2
            + len("...")
        )
        for attachment in projection["attachments"]
    )
    assert len(rendered) <= MAX_PROMPT_MESSAGE_CONTEXT_CHARS


def test_projection_raises_when_required_metadata_cannot_fit() -> None:
    """Oversized load-bearing addressing metadata should fail instead of leak."""

    envelope = {
        "body_text": "body",
        "raw_wire_text": "wire",
        "mentions": [
            {
                "global_user_id": f"user-{index}-" + ("x" * 200),
                "entity_kind": "user",
            }
            for index in range(40)
        ],
        "attachments": [],
        "addressed_to_global_user_ids": [
            f"user-{index}-" + ("y" * 200)
            for index in range(40)
        ],
        "broadcast": False,
    }

    with pytest.raises(PromptContextTooLargeError, match="body_text.*reply.excerpt"):
        project_prompt_message_context(message_envelope=envelope)


def test_safety_assertion_rejects_foreign_nested_keys() -> None:
    """Whitelist validation should reject smuggled storage fields."""

    payload = {
        "body_text": "",
        "addressed_to_global_user_ids": [],
        "broadcast": False,
        "mentions": [],
        "attachments": [{
            "media_kind": "image",
            "description": "safe",
            "summary_status": "available",
            "inline_bytes": "not allowed",
        }],
    }

    with pytest.raises(ValueError, match="inline_bytes"):
        assert_prompt_message_context_safe(payload)


def test_safety_assertion_rejects_foreign_keys_inside_leaf_values() -> None:
    """Whitelist validation should reject keys hidden below allowed leaves."""

    payload = {
        "body_text": "",
        "addressed_to_global_user_ids": [],
        "broadcast": False,
        "mentions": [],
        "attachments": [{
            "media_kind": "image",
            "description": {"inline_bytes": "not allowed"},
            "summary_status": "available",
        }],
    }

    with pytest.raises(ValueError, match="inline_bytes"):
        assert_prompt_message_context_safe(payload)
