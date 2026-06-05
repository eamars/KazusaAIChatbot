"""Deterministic tests for adapter-owned envelope normalizers."""

from __future__ import annotations

from types import SimpleNamespace

from adapters import napcat_qq_adapter as napcat_module
from adapters.discord_adapter import DiscordEnvelopeNormalizer
from adapters.napcat_qq_adapter import QQEnvelopeNormalizer
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.message_envelope import (
    PassthroughMentionResolver,
    build_default_attachment_handler_registry,
)


def test_qq_normalizer_rewrites_cq_mentions_as_readable_tokens() -> None:
    """QQ normalizer should keep CQ syntax out while preserving mention labels."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content=(
            "[CQ:reply,id=1733223276]"
            "[CQ:at,qq=3768713357] what are these "
            "[CQ:at,qq=673225019][CQ:face,id=1]"
        ),
        platform_bot_id="3768713357",
        mention_display_names={
            "3768713357": "Kazusa",
            "673225019": "Other User",
        },
        reply_context={
            "reply_to_message_id": "1733223276",
            "reply_to_platform_user_id": "3768713357",
            "reply_excerpt": "previous bot message",
        },
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == (
        '@Kazusa what are these @Other User <image>表情</image>'
    )
    assert "<@3768713357>" not in envelope["body_text"]
    assert "[CQ:" not in envelope["body_text"]
    assert envelope["mentions"][0]["entity_kind"] == "bot"
    assert envelope["mentions"][0]["global_user_id"] == CHARACTER_GLOBAL_USER_ID
    assert envelope["mentions"][0]["display_name"] == "Kazusa"
    assert envelope["mentions"][1]["display_name"] == "Other User"
    assert envelope["reply"]["global_user_id"] == CHARACTER_GLOBAL_USER_ID
    assert envelope["reply"]["platform_message_id"] == "1733223276"
    assert envelope["addressed_to_global_user_ids"] == [CHARACTER_GLOBAL_USER_ID]


def test_qq_normalizer_uses_occurrence_label_without_display_name() -> None:
    """QQ fallback labels should not leak platform IDs into body text."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:at,qq=673225019] hello",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == "@mentioned-user-1 hello"
    assert "673225019" not in envelope["body_text"]
    assert "[CQ:" not in envelope["body_text"]
    assert envelope["mentions"][0]["platform_user_id"] == "673225019"
    assert envelope["mentions"][0]["display_name"] == ""


def test_qq_normalizer_drops_image_only_reply_excerpt_cq() -> None:
    """QQ image wire syntax should not cross as semantic reply text."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:reply,id=1733223276]what is this?",
        platform_bot_id="3768713357",
        reply_context={
            "reply_to_message_id": "1733223276",
            "reply_excerpt": "[CQ:image,file=sam.png]",
        },
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert "excerpt" not in envelope["reply"]
    assert envelope["reply"]["platform_message_id"] == "1733223276"


def test_qq_normalizer_sanitizes_mixed_reply_excerpt_cq() -> None:
    """QQ reply excerpts should keep text while removing transport syntax."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:reply,id=1733223276]what is this?",
        platform_bot_id="3768713357",
        reply_context={
            "reply_to_message_id": "1733223276",
            "reply_excerpt": "look[CQ:image,file=sam.png]nice",
        },
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["reply"]["excerpt"] == "look nice"
    assert "[CQ:" not in envelope["reply"]["excerpt"]


def test_qq_normalizer_projects_known_face_as_inline_image_block() -> None:
    """QQ face-only messages should preserve the visible expression."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:face,id=344]",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == '<image>大怨种表情</image>'
    assert "[CQ:" not in envelope["body_text"]
    assert "344" not in envelope["body_text"]
    assert envelope["raw_wire_text"] == "[CQ:face,id=344]"


def test_qq_normalizer_preserves_inline_face_position() -> None:
    """QQ faces may appear between authored text fragments."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content='我[CQ:face,id=344]服了',
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == '我 <image>大怨种表情</image> 服了'


def test_qq_normalizer_projects_unknown_face_as_generic_expression() -> None:
    """Unmapped QQ faces should stay visible without invented meaning."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:face,id=999999]",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == '<image>表情</image>'


def test_qq_normalizer_projects_face_without_id_as_generic_expression() -> None:
    """Closed face segments without usable IDs should not disappear."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:face]",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == '<image>表情</image>'


def test_qq_normalizer_reads_face_id_from_any_parameter_position() -> None:
    """QQ face projection should not depend on CQ parameter order."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:face,foo=bar,id=344]",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == '<image>大怨种表情</image>'


def test_qq_normalizer_preserves_multiple_adjacent_faces() -> None:
    """Adjacent QQ faces should remain ordered and separated."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:face,id=344][CQ:face,id=999999]",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == (
        '<image>大怨种表情</image> <image>表情</image>'
    )


def test_qq_normalizer_escapes_face_description_boundaries(monkeypatch) -> None:
    """Face mapping descriptions should not break image boundaries."""

    monkeypatch.setitem(
        napcat_module._QQ_FACE_IMAGE_DESCRIPTIONS,
        "888",
        "A < B & C",
    )

    projected = napcat_module.project_qq_semantic_text(
        "[CQ:face,id=888]",
        "3768713357",
        {},
    )

    assert projected == "<image>A &lt; B &amp; C</image>"


def test_qq_normalizer_projects_faces_in_reply_excerpt() -> None:
    """QQ face reply excerpts should keep the visible expression."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="[CQ:reply,id=1733223276]what is this?",
        platform_bot_id="3768713357",
        reply_context={
            "reply_to_message_id": "1733223276",
            "reply_excerpt": "[CQ:face,id=344]",
        },
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["reply"]["excerpt"] == '<image>大怨种表情</image>'


def test_qq_semantic_text_projection_is_adapter_owned() -> None:
    """QQ transport stripping should be available without private method calls."""

    project_text = getattr(napcat_module, "project_qq_semantic_text", None)

    assert callable(project_text)
    assert project_text(
        "[CQ:reply,id=1733223276]look[CQ:image,file=sam.png]nice",
        "3768713357",
        {},
    ) == "look nice"


def test_discord_normalizer_rewrites_tags_as_readable_tokens() -> None:
    """Discord normalizer should keep tags out while preserving mention labels."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="discord",
        channel_type="group",
        content=(
            "<@12345> hello <@&777> <#888> "
            "<:sparkle:999> @everyone @here"
        ),
        platform_bot_id="12345",
        user_mention_display_names={"12345": "Character"},
        role_mention_display_names={"777": "Regulars"},
        channel_mention_display_names={"888": "general"},
        reply_context={},
        attachments=[],
    )

    envelope = DiscordEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == "@Character hello @Regulars #general @everyone @here"
    assert "<@12345>" not in envelope["body_text"]
    assert "<#888>" not in envelope["body_text"]
    assert envelope["mentions"][0]["entity_kind"] == "bot"
    assert envelope["mentions"][0]["display_name"] == "Character"
    assert {mention["entity_kind"] for mention in envelope["mentions"]} == {
        "bot",
        "platform_role",
        "channel",
        "everyone",
    }


def test_group_message_without_target_is_not_inbound_broadcast() -> None:
    """Inbound user envelopes should not set assistant broadcast semantics."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content="plain group chatter",
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["addressed_to_global_user_ids"] == []
    assert envelope["broadcast"] is False
