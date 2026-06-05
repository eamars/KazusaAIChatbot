"""Deterministic tests for adapter-owned envelope normalizers."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from adapters import napcat_qq_adapter as napcat_module
from adapters.discord_adapter import DiscordEnvelopeNormalizer
from adapters.napcat_qq_adapter import QQEnvelopeNormalizer
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.message_envelope import (
    PassthroughMentionResolver,
    build_default_attachment_handler_registry,
)


_ROOT = Path(__file__).resolve().parents[1]
_NAPCAT_FACE_SOURCE_SNAPSHOT = (
    _ROOT / "tests" / "fixtures" / "napcat_qq_face_source_snapshot.json"
)
_REQUIRED_ADAPTER_ICD_SECTIONS = (
    "Adapter Responsibility Boundary",
    "Required Adapter Lifecycle",
    "Optional Runtime Send Interface",
    "Message Envelope Contract",
    "Runtime Registration Contract",
    "Forbidden Adapter Behavior",
    "Testing Expectations",
)
_REQUIRED_NAPCAT_ICD_SECTIONS = (
    "Public Imports And CLI",
    "Submodule Responsibility Table",
    "Inbound QQ Segment Flow",
    "CQ Projection And Face Catalog Contract",
    "Runtime API Binding Contract",
    "Unknown Face Omission Contract",
    "Source Snapshot And Label Maintenance",
    "Verification Commands",
)


def _face_catalog_module():
    catalog_module = importlib.import_module(
        "adapters.napcat_qq_adapter.face_catalog",
    )
    return catalog_module


def test_napcat_adapter_package_exposes_public_surface_only() -> None:
    """NapCat adapter should be a package with stable public imports."""

    assert hasattr(napcat_module, "__path__")
    assert callable(napcat_module.NapCatWSAdapter)
    assert callable(napcat_module.QQEnvelopeNormalizer)
    assert callable(napcat_module.project_qq_semantic_text)
    assert hasattr(napcat_module, "runtime_app")
    assert callable(napcat_module.main)
    private_catalog_name = "_" + "QQ_FACE_IMAGE_DESCRIPTIONS"
    assert not hasattr(napcat_module, private_catalog_name)
    assert not hasattr(napcat_module, "_MENTION_DISPLAY_CACHE_LIMIT")
    assert not hasattr(napcat_module, "asyncio")


def test_adapter_icds_define_required_boundaries() -> None:
    """Adapter ICDs should name the contracts needed for future adapter work."""

    adapter_readme = (_ROOT / "src" / "adapters" / "README.md").read_text(
        encoding="utf-8",
    )
    napcat_readme = (
        _ROOT / "src" / "adapters" / "napcat_qq_adapter" / "README.md"
    ).read_text(encoding="utf-8")

    for section in _REQUIRED_ADAPTER_ICD_SECTIONS:
        assert f"## {section}" in adapter_readme
    for section in _REQUIRED_NAPCAT_ICD_SECTIONS:
        assert f"## {section}" in napcat_readme


def test_qq_face_catalog_matches_reviewed_source_snapshot() -> None:
    """The production QQ face catalog should cover the reviewed numeric source."""

    catalog_module = _face_catalog_module()
    snapshot = json.loads(
        _NAPCAT_FACE_SOURCE_SNAPSHOT.read_text(encoding="utf-8"),
    )
    source = snapshot["source"]
    assert source == {
        "repository": "https://github.com/koishijs/QFace",
        "commit": "e476a706a7e508849c6031c3654051a02639964f",
        "path": "public/assets/qq_emoji/_index.json",
        "captured_at": "2026-06-05",
        "total_rows": 482,
        "numeric_rows": 317,
        "unicode_emoji_rows": 165,
    }

    faces = snapshot["faces"]
    numeric_rows = [
        face for face in faces
        if face["id"].isascii() and face["id"].isdecimal()
    ]
    expected_labels = {
        face["id"]: face["semantic_label"]
        for face in numeric_rows
    }

    assert len(faces) == 482
    assert len(numeric_rows) == 317
    assert catalog_module.QQ_FACE_IMAGE_DESCRIPTIONS == expected_labels
    assert catalog_module.qq_face_image_description("344") == '大怨种表情'


def test_qq_face_catalog_labels_are_reviewed_and_prompt_safe() -> None:
    """Reviewed catalog labels should be meaningful prompt-facing image text."""

    snapshot = json.loads(
        _NAPCAT_FACE_SOURCE_SNAPSHOT.read_text(encoding="utf-8"),
    )
    placeholder_labels = {
        'QQ表情',
        '表情',
        '未知表情',
        '未命名表情',
    }
    raw_id_label_pattern = re.compile(r"^#?\d+$")

    for face in snapshot["faces"]:
        face_id = face["id"]
        if not face_id.isascii() or not face_id.isdecimal():
            continue
        semantic_label = face["semantic_label"].strip()
        source_describe = face["source_describe"].strip()

        assert face["review_status"] == "reviewed"
        assert semantic_label
        assert semantic_label not in placeholder_labels
        assert raw_id_label_pattern.fullmatch(semantic_label) is None
        assert "[CQ:" not in semantic_label
        assert "<image>" not in semantic_label
        assert "</image>" not in semantic_label
        assert "<" not in semantic_label
        assert ">" not in semantic_label
        assert "&" not in semantic_label
        if not source_describe:
            assert face["label_basis"] == "asset_review"


def test_qq_normalizer_rewrites_cq_mentions_as_readable_tokens() -> None:
    """QQ normalizer should keep CQ syntax out while preserving mention labels."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content=(
            "[CQ:reply,id=1733223276]"
            "[CQ:at,qq=3768713357] what are these "
            "[CQ:at,qq=673225019][CQ:face,id=344]"
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
        '@Kazusa what are these @Other User <image>大怨种表情</image>'
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


def test_qq_normalizer_omits_unknown_face() -> None:
    """Unmapped QQ faces should not invent visible meaning."""

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

    assert envelope["body_text"] == ""


def test_qq_normalizer_omits_face_without_id() -> None:
    """Closed face segments without usable IDs should not invent meaning."""

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

    assert envelope["body_text"] == ""


def test_qq_normalizer_omits_unknown_inline_face() -> None:
    """Unknown QQ faces can appear between adjacent authored text."""

    handlers = build_default_attachment_handler_registry()
    request = SimpleNamespace(
        platform="qq",
        channel_type="group",
        content='我[CQ:face,id=999999]服了',
        platform_bot_id="3768713357",
        reply_context={},
        attachments=[],
    )

    envelope = QQEnvelopeNormalizer().normalize(
        request,
        PassthroughMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == '我服了'


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

    assert envelope["body_text"] == '<image>大怨种表情</image>'


def test_qq_normalizer_escapes_face_description_boundaries(monkeypatch) -> None:
    """Face mapping descriptions should not break image boundaries."""

    monkeypatch.setitem(
        _face_catalog_module().QQ_FACE_IMAGE_DESCRIPTIONS,
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
