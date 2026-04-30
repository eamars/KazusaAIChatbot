"""Deterministic tests for the message-envelope public contract."""

from __future__ import annotations

import typing
from pathlib import Path

from kazusa_ai_chatbot.db.schemas import ConversationMessageDoc
from kazusa_ai_chatbot.message_envelope import (
    AttachmentHandlerRegistry,
    AttachmentRef,
    MessageEnvelope,
    NormalizerRegistry,
    RawMention,
    ResolvedMention,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.state import IMProcessState


class DummyMentionResolver:
    """Resolver used by the normalizer protocol contract test."""

    def resolve(self, raw_mention: RawMention) -> ResolvedMention:
        """Resolve a raw mention into a stable fixture identity.

        Args:
            raw_mention: Raw mention emitted by the dummy normalizer.

        Returns:
            Resolved mention fixture.
        """

        return_value: ResolvedMention = {
            "platform_user_id": raw_mention["platform_user_id"],
            "global_user_id": "user-1",
            "entity_kind": "user",
        }
        return return_value


class DummyAttachmentHandler:
    """Attachment handler used by registry tests."""

    def supports(self, media_type: str) -> bool:
        """Return whether the media type is accepted.

        Args:
            media_type: MIME type to inspect.

        Returns:
            True for image media types.
        """

        return_value = media_type.startswith("image/")
        return return_value

    def build_ref(self, raw_attachment: object) -> AttachmentRef:
        """Build a fixture attachment reference.

        Args:
            raw_attachment: Unused fixture input.

        Returns:
            Attachment reference preserving description text.
        """

        return_value: AttachmentRef = {
            "media_type": "image/png",
            "description": "portrait",
            "storage_shape": "inline",
        }
        return return_value

    def storage_shape(self, attachment: AttachmentRef) -> str:
        """Return the fixture storage shape.

        Args:
            attachment: Attachment reference to inspect.

        Returns:
            The storage shape from the attachment.
        """

        return_value = attachment["storage_shape"]
        return return_value


class DummyEnvelopeNormalizer:
    """Normalizer used by the protocol contract test."""

    def normalize(
        self,
        request: object,
        mention_resolver: DummyMentionResolver,
        attachment_handlers: AttachmentHandlerRegistry,
    ) -> MessageEnvelope:
        """Build a minimal envelope without platform-specific parsing.

        Args:
            request: Fixture request object.
            mention_resolver: Fixture mention resolver.
            attachment_handlers: Registry passed into the normalizer.

        Returns:
            Message envelope fixture.
        """

        raw_mention: RawMention = {
            "platform_user_id": "platform-user-1",
            "entity_kind": "user",
        }
        resolved = mention_resolver.resolve(raw_mention)
        handler = attachment_handlers.handler_for("image/png")
        attachments: list[AttachmentRef] = []
        if handler is not None:
            attachments.append(handler.build_ref({}))

        return_value: MessageEnvelope = {
            "body_text": "hello",
            "raw_wire_text": "<@bot> hello",
            "mentions": [resolved],
            "attachments": attachments,
            "addressed_to_global_user_ids": ["bot-global"],
            "broadcast": False,
        }
        return return_value


def test_message_envelope_shape_has_contract_fields() -> None:
    """MessageEnvelope should expose the adapter-to-brain contract fields."""

    hints = typing.get_type_hints(MessageEnvelope)

    for field in [
        "body_text",
        "raw_wire_text",
        "mentions",
        "reply",
        "attachments",
        "addressed_to_global_user_ids",
        "broadcast",
    ]:
        assert field in hints


def test_conversation_message_doc_has_additive_envelope_fields() -> None:
    """Conversation rows should accept typed envelope storage fields."""

    hints = typing.get_type_hints(ConversationMessageDoc)

    for field in [
        "body_text",
        "raw_wire_text",
        "addressed_to_global_user_ids",
        "mentions",
        "broadcast",
    ]:
        assert field in hints


def test_bootstrap_declares_typed_envelope_indexes() -> None:
    """Database bootstrap should create typed-addressing and body-text indexes."""

    source = Path("src/kazusa_ai_chatbot/db/bootstrap.py").read_text(encoding="utf-8")

    assert "conv_platform_channel_addressee_ts" in source
    assert '"addressed_to_global_user_ids"' in source
    assert "conv_body_text" in source
    assert '"body_text"' in source


def test_state_shapes_accept_message_envelope_and_addressee_fields() -> None:
    """Graph state contracts should carry envelope and typed addressing fields."""

    im_hints = typing.get_type_hints(IMProcessState)
    global_hints = typing.get_type_hints(GlobalPersonaState)
    cognition_hints = typing.get_type_hints(CognitionState)

    assert "message_envelope" in im_hints
    assert "target_addressed_user_ids" in im_hints
    assert "target_broadcast" in im_hints
    assert "message_envelope" in global_hints
    assert "referents" in global_hints
    assert "target_addressed_user_ids" in global_hints
    assert "target_broadcast" in global_hints
    assert "message_envelope" in cognition_hints
    assert "referents" in cognition_hints
    assert "target_addressed_user_ids" in cognition_hints
    assert "target_broadcast" in cognition_hints


def test_registries_return_protocol_typed_implementations() -> None:
    """Registries should route by platform and media-type prefix."""

    normalizer = DummyEnvelopeNormalizer()
    normalizers = NormalizerRegistry()
    normalizers.register("QQ", normalizer)

    attachment_handler = DummyAttachmentHandler()
    handlers = AttachmentHandlerRegistry()
    handlers.register("image/", attachment_handler)

    assert normalizers.get("qq") is normalizer
    assert handlers.handler_for("image/png") is attachment_handler
    assert handlers.handler_for("audio/ogg") is None


def test_envelope_normalizer_protocol_builds_message_contract() -> None:
    """Normalizer implementations should produce the envelope contract."""

    handlers = AttachmentHandlerRegistry()
    handlers.register("image/", DummyAttachmentHandler())

    envelope = DummyEnvelopeNormalizer().normalize(
        object(),
        DummyMentionResolver(),
        handlers,
    )

    assert envelope["body_text"] == "hello"
    assert envelope["raw_wire_text"] == "<@bot> hello"
    assert envelope["mentions"][0]["global_user_id"] == "user-1"
    assert envelope["attachments"][0]["description"] == "portrait"
