"""Public message-envelope contract for adapter-to-brain intake.

This package owns the typed message envelope consumed by the brain graph. Its
inputs are platform adapter request objects and its outputs are dict-shaped
TypedDict payloads suitable for LangGraph state, conversation storage, RAG, and
cognition. Platform-specific normalizers and attachment handlers plug in
through the public Protocols and registries exported here.
"""

from __future__ import annotations

from kazusa_ai_chatbot.message_envelope.attachment_policy import (
    INLINE_ATTACHMENT_BYTE_LIMIT,
)
from kazusa_ai_chatbot.message_envelope.protocols import (
    AttachmentHandler,
    AttachmentHandlerRegistryProtocol,
    EnvelopeNormalizer,
    MentionResolver,
)
from kazusa_ai_chatbot.message_envelope.registry import (
    AttachmentHandlerRegistry,
    NormalizerRegistry,
    build_default_attachment_handler_registry,
)
from kazusa_ai_chatbot.message_envelope.resolvers import (
    PassthroughMentionResolver,
)
from kazusa_ai_chatbot.message_envelope.types import (
    AttachmentRef,
    ConversationAuthorRole,
    Mention,
    MentionEntityKind,
    MessageEnvelope,
    NormalizedEnvelopeFragment,
    RawMention,
    RawReply,
    ReplyTarget,
    ResolvedMention,
)

__all__ = [
    "AttachmentHandler",
    "AttachmentHandlerRegistry",
    "AttachmentHandlerRegistryProtocol",
    "AttachmentRef",
    "ConversationAuthorRole",
    "EnvelopeNormalizer",
    "INLINE_ATTACHMENT_BYTE_LIMIT",
    "Mention",
    "MentionEntityKind",
    "MentionResolver",
    "MessageEnvelope",
    "NormalizedEnvelopeFragment",
    "NormalizerRegistry",
    "PassthroughMentionResolver",
    "RawMention",
    "RawReply",
    "ReplyTarget",
    "ResolvedMention",
    "build_default_attachment_handler_registry",
]
