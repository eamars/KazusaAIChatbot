"""Tests for deterministic referent-resolution helper policy."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes.referent_resolution import (
    needs_referent_clarification,
    normalize_referents,
    should_skip_rag_for_unresolved_referents,
    unresolved_referent_reason,
)


def test_normalize_referents_drops_malformed_rows() -> None:
    """Malformed referents should not become trusted clarification signals."""

    referents = normalize_referents([
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        {"phrase": "broken", "referent_role": "thing", "status": "unknown"},
        {"phrase": "", "referent_role": "object", "status": "resolved"},
    ])

    assert referents == [
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
    ]


def test_resolved_referents_do_not_clarify() -> None:
    """Resolved structured referents should not trigger clarification."""

    referents = [
        {"phrase": "这些", "referent_role": "object", "status": "resolved"}
    ]

    assert needs_referent_clarification(referents) is False
    assert should_skip_rag_for_unresolved_referents(referents) is False


def test_mixed_referents_clarify_without_skipping_rag() -> None:
    """Mixed referents should ask narrowly while preserving retrieval access."""

    referents = [
        {"phrase": "他", "referent_role": "subject", "status": "resolved"},
        {"phrase": "那些话", "referent_role": "object", "status": "unresolved"},
    ]

    assert needs_referent_clarification(referents) is True
    assert should_skip_rag_for_unresolved_referents(referents) is False
    assert unresolved_referent_reason(referents) == "缺少以下指代对象: 那些话"


def test_empty_referents_do_not_clarify() -> None:
    """Empty referents should not synthesize clarification signals."""

    referents = []

    assert needs_referent_clarification(referents) is False
    assert should_skip_rag_for_unresolved_referents(referents) is False
    assert unresolved_referent_reason(referents) == ""
