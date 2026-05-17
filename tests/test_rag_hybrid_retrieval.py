from __future__ import annotations

from kazusa_ai_chatbot.rag.hybrid_retrieval import (
    candidate_prompt_text,
    hybrid_row_identity,
    merge_hybrid_candidates,
    select_neighbor_seed_candidates,
)


def _row(
    message_id: str,
    text: str,
    *,
    score: float = 0.0,
    timestamp_utc: str = "2026-05-11T09:00:00+00:00",
) -> dict[str, object]:
    return {
        "platform_message_id": message_id,
        "body_text": text,
        "display_name": "Tester",
        "timestamp": timestamp_utc,
        "score": score,
    }


def test_merge_hybrid_candidates_prioritizes_cross_supported_rows() -> None:
    """Rows supported by semantic and keyword paths should outrank drift."""

    semantic_rows = [
        _row("semantic-only", "broad topic echo", score=0.95),
        _row("cross-supported", "specific direct evidence", score=0.80),
    ]
    keyword_rows = [
        _row("cross-supported", "specific direct evidence"),
    ]

    candidates = merge_hybrid_candidates(
        semantic_rows,
        keyword_rows,
        semantic_only_floor=0.70,
        selected_limit=10,
    )

    assert [candidate.identity for candidate in candidates] == [
        "message:cross-supported",
        "message:semantic-only",
    ]
    assert candidates[0].methods == ("semantic", "keyword")


def test_merge_hybrid_candidates_rejects_weak_semantic_only_rows() -> None:
    """Semantic rows below the floor should not become evidence alone."""

    semantic_rows = [
        _row("weak-semantic", "unrelated echo", score=0.65),
        _row("strong-semantic", "direct memory", score=0.81),
    ]

    candidates = merge_hybrid_candidates(
        semantic_rows,
        [],
        semantic_only_floor=0.70,
        selected_limit=10,
    )

    assert [candidate.identity for candidate in candidates] == [
        "message:strong-semantic",
    ]


def test_merge_hybrid_candidates_keeps_keyword_only_before_semantic_only() -> None:
    """Exact-anchor hits should beat unsupported semantic rows."""

    semantic_rows = [
        _row("semantic-only", "topic mention", score=0.91),
    ]
    keyword_rows = [
        _row("keyword-only", "literal anchor mention"),
    ]

    candidates = merge_hybrid_candidates(
        semantic_rows,
        keyword_rows,
        semantic_only_floor=0.70,
        selected_limit=10,
    )

    assert [candidate.identity for candidate in candidates] == [
        "message:keyword-only",
        "message:semantic-only",
    ]


def test_select_neighbor_seed_candidates_uses_direct_literal_support_first() -> None:
    """Neighbor expansion should not follow broad semantic rows when anchors hit."""

    candidates = merge_hybrid_candidates(
        [_row("semantic-only", "broad topic echo", score=0.95)],
        [_row("keyword-only", "literal anchor mention")],
        semantic_only_floor=0.70,
        selected_limit=10,
    )

    seeds = select_neighbor_seed_candidates(
        candidates,
        keyword_rows_present=True,
        semantic_only_floor=0.70,
        seed_limit=5,
    )

    assert [candidate.identity for candidate in seeds] == ["message:keyword-only"]


def test_hybrid_identity_falls_back_to_memory_name_and_timestamp() -> None:
    """Persistent-memory dedupe should remain stable without Mongo IDs."""

    identity = hybrid_row_identity(
        {
            "memory_name": "hardware-note",
            "timestamp": "2026-05-11T09:00:00+00:00",
            "content": "content",
        },
        source="persistent_memory",
    )

    assert identity == "memory:hardware-note:2026-05-11T09:00:00+00:00"


def test_candidate_prompt_text_includes_attachments_and_reply_context() -> None:
    """Conversation projection should not drop attachment-only evidence."""

    text = candidate_prompt_text(
        {
            "display_name": "Tester",
            "timestamp": "2026-05-11T09:00:00+00:00",
            "body_text": "",
            "attachments": [
                {
                    "description": "screenshot showing GPU market-share chart",
                }
            ],
            "reply_context": {
                "reply_excerpt": "previous message about Steam hardware survey",
            },
        },
        source="conversation",
        text_limit=300,
    )

    assert "Tester at" in text
    assert "GPU market-share chart" in text
    assert "Steam hardware survey" in text
