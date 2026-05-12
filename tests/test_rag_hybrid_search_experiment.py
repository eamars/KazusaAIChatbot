import pytest

from experiments.rag_hybrid_search import hybrid_search
from experiments.rag_hybrid_search.hybrid_search import (
    SearchCase,
    build_hybrid_seed_rows,
    build_neighbor_seed_rows,
    evaluate_case_method,
    merge_candidates,
    summarize_results,
)
from experiments.rag_hybrid_search.run import _parser
from kazusa_ai_chatbot.config import RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR


def _positive_case() -> SearchCase:
    """Build a compact positive case for deterministic metric tests."""

    case = SearchCase(
        case_id="case-positive",
        kind="positive",
        platform="qq",
        platform_channel_id="channel",
        query="query",
        keywords=("gpu",),
        expected_any=("expected",),
        expected_message_ids=("message-expected",),
        forbidden_any=("forbidden",),
    )
    return case


def _negative_case() -> SearchCase:
    """Build a compact absent-topic case for deterministic metric tests."""

    case = SearchCase(
        case_id="case-negative",
        kind="negative",
        platform="qq",
        platform_channel_id="channel",
        query="absent query",
        keywords=("absent",),
        expected_any=(),
        expected_message_ids=(),
        forbidden_any=("forbidden",),
    )
    return case


def test_merge_candidates_deduplicates_and_preserves_methods() -> None:
    """Hybrid merging should retain provenance from multiple retrieval paths."""

    semantic_rows = [
        {
            "platform_message_id": "message-1",
            "body_text": "expected evidence",
            "score": 0.8,
            "rank": 3,
            "methods": ["semantic"],
        }
    ]
    keyword_rows = [
        {
            "platform_message_id": "message-1",
            "body_text": "expected evidence",
            "rank": 1,
            "methods": ["keyword:gpu"],
        }
    ]

    merged = merge_candidates(semantic_rows, keyword_rows)

    assert len(merged) == 1
    assert merged[0]["methods"] == ["semantic", "keyword:gpu"]
    assert merged[0]["best_rank"] == 1
    assert merged[0]["score"] == 0.8


def test_hybrid_seed_rows_rank_cross_supported_literal_hits_first() -> None:
    """Hybrid seeds should prefer semantic rows with literal-anchor support."""

    semantic_rows = [
        {
            "platform_message_id": "semantic-only",
            "body_text": "broad topic echo",
            "score": 0.95,
            "rank": 1,
            "methods": ["semantic"],
        },
        {
            "platform_message_id": "cross-supported",
            "body_text": "specific expected evidence",
            "score": 0.80,
            "rank": 5,
            "methods": ["semantic"],
        },
    ]
    keyword_rows = [
        {
            "platform_message_id": "cross-supported",
            "body_text": "specific expected evidence",
            "rank": 1,
            "methods": ["keyword:gpu"],
        }
    ]

    seeds = build_hybrid_seed_rows(
        semantic_rows,
        keyword_rows,
        semantic_only_floor=0.70,
    )

    assert seeds[0]["platform_message_id"] == "cross-supported"
    assert seeds[0]["methods"] == ["semantic", "keyword:gpu"]


def test_hybrid_seed_rows_filter_weak_semantic_only_without_keyword_hit() -> None:
    """Weak semantic-only drift should not seed hybrid evidence by itself."""

    semantic_rows = [
        {
            "platform_message_id": "weak-semantic",
            "body_text": "unrelated topic",
            "score": 0.65,
            "rank": 1,
            "methods": ["semantic"],
        },
        {
            "platform_message_id": "strong-semantic",
            "body_text": "direct answer",
            "score": 0.82,
            "rank": 2,
            "methods": ["semantic"],
        },
    ]

    seeds = build_hybrid_seed_rows(
        semantic_rows,
        [],
        semantic_only_floor=0.70,
    )

    assert [row["platform_message_id"] for row in seeds] == ["strong-semantic"]


def test_neighbor_seed_rows_use_direct_evidence_when_keywords_exist() -> None:
    """Neighbor expansion should not promote semantic-only topical echoes."""

    candidate_rows = [
        {
            "platform_message_id": "cross-supported",
            "body_text": "specific expected evidence",
            "score": 0.80,
            "methods": ["semantic", "keyword:gpu"],
        },
        {
            "platform_message_id": "semantic-only",
            "body_text": "broad topic echo",
            "score": 0.95,
            "methods": ["semantic"],
        },
    ]

    seeds = build_neighbor_seed_rows(
        candidate_rows,
        keyword_rows_present=True,
        semantic_only_floor=0.70,
    )

    assert [row["platform_message_id"] for row in seeds] == ["cross-supported"]


def test_evaluate_case_method_detects_expected_message_hit() -> None:
    """Positive cases resolve when expected text and message identity are present."""

    result = evaluate_case_method(
        _positive_case(),
        "hybrid",
        [
            {
                "platform_message_id": "message-expected",
                "body_text": "expected evidence",
            }
        ],
        text_limit=50,
    )

    assert result["resolved"] is True
    assert result["false_negative"] is False
    assert result["expected_message_rank"] == 1


def test_evaluate_case_method_marks_missing_expected_message() -> None:
    """Expected text alone is not enough when a case names source messages."""

    result = evaluate_case_method(
        _positive_case(),
        "semantic",
        [
            {
                "platform_message_id": "other-message",
                "body_text": "expected evidence",
            }
        ],
        text_limit=50,
    )

    assert result["resolved"] is False
    assert result["false_negative"] is True
    assert result["expected_message_rank"] is None


def test_evaluate_case_method_marks_any_negative_rows_false_positive() -> None:
    """Absent-topic cases should fail when retrieval returns any evidence."""

    result = evaluate_case_method(
        _negative_case(),
        "semantic",
        [
            {
                "platform_message_id": "unrelated-message",
                "body_text": "unrelated evidence",
            }
        ],
        text_limit=50,
    )

    assert result["resolved"] is False
    assert result["false_positive"] is True
    assert result["result_count"] == 1


def test_summarize_results_counts_method_metrics() -> None:
    """Experiment summaries should aggregate method-level quality metrics."""

    summary = summarize_results([
        {
            "method": "hybrid",
            "kind": "positive",
            "resolved": True,
            "false_positive": False,
            "false_negative": False,
            "expected_message_rank": 2,
        },
        {
            "method": "hybrid",
            "kind": "negative",
            "resolved": False,
            "false_positive": True,
            "false_negative": False,
            "expected_message_rank": None,
        },
    ])

    assert summary["hybrid"]["case_count"] == 2
    assert summary["hybrid"]["resolved_count"] == 1
    assert summary["hybrid"]["false_positive_count"] == 1
    assert summary["hybrid"]["expected_message_hit_count"] == 1
    assert summary["hybrid"]["average_expected_message_rank"] == 2.0


@pytest.mark.asyncio
async def test_search_case_methods_uses_production_hybrid_entrypoint(
    monkeypatch,
) -> None:
    """Hybrid experiment rows should come from production retrieval code."""

    production_calls: list[dict[str, object]] = []

    async def fake_semantic_rows(
        case: SearchCase,
        limit: int,
    ) -> list[dict[str, object]]:
        return [{"platform_message_id": "semantic", "body_text": "semantic"}]

    async def fake_keyword_rows(
        case: SearchCase,
        limit: int,
    ) -> list[dict[str, object]]:
        return [{"platform_message_id": "keyword", "body_text": "keyword"}]

    async def fake_hybrid_rows(
        args: dict[str, object],
        **kwargs: object,
    ) -> list[dict[str, object]]:
        production_calls.append({"args": dict(args), "kwargs": dict(kwargs)})
        return [{"platform_message_id": "hybrid", "body_text": "hybrid"}]

    monkeypatch.setattr(hybrid_search, "_semantic_rows", fake_semantic_rows)
    monkeypatch.setattr(hybrid_search, "_keyword_rows", fake_keyword_rows)
    monkeypatch.setattr(
        hybrid_search.conversation_search_agent,
        "run_hybrid_conversation_search",
        fake_hybrid_rows,
    )

    methods = await hybrid_search.search_case_methods(
        _positive_case(),
        hybrid_search.HybridSearchConfig(),
    )

    assert methods["hybrid"][0]["platform_message_id"] == "hybrid"
    assert production_calls[0]["args"]["literal_anchors"] == ["gpu"]
    assert "semantic_only_floor" in production_calls[0]["kwargs"]


def test_experiment_cli_defaults_use_production_semantic_floor() -> None:
    """Experiment defaults should not silently drift from production config."""

    parser = _parser()
    args = parser.parse_args([])

    assert args.semantic_only_floor == RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR
