from kazusa_ai_chatbot.rag.continuation import (
    MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    empty_continuation_decision,
    normalize_continuation_decision,
    validate_refined_query,
)


def test_empty_continuation_decision_shape() -> None:
    decision = empty_continuation_decision()

    assert set(decision.keys()) == {
        "promote_candidate",
        "promoted_candidate_indexes",
        "promotion_summary",
        "should_continue",
        "refined_query",
        "reason",
    }
    assert decision["promote_candidate"] is False
    assert decision["promoted_candidate_indexes"] == []
    assert decision["promotion_summary"] == ""
    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""
    assert decision["reason"] == ""


def test_normalize_stop_decision_drops_refined_query() -> None:
    raw_decision = {
        "should_continue": False,
        "refined_query": "Retry with another route.",
        "reason": 123,
    }

    decision = normalize_continuation_decision(raw_decision)

    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""
    assert decision["reason"] == "123"


def test_validate_accepts_self_contained_refined_query() -> None:
    raw_decision = {
        "should_continue": True,
        "refined_query": (
            "The user wants a current comparison between model A and model B. "
            "Prior memory only says this kind of benchmark can expire, so "
            "retrieve fresh public evidence for the comparison."
        ),
        "reason": "memory provided source strategy",
    }

    decision = validate_refined_query(
        raw_decision,
        original_query="Compare model A and model B.",
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert decision["should_continue"] is True
    assert "fresh public evidence" in decision["refined_query"]
    assert decision["reason"] == "memory provided source strategy"


def test_validate_rejects_empty_same_or_duplicate_refined_query() -> None:
    same_decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": "Compare model A and model B.",
            "reason": "same query",
        },
        original_query="Compare model A and model B.",
        previous_refined_queries=[],
        continuation_count=0,
    )
    duplicate_decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": "Retrieve fresh evidence for model A and model B.",
            "reason": "duplicate query",
        },
        original_query="Compare model A and model B.",
        previous_refined_queries=[
            "Retrieve fresh evidence for model A and model B.",
        ],
        continuation_count=0,
    )
    empty_decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": "",
            "reason": "missing query",
        },
        original_query="Compare model A and model B.",
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert same_decision["should_continue"] is False
    assert same_decision["refined_query"] == ""
    assert duplicate_decision["should_continue"] is False
    assert duplicate_decision["refined_query"] == ""
    assert empty_decision["should_continue"] is False
    assert empty_decision["refined_query"] == ""


def test_validate_rejects_slot_or_backend_shaped_refined_query() -> None:
    slot_decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": "Live-context: answer current benchmark",
            "reason": "slot output is not allowed",
        },
        original_query="Compare model A and model B.",
        previous_refined_queries=[],
        continuation_count=0,
    )
    backend_decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": "Query MongoDB collection memory with $vectorSearch.",
            "reason": "backend syntax is not allowed",
        },
        original_query="Compare model A and model B.",
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert slot_decision["should_continue"] is False
    assert slot_decision["refined_query"] == ""
    assert backend_decision["should_continue"] is False
    assert backend_decision["refined_query"] == ""


def test_validate_rejects_missing_user_placeholder_query() -> None:
    decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": (
                '请根据我的具体用途、预算、硬件约束和个人偏好，'
                '为我推荐合适的开发板。'
            ),
            "reason": "missing user constraints were rewritten as placeholders",
        },
        original_query='我该买哪个开发板？',
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""


def test_validate_applies_continuation_cap() -> None:
    decision = validate_refined_query(
        {
            "should_continue": True,
            "refined_query": "Retrieve fresh evidence for model A and model B.",
            "reason": "fresh evidence is needed",
        },
        original_query="Compare model A and model B.",
        previous_refined_queries=[],
        continuation_count=MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    )

    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""
