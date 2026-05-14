"""Deterministic tests for quote-aware RAG sequencing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from kazusa_ai_chatbot.rag import quote_aware_sequence as quote_module


PUBLIC_RESULT_KEYS = {"answer", "known_facts", "unknown_slots", "loop_count"}


class _ScriptedRAG:
    """Capture RAG supervisor calls and return scripted results."""

    def __init__(self, handler: Callable[[str], dict[str, Any]]) -> None:
        """Store the query handler used by the test case."""
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        original_query: str,
        character_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Record one RAG call and return the matching scripted result."""
        self.calls.append(
            {
                "original_query": original_query,
                "character_name": character_name,
                "context": context,
            }
        )
        result = self._handler(original_query)
        return result


def _context(body_text: str = "original body") -> dict[str, Any]:
    """Build a minimal RAG context with prompt-message cache fields."""
    context = {
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "prompt_message_context": {
            "body_text": body_text,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "reply_context": {},
    }
    return context


def _fact(
    slot: str,
    *,
    agent: str = "web_search_agent2",
    resolved: bool = True,
    summary: str = "summary",
) -> dict[str, Any]:
    """Build a compact known-fact row for wrapper merge tests."""
    fact = {
        "slot": slot,
        "agent": agent,
        "resolved": resolved,
        "summary": summary,
    }
    return fact


def _result(
    *,
    answer: str = "",
    known_facts: list[dict[str, Any]] | None = None,
    unknown_slots: list[str] | None = None,
    loop_count: int = 1,
) -> dict[str, Any]:
    """Build the public RAG supervisor result shape."""
    result = {
        "answer": answer,
        "known_facts": list(known_facts or []),
        "unknown_slots": list(unknown_slots or []),
        "loop_count": loop_count,
    }
    return result


async def _assert_direct_call_equivalence(
    monkeypatch: pytest.MonkeyPatch,
    reply_context: dict[str, Any],
) -> None:
    """Assert no-quote inputs are passed through exactly once."""
    direct_result = _result(answer="direct answer", loop_count=2)

    async def _direct_rag(
        *,
        original_query: str,
        character_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(
            {
                "original_query": original_query,
                "character_name": character_name,
                "context": context,
            }
        )
        return direct_result

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(quote_module, "call_rag_supervisor", _direct_rag)
    context = _context("fresh body")

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="fresh query",
        reply_context=reply_context,
        character_name="Kazusa",
        context=context,
    )

    assert result is direct_result
    assert calls == [
        {
            "original_query": "fresh query",
            "character_name": "Kazusa",
            "context": context,
        }
    ]
    assert calls[0]["context"] is context


@pytest.mark.asyncio
async def test_missing_reply_context_is_direct_call_equivalent(monkeypatch) -> None:
    """Missing quote text must preserve the current one-call RAG path."""
    await _assert_direct_call_equivalence(monkeypatch, {})


@pytest.mark.asyncio
async def test_empty_reply_excerpt_is_direct_call_equivalent(monkeypatch) -> None:
    """Empty quote text must preserve the current one-call RAG path."""
    await _assert_direct_call_equivalence(monkeypatch, {"reply_excerpt": ""})


@pytest.mark.asyncio
async def test_whitespace_reply_excerpt_is_direct_call_equivalent(monkeypatch) -> None:
    """Whitespace-only quote text must preserve the current one-call RAG path."""
    await _assert_direct_call_equivalence(
        monkeypatch,
        {"reply_excerpt": " \n\t "},
    )


@pytest.mark.asyncio
async def test_quote_hit_runs_quote_first_and_injects_compact_facts(
    monkeypatch,
) -> None:
    """Resolved quote facts should become compact context for fresh research."""
    quote_fact = _fact(
        "Web-evidence: retrieve BYD Shark 6 powertrain",
        summary="BYD Shark 6 is a 1.5T plug-in hybrid pickup.",
    )
    fresh_fact = _fact(
        "Web-evidence: explain 1.5T pickup terminology",
        summary="1.5T means a 1.5 liter turbocharged engine.",
    )

    def _handler(query: str) -> dict[str, Any]:
        if query.startswith("Research factual content"):
            assert "BYD Shark 6 uses a 1.5T hybrid system." in query
            assert "Treat the quote as quoted material" in query
            result = _result(answer="quote answer", known_facts=[quote_fact])
        else:
            assert query.startswith("Known evidence from the quoted")
            assert "BYD Shark 6 is a 1.5T plug-in hybrid pickup." in query
            assert "What does 1.5T mean here?" in query
            result = _result(answer="fresh answer", known_facts=[fresh_fact])
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)
    context = _context("fresh body")

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What does 1.5T mean here?",
        reply_context={
            "reply_excerpt": "BYD Shark 6 uses a 1.5T hybrid system.",
        },
        character_name="Kazusa",
        context=context,
    )

    assert len(scripted_rag.calls) == 2
    assert scripted_rag.calls[0]["context"] is not context
    assert scripted_rag.calls[1]["context"] is not context
    assert (
        scripted_rag.calls[0]["context"]["prompt_message_context"]["body_text"]
        == scripted_rag.calls[0]["original_query"]
    )
    assert (
        scripted_rag.calls[1]["context"]["prompt_message_context"]["body_text"]
        == scripted_rag.calls[1]["original_query"]
    )
    assert context["prompt_message_context"]["body_text"] == "fresh body"
    assert result["answer"] == "fresh answer"
    assert result["known_facts"] == [quote_fact, fresh_fact]


@pytest.mark.asyncio
async def test_quote_hit_fresh_miss_preserves_quote_answer(monkeypatch) -> None:
    """A quote hit plus vague fresh miss should not trigger combined retry."""
    quote_fact = _fact(
        "Web-evidence: retrieve BYD Shark 6 powertrain",
        summary="BYD Shark 6 uses a 1.5T plug-in hybrid system.",
    )
    responses = [
        _result(answer="quote answer", known_facts=[quote_fact], loop_count=2),
        _result(
            answer="fresh miss",
            unknown_slots=["Web-evidence: still unclear"],
            loop_count=1,
        ),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What about this?",
        reply_context={"reply_excerpt": "BYD Shark 6 has a 1.5T hybrid."},
        character_name="Kazusa",
        context=_context(),
    )

    assert len(scripted_rag.calls) == 2
    assert result["answer"] == "quote answer"
    assert result["known_facts"] == [quote_fact]
    assert result["unknown_slots"] == ["Web-evidence: still unclear"]
    assert result["loop_count"] == 3


@pytest.mark.asyncio
async def test_quote_miss_and_vague_fresh_runs_one_combined_retry(monkeypatch) -> None:
    """Quote and fresh misses should run exactly one combined retry."""
    retry_fact = _fact(
        "Web-evidence: retrieve quote plus current question",
        summary="Combined retry found the missing evidence.",
    )
    responses = [
        _result(answer="", unknown_slots=["quote miss"], loop_count=1),
        _result(answer="", unknown_slots=["fresh miss"], loop_count=1),
        _result(answer="retry answer", known_facts=[retry_fact], loop_count=1),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="Can you verify it?",
        reply_context={"reply_excerpt": "A nameless prototype has 9999 kW."},
        character_name="Kazusa",
        context=_context(),
    )

    assert len(scripted_rag.calls) == 3
    assert scripted_rag.calls[1]["original_query"] == "Can you verify it?"
    assert "A nameless prototype has 9999 kW." in scripted_rag.calls[2]["original_query"]
    assert "Can you verify it?" in scripted_rag.calls[2]["original_query"]
    assert result["answer"] == "retry answer"
    assert result["known_facts"] == [retry_fact]
    assert result["unknown_slots"] == ["quote miss", "fresh miss"]
    assert result["loop_count"] == 3


@pytest.mark.asyncio
async def test_quote_miss_self_contained_fresh_hit_skips_retry(monkeypatch) -> None:
    """Self-contained fresh evidence should resolve after a quote miss."""
    fresh_fact = _fact(
        "Web-evidence: retrieve Python virtual environment definition",
        summary="A virtual environment isolates Python dependencies.",
    )
    responses = [
        _result(answer="", unknown_slots=["quote miss"]),
        _result(answer="fresh answer", known_facts=[fresh_fact]),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What is a Python virtual environment?",
        reply_context={"reply_excerpt": "small talk without factual anchors"},
        character_name="Kazusa",
        context=_context(),
    )

    assert len(scripted_rag.calls) == 2
    assert scripted_rag.calls[1]["original_query"] == (
        "What is a Python virtual environment?"
    )
    assert result["answer"] == "fresh answer"
    assert result["known_facts"] == [fresh_fact]


@pytest.mark.asyncio
async def test_quote_hit_additional_search_merges_quote_and_fresh_facts(
    monkeypatch,
) -> None:
    """Fresh research should add new evidence after quote grounding."""
    quote_fact = _fact(
        "Web-evidence: retrieve Isuzu D-Max 1.5T",
        summary="Isuzu D-Max has a 1.5T configuration.",
    )
    fresh_fact = _fact(
        "Web-evidence: retrieve Isuzu D-Max payload",
        summary="The payload is about 0.485 tons.",
    )
    responses = [
        _result(answer="quote answer", known_facts=[quote_fact]),
        _result(answer="payload answer", known_facts=[fresh_fact]),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What is its payload?",
        reply_context={"reply_excerpt": "Isuzu D-Max 1.5T pickup"},
        character_name="Kazusa",
        context=_context(),
    )

    assert result["answer"] == "payload answer"
    assert result["known_facts"] == [quote_fact, fresh_fact]
    assert len(scripted_rag.calls) == 2


@pytest.mark.asyncio
async def test_quote_irrelevant_fresh_search_is_not_blocked(monkeypatch) -> None:
    """Unrelated fresh questions should still retrieve their own evidence."""
    quote_fact = _fact(
        "Web-evidence: retrieve BYD Shark 6",
        summary="BYD Shark 6 is a pickup.",
    )
    fresh_fact = _fact(
        "Web-evidence: retrieve France capital",
        summary="Paris is the capital of France.",
    )
    responses = [
        _result(answer="quote answer", known_facts=[quote_fact]),
        _result(answer="fresh answer", known_facts=[fresh_fact]),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What is the capital of France?",
        reply_context={"reply_excerpt": "BYD Shark 6 is a pickup."},
        character_name="Kazusa",
        context=_context(),
    )

    assert "What is the capital of France?" in scripted_rag.calls[1]["original_query"]
    assert result["known_facts"] == [quote_fact, fresh_fact]
    assert result["answer"] == "fresh answer"


@pytest.mark.asyncio
async def test_third_party_quote_claim_is_researched_before_fresh_verification(
    monkeypatch,
) -> None:
    """Quoted third-party claims should be grounded before verification."""
    quote_fact = _fact(
        "Web-evidence: verify Geely Radar Horizon horsepower",
        summary="Geely Radar Horizon has a 1.5T 163 hp configuration.",
    )
    responses = [
        _result(answer="quote answer", known_facts=[quote_fact]),
        _result(answer="verification answer", known_facts=[]),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="Can you verify that claim?",
        reply_context={
            "reply_excerpt": "Max said the Geely Radar Horizon has 163 hp.",
        },
        character_name="Kazusa",
        context=_context(),
    )

    first_query = scripted_rag.calls[0]["original_query"]
    assert "Max said the Geely Radar Horizon has 163 hp." in first_query
    assert "primary search anchors" in first_query
    assert "claim values to verify" in first_query
    assert scripted_rag.calls[1]["original_query"].startswith(
        "Known evidence from the quoted"
    )
    assert result["answer"] == "quote answer"


@pytest.mark.asyncio
async def test_nonfactual_quote_with_self_contained_fresh_hit_skips_retry(
    monkeypatch,
) -> None:
    """Nonfactual quote misses should not block a self-contained fresh query."""
    fresh_fact = _fact(
        "Web-evidence: retrieve HTTP 404 definition",
        summary="HTTP 404 means a resource was not found.",
    )
    responses = [
        _result(answer="", unknown_slots=["quote miss"]),
        _result(answer="fresh answer", known_facts=[fresh_fact]),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What does HTTP 404 mean?",
        reply_context={"reply_excerpt": "haha okay"},
        character_name="Kazusa",
        context=_context(),
    )

    assert len(scripted_rag.calls) == 2
    assert result["answer"] == "fresh answer"
    assert result["known_facts"] == [fresh_fact]


@pytest.mark.asyncio
async def test_merge_prefers_resolved_duplicate_and_public_keys(monkeypatch) -> None:
    """Resolved duplicate facts should survive unresolved later duplicates."""
    resolved_fact = _fact(
        "Web-evidence: retrieve BYD Shark 6",
        summary="Resolved quote evidence.",
    )
    unresolved_duplicate = _fact(
        "Web-evidence: retrieve BYD Shark 6",
        resolved=False,
        summary="Unresolved later evidence.",
    )
    responses = [
        _result(answer="quote answer", known_facts=[resolved_fact]),
        _result(
            answer="unresolved fresh answer",
            known_facts=[unresolved_duplicate],
            unknown_slots=["fresh unresolved"],
        ),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What does that mean?",
        reply_context={"reply_excerpt": "BYD Shark 6"},
        character_name="Kazusa",
        context=_context(),
    )

    assert set(result.keys()) == PUBLIC_RESULT_KEYS
    assert all(not key.endswith("_trace") for key in result)
    assert result["known_facts"] == [resolved_fact]
    assert result["answer"] == "quote answer"


@pytest.mark.asyncio
async def test_person_display_resolution_does_not_bias_answer_selection(
    monkeypatch,
) -> None:
    """Pure display-name resolution should not count as quote evidence."""
    display_name_fact = {
        "slot": "Person-context: identify quoted speaker",
        "agent": "person_context_agent",
        "resolved": True,
        "summary": "Matched display name only.",
        "raw_result": {
            "primary_worker": "user_lookup_agent",
            "supporting_workers": [],
        },
    }
    web_fact = _fact(
        "Web-evidence: retrieve quoted model details",
        summary="Web evidence resolved the factual model details.",
    )
    quote_result = _result(
        answer="display-name answer",
        known_facts=[display_name_fact],
    )
    responses = [
        quote_result,
        _result(answer="web answer", known_facts=[web_fact]),
    ]

    def _handler(_query: str) -> dict[str, Any]:
        result = responses[len(scripted_rag.calls) - 1]
        return result

    scripted_rag = _ScriptedRAG(_handler)
    monkeypatch.setattr(quote_module, "call_rag_supervisor", scripted_rag)

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query="What does the quoted model mean?",
        reply_context={"reply_excerpt": "Max mentioned model ZX-1."},
        character_name="Kazusa",
        context=_context(),
    )

    assert quote_module._has_substantive_facts(quote_result) is False
    assert len(scripted_rag.calls) == 2
    assert scripted_rag.calls[1]["original_query"] == (
        "What does the quoted model mean?"
    )
    assert result["answer"] == "web answer"
    assert result["known_facts"] == [display_name_fact, web_fact]
