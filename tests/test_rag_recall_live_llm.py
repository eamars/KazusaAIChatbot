"""Real LLM checks for RAG2 Recall routing."""

from __future__ import annotations

import logging

import httpx
import pytest

from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as supervisor2_module
from kazusa_ai_chatbot.rag import recall_agent as recall_module
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def _skip_if_llm_unavailable() -> None:
    """Skip the live Recall tests when the planner endpoint is unavailable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{RAG_PLANNER_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {RAG_PLANNER_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{RAG_PLANNER_LLM_BASE_URL}"
        )


async def _noop_async(*args, **kwargs) -> None:
    """Avoid persistent Cache2 writes in focused live prompt checks."""
    del args, kwargs


async def _empty_active_commitments(*args, **kwargs) -> list[dict]:
    """Keep the live RAG2 case focused on progress-fed Recall evidence."""
    del args, kwargs
    return_value: list[dict] = []
    return return_value


async def _empty_scheduled_events(*args, **kwargs) -> list[dict]:
    """Avoid unrelated scheduler/database dependency in live route checks."""
    del args, kwargs
    return_value: list[dict] = []
    return return_value


async def _empty_history(*args, **kwargs) -> list[dict]:
    """Avoid transcript database dependency unless a case explicitly needs it."""
    del args, kwargs
    return_value: list[dict] = []
    return return_value


def _initializer_state(query: str) -> dict:
    """Build a minimal initializer state for one live route check.

    Args:
        query: Current user query to route.

    Returns:
        State dictionary consumed by ``rag_initializer``.
    """

    return_value = {
        "original_query": query,
        "character_name": "the active character",
        "context": {
            "platform": "qq",
            "platform_channel_id": "recall-live-route-test",
            "global_user_id": "user-live-recall",
            "user_name": "User",
            "current_timestamp": "2026-05-02T00:00:00+00:00",
            "prompt_message_context": {
                "body_text": query,
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
            "conversation_progress": {
                "status": "active",
                "continuity": "same_episode",
                "current_thread": "The user will pick up the character at 9:30.",
            },
        },
    }
    return return_value


async def _run_initializer_case(monkeypatch, case_id: str, query: str) -> list[str]:
    """Run the live initializer and write an inspectable trace artifact.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        case_id: Stable test case identifier.
        query: User query to route.

    Returns:
        Parsed unknown slots emitted by the initializer.
    """

    await _skip_if_llm_unavailable()
    await get_rag_cache2_runtime().clear()
    monkeypatch.setattr(supervisor2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(supervisor2_module, "record_initializer_hit", _noop_async)
    monkeypatch.setattr(recall_module, "query_user_memory_units", _empty_active_commitments)
    monkeypatch.setattr(recall_module, "query_pending_scheduled_events", _empty_scheduled_events)
    monkeypatch.setattr(recall_module, "get_conversation_history", _empty_history)

    state = _initializer_state(query)
    result = await supervisor2_module.rag_initializer(state)
    unknown_slots = result["unknown_slots"]
    trace_path = write_llm_trace(
        "rag_recall_live_llm",
        case_id,
        {
            "query": query,
            "raw_initializer_output": result,
            "parsed_slots": unknown_slots,
            "judgment": "manual_review_required_for_recall_route_quality",
        },
    )
    logger.info(
        f"RAG_RECALL_LIVE case={case_id} trace={trace_path} "
        f"slots={unknown_slots}"
    )
    return unknown_slots


async def test_live_initializer_routes_active_agreement_to_recall(monkeypatch) -> None:
    """The live initializer should route active agreement recall to Recall."""
    unknown_slots = await _run_initializer_case(
        monkeypatch,
        "active_agreement",
        '早上好呀，还记得今天的约定么',
    )

    assert any(slot.startswith("Recall:") for slot in unknown_slots)


async def test_live_initializer_keeps_exact_phrase_on_conversation_evidence(monkeypatch) -> None:
    """Exact phrase provenance should remain a conversation evidence route."""
    unknown_slots = await _run_initializer_case(
        monkeypatch,
        "exact_phrase",
        '谁说过"约定就是约定"？',
    )

    assert any(slot.startswith("Conversation-evidence:") for slot in unknown_slots)
    assert not any(slot.startswith("Recall:") for slot in unknown_slots)


async def test_live_rag2_recall_answers_today_agreement(monkeypatch) -> None:
    """End-to-end RAG2 should answer active-agreement recall after wiring."""
    await _skip_if_llm_unavailable()
    await get_rag_cache2_runtime().clear()
    monkeypatch.setattr(supervisor2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(supervisor2_module, "record_initializer_hit", _noop_async)

    query = '早上好呀，还记得今天的约定么'
    result = await supervisor2_module.call_rag_supervisor(
        original_query=query,
        character_name="the active character",
        context=_initializer_state(query)["context"],
    )
    trace_path = write_llm_trace(
        "rag_recall_live_llm",
        "rag2_today_agreement",
        {
            "query": query,
            "result": result,
            "parsed_slots": result.get("unknown_slots", []),
            "known_facts": result.get("known_facts", []),
            "final_answer": result.get("answer", ""),
            "judgment": "manual_review_required_for_end_to_end_recall_quality",
        },
    )
    logger.info(
        f"RAG_RECALL_LIVE case=rag2_today_agreement trace={trace_path} "
        f"answer={result.get('answer', '')}"
    )

    assert any(
        fact.get("agent") == "recall_agent"
        for fact in result.get("known_facts", [])
    )
    assert "9:30" in result.get("answer", "")
