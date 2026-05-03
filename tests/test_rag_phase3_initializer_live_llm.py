"""Real LLM route checks for the RAG2 capability-layer initializer."""

from __future__ import annotations

import logging

import httpx
import pytest

from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as supervisor2_module
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def _noop_async(*args, **kwargs) -> None:
    """Avoid persistent Cache2 writes in focused live prompt checks."""
    del args, kwargs


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured planner endpoint is unavailable."""
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


def _initializer_state(query: str) -> dict:
    """Build a minimal prompt-safe initializer state for one route check."""

    return_value = {
        "original_query": query,
        "character_name": "the active character",
        "context": {
            "platform": "qq",
            "platform_channel_id": "rag-phase3-live-route",
            "platform_user_id": "673225019",
            "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
            "user_name": "蚝爹油",
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
            "chat_history_recent": [
                {
                    "role": "user",
                    "display_name": "蚝爹油",
                    "body_text": "我在基督城，明天去游乐场前先看天气。",
                }
            ],
            "chat_history_wide": [],
        },
    }
    return return_value


async def _run_initializer_case(
    monkeypatch,
    case_id: str,
    query: str,
    expected_prefixes: list[str],
    required_slot_fragments: list[str] | None = None,
    forbidden_prefixes: list[str] | None = None,
    forbidden_slot_fragments: list[str] | None = None,
) -> list[str]:
    """Run one live initializer case and write an inspectable trace."""

    await _skip_if_llm_unavailable()
    await get_rag_cache2_runtime().clear()
    monkeypatch.setattr(supervisor2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(supervisor2_module, "record_initializer_hit", _noop_async)

    result = await supervisor2_module.rag_initializer(_initializer_state(query))
    unknown_slots = result["unknown_slots"]
    trace_path = write_llm_trace(
        "rag_phase3_initializer_live_llm",
        case_id,
        {
            "query": query,
            "raw_initializer_output": result,
            "parsed_slots": unknown_slots,
            "expected_prefixes": expected_prefixes,
            "required_slot_fragments": required_slot_fragments or [],
            "forbidden_prefixes": forbidden_prefixes or [],
            "forbidden_slot_fragments": forbidden_slot_fragments or [],
            "judgment": "manual_review_required_for_capability_route_quality",
        },
    )
    logger.info(
        f"RAG_PHASE3_INITIALIZER_LIVE case={case_id} trace={trace_path} "
        f"slots={unknown_slots}"
    )

    for prefix in expected_prefixes:
        assert any(slot.startswith(prefix) for slot in unknown_slots)
    for fragment in required_slot_fragments or []:
        assert any(fragment in slot for slot in unknown_slots)
    for prefix in forbidden_prefixes or []:
        assert not any(slot.startswith(prefix) for slot in unknown_slots)
    for fragment in forbidden_slot_fragments or []:
        assert not any(fragment in slot for slot in unknown_slots)

    return unknown_slots


async def test_live_initializer_routes_current_time_to_runtime_live_context(
    monkeypatch,
) -> None:
    """Bare current time should route to the runtime-backed Live-context form."""

    await _run_initializer_case(
        monkeypatch,
        "current_time_to_runtime_live_context",
        '现在几点？',
        ["Live-context:"],
        required_slot_fragments=["active character current local time"],
        forbidden_slot_fragments=["unknown location", "Runtime-context:"],
    )


async def test_live_initializer_routes_current_date_to_runtime_live_context(
    monkeypatch,
) -> None:
    """Bare current date should route to the runtime-backed Live-context form."""

    await _run_initializer_case(
        monkeypatch,
        "current_date_to_runtime_live_context",
        '今天几号？',
        ["Live-context:"],
        required_slot_fragments=["active character current local date"],
        forbidden_slot_fragments=[
            "unknown target",
            "unknown location",
            "Runtime-context:",
        ],
    )


async def test_live_initializer_routes_current_weekday_to_runtime_live_context(
    monkeypatch,
) -> None:
    """Bare current weekday should route to the runtime-backed Live-context form."""

    await _run_initializer_case(
        monkeypatch,
        "current_weekday_to_runtime_live_context",
        '今天星期几？',
        ["Live-context:"],
        required_slot_fragments=["active character current local weekday"],
        forbidden_slot_fragments=[
            "unknown target",
            "unknown location",
            "Runtime-context:",
        ],
    )


async def test_live_initializer_routes_active_agreement_to_recall(monkeypatch) -> None:
    """Active agreement recall remains a Recall route."""

    await _run_initializer_case(
        monkeypatch,
        "active_agreement_to_recall",
        '早上好呀，还记得今天的约定么',
        ["Recall:"],
    )


async def test_live_initializer_routes_exact_phrase_to_conversation_evidence(monkeypatch) -> None:
    """Exact phrase provenance moves to Conversation-evidence, not Recall."""

    await _run_initializer_case(
        monkeypatch,
        "exact_phrase_to_conversation_evidence",
        '谁说过"约定就是约定"？',
        ["Conversation-evidence:"],
        forbidden_prefixes=["Recall:"],
    )


async def test_live_initializer_routes_character_local_temperature_to_live_context(monkeypatch) -> None:
    """Character-local current temperature should route to Live-context."""

    await _run_initializer_case(
        monkeypatch,
        "character_local_temperature_to_live_context",
        '你那边现在多少度？',
        ["Live-context:"],
    )


async def test_live_initializer_routes_user_local_temperature_to_live_context_without_character_fallback(
    monkeypatch,
) -> None:
    """User-local current temperature should not silently use character location."""

    await _run_initializer_case(
        monkeypatch,
        "user_local_temperature_to_live_context",
        '我这边现在多少度？',
        ["Live-context:"],
    )


async def test_live_initializer_preserves_cascaded_phrase_person_link_web_chain(monkeypatch) -> None:
    """The initializer must still decode a cascaded phrase -> person -> link -> web chain."""

    await _run_initializer_case(
        monkeypatch,
        "cascaded_phrase_person_link_web_chain",
        '说版权保护是play一环的那个人，他发过什么链接，链接里是什么内容',
        [
            "Conversation-evidence:",
            "Person-context:",
            "Web-evidence:",
        ],
    )
