"""Real-conversation-derived live LLM route checks for RAG2 capability slots."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.llm_trace import write_llm_trace
from tests.test_rag_phase3_initializer_live_llm import _run_initializer_case

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rag_phase3_real_conversation_cases.json"
)


def _load_case(case_id: str) -> dict:
    """Load one compact real-conversation case by id."""

    cases = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(f"Missing real conversation case fixture: {case_id}")


async def _run_real_conversation_case(monkeypatch, case_id: str) -> None:
    """Run a compact case and write a second case-focused trace."""

    case = _load_case(case_id)
    unknown_slots = await _run_initializer_case(
        monkeypatch,
        case_id,
        case["query"],
        case["expected_prefixes"],
        forbidden_prefixes=case.get("forbidden_prefixes"),
    )
    forbidden_slot_terms = case.get("forbidden_slot_terms", [])
    for term in forbidden_slot_terms:
        assert not any(term in slot for slot in unknown_slots), (
            f"Forbidden slot term {term!r} found in slots: {unknown_slots}"
        )
    trace_path = write_llm_trace(
        "rag_phase3_real_conversation_live_llm",
        case_id,
        {
            "case_id": case_id,
            "query": case["query"],
            "reviewer_note": case["reviewer_note"],
            "expected_prefixes": case["expected_prefixes"],
            "forbidden_prefixes": case.get("forbidden_prefixes", []),
            "forbidden_slot_terms": forbidden_slot_terms,
            "parsed_slots": unknown_slots,
            "judgment": "manual_review_required_for_real_conversation_route_quality",
        },
    )
    logger.info(
        f"RAG_PHASE3_REAL_CONVERSATION_LIVE case={case_id} "
        f"trace={trace_path} slots={unknown_slots}"
    )


async def test_real_conversation_christchurch_weather_routes_to_live_context(monkeypatch) -> None:
    await _run_real_conversation_case(monkeypatch, "christchurch_weekend_weather")


async def test_real_conversation_amusement_park_opening_routes_to_live_context(monkeypatch) -> None:
    await _run_real_conversation_case(monkeypatch, "amusement_park_opening")


async def test_real_conversation_recent_address_confirmation_routes_to_conversation_evidence(
    monkeypatch,
) -> None:
    await _run_real_conversation_case(monkeypatch, "recent_address_confirmation")


async def test_real_conversation_official_address_routes_to_memory_evidence(monkeypatch) -> None:
    await _run_real_conversation_case(monkeypatch, "official_address_memory")


async def test_real_conversation_today_agreement_routes_to_recall(monkeypatch) -> None:
    await _run_real_conversation_case(monkeypatch, "today_agreement")


async def test_real_conversation_episode_next_step_routes_to_recall(monkeypatch) -> None:
    await _run_real_conversation_case(monkeypatch, "episode_position_next_step")


async def test_real_conversation_exact_phrase_boundary_routes_to_conversation_evidence(
    monkeypatch,
) -> None:
    await _run_real_conversation_case(monkeypatch, "exact_phrase_boundary")
