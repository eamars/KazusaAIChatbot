"""Live LLM smoke tests for the cognition referents migration."""

from __future__ import annotations

import logging
from time import perf_counter

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_content_anchor_agent,
)
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm


async def _skip_if_llm_unavailable() -> None:
    """Skip live cognition tests when the local LLM endpoint is unavailable.

    Returns:
        None. The function calls ``pytest.skip`` if the endpoint cannot be used.
    """

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the live local cognition LLM endpoint is reachable.

    Returns:
        None.
    """

    await _skip_if_llm_unavailable()


@pytest.mark.asyncio
async def test_live_judgment_core_prefers_referents(
    ensure_live_llm: None,
) -> None:
    """Judgment Core should clarify from structured referents."""

    del ensure_live_llm
    state = {
        "character_profile": {"name": "Kazusa"},
        "user_profile": {"affinity": 500},
        "internal_monologue": "I could answer if I knew what the object was.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "boundary_summary": "No boundary issue.",
            "behavior_primary": "comply",
            "behavior_secondary": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
            "identity_policy": "accept",
            "pressure_policy": "absorb",
            "trajectory": "stable",
        },
    }

    started_at = perf_counter()
    result = await call_judgment_core_agent(state)
    duration_seconds = perf_counter() - started_at
    trace_path = write_llm_trace(
        "cognition_referents_live",
        "judgment_unresolved_referent",
        {
            "input": state,
            "output": result,
            "duration_seconds": duration_seconds,
            "judgment": (
                "E3 Judgment Core must treat structured unresolved referents "
                "as authoritative."
            ),
        },
    )

    logger.info(
        f"live_cognition_referents judgment trace={trace_path} "
        f"duration_seconds={duration_seconds:.3f} result={result!r}"
    )
    assert result["logical_stance"] == "TENTATIVE"
    assert result["character_intent"] == "CLARIFY"
    assert "这些" in result["judgment_note"]
    assert duration_seconds < 30.0


@pytest.mark.asyncio
async def test_live_content_anchor_clarifies_live_failure_input(
    ensure_live_llm: None,
) -> None:
    """Content Anchor should ask what the unresolved demonstrative means."""

    del ensure_live_llm
    state = {
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
        "rag_result": {
            "answer": "",
            "user_image": {},
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {
                "unknown_slots": [],
                "loop_count": 0,
                "dispatched": [],
            },
        },
        "internal_monologue": "The referent is missing; ask what these refers to.",
        "logical_stance": "TENTATIVE",
        "character_intent": "CLARIFY",
        "conversation_progress": None,
    }

    started_at = perf_counter()
    result = await call_content_anchor_agent(state)
    duration_seconds = perf_counter() - started_at
    anchors = result["content_anchors"]
    trace_path = write_llm_trace(
        "cognition_referents_live",
        "content_anchor_live_failure_input",
        {
            "input": state,
            "output": result,
            "duration_seconds": duration_seconds,
            "judgment": (
                "E3 Content Anchor must turn unresolved referents into a "
                "narrow clarification question for the live failure input."
            ),
        },
    )

    logger.info(
        f"live_cognition_referents content_anchor trace={trace_path} "
        f"duration_seconds={duration_seconds:.3f} result={result!r}"
    )
    answer_anchors = [
        anchor for anchor in anchors if anchor.startswith("[ANSWER]")
    ]
    assert answer_anchors
    assert "这些" in " ".join(answer_anchors)
    assert all(not anchor.startswith("[FACT]") for anchor in anchors)
    assert duration_seconds < 30.0


@pytest.mark.asyncio
async def test_live_content_anchor_keeps_mixed_referent_question_narrow(
    ensure_live_llm: None,
) -> None:
    """Content Anchor should clarify only the unresolved part of mixed referents."""

    del ensure_live_llm
    state = {
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "他上次说的那些关于X的话是什么意思？",
        "referents": [
            {"phrase": "他", "referent_role": "subject", "status": "resolved"},
            {"phrase": "那些话", "referent_role": "object", "status": "unresolved"},
        ],
        "rag_result": {
            "answer": "检索找到关于这个人的上下文，但没有找到那些话的具体原文。",
            "user_image": {},
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [
                {
                    "summary": "用户之前提到过这个人。",
                    "content": "用户之前提到过这个人。",
                }
            ],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {
                "unknown_slots": [],
                "loop_count": 1,
                "dispatched": [],
            },
        },
        "internal_monologue": "The subject is known, but the quoted words are missing.",
        "logical_stance": "TENTATIVE",
        "character_intent": "CLARIFY",
        "conversation_progress": None,
    }

    started_at = perf_counter()
    result = await call_content_anchor_agent(state)
    duration_seconds = perf_counter() - started_at
    anchors = result["content_anchors"]
    trace_path = write_llm_trace(
        "cognition_referents_live",
        "content_anchor_mixed_referents",
        {
            "input": state,
            "output": result,
            "duration_seconds": duration_seconds,
            "judgment": (
                "E3 Content Anchor should narrow the clarification to the "
                "unresolved object while preserving resolved context."
            ),
        },
    )

    logger.info(
        f"live_cognition_referents mixed_content_anchor trace={trace_path} "
        f"duration_seconds={duration_seconds:.3f} result={result!r}"
    )
    answer_anchors = [
        anchor for anchor in anchors if anchor.startswith("[ANSWER]")
    ]
    answer_text = " ".join(answer_anchors)
    assert answer_anchors
    assert "话" in answer_text
    assert "哪" in answer_text or "具体" in answer_text
    assert all(not anchor.startswith("[FACT]") for anchor in anchors)
    assert duration_seconds < 30.0
