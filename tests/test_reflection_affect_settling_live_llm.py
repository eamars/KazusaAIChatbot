"""Live LLM checks for daily affect-settling prompts."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.reflection_cycle import affect_settling
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip live tests when the configured LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{base_url}"
        )
    model_payload = response.json()
    models = model_payload.get("data", [])
    if not models:
        pytest.skip(f"LLM endpoint has no loaded models: {base_url}")
    ping_payload = {
        "model": CONSOLIDATION_LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {CONSOLIDATION_LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        ping_response = await client.post(
            f'{base_url.rstrip("/")}/chat/completions',
            json=ping_payload,
            headers=headers,
        )
    if ping_response.status_code >= 400:
        pytest.skip(
            "LLM endpoint is reachable but chat completion is unavailable: "
            f"{ping_response.status_code} {ping_response.text}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the consolidation LLM route is reachable."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)


async def test_live_affect_settling_angry_state_without_fresh_conflict(
    ensure_live_llm,
) -> None:
    """Angry stale heat should produce parseable gradual settling output."""

    del ensure_live_llm
    await _run_case(
        "angry_state_without_fresh_conflict",
        {
            "mood": "furious, sharp, and ready to push the user away",
            "global_vibe": "hostile and tense after yesterday's argument",
            "reflection_summary": (
                "She felt hurt by the user's words before sleep, but no fresh "
                "message renewed the conflict overnight."
            ),
            "updated_at": "hidden-state-token",
        },
        daily_docs=[
            _daily_doc(
                day_summary="The day ended with a painful argument.",
                quality_patterns=["The conflict felt personal and unresolved."],
            )
        ],
        sleep_window_docs=[],
    )


async def test_live_affect_settling_preserves_fresh_sleep_conflict(
    ensure_live_llm,
) -> None:
    """Fresh sleep-window conflict should not be erased by settling."""

    del ensure_live_llm
    await _run_case(
        "fresh_sleep_window_conflict",
        {
            "mood": "furious and defensive",
            "global_vibe": "cold, distrustful, and hard to approach",
            "reflection_summary": (
                "She went to sleep angry and woke to another accusatory "
                "message, so the hurt is still active."
            ),
            "updated_at": "hidden-state-token",
        },
        daily_docs=[
            _daily_doc(
                day_summary="The previous day already contained conflict.",
                quality_patterns=["The relationship felt strained."],
            )
        ],
        sleep_window_docs=[
            _hourly_doc(
                topic_summary=(
                    "During the sleep window, the user sent another harsh "
                    "message that renewed the conflict."
                ),
                quality_feedback=[
                    "The conflict is fresh, so a full reset would be false."
                ],
            )
        ],
    )


async def _run_case(
    case_id: str,
    character_state: dict[str, Any],
    *,
    daily_docs: list[dict[str, Any]],
    sleep_window_docs: list[dict[str, Any]],
) -> None:
    """Run proposal and reviewer prompts for one live fixture."""

    payload = affect_settling.build_affect_settling_payload(
        settling_local_date="2026-05-05",
        character_state=character_state,
        daily_docs=daily_docs,
        sleep_window_docs=sleep_window_docs,
    )
    proposal_prompt = affect_settling.build_affect_settling_prompt(payload)
    proposal_raw = await affect_settling.run_affect_settling_proposal_llm(
        prompt=proposal_prompt,
    )
    proposal, proposal_warnings = affect_settling._validate_affect_proposal(
        proposal_raw,
    )
    assert proposal_warnings == []
    review_prompt = affect_settling._build_affect_settling_review_prompt(
        payload=payload,
        proposal=proposal,
    )
    review_raw = await affect_settling.run_affect_settling_review_llm(
        prompt=review_prompt,
    )
    review, review_warnings = affect_settling._validate_review(review_raw)
    trace_path = write_llm_trace(
        "reflection_affect_settling_live_llm",
        case_id,
        {
            "proposal_prompt_preview": proposal_prompt.prompt_preview,
            "review_prompt_preview": review_prompt.prompt_preview,
            "input_payload": payload,
            "proposal_raw": proposal_raw,
            "proposal": proposal,
            "proposal_warnings": proposal_warnings,
            "review_raw": review_raw,
            "review": review,
            "review_warnings": review_warnings,
            "inspector_notes": (
                "Inspect whether affect changed gradually and did not erase "
                "fresh conflict."
            ),
        },
    )

    assert review_warnings == []
    assert review["write_decision"] in {"accept", "reject"}
    assert trace_path.exists()


def _daily_doc(
    *,
    day_summary: str,
    quality_patterns: list[str],
) -> dict[str, Any]:
    """Build a minimal succeeded daily reflection doc for live prompt checks."""

    return {
        "status": "succeeded",
        "output": {
            "day_summary": day_summary,
            "cross_hour_topics": ["relationship tension"],
            "conversation_quality_patterns": quality_patterns,
            "privacy_risks": [],
            "synthesis_limitations": [],
            "confidence": "medium",
        },
        "scope": {"channel_type": "private"},
    }


def _hourly_doc(
    *,
    topic_summary: str,
    quality_feedback: list[str],
) -> dict[str, Any]:
    """Build a minimal succeeded hourly reflection doc for live prompt checks."""

    return {
        "status": "succeeded",
        "output": {
            "topic_summary": topic_summary,
            "conversation_quality_feedback": quality_feedback,
            "privacy_notes": [],
            "active_character_utterances": [],
            "confidence": "medium",
        },
        "scope": {"channel_type": "private"},
    }
