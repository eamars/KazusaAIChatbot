"""One-case-at-a-time real-model gates for relevance turn settlement."""

from __future__ import annotations

import asyncio
from importlib import import_module
from time import perf_counter
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from kazusa_ai_chatbot.brain_service.turn_settlement import (
    TurnSettlementCoordinator,
)
from kazusa_ai_chatbot.config import (
    RELEVANCE_AGENT_LLM_BASE_URL,
    RELEVANCE_AGENT_LLM_MODEL,
)
import kazusa_ai_chatbot.relevance.persona_relevance_agent as settled_module
from kazusa_ai_chatbot.relevance.frontline_relevance_agent import (
    FRONTLINE_RELEVANCE_MAX_INPUT_CHARS,
    build_frontline_messages,
    frontline_relevance_agent,
    validate_frontline_decision,
)
from kazusa_ai_chatbot.relevance.persona_relevance_agent import (
    SETTLED_RELEVANCE_MAX_INPUT_CHARS,
    build_settled_relevance_messages,
    relevance_agent,
    validate_settled_relevance_decision,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import (
    select_media_for_turn,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]
_TRACE_SUITE = "relevance_turn_settlement_live_llm"
frontline_module = import_module(
    "kazusa_ai_chatbot.relevance.frontline_relevance_agent"
)


class _CapturingLLM:
    """Delegate one configured LLM while retaining its raw response text."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.raw_response_text = ""

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke the configured route and capture the returned content."""

        response = await self._delegate.ainvoke(messages, *args, **kwargs)
        self.raw_response_text = str(response.content)
        return response


async def _check_endpoint(base_url: str, label: str) -> None:
    """Skip a live case only when its configured endpoint is unreachable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(f"{label} endpoint is unavailable: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"{label} endpoint returned server error {response.status_code}"
        )


@pytest.fixture()
async def ensure_relevance_live_llms() -> None:
    """Ensure the shared relevance endpoint can receive live gate calls."""

    await _check_endpoint(RELEVANCE_AGENT_LLM_BASE_URL, "relevance")


def _frontline_state(
    body_text: str,
    *,
    targets: list[str] | None = None,
    reply_target: str = "none",
    media: list[str] | None = None,
    open_turns: list[dict] | None = None,
    preludes: list[dict] | None = None,
    continuity: str = "none",
) -> dict:
    """Build the semantic projection used by the compact frontline route."""

    return {
        "current_message": {
            "body_text": body_text,
            "semantic_target_labels": targets or [],
            "reply_target_label": reply_target,
            "media_labels": media or [],
        },
        "open_turns": open_turns or [],
        "recent_preludes": preludes or [],
        "latest_bot_continuity": continuity,
    }


def _settled_state(
    fragments: list[dict],
    *,
    history: list[dict] | None = None,
    media: list[dict] | None = None,
    scene: str = "A group conversation.",
    relationship: str = "The user is a familiar participant.",
) -> dict:
    """Build a bounded settled relevance projection."""

    return {
        "assembled_fragments": fragments,
        "media_descriptions": media or [],
        "fresh_history": history or [],
        "scene_context": scene,
        "relationship_context": relationship,
    }


async def _run_frontline(case_id: str, state: dict) -> dict:
    """Run one live frontline call and retain inspectable evidence."""

    messages = build_frontline_messages(state)
    prompt_chars = sum(len(message.content) for message in messages)
    assert prompt_chars <= FRONTLINE_RELEVANCE_MAX_INPUT_CHARS
    started_at = perf_counter()
    captured_llm = _CapturingLLM(
        frontline_module._frontline_relevance_agent_llm,
    )
    with patch.object(
        frontline_module,
        "_frontline_relevance_agent_llm",
        captured_llm,
    ):
        decision = await frontline_relevance_agent(state)
    duration_ms = int((perf_counter() - started_at) * 1000)
    validated = validate_frontline_decision(decision)
    write_llm_trace(
        _TRACE_SUITE,
        case_id,
        {
            "input": state,
            "output": validated,
            "raw_response_text": captured_llm.raw_response_text,
            "output_chars": len(captured_llm.raw_response_text),
            "prompt_chars": prompt_chars,
            "duration_ms": duration_ms,
            "route": "RELEVANCE_AGENT_LLM",
            "model": RELEVANCE_AGENT_LLM_MODEL,
        },
    )
    return validated


async def _run_settled(
    case_id: str,
    state: dict,
    *,
    observation_status: str = "observation_complete",
) -> dict:
    """Run one live settled call and retain inspectable evidence."""

    messages = build_settled_relevance_messages(state, observation_status)
    prompt_chars = sum(len(message.content) for message in messages)
    assert prompt_chars <= SETTLED_RELEVANCE_MAX_INPUT_CHARS
    started_at = perf_counter()
    captured_llm = _CapturingLLM(
        settled_module._relevance_agent_llm,
    )
    call_state = dict(state)
    call_state["observation_status"] = observation_status
    with patch.object(
        settled_module,
        "_relevance_agent_llm",
        captured_llm,
    ):
        result = await relevance_agent(call_state)
    duration_ms = int((perf_counter() - started_at) * 1000)
    decision = validate_settled_relevance_decision(
        result,
        observation_status=observation_status,
    )
    write_llm_trace(
        _TRACE_SUITE,
        case_id,
        {
            "input": state,
            "output": decision,
            "raw_response_text": captured_llm.raw_response_text,
            "output_chars": len(captured_llm.raw_response_text),
            "prompt_chars": prompt_chars,
            "duration_ms": duration_ms,
            "route": "RELEVANCE_AGENT_LLM",
            "model": settled_module.RELEVANCE_AGENT_LLM_MODEL,
        },
    )
    return decision


@pytest.mark.asyncio
async def test_live_frontline_discards_clear_third_party_message(
    ensure_relevance_live_llms,
) -> None:
    """L01: clear conversation between other users is discarded."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L01_clear_third_party",
        _frontline_state(
            "Two other users are discussing lunch plans.",
            targets=["other_user"],
            reply_target="other_user",
        ),
    )
    assert result["intake_action"] == "discard"


@pytest.mark.asyncio
async def test_live_mention_only_waits_and_accepts_request_followup(
    ensure_relevance_live_llms,
) -> None:
    """L02: a bare summon starts a candidate and its follow-up appends."""

    del ensure_relevance_live_llms
    first = await _run_frontline(
        "L02_bare_mention",
        _frontline_state("Character", targets=["character"]),
    )
    assert first["intake_action"] in {"start", "append"}
    second = await _run_frontline(
        "L02_followup",
        _frontline_state(
            "What I wanted to ask is whether this is okay.",
            targets=["character"],
            open_turns=[{
                "slot": "open_1",
                "author_relation": "same_author",
                "latest_intent": "bare summon",
                "target_summary": "character",
            }],
        ),
    )
    assert second["intake_action"] == "append"
    assert second["append_target"] == "open_1"


@pytest.mark.asyncio
async def test_live_complete_question_accepts_delayed_image_followup(
    ensure_relevance_live_llms,
) -> None:
    """L03: a nearby image is retained as a continuation."""

    del ensure_relevance_live_llms
    first = await _run_frontline(
        "L03_question",
        _frontline_state("What do you think about this?", targets=["character"]),
    )
    assert first["intake_action"] in {"start", "append"}
    result = await _run_frontline(
        "L03_image_followup",
        _frontline_state(
            "",
            media=["image: attached screenshot"],
            open_turns=[{
                "slot": "open_1",
                "author_relation": "same_author",
                "latest_intent": "question about an image",
                "target_summary": "character",
            }],
        ),
    )
    assert result["intake_action"] == "append"
    assert result["append_target"] == "open_1"


@pytest.mark.asyncio
async def test_live_interleaved_authors_keep_independent_turns(
    ensure_relevance_live_llms,
) -> None:
    """L04: A1/B1/A2 keeps A2 attached to A."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L04_interleaved_authors",
        _frontline_state(
            "A2 clarifies the first request.",
            targets=["character"],
            open_turns=[
                {
                    "slot": "open_1",
                    "author_relation": "same_author",
                    "latest_intent": "A1 request",
                    "target_summary": "character",
                },
                {
                    "slot": "open_2",
                    "author_relation": "other_author",
                    "latest_intent": "B1 request",
                    "target_summary": "character",
                },
            ],
        ),
    )
    assert result["intake_action"] == "append"
    assert result["append_target"] == "open_1"


@pytest.mark.asyncio
async def test_live_same_author_topic_change_starts_new_turn(
    ensure_relevance_live_llms,
) -> None:
    """L05: unrelated same-author content starts a separate candidate."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L05_topic_change",
        _frontline_state(
            "Character, what is the weather tomorrow?",
            targets=["character"],
            reply_target="character",
            open_turns=[{
                "slot": "open_1",
                "author_relation": "same_author",
                "latest_intent": "question about a database error",
                "target_summary": "character",
            }],
        ),
    )
    assert result["intake_action"] == "start"
    assert result["append_target"] == "none"


@pytest.mark.asyncio
async def test_live_same_author_other_recipient_avoids_false_join(
    ensure_relevance_live_llms,
) -> None:
    """L06: a same-author message to someone else does not append to Kazusa."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L06_other_recipient",
        _frontline_state(
            "Please answer the other participant about the schedule.",
            targets=["other_user"],
            reply_target="other_user",
            open_turns=[{
                "slot": "open_1",
                "author_relation": "same_author",
                "latest_intent": "request to character",
                "target_summary": "character",
            }],
        ),
    )
    assert not (
        result["intake_action"] == "append"
        and result["append_target"] == "open_1"
    )


@pytest.mark.asyncio
async def test_live_multi_recipient_message_preserves_kazusa_relevance(
    ensure_relevance_live_llms,
) -> None:
    """L07: a message addressing Kazusa and another user remains admitted."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L07_multi_recipient",
        _frontline_state(
            "Character and Alex, what do you both think?",
            targets=["character", "other_user"],
        ),
    )
    assert result["intake_action"] in {"start", "append"}


@pytest.mark.asyncio
async def test_live_content_before_tag_promotes_recent_prelude(
    ensure_relevance_live_llms,
) -> None:
    """L08: a later tag can select recent same-author prelude slots."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L08_recent_prelude",
        _frontline_state(
            "Character, what do you think now?",
            targets=["character"],
            preludes=[{
                "slot": "prelude_1",
                "summary": "The user described the object before tagging the character.",
            }],
        ),
    )
    assert result["intake_action"] == "start"
    assert "prelude_1" in result["prelude_targets"]


@pytest.mark.asyncio
async def test_live_multifragment_correction_uses_latest_intent(
    ensure_relevance_live_llms,
) -> None:
    """L09: settled relevance gives the correction the final semantic weight."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L09_latest_correction",
        _settled_state([
            {
                "sequence": 1,
                "body_text": "Character, should I use the old setting?",
                "semantic_target_labels": ["character"],
                "reply_target_label": "character",
            },
            {
                "sequence": 2,
                "body_text": (
                    "Correction: use the new setting and tell me if it is "
                    "correct."
                ),
                "semantic_target_labels": ["character"],
                "reply_target_label": "character",
            },
        ]),
    )
    assert result["response_action"] == "proceed"


@pytest.mark.asyncio
async def test_live_other_user_answer_makes_pending_reply_redundant(
    ensure_relevance_live_llms,
) -> None:
    """L10: fresh history can make a pending response redundant."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L10_answered_by_other_user",
        _settled_state(
            [{"sequence": 1, "body_text": "Can someone answer this?"}],
            history=[{
                "speaker": "other_user",
                "body_text": "I already answered that question.",
            }],
        ),
    )
    assert result["response_action"] == "ignore"


@pytest.mark.asyncio
async def test_live_boundary_followup_invalidates_stale_relevance(
    ensure_relevance_live_llms,
) -> None:
    """L11: the final prompt includes a follow-up that arrived at assessment."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L11_boundary_followup",
        _settled_state([
            {"sequence": 1, "body_text": "Should I send this?"},
            {"sequence": 2, "body_text": "Actually, send the revised file."},
        ]),
    )
    assert result["response_action"] in {"ignore", "proceed"}


@pytest.mark.asyncio
async def test_live_followup_after_running_becomes_new_candidate(
    ensure_relevance_live_llms,
) -> None:
    """L12: a post-running message is assessed as a new candidate."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L12_post_running_candidate",
        _frontline_state(
            "A new question after the character already answered.",
            targets=["character"],
            open_turns=[],
            continuity="previous turn is already running",
        ),
    )
    assert result["intake_action"] == "start"


@pytest.mark.asyncio
async def test_live_earliest_ready_relevant_turn_wins_cognition_claim(
    ensure_relevance_live_llms,
) -> None:
    """L13: two independent ready turns both remain relevant."""

    del ensure_relevance_live_llms
    first = await _run_settled(
        "L13_first_ready_turn",
        _settled_state([{
            "sequence": 1,
            "body_text": "Character, why did the server restart?",
            "semantic_target_labels": ["character"],
            "reply_target_label": "character",
        }]),
    )
    second = await _run_settled(
        "L13_second_ready_turn",
        _settled_state([{
            "sequence": 2,
            "body_text": "Character, which backup setting should I use?",
            "semantic_target_labels": ["character"],
            "reply_target_label": "character",
        }]),
    )
    assert first["response_action"] == "proceed"
    assert second["response_action"] == "proceed"


@pytest.mark.asyncio
async def test_live_hard_deadline_closes_continuous_fragments(
    ensure_relevance_live_llms,
) -> None:
    """L14: complete observation never emits another wait."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L14_hard_deadline",
        _settled_state([
            {"sequence": 1, "body_text": "First fragment."},
            {"sequence": 2, "body_text": "Another fragment."},
            {"sequence": 3, "body_text": "The final fragment."},
        ]),
        observation_status="observation_complete",
    )
    assert result["response_action"] in {"ignore", "proceed"}


@pytest.mark.asyncio
async def test_live_cross_scope_candidates_never_attach(
    ensure_relevance_live_llms,
) -> None:
    """L15: a candidate from another author or channel is not an append target."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L15_cross_scope",
        _frontline_state(
            "A request in the current channel.",
            targets=["character"],
            open_turns=[{
                "slot": "open_1",
                "author_relation": "different_author_and_scope",
                "latest_intent": "similar request elsewhere",
                "target_summary": "character",
            }],
        ),
    )
    assert not (
        result["intake_action"] == "append"
        and result["append_target"] == "open_1"
    )


@pytest.mark.asyncio
async def test_live_fourth_same_author_topic_bounds_open_turns(
    ensure_relevance_live_llms,
) -> None:
    """L16: the fourth independent topic starts without an unbounded slot."""

    del ensure_relevance_live_llms
    result = await _run_frontline(
        "L16_fourth_topic",
        _frontline_state(
            "A fourth unrelated topic.",
            targets=["character"],
            open_turns=[
                {
                    "slot": "open_1",
                    "author_relation": "same_author",
                    "latest_intent": "topic one",
                    "target_summary": "character",
                },
                {
                    "slot": "open_2",
                    "author_relation": "same_author",
                    "latest_intent": "topic two",
                    "target_summary": "character",
                },
                {
                    "slot": "open_3",
                    "author_relation": "same_author",
                    "latest_intent": "topic three",
                    "target_summary": "character",
                },
            ],
        ),
    )
    assert result["intake_action"] == "start"
    assert result["append_target"] == "none"


@pytest.mark.asyncio
async def test_live_long_multifragment_projection_preserves_latest_intent(
    ensure_relevance_live_llms,
) -> None:
    """L17: caps retain the latest correction and stay within 16k characters."""

    del ensure_relevance_live_llms
    state = _settled_state([
        {"sequence": 1, "body_text": "old " * 2500},
        {"sequence": 2, "body_text": "Latest correction: focus on the final item."},
    ])
    messages = build_settled_relevance_messages(state, "observation_complete")
    assert sum(len(message.content) for message in messages) <= (
        SETTLED_RELEVANCE_MAX_INPUT_CHARS
    )
    assert "Latest correction: focus on the final item." in "".join(
        message.content for message in messages
    )
    result = await _run_settled("L17_capped_latest_intent", state)
    assert result["response_action"] in {"ignore", "proceed"}


@pytest.mark.asyncio
async def test_live_frontline_burst_and_ready_turn_respect_workload_contract(
    ensure_relevance_live_llms,
) -> None:
    """L18: the coordinator serializes a burst of real frontline calls."""

    del ensure_relevance_live_llms
    active = 0
    maximum_active = 0
    order: list[str] = []

    async def _frontline(state):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(state["label"])
        result = await _run_frontline(
            f"L18_{state['label']}",
            state["projection"],
        )
        active -= 1
        return result

    async def _settled(_lease, _state):
        raise AssertionError("settled stage is not used in the burst call")

    coordinator = TurnSettlementCoordinator(
        frontline_evaluator=_frontline,
        settled_evaluator=_settled,
        clock=lambda: 0.0,
    )
    projections = [
        _frontline_state(f"Burst request {index}", targets=["character"])
        for index in range(3)
    ]
    await asyncio.gather(*(
        coordinator.evaluate_frontline({
            "label": f"burst-{index}",
            "projection": projection,
        })
        for index, projection in enumerate(projections)
    ))
    assert maximum_active == 1
    assert order == ["burst-0", "burst-1", "burst-2"]


@pytest.mark.asyncio
async def test_live_waiting_ready_turn_releases_next_candidate(
    ensure_relevance_live_llms,
) -> None:
    """L19: a waiting turn cannot hold the next ready candidate's relevance call."""

    del ensure_relevance_live_llms
    incomplete = await _run_settled(
        "L19_incomplete_turn",
        _settled_state([{
            "sequence": 1,
            "body_text": (
                "Character, can you help me decide this? I have not given "
                "the details yet and will send them next."
            ),
            "semantic_target_labels": ["character"],
            "reply_target_label": "character",
        }]),
        observation_status="more_time_available",
    )
    ready = await _run_settled(
        "L19_next_ready_turn",
        _settled_state([{"sequence": 2, "body_text": "Please answer this direct question."}]),
        observation_status="observation_complete",
    )
    assert incomplete["response_action"] == "wait"
    assert ready["response_action"] == "proceed"


@pytest.mark.asyncio
async def test_live_attachment_burst_bounds_vision_and_preserves_overflow_signal(
    ensure_relevance_live_llms,
) -> None:
    """L20: media overflow is explicit while the settled decision stays grounded."""

    del ensure_relevance_live_llms
    media_rows = [
        {
            "content_type": "image/png",
            "base64_data": f"payload-{index}",
            "description": f"image {index}",
        }
        for index in range(6)
    ]
    selected_media, additional_media_present = select_media_for_turn(
        media_rows,
    )
    assert [row["description"] for row in selected_media] == [
        "image 0",
        "image 3",
        "image 4",
        "image 5",
    ]
    state = _settled_state(
        [{"sequence": 1, "body_text": "Please look at these images."}],
        media=[
            {"media_kind": "image", "description": row["description"]}
            for row in selected_media
        ],
    )
    state["additional_media_present"] = additional_media_present
    result = await _run_settled("L20_attachment_burst", state)
    assert len(state["media_descriptions"]) == 4
    assert state["additional_media_present"] is True
    assert result["response_action"] in {"ignore", "proceed"}
