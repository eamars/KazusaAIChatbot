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
    PersistedChatFragment,
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
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontextualizer import (
    select_media_for_turn,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]
_TRACE_SUITE = "relevance_turn_settlement_live_llm"
frontline_module = import_module(
    "kazusa_ai_chatbot.relevance.frontline_relevance_agent"
)


class _ManualClock:
    """Small monotonic clock for live coordinator gates."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _fragment(
    sequence: int,
    *,
    author: str = "author-a",
    channel: str = "channel-1",
    body: str = "request",
    enqueue_monotonic: float = 0.0,
    targets: tuple[str, ...] = ("character",),
    reply_target: str = "character",
) -> PersistedChatFragment:
    """Build a production-shaped persisted fragment for live gates."""

    return PersistedChatFragment(
        arrival_sequence=sequence,
        scope=("debug", channel, "group"),
        author_platform_user_id=author,
        author_global_user_id=f"global-{author}",
        platform_message_id=f"message-{sequence}",
        conversation_row_id=f"row-{sequence}",
        storage_timestamp_utc="2026-07-16T00:00:00+00:00",
        enqueue_monotonic=enqueue_monotonic,
        body_text=body,
        semantic_target_labels=targets,
        reply_target_label=reply_target,
    )


def _start_decision() -> dict:
    """Return the deterministic candidate-start action used for setup."""

    return {
        "intake_action": "start",
        "append_target": "none",
        "prelude_targets": [],
        "reason": "live gate setup",
    }


def _coordinator(
    clock: _ManualClock,
    *,
    frontline_evaluator=None,
    settled_evaluator=None,
) -> TurnSettlementCoordinator:
    """Build a coordinator whose unused roles fail loudly."""

    async def _unused_frontline(_state):
        raise AssertionError("frontline evaluator is not used in this gate")

    async def _unused_settled(_lease, _state):
        raise AssertionError("settled evaluator is not used in this gate")

    return TurnSettlementCoordinator(
        frontline_evaluator=frontline_evaluator or _unused_frontline,
        settled_evaluator=settled_evaluator or _unused_settled,
        clock=clock,
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
        "conversation_scope": "group",
        "active_character_name": "Character",
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
    conversation_scope: str = "group",
    group_attention: str = "",
) -> dict:
    """Build a bounded settled relevance projection."""

    return {
        "conversation_scope": conversation_scope,
        "active_character_name": "Character",
        "assembled_fragments": fragments,
        "media_descriptions": media or [],
        "fresh_history": history or [],
        "scene_context": scene,
        "relationship_context": relationship,
        "group_attention": group_attention,
    }


def _settled_state_from_lease(lease) -> dict:
    """Project a coordinator lease into the live settled-agent fixture."""

    return _settled_state([
        {
            "body_text": fragment.body_text,
            "semantic_target_labels": list(
                fragment.semantic_target_labels
            ),
            "reply_target_label": fragment.reply_target_label,
            "media_labels": list(fragment.media_labels),
        }
        for fragment in lease.fragments
    ])


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
            targets=["other_participant"],
            reply_target="other_participant",
        ),
    )
    assert result["intake_action"] == "discard"


@pytest.mark.asyncio
async def test_live_mention_only_settles_safely_and_accepts_followup(
    ensure_relevance_live_llms,
) -> None:
    """L02: a bare summon waits or discards safely before its full request."""

    del ensure_relevance_live_llms
    clock = _ManualClock()
    coordinator = _coordinator(clock)
    bare_mention = _fragment(
        1,
        body="Character",
        targets=("character",),
        reply_target="none",
    )
    bare_state = await coordinator.build_frontline_state(bare_mention)
    bare_state["active_character_name"] = "Character"
    first = await _run_frontline("L02_bare_mention", bare_state)
    assert first["intake_action"] in {"start", "discard"}
    await coordinator.apply_frontline_decision(bare_mention, first)
    followup = _fragment(
        2,
        body="What I wanted to ask is whether this is okay.",
        enqueue_monotonic=8.414,
        targets=("character",),
        reply_target="none",
    )
    followup_state = await coordinator.build_frontline_state(followup)
    followup_state["active_character_name"] = "Character"
    if first["intake_action"] == "start":
        assert len(followup_state["open_turns"]) == 1
    else:
        assert followup_state["open_turns"] == []
        assert len(followup_state["recent_preludes"]) == 1
    second = await _run_frontline(
        "L02_followup",
        followup_state,
    )
    if first["intake_action"] == "start":
        assert second["intake_action"] == "append"
        assert second["append_target"] == "open_1"
    else:
        assert second["intake_action"] == "start"


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
    assert first["intake_action"] == "start"
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
    clock = _ManualClock()
    coordinator = _coordinator(clock)
    await coordinator.apply_frontline_decision(
        _fragment(1, body="A1 request"),
        _start_decision(),
    )
    await coordinator.apply_frontline_decision(
        _fragment(2, author="author-b", body="B1 request"),
        _start_decision(),
    )
    state = await coordinator.build_frontline_state(
        _fragment(3, body="A2 clarifies the first request", enqueue_monotonic=2.0),
    )
    state["active_character_name"] = "Character"
    assert len(state["open_turns"]) == 1

    result = await _run_frontline(
        "L04_interleaved_authors",
        state,
    )
    assert result["intake_action"] == "append"
    assert result["append_target"] == "open_1"


@pytest.mark.asyncio
async def test_live_same_author_topic_change_never_false_appends(
    ensure_relevance_live_llms,
) -> None:
    """L05: unrelated same-author content never contaminates an open turn."""

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
    assert result["intake_action"] in {"start", "discard"}
    assert result["intake_action"] != "append"


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
            targets=["other_participant"],
            reply_target="other_participant",
            open_turns=[{
                "slot": "open_1",
                "author_relation": "same_author",
                "latest_intent": "request to character",
                "target_summary": "character",
            }],
        ),
    )
    assert result["intake_action"] == "discard"


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
            targets=["character", "other_participant"],
        ),
    )
    assert result["intake_action"] == "start"


@pytest.mark.asyncio
async def test_live_content_before_tag_promotes_recent_prelude(
    ensure_relevance_live_llms,
) -> None:
    """L08: a later tag can select recent same-author prelude slots."""

    del ensure_relevance_live_llms
    clock = _ManualClock()
    coordinator = _coordinator(clock)
    await coordinator.apply_frontline_decision(
        _fragment(
            1,
            body="The user described the object before tagging the character.",
            targets=("none",),
            reply_target="none",
        ),
        {
            "intake_action": "discard",
            "append_target": "none",
            "prelude_targets": [],
            "reason": "silent prelude",
        },
    )
    current = _fragment(
        2,
        body="Character, what do you think now?",
        enqueue_monotonic=4.0,
    )
    state = await coordinator.build_frontline_state(current)
    state["active_character_name"] = "Character"
    result = await _run_frontline(
        "L08_recent_prelude",
        state,
    )
    assert result["intake_action"] == "start"
    assert "prelude_1" in result["prelude_targets"]
    await coordinator.apply_frontline_decision(current, result)
    clock.value = 10.0
    lease = await coordinator.wait_for_assessment_ready()
    assert [fragment.arrival_sequence for fragment in lease.fragments] == [1, 2]
    assert lease.response_owner_sequence == 2


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
    clock = _ManualClock()

    async def _settled(lease, state):
        del state
        return await _run_settled(
            f"L11_assessment_v{lease.version}",
            _settled_state_from_lease(lease),
            observation_status=lease.observation_status,
        )

    coordinator = _coordinator(clock, settled_evaluator=_settled)
    start = await coordinator.apply_frontline_decision(
        _fragment(1, body="Should I send this?"),
        _start_decision(),
    )
    clock.value = 6.0
    first_lease = await coordinator.wait_for_assessment_ready()
    first_decision = await coordinator.evaluate_settled(first_lease, {})
    append = await coordinator.apply_frontline_decision(
        _fragment(
            2,
            body="Actually, send the revised file.",
            enqueue_monotonic=6.0,
        ),
        {
            "intake_action": "append",
            "append_target": "open_1",
            "prelude_targets": [],
            "reason": "newest correction",
        },
    )
    stale = await coordinator.apply_settled_decision(
        first_lease,
        first_decision,
    )
    assert stale.stale is True
    clock.value = 10.0
    final_lease = await coordinator.wait_for_assessment_ready()
    result = await coordinator.evaluate_settled(final_lease, {})

    assert final_lease.turn_id == start.turn_id
    assert final_lease.version == append.version
    assert len(final_lease.fragments) == 2
    assert result["response_action"] in {"ignore", "proceed"}


@pytest.mark.asyncio
async def test_live_followup_after_running_becomes_new_candidate(
    ensure_relevance_live_llms,
) -> None:
    """L12: a post-running message is assessed as a new candidate."""

    del ensure_relevance_live_llms
    clock = _ManualClock()
    coordinator = _coordinator(clock)
    await coordinator.record_bot_continuity(
        scope=("debug", "channel-1", "group"),
        author_platform_user_id="author-a",
        dialog_text="The character already answered the previous turn.",
    )
    state = await coordinator.build_frontline_state(
        _fragment(
            2,
            body="A new question after the character already answered.",
            enqueue_monotonic=7.0,
        ),
    )
    state["active_character_name"] = "Character"
    assert state["open_turns"] == []
    assert "already answered" in state["latest_bot_continuity"]

    result = await _run_frontline(
        "L12_post_running_candidate",
        state,
    )
    assert result["intake_action"] == "start"


@pytest.mark.asyncio
async def test_live_earliest_ready_relevant_turn_wins_cognition_claim(
    ensure_relevance_live_llms,
) -> None:
    """L13: two independent ready turns both remain relevant."""

    del ensure_relevance_live_llms
    clock = _ManualClock()

    async def _settled(lease, state):
        del state
        return await _run_settled(
            f"L13_turn_{lease.leader_sequence}",
            _settled_state_from_lease(lease),
            observation_status=lease.observation_status,
        )

    coordinator = _coordinator(clock, settled_evaluator=_settled)
    first_turn = await coordinator.apply_frontline_decision(
        _fragment(1, body="Character, why did the server restart?"),
        _start_decision(),
    )
    await coordinator.apply_frontline_decision(
        _fragment(
            2,
            author="author-b",
            body="Character, which backup setting should I use?",
        ),
        _start_decision(),
    )
    clock.value = 6.0
    first_lease = await coordinator.wait_for_assessment_ready()
    first = await coordinator.evaluate_settled(first_lease, {})
    await coordinator.apply_settled_decision(first_lease, first)
    assert await coordinator.claim_for_cognition(
        first_lease.turn_id,
        first_lease.version,
    ) is True
    second_lease = await coordinator.wait_for_assessment_ready()
    second = await coordinator.evaluate_settled(second_lease, {})

    assert first_lease.turn_id == first_turn.turn_id
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
    assert result["response_action"] == "ignore"


@pytest.mark.asyncio
async def test_live_cross_scope_candidates_never_attach(
    ensure_relevance_live_llms,
) -> None:
    """L15: a candidate from another author or channel is not an append target."""

    del ensure_relevance_live_llms
    clock = _ManualClock()
    coordinator = _coordinator(clock)
    await coordinator.apply_frontline_decision(
        _fragment(1, author="other-author", channel="other-channel"),
        _start_decision(),
    )
    state = await coordinator.build_frontline_state(
        _fragment(2, body="A request in the current channel."),
    )
    state["active_character_name"] = "Character"
    assert state["open_turns"] == []

    result = await _run_frontline(
        "L15_cross_scope",
        state,
    )
    assert result["intake_action"] != "append"


@pytest.mark.asyncio
async def test_live_fourth_same_author_topic_avoids_false_append(
    ensure_relevance_live_llms,
) -> None:
    """L16: a fourth topic never contaminates one of three bounded slots."""

    del ensure_relevance_live_llms
    clock = _ManualClock()
    coordinator = _coordinator(clock)
    for sequence in range(1, 4):
        await coordinator.apply_frontline_decision(
            _fragment(sequence, body=f"topic {sequence}"),
            _start_decision(),
        )
    state = await coordinator.build_frontline_state(
        _fragment(4, body="A fourth unrelated topic.", enqueue_monotonic=1.0),
    )
    state["active_character_name"] = "Character"
    assert len(state["open_turns"]) == 3

    result = await _run_frontline(
        "L16_fourth_topic",
        state,
    )
    assert result["intake_action"] in {"start", "discard"}
    assert result["intake_action"] != "append"


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
    assert result["response_action"] == "ignore"


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

    async def _settled(lease, state):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append("settled-ready")
        result = await _run_settled(
            "L18_settled_ready",
            dict(state),
            observation_status=lease.observation_status,
        )
        active -= 1
        return result

    clock = _ManualClock()
    coordinator = _coordinator(
        clock,
        frontline_evaluator=_frontline,
        settled_evaluator=_settled,
    )
    await coordinator.apply_frontline_decision(
        _fragment(20, body="Character, assess this ready request."),
        _start_decision(),
    )
    clock.value = 10.0
    lease = await coordinator.wait_for_assessment_ready()
    projections = [
        _frontline_state(f"Burst request {index}", targets=["character"])
        for index in range(3)
    ]
    tasks = []
    for index, projection in enumerate(projections):
        tasks.append(asyncio.create_task(coordinator.evaluate_frontline({
            "label": f"burst-{index}",
            "projection": projection,
        })))
        if index == 0:
            await asyncio.sleep(0)
            tasks.append(asyncio.create_task(coordinator.evaluate_settled(
                lease,
                _settled_state([{
                    "body_text": "Character, assess this ready request.",
                    "semantic_target_labels": ["character"],
                    "reply_target_label": "character",
                }]),
            )))
    await asyncio.gather(*tasks)
    assert maximum_active == 1
    assert order == [
        "burst-0",
        "settled-ready",
        "burst-1",
        "burst-2",
    ]


@pytest.mark.asyncio
async def test_live_waiting_ready_turn_releases_next_candidate(
    ensure_relevance_live_llms,
) -> None:
    """L19: a waiting turn cannot hold the next ready candidate's relevance call."""

    del ensure_relevance_live_llms
    clock = _ManualClock()

    async def _settled(lease, state):
        del state
        return await _run_settled(
            f"L19_turn_{lease.leader_sequence}",
            _settled_state_from_lease(lease),
            observation_status=lease.observation_status,
        )

    coordinator = _coordinator(clock, settled_evaluator=_settled)
    await coordinator.apply_frontline_decision(
        _fragment(
            1,
            body=(
                "Character, can you help me decide this? I have not given "
                "the details yet and will send them next."
            ),
        ),
        _start_decision(),
    )
    await coordinator.apply_frontline_decision(
        _fragment(
            2,
            author="author-b",
            body="Character, which backup should I use before the restart?",
        ),
        _start_decision(),
    )
    clock.value = 6.0
    incomplete_lease = await coordinator.wait_for_assessment_ready()
    incomplete = await coordinator.evaluate_settled(incomplete_lease, {})
    await coordinator.apply_settled_decision(incomplete_lease, incomplete)
    ready_lease = await coordinator.wait_for_assessment_ready()
    ready = await coordinator.evaluate_settled(ready_lease, {})

    assert incomplete["response_action"] == "wait"
    assert ready_lease.leader_sequence == 2
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
    assert result["response_action"] == "ignore"


@pytest.mark.asyncio
async def test_live_specific_group_question_requests_native_reply_anchor(
    ensure_relevance_live_llms,
) -> None:
    """L21: a specific direct question in a noisy group requests anchoring."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L21_specific_group_reply_anchor",
        _settled_state(
            [{
                "body_text": (
                    "Character, can you check whether the deployment window "
                    "is still 9 PM?"
                ),
                "semantic_target_labels": ["character"],
                "reply_target_label": "none",
            }],
            scene="A busy operations group with several active discussions.",
            group_attention="high_noise",
        ),
    )

    assert result["response_action"] == "proceed"
    assert result["use_reply_feature"] is True


@pytest.mark.asyncio
async def test_live_private_question_avoids_native_reply_anchor(
    ensure_relevance_live_llms,
) -> None:
    """L22: ordinary private conversation proceeds without native anchoring."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L22_private_without_reply_anchor",
        _settled_state(
            [{
                "body_text": "Can you help me check tonight's schedule?",
                "semantic_target_labels": [],
                "reply_target_label": "none",
            }],
            scene="A private conversation with the character.",
            conversation_scope="private",
        ),
    )

    assert result["response_action"] == "proceed"
    assert result["use_reply_feature"] is False


@pytest.mark.asyncio
async def test_live_whole_group_invitation_avoids_native_reply_anchor(
    ensure_relevance_live_llms,
) -> None:
    """L23: a whole-group invitation proceeds without a personal anchor."""

    del ensure_relevance_live_llms
    result = await _run_settled(
        "L23_group_invitation_without_reply_anchor",
        _settled_state(
            [{
                "body_text": (
                    "Everyone, which release option should we choose tonight?"
                ),
                "semantic_target_labels": ["broadcast"],
                "reply_target_label": "none",
            }],
            scene="A release-planning group conversation.",
            group_attention="medium_noise",
        ),
    )

    assert result["response_action"] == "proceed"
    assert result["use_reply_feature"] is False
