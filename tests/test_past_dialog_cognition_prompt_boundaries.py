"""Prompt boundary tests for past-dialog cognition residual."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_chain_core.stages import l1 as l1_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from llm_test_helpers import bind_test_llm


class _CapturingLLM:
    """Capture one stage call while returning a fixed JSON response."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.messages: list[list[object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config,
    ) -> SimpleNamespace:
        del config
        self.messages.append(messages)
        return SimpleNamespace(
            content=json.dumps(self._payload, ensure_ascii=False),
        )


def _episode() -> dict:
    turn_clock = build_turn_clock("2026-06-24 10:15:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="past-dialog-prompt-boundary-episode",
        percept_id="past-dialog-prompt-boundary-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="What did you mean in that earlier message?",
        platform="debug",
        platform_channel_id="channel-1",
        channel_type="private",
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=["row-current"],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    return episode


def _base_l2_state() -> dict:
    episode = _episode()
    return {
        "user_input": "What did you mean in that earlier message?",
        "prompt_message_context": {
            "body_text": "What did you mean in that earlier message?",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "reply_context": {
            "reply_to_message_id": "past-message-1",
            "reply_to_display_name": "Kazusa",
            "reply_excerpt": "I meant that the idea needed time.",
        },
        "user_name": "User",
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-1",
            "mood": "Neutral",
            "global_vibe": "Calm",
            "personality_brief": {"mbti": "INTJ"},
        },
        "local_time_context": episode["local_time_context"],
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "",
        },
        "cognitive_episode": episode,
        "decontexualized_input": (
            "The user asks about the meaning of an earlier Kazusa message."
        ),
        "rag_result": {
            "answer": "",
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "user_image": {},
        },
        "past_dialog_cognition_context": (
            "Prior private context: she was tentative, not making a promise."
        ),
        "indirect_speech_context": "",
        "emotional_appraisal": {"valence": "neutral"},
        "interaction_subtext": {"summary": "past-message clarification"},
    }


def _captured_payload(fake_llm: _CapturingLLM) -> dict:
    messages = fake_llm.messages[0]
    payload = json.loads(messages[1].content)
    return payload


@pytest.mark.asyncio
async def test_l2a_receives_past_dialog_cognition_context_only_in_human_payload() -> None:
    fake_llm = _CapturingLLM({
        "internal_monologue": "The prior message was tentative.",
        "logical_stance": "Clarify without overstating.",
        "character_intent": "Explain continuity.",
    })
    token = l2_module.set_conscious_llm(
        bind_test_llm(fake_llm, "conscious_llm"),
    )
    try:
        await l2_module.call_cognition_consciousness(_base_l2_state())
    finally:
        l2_module.reset_conscious_llm(token)

    payload = _captured_payload(fake_llm)
    assert payload["past_dialog_cognition_context"] == (
        "Prior private context: she was tentative, not making a promise."
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "trace-" not in serialized
    assert "raw_messages" not in serialized
    assert "raw_response_text" not in serialized
    assert "l2a_conscious_framing" not in serialized
    assert "l2c1_judgment_synthesis" not in serialized


@pytest.mark.asyncio
async def test_l2a_omits_empty_past_dialog_cognition_context() -> None:
    fake_llm = _CapturingLLM({
        "internal_monologue": "No prior private context is available.",
        "logical_stance": "Clarify from visible context.",
        "character_intent": "Answer carefully.",
    })
    state = _base_l2_state()
    state["past_dialog_cognition_context"] = ""
    token = l2_module.set_conscious_llm(
        bind_test_llm(fake_llm, "conscious_llm"),
    )
    try:
        await l2_module.call_cognition_consciousness(state)
    finally:
        l2_module.reset_conscious_llm(token)

    payload = _captured_payload(fake_llm)
    assert "past_dialog_cognition_context" not in payload


@pytest.mark.asyncio
async def test_l2c1_does_not_receive_past_dialog_cognition_context() -> None:
    fake_llm = _CapturingLLM({
        "logical_stance": "Clarify without overstating.",
        "character_intent": "Explain continuity.",
        "judgment_note": "The prior message was tentative.",
    })
    state = _base_l2_state()
    state.update({
        "referents": [],
        "internal_monologue": "The prior message was tentative.",
        "logical_stance": "Clarify without overstating.",
        "character_intent": "Explain continuity.",
        "boundary_core_assessment": {
            "boundary_issue": False,
            "boundary_summary": "No boundary issue.",
            "behavior_primary": "Clarify.",
            "behavior_secondary": "Avoid overclaiming.",
            "acceptance": "Allowed.",
            "stance_bias": "Clarify.",
            "identity_policy": "No identity pressure.",
            "pressure_policy": "No pressure.",
            "trajectory": "Normal clarification.",
        },
    })

    token = l2_module.set_judgement_core_llm(
        bind_test_llm(fake_llm, "judgement_core_llm"),
    )
    try:
        await l2_module.call_judgment_core_agent(state)
    finally:
        l2_module.reset_judgement_core_llm(token)

    payload = _captured_payload(fake_llm)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "past_dialog_cognition_context" not in serialized
    assert "Prior private context" not in serialized


def test_l1_system_prompt_does_not_reference_past_dialog_cognition_context() -> None:
    assert "past_dialog_cognition_context" not in (
        l1_module._COGNITION_SUBCONSCIOUS_PROMPT
    )
