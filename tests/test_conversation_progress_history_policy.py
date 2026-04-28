"""Tests for Phase 2 raw-history exposure policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module

_ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    """Small LLM response object for patched prompt tests."""

    def __init__(self, payload: dict):
        self.content = json.dumps(payload)


class _CapturingLLM:
    """Capture messages sent to a patched LLM call."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return _FakeResponse(self.payload)


def _history(count: int = 8) -> list[dict]:
    """Build alternating user/assistant history fixtures.

    Args:
        count: Number of messages to return.

    Returns:
        Prompt-facing history list.
    """

    result = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        result.append({
            "role": role,
            "content": f"{role} message {index}",
            "platform_user_id": "user-1" if role == "user" else "bot-1",
        })
    return result


def _character_profile() -> dict:
    """Return the minimal profile required by L3 and dialog prompts.

    Args:
        None.

    Returns:
        Character profile fixture.
    """

    return {
        "name": "Kazusa",
        "mood": "Neutral",
        "global_vibe": "Calm",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "precise",
            "tempo": "measured",
            "defense": "guarded",
            "quirks": "dry",
            "taboos": "physical action narration",
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.4,
            "hesitation_density": 0.2,
            "counter_questioning": 0.2,
            "softener_density": 0.3,
            "formalism_avoidance": 0.6,
            "abstraction_reframing": 0.4,
            "direct_assertion": 0.6,
            "emotional_leakage": 0.3,
            "rhythmic_bounce": 0.2,
            "self_deprecation": 0.1,
        },
    }


def _base_l3_state() -> dict:
    """Build a reusable L3 state fixture.

    Args:
        None.

    Returns:
        Cognition-state subset for Contextual and Style agents.
    """

    return {
        "character_profile": _character_profile(),
        "user_profile": {"affinity": 700, "last_relationship_insight": "friendly task support"},
        "chat_history_recent": _history(),
        "internal_monologue": "answer directly",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
    }


def _payload_chars(value: object) -> int:
    """Return compact JSON character count for a prompt payload.

    Args:
        value: Prompt payload or message list.

    Returns:
        Serialized character count with CJK preserved.
    """

    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


@pytest.mark.asyncio
async def test_contextual_agent_receives_at_most_four_history_messages(monkeypatch) -> None:
    """Contextual Agent uses only the approved social surface window."""

    fake_llm = _CapturingLLM({
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "cooperative",
        "expression_willingness": "open",
    })
    monkeypatch.setattr(l3_module, "_contextual_agent_llm", fake_llm)

    await l3_module.call_contextual_agent(_base_l3_state())

    human_payload = json.loads(fake_llm.messages[1].content)
    assert len(human_payload["chat_history"]) == 4
    assert human_payload["chat_history"] == _history()[-4:]


@pytest.mark.asyncio
async def test_style_agent_receives_at_most_two_history_messages(monkeypatch) -> None:
    """Style Agent uses only the approved wording buffer."""

    fake_llm = _CapturingLLM({
        "rhetorical_strategy": "brief direct help",
        "linguistic_style": "short sentences",
        "forbidden_phrases": [],
    })
    monkeypatch.setattr(l3_module, "_style_agent_llm", fake_llm)

    await l3_module.call_style_agent(_base_l3_state())

    human_payload = json.loads(fake_llm.messages[1].content)
    assert len(human_payload["chat_history"]) == 2
    assert human_payload["chat_history"] == _history()[-2:]


@pytest.mark.asyncio
async def test_dialog_generator_tone_history_is_capped_to_two(monkeypatch) -> None:
    """Dialog Generator receives only the final local tone pair."""

    fake_llm = _CapturingLLM({"final_dialog": ["answer"]})
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    await dialog_module.dialog_generator({
        "internal_monologue": "answer",
        "action_directives": {
            "linguistic_directives": {
                "rhetorical_strategy": "direct",
                "linguistic_style": "brief",
                "accepted_user_preferences": [],
                "content_anchors": ["[DECISION] answer", "[SCOPE] short"],
                "forbidden_phrases": [],
            },
            "contextual_directives": {
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "calm",
                "relational_dynamic": "cooperative",
                "expression_willingness": "open",
            },
        },
        "chat_history_wide": _history(),
        "chat_history_recent": _history(),
        "platform_user_id": "user-1",
        "platform_bot_id": "bot-1",
        "user_name": "User",
        "user_profile": {"affinity": 700},
        "character_profile": _character_profile(),
        "messages": [],
    })

    human_payload = json.loads(fake_llm.messages[1].content)
    assert len(human_payload["tone_history"]) == 2
    assert human_payload["tone_history"] == _history()[-2:]


@pytest.mark.asyncio
async def test_dialog_evaluator_receives_last_user_message_only(monkeypatch) -> None:
    """Dialog Evaluator does not receive a raw history list."""

    fake_llm = _CapturingLLM({"feedback": "Passed", "should_stop": True})
    monkeypatch.setattr(dialog_module, "_dialog_evaluator_llm", fake_llm)

    await dialog_module.dialog_evaluator({
        "internal_monologue": "answer",
        "final_dialog": ["answer"],
        "action_directives": {
            "linguistic_directives": {
                "rhetorical_strategy": "direct",
                "linguistic_style": "brief",
                "accepted_user_preferences": [],
                "content_anchors": ["[DECISION] answer", "[SCOPE] short"],
                "forbidden_phrases": [],
            },
            "contextual_directives": {
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "calm",
                "relational_dynamic": "cooperative",
                "expression_willingness": "open",
            },
        },
        "chat_history_wide": _history(),
        "chat_history_recent": _history(),
        "platform_user_id": "user-1",
        "platform_bot_id": "bot-1",
        "character_profile": _character_profile(),
        "messages": [],
        "retry": 0,
    })

    human_payload = json.loads(fake_llm.messages[1].content)
    assert "chat_history" not in human_payload
    assert "tone_history" not in human_payload
    assert human_payload["last_user_message"] == "user message 6"


def test_phase2_workload_summary_records_context_budget() -> None:
    """Record affected LLM payload counts for the Phase 2 context budget."""

    history = _history()
    phase1 = {
        "contextual_history_messages": len(history),
        "style_history_messages": len(history),
        "dialog_generator_tone_messages": len(history),
    }
    phase2 = {
        "contextual_history_messages": len(l3_module._surface_history_for_contextual(history)),
        "style_history_messages": len(l3_module._surface_history_for_style(history)),
        "dialog_generator_tone_messages": len(dialog_module._tone_history_for_generator(history)),
        "content_anchor_progress_cap_chars": 5000,
        "content_anchor_raw_history_messages": 0,
        "dialog_evaluator_raw_history_messages": 0,
        "dialog_evaluator_last_user_message_only": True,
        "recorder_response_path_calls": 0,
        "recorder_runs_in_background": True,
        "new_response_path_llm_calls": 0,
    }
    summary = {
        "context_window_cap_tokens": 50000,
        "phase1_dynamic_payload_chars": {
            "contextual_history": _payload_chars(history),
            "style_history": _payload_chars(history),
            "dialog_generator_tone_history": _payload_chars(history),
        },
        "phase2_dynamic_payload_chars": {
            "contextual_history": _payload_chars(l3_module._surface_history_for_contextual(history)),
            "style_history": _payload_chars(l3_module._surface_history_for_style(history)),
            "dialog_generator_tone_history": _payload_chars(dialog_module._tone_history_for_generator(history)),
            "content_anchor_conversation_progress_cap": 5000,
            "dialog_evaluator_raw_history": 0,
            "recorder_response_path_payload": 0,
        },
        "phase1": phase1,
        "phase2": phase2,
    }

    output_path = _ROOT / "test_artifacts" / "conversation_progress_phase2_workload_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    assert phase2["new_response_path_llm_calls"] == 0
    assert phase2["contextual_history_messages"] <= 4
    assert phase2["style_history_messages"] <= 2
    assert phase2["dialog_generator_tone_messages"] <= 2
    assert phase2["content_anchor_raw_history_messages"] == 0
    assert phase2["dialog_evaluator_raw_history_messages"] == 0
    assert phase2["dialog_evaluator_last_user_message_only"]
    assert phase2["recorder_response_path_calls"] == 0
