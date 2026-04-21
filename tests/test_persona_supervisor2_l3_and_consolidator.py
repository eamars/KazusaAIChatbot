from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.config import AFFINITY_RAW_DEAD_ZONE
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator as consolidator_module


class _DummyResponse:
    """Minimal async LLM response wrapper for unit tests."""

    def __init__(self, content: str):
        self.content = content


class _CapturingAsyncLLM:
    """Capture the last message list and return a fixed response payload."""

    def __init__(self, response_payload: dict):
        self.messages = None
        self._response_payload = response_payload

    async def ainvoke(self, messages):
        self.messages = messages
        return _DummyResponse(json.dumps(self._response_payload, ensure_ascii=False))


@pytest.mark.asyncio
async def test_call_linguistic_agent_sends_decontexualized_input_key(monkeypatch):
    """The L3 linguistic-agent payload should match the prompt's input-key contract."""
    llm = _CapturingAsyncLLM(
        {
            "rhetorical_strategy": "plain",
            "linguistic_style": "brief",
            "content_anchors": ["[DECISION] 接受", "[SCOPE] ~15字，说完[DECISION]即止"],
            "forbidden_phrases": [],
        }
    )
    monkeypatch.setitem(cognition_module.call_linguistic_agent.__globals__, "_linguistic_agent_llm", llm)

    state = {
        "character_profile": {
            "name": "Kazusa",
            "mood": "Neutral",
            "global_vibe": "Calm",
            "personality_brief": {
                "logic": "guarded",
                "tempo": "slow",
                "defense": "reserved",
                "quirks": "minimal",
                "taboos": "none",
            },
            "linguistic_texture_profile": {
                "fragmentation": 0.4,
                "hesitation_density": 0.4,
                "counter_questioning": 0.4,
                "softener_density": 0.4,
                "formalism_avoidance": 0.4,
                "abstraction_reframing": 0.4,
                "direct_assertion": 0.4,
                "emotional_leakage": 0.4,
                "rhythmic_bounce": 0.4,
                "self_deprecation": 0.4,
            },
        },
        "user_profile": {"last_relationship_insight": "neutral stranger"},
        "internal_monologue": "Just answer simply.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "research_facts": {},
        "chat_history": [],
        "decontexualized_input": "Tell me your opinion.",
    }

    await cognition_module.call_linguistic_agent(state)

    human_payload = json.loads(llm.messages[1].content)
    assert human_payload["decontexualized_input"] == "Tell me your opinion."
    assert "decontextualized_input" not in human_payload


@pytest.mark.asyncio
async def test_relationship_recorder_honors_skip(monkeypatch):
    """Recorder skip should force affinity delta back to zero."""
    llm = _CapturingAsyncLLM(
        {
            "skip": True,
            "diary_entry": ["没什么特别的。"],
            "affinity_delta": 4,
            "last_relationship_insight": "ordinary",
        }
    )
    monkeypatch.setattr(consolidator_module, "_relationship_recorder_llm", llm)

    state = {
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {"mbti": "INTJ"},
        },
        "user_name": "TestUser",
        "user_profile": {"affinity": 500},
        "internal_monologue": "Nothing much happened.",
        "emotional_appraisal": "Flat.",
        "interaction_subtext": "routine",
        "logical_stance": "CONFIRM",
    }

    result = await consolidator_module.relationship_recorder(state)

    assert result["affinity_delta"] == 0
    assert result["last_relationship_insight"] == "ordinary"


@pytest.mark.asyncio
async def test_relationship_recorder_invalid_affinity_delta_falls_back_to_zero(monkeypatch):
    """Malformed external LLM affinity deltas should not crash the recorder."""
    llm = _CapturingAsyncLLM(
        {
            "skip": False,
            "diary_entry": ["有点奇怪。"],
            "affinity_delta": "not-a-number",
            "last_relationship_insight": "unclear",
        }
    )
    monkeypatch.setattr(consolidator_module, "_relationship_recorder_llm", llm)

    state = {
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {"mbti": "INTJ"},
        },
        "user_name": "TestUser",
        "user_profile": {"affinity": 500},
        "internal_monologue": "The feeling is fuzzy.",
        "emotional_appraisal": "Unclear.",
        "interaction_subtext": "routine",
        "logical_stance": "TENTATIVE",
    }

    result = await consolidator_module.relationship_recorder(state)

    assert result["affinity_delta"] == 0


def test_process_affinity_delta_uses_dead_zone():
    """Small raw deltas inside the dead zone should not move affinity."""
    assert consolidator_module.process_affinity_delta(500, 0) == 0
    assert consolidator_module.process_affinity_delta(500, AFFINITY_RAW_DEAD_ZONE) == 0
    assert consolidator_module.process_affinity_delta(500, -AFFINITY_RAW_DEAD_ZONE) == 0


def test_process_affinity_delta_preserves_meaningful_change_outside_dead_zone():
    """Deltas outside the dead zone should still move affinity with preserved sign."""
    positive = consolidator_module.process_affinity_delta(500, AFFINITY_RAW_DEAD_ZONE + 1)
    negative = consolidator_module.process_affinity_delta(500, -(AFFINITY_RAW_DEAD_ZONE + 1))

    assert positive > 0
    assert negative < 0
