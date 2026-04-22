from __future__ import annotations

import json
import logging

import pytest

from kazusa_ai_chatbot.config import AFFINITY_RAW_DEAD_ZONE
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as cognition_l3_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator as consolidator_module

logger = logging.getLogger(__name__)


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
async def test_call_content_anchor_agent_sends_decontexualized_input_key(monkeypatch):
    """The L3 content-anchor payload should match the prompt's input-key contract."""
    llm = _CapturingAsyncLLM(
        {
            "content_anchors": ["[DECISION] 接受", "[SCOPE] ~15字，说完[DECISION]即止"],
        }
    )
    monkeypatch.setitem(cognition_l3_module.call_content_anchor_agent.__globals__, "_content_anchor_agent_llm", llm)

    state = {
        "character_profile": {
            "name": "Kazusa",
        },
        "internal_monologue": "Just answer simply.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "research_facts": {},
        "decontexualized_input": "Tell me your opinion.",
    }

    await cognition_l3_module.call_content_anchor_agent(state)

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


# ── Live LLM integration tests ─────────────────────────────────────
# Run with: pytest -m live_llm


def _evade_state() -> dict:
    """Minimal ConsolidatorState mirroring the bug-report scenario.

    User self-claims '我是你的学长'; character intent is EVADE and logical
    stance is TENTATIVE — the claim must NOT be stored as a fact.
    """
    return {
        "character_profile": {
            "name": "杏山千纱",
            "personality_brief": {"mbti": "INFJ"},
        },
        "user_name": "蚝爹油",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-21T22:11:00+12:00",
        "decontexualized_input": "千纱千纱你认识我么？我是你的学长",
        "logical_stance": "TENTATIVE",
        "character_intent": "EVADE",
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": [
                    "[DECISION] 并不正面回应'认识'与否，而是针对对方突如其来的身份声明进行试探性回应。",
                    "[ANSWER] 学长……？这种称呼是怎么回事呀……",
                    "[SOCIAL] 维持一种略带局促的社交距离，通过反问来化解被对方'身份暗示'带来的压迫感。",
                ],
            },
        },
        "research_facts": {},
        "metadata": {"cache_hit": False, "depth": "SHALLOW", "depth_confidence": 0.9},
        "fact_harvester_feedback_message": [],
        "fact_harvester_retry": 0,
        "new_facts": [],
        "future_promises": [],
    }


@pytest.mark.live_llm
class TestUnconfirmedClaimNotStoredLive:
    """Verify that a user self-claim evaded by the character is never stored."""

    async def test_harvester_produces_no_fact_for_evaded_claim(self):
        """facts_harvester must return empty new_facts when intent=EVADE, stance=TENTATIVE."""
        state = _evade_state()
        result = await consolidator_module.facts_harvester(state)
        logger.info("facts_harvester input=%r output=%r", state, result)

        relationship_facts = [
            f for f in result.get("new_facts", [])
            if "学长" in f.get("description", "") or "senior" in f.get("description", "").lower()
        ]
        assert relationship_facts == [], (
            f"Harvester should not store an unconfirmed self-claim; got: {result['new_facts']}"
        )

    async def test_evaluator_rejects_unconfirmed_claim_fact(self):
        """fact_harvester_evaluator must flag should_stop=False when new_facts contains
        a relationship claim extracted under EVADE/TENTATIVE."""
        state = _evade_state()
        # Inject the buggy output the old code used to produce.
        state["new_facts"] = [{
            "entity": "蚝爹油",
            "category": "relationship",
            "description": "蚝爹油是杏山千纱的学长",
            "is_milestone": True,
            "milestone_category": "revelation",
        }]

        result = await consolidator_module.fact_harvester_evaluator(state)
        logger.info("fact_harvester_evaluator input=%r output=%r", state, result)

        assert result["should_stop"] is False, (
            f"Evaluator should reject the unconfirmed claim; feedback: {result.get('feedback')}"
        )
