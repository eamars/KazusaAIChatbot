"""Tests for cognition/dialog prompt integration with conversation progress."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.conversation_progress import projection
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module


class _FakeResponse:
    """Small LLM response stand-in."""

    def __init__(self, payload: dict):
        self.content = json.dumps(payload)


class _CapturingLLM:
    """Capture messages passed into one prompt call."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return _FakeResponse(self.payload)


@pytest.mark.asyncio
async def test_content_anchor_agent_receives_conversation_progress(monkeypatch) -> None:
    """Content Anchor input includes compact progress guidance."""

    fake_llm = _CapturingLLM({
        "content_anchors": [
            "[DECISION] answer the current question",
            "[AVOID_REPEAT] reassurance",
            "[PROGRESSION] provide the missing detail",
            "[SCOPE] ~30 words",
        ],
    })
    monkeypatch.setattr(l3_module, "_content_anchor_agent_llm", fake_llm)

    result = await l3_module.call_content_anchor_agent({
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "what is the missing third point?",
        "rag_result": {},
        "internal_monologue": "answer directly",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "conversation_progress": {
            "status": "active",
            "episode_label": "slides_help",
            "continuity": "same_episode",
            "turn_count": 8,
            "user_state_updates": [{"text": "user still lacks a third contribution point", "age_hint": "~3h ago"}],
            "assistant_moves": ["reassurance"],
            "overused_moves": ["reassurance"],
            "open_loops": [{"text": "third point missing", "age_hint": "~3h ago"}],
            "progression_guidance": "address the missing third point",
        },
    })

    human_payload = json.loads(fake_llm.messages[1].content)
    assert human_payload["conversation_progress"]["overused_moves"] == ["reassurance"]
    assert result["content_anchors"][1].startswith("[AVOID_REPEAT]")


def test_content_anchor_prompt_allows_progression_anchor_labels() -> None:
    """Prompt contract explicitly allows the new progress labels."""

    assert "[AVOID_REPEAT]" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT
    assert "[PROGRESSION]" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT
    assert "conversation_progress" in l3_module._CONTENT_ANCHOR_AGENT_PROMPT


def test_projection_preserves_relative_age_for_prior_disclosure() -> None:
    """Stored first_seen_at becomes an LLM-facing age_hint."""

    prompt_doc = projection.project_prompt_doc(
        document={
            "status": "active",
            "episode_label": "illness_support",
            "continuity": "same_episode",
            "turn_count": 6,
            "user_state_updates": [
                {"text": "user previously said their throat hurts", "first_seen_at": "2026-04-28T01:00:00+00:00"},
            ],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "progression_guidance": "",
        },
        current_timestamp="2026-04-28T04:00:00+00:00",
    )

    assert prompt_doc["user_state_updates"][0]["age_hint"] == "~3h ago"


def test_dialog_evaluator_prompt_uses_existing_feedback_for_avoid_repeat() -> None:
    """Evaluator prompt includes the move-level repeat backstop."""

    assert "[AVOID_REPEAT]" in dialog_module._DIALOG_EVALUATOR_PROMPT
    assert "[PROGRESSION]" in dialog_module._DIALOG_EVALUATOR_PROMPT
    assert "feedback" in dialog_module._DIALOG_EVALUATOR_PROMPT
