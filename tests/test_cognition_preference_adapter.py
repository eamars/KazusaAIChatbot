from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import call_preference_adapter


class _FakePreferenceAdapterLlm:
    """Small async fake for the preference adapter LLM."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: list) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


class _CapturingPreferenceAdapterLlm:
    """Async fake that captures the preference adapter prompt payload."""

    def __init__(self, content: str) -> None:
        """Create the fake with a fixed response."""

        self._content = content
        self.payload: dict | None = None

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        """Capture the human JSON payload and return the configured content."""

        self.payload = json.loads(messages[1].content)
        return_value = SimpleNamespace(content=self._content)
        return return_value


def _preference_state() -> dict:
    """Build the minimal state consumed by ``call_preference_adapter``."""

    return {
        "decontexualized_input": "please keep replies short",
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "milestones": [],
                    "active_commitments": [],
                },
            },
            "user_memory_unit_candidates": [],
        },
        "user_profile": {},
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {"taboos": []},
        },
        "internal_monologue": "The request is harmless.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "linguistic_style": "concise",
        "content_anchors": [],
    }


@pytest.mark.asyncio
async def test_preference_adapter_accepts_string_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preference adapter preserves native string preference items."""

    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3._preference_adapter_llm",
        _FakePreferenceAdapterLlm('{"accepted_user_preferences":[" concise replies ", "soft tone"]}'),
    )

    result = await call_preference_adapter(_preference_state())

    assert result["accepted_user_preferences"] == ["concise replies", "soft tone"]


@pytest.mark.asyncio
async def test_preference_adapter_does_not_stringify_container_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preference adapter must not turn dict/list preference items into repr text."""

    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3._preference_adapter_llm",
        _FakePreferenceAdapterLlm(
            '{"accepted_user_preferences":[{"text":"do not stringify"},["bad"]," keep me "]}'
        ),
    )

    result = await call_preference_adapter(_preference_state())

    assert result["accepted_user_preferences"] == ["keep me"]


@pytest.mark.asyncio
async def test_preference_adapter_preserves_commitment_over_style_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted commitments remain higher authority than style guidance."""

    fake_llm = _CapturingPreferenceAdapterLlm(
        '{"accepted_user_preferences":["讨论工作时尽量使用更短的句子。"]}'
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3._preference_adapter_llm",
        fake_llm,
    )
    state = _preference_state()
    active_commitment = {
        "fact": "用户和角色已约定讨论工作时使用更短的句子。",
        "subjective_appraisal": "角色认为这是清楚且可执行的表达约定。",
        "relationship_signal": "讨论工作时优先短句。",
        "updated_at": "2026-05-06 10:00",
    }
    user_image = state["rag_result"]["user_image"]
    user_memory_context = user_image["user_memory_context"]
    user_memory_context["active_commitments"] = [active_commitment]
    state["interaction_style_context"] = {
        "user_style": {
            "speech_guidelines": ["可使用更长、更流动的表达。"],
            "social_guidelines": [],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "medium",
        },
        "application_order": ["user_style"],
    }

    result = await call_preference_adapter(state)

    assert result["accepted_user_preferences"] == [
        "讨论工作时尽量使用更短的句子。"
    ]
    assert "更长" not in "".join(result["accepted_user_preferences"])
    assert fake_llm.payload["active_commitments"] == [active_commitment]
    assert fake_llm.payload["interaction_style_context"] == state[
        "interaction_style_context"
    ]
