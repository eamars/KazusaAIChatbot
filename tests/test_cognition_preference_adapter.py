from __future__ import annotations

from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import call_preference_adapter


class _FakePreferenceAdapterLlm:
    """Small async fake for the preference adapter LLM."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: list) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


def _preference_state() -> dict:
    """Build the minimal state consumed by ``call_preference_adapter``."""

    return {
        "decontexualized_input": "please keep replies short",
        "rag_result": {},
        "user_profile": {"active_commitments": []},
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
async def test_preference_adapter_accepts_string_preferences(monkeypatch) -> None:
    """Preference adapter preserves native string preference items."""

    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3._preference_adapter_llm",
        _FakePreferenceAdapterLlm('{"accepted_user_preferences":[" concise replies ", "soft tone"]}'),
    )

    result = await call_preference_adapter(_preference_state())

    assert result["accepted_user_preferences"] == ["concise replies", "soft tone"]


@pytest.mark.asyncio
async def test_preference_adapter_does_not_stringify_container_items(monkeypatch) -> None:
    """Preference adapter must not turn dict/list preference items into repr text."""

    monkeypatch.setattr(
        "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3._preference_adapter_llm",
        _FakePreferenceAdapterLlm(
            '{"accepted_user_preferences":[{"text":"do not stringify"},["bad"]," keep me "]}'
        ),
    )

    result = await call_preference_adapter(_preference_state())

    assert result["accepted_user_preferences"] == ["keep me"]
