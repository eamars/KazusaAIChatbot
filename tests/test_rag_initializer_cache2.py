from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as supervisor2_module
from kazusa_ai_chatbot.rag.cache2_policy import build_initializer_cache_key
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime


class _DummyResponse:
    """Small LangChain-like response wrapper for initializer tests."""

    def __init__(self, content: str) -> None:
        """Create a dummy response.

        Args:
            content: LLM response content.
        """
        self.content = content


class _CountingAsyncLLM:
    """Async LLM fake that counts calls and returns one JSON payload."""

    def __init__(self, payload: dict) -> None:
        """Create a counting fake LLM.

        Args:
            payload: JSON-serializable response payload.
        """
        self.calls = 0
        self.payload = payload

    async def ainvoke(self, _messages: list) -> _DummyResponse:
        """Return the configured payload and increment the call count.

        Args:
            _messages: Prompt messages supplied by the caller.

        Returns:
            Dummy response with JSON content.
        """
        self.calls += 1
        return _DummyResponse(json.dumps(self.payload))


def test_initializer_cache_key_ignores_volatile_timestamp() -> None:
    """Initializer cache keys should not miss just because wall time changed."""
    base_context = {
        "platform": "discord",
        "platform_channel_id": "chan-1",
        "user_name": "Alice",
        "current_timestamp": "2026-04-26T00:00:00+00:00",
    }
    later_context = {
        **base_context,
        "current_timestamp": "2026-04-26T00:01:00+00:00",
    }

    key_a = build_initializer_cache_key(
        original_query="what did Alice say?",
        character_name="Kazusa",
        context=base_context,
    )
    key_b = build_initializer_cache_key(
        original_query=" what   did alice say? ",
        character_name="kazusa",
        context=later_context,
    )

    assert key_a == key_b


def test_initializer_prompt_documents_profile_evidence_dependency() -> None:
    """Initializer prompt should distinguish profile-needed and no-retrieval acts."""
    rendered_prompt = supervisor2_module._INITIALIZER_PROMPT.format(character_name="千纱")

    assert "Evidence-dependency gate" in rendered_prompt
    assert "千纱能做一个自我介绍么" in rendered_prompt
    assert "Profile: retrieve full user profile" in rendered_prompt
    assert "千纱千纱欢迎回来" in rendered_prompt
    assert "No profile, memory, identity, or conversation evidence is needed" in rendered_prompt


@pytest.mark.asyncio
async def test_rag_initializer_serves_second_identical_call_from_cache(monkeypatch) -> None:
    """A repeated initializer call should reuse Cache 2 and skip the LLM."""
    runtime = RAGCache2Runtime(max_entries=10)
    llm = _CountingAsyncLLM({
        "unknown_slots": ["Identity: look up display name 'Alice' to get global_user_id"]
    })
    monkeypatch.setattr(supervisor2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(supervisor2_module, "_initializer_llm", llm)

    state = {
        "original_query": "what did Alice say?",
        "character_name": "Kazusa",
        "context": {
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "user_name": "Alice",
        },
    }

    first = await supervisor2_module.rag_initializer(state)
    second = await supervisor2_module.rag_initializer(state)

    assert first["unknown_slots"] == second["unknown_slots"]
    assert first["initializer_cache"]["hit"] is False
    assert second["initializer_cache"]["hit"] is True
    assert llm.calls == 1
    assert runtime.get_stats()["hits"] == 1
