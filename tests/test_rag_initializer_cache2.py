from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as supervisor2_module
from kazusa_ai_chatbot.message_envelope import project_prompt_message_context
from kazusa_ai_chatbot.rag.cache2_policy import build_initializer_cache_key
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime


class _ClosedTask:
    """Placeholder returned by a fake ``asyncio.create_task``."""


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
        self.messages: list[list] = []

    async def ainvoke(self, _messages: list) -> _DummyResponse:
        """Return the configured payload and increment the call count.

        Args:
            _messages: Prompt messages supplied by the caller.

        Returns:
            Dummy response with JSON content.
        """
        self.calls += 1
        self.messages.append(_messages)
        return _DummyResponse(json.dumps(self.payload))


def _capture_create_task(created: list) -> object:
    """Build a fake create_task function that records and closes coroutines.

    Args:
        created: List populated with received coroutine objects.

    Returns:
        Function compatible with ``asyncio.create_task`` for these tests.
    """

    def fake_create_task(coro):
        created.append(coro)
        coro.close()
        return_value = _ClosedTask()
        return return_value

    return fake_create_task


def _prompt_context(body_text: str = "what did current user say?") -> dict:
    """Build the strict initializer prompt context required by cache keys.

    Args:
        body_text: Clean user-authored message text.

    Returns:
        Prompt-safe current-message context dict.
    """

    return_value = {
        "body_text": body_text,
        "mentions": [],
        "attachments": [],
        "addressed_to_global_user_ids": [],
        "broadcast": True,
    }
    return return_value


def test_initializer_cache_key_ignores_volatile_timestamp() -> None:
    """Initializer cache keys should not miss just because wall time changed."""
    base_context = {
        "platform": "discord",
        "platform_channel_id": "chan-1",
        "user_name": "<current user>",
        "current_timestamp": "2026-04-26T00:00:00+00:00",
        "prompt_message_context": _prompt_context(),
    }
    later_context = {
        **base_context,
        "current_timestamp": "2026-04-26T00:01:00+00:00",
    }

    key_a = build_initializer_cache_key(
        original_query="what did current user say?",
        character_name="<active character>",
        context=base_context,
    )
    key_b = build_initializer_cache_key(
        original_query=" what   did current user say? ",
        character_name="<active character>",
        context=later_context,
    )

    assert key_a == key_b


def test_initializer_cache_key_uses_body_text_and_addressing() -> None:
    """Initializer keys should reflect semantic body text and typed addressing."""

    base_context = {
        "platform": "discord",
        "platform_channel_id": "chan-1",
        "global_user_id": "user-a",
        "user_name": "<current user>",
        "prompt_message_context": {
            "body_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-global"],
            "broadcast": False,
        },
    }
    different_body_context = {
        **base_context,
        "prompt_message_context": {
            "body_text": "other body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-global"],
            "broadcast": False,
        },
    }
    different_addressing_context = {
        **base_context,
        "prompt_message_context": {
            "body_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
    }

    key_a = build_initializer_cache_key(
        original_query="clean body",
        character_name="<active character>",
        context=base_context,
    )
    key_b = build_initializer_cache_key(
        original_query="clean body",
        character_name="<active character>",
        context=different_body_context,
    )
    key_c = build_initializer_cache_key(
        original_query="clean body",
        character_name="<active character>",
        context=different_addressing_context,
    )

    assert key_a != key_b
    assert key_a != key_c


def test_initializer_cache_key_requires_prompt_message_context() -> None:
    """Initializer cache keys should fail clearly without prompt context."""

    with pytest.raises(KeyError, match="prompt_message_context is required"):
        build_initializer_cache_key(
            original_query="clean body",
            character_name="<active character>",
            context={
                "platform": "discord",
                "platform_channel_id": "chan-1",
            },
        )


def test_initializer_prompt_documents_profile_evidence_dependency() -> None:
    """Initializer prompt should distinguish profile-needed and no-retrieval acts."""
    rendered_prompt = supervisor2_module._INITIALIZER_PROMPT.format(
        character_name="<active character>",
    )

    assert "Evidence-dependency gate" in rendered_prompt
    assert "<character mention>能做一个自我介绍么" in rendered_prompt
    assert "Profile: retrieve full user profile" in rendered_prompt
    assert "<character mention><character mention>欢迎回来" in rendered_prompt
    assert "No profile, memory, identity, or conversation evidence is needed" in rendered_prompt


def test_memory_search_prompt_uses_evidence_contract() -> None:
    """Memory-search prompt wording should not pre-classify memory evidence."""
    rendered_initializer = supervisor2_module._INITIALIZER_PROMPT.format(
        character_name="<active character>",
    )
    rendered_dispatcher = supervisor2_module._DISPATCHER_PROMPT.format(
        agent_name_union=supervisor2_module._build_agent_name_union(),
    )

    assert (
        "Memory-search: search persistent memory for evidence relevant to answering"
        in rendered_initializer
    )
    assert "Memory-search: search persistent memory for impressions or opinions" not in rendered_initializer
    assert "Do not pre-classify the evidence as fact, impression, opinion" in rendered_initializer
    assert "Handles durable memory evidence relevant to answering the slot" in rendered_dispatcher


def test_normalize_initializer_slots_does_not_stringify_container_items() -> None:
    """Initializer slots should keep only native strings."""

    slots = supervisor2_module._normalize_initializer_slots([
        {"slot": "do not stringify"},
        ["bad"],
        " keep me ",
    ])

    assert slots == ["keep me"]


def test_normalize_dispatch_does_not_stringify_task_or_agent() -> None:
    """Dispatcher payloads should not turn malformed fields into executable work."""

    dispatch = supervisor2_module._normalize_dispatch(
        {
            "agent_name": {"bad": "agent"},
            "task": {"bad": "task"},
            "context": '{"bad":"context"}',
            "max_attempts": 2,
        },
        current_slot="fallback slot",
    )

    assert dispatch == {
        "agent_name": "",
        "task": "fallback slot",
        "context": {},
        "max_attempts": 2,
    }


def test_normalize_dispatch_accepts_valid_payload() -> None:
    """Dispatcher payloads should preserve valid native fields."""

    dispatch = supervisor2_module._normalize_dispatch(
        {
            "agent_name": "user_lookup_agent",
            "task": " look up <named user> ",
            "context": {"known_facts": []},
            "max_attempts": 2,
        },
        current_slot="fallback slot",
    )

    assert dispatch == {
        "agent_name": "user_lookup_agent",
        "task": "look up <named user>",
        "context": {"known_facts": []},
        "max_attempts": 2,
    }


@pytest.mark.asyncio
async def test_rag_initializer_serves_second_identical_call_from_cache(monkeypatch) -> None:
    """A repeated initializer call should reuse Cache 2 and skip the LLM."""
    runtime = RAGCache2Runtime(max_entries=10)
    created_tasks: list = []
    llm = _CountingAsyncLLM({
        "unknown_slots": ["Identity: look up display name '<named user>' to get global_user_id"]
    })
    monkeypatch.setattr(supervisor2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(supervisor2_module, "_initializer_llm", llm)
    monkeypatch.setattr(
        supervisor2_module.asyncio,
        "create_task",
        _capture_create_task(created_tasks),
    )

    state = {
        "original_query": "what did current user say?",
        "character_name": "<active character>",
        "context": {
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "user_name": "<current user>",
            "prompt_message_context": _prompt_context(),
        },
    }

    first = await supervisor2_module.rag_initializer(state)
    second = await supervisor2_module.rag_initializer(state)

    assert first["unknown_slots"] == second["unknown_slots"]
    assert first["initializer_cache"]["hit"] is False
    assert second["initializer_cache"]["hit"] is True
    assert llm.calls == 1
    assert runtime.get_stats()["hits"] == 1
    assert len(created_tasks) == 2


@pytest.mark.asyncio
async def test_rag_initializer_hit_schedules_persistent_hit(monkeypatch) -> None:
    """A memory cache hit should schedule hit recording without calling the LLM."""
    runtime = RAGCache2Runtime(max_entries=10)
    created_tasks: list = []
    llm = _CountingAsyncLLM({"unknown_slots": ["should not run"]})
    state = {
        "original_query": "what did current user say?",
        "character_name": "<active character>",
        "context": {
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "user_name": "<current user>",
            "prompt_message_context": _prompt_context(),
        },
    }
    cache_key = build_initializer_cache_key(
        original_query=state["original_query"],
        character_name=state["character_name"],
        context=state["context"],
    )
    await runtime.store(
        cache_key=cache_key,
        cache_name=supervisor2_module.INITIALIZER_CACHE_NAME,
        result={"unknown_slots": ["cached slot"], "confidence": 1.0},
        dependencies=[],
        metadata={"stage": "rag_initializer"},
    )
    monkeypatch.setattr(supervisor2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(supervisor2_module, "_initializer_llm", llm)
    monkeypatch.setattr(
        supervisor2_module.asyncio,
        "create_task",
        _capture_create_task(created_tasks),
    )

    result = await supervisor2_module.rag_initializer(state)

    assert result["unknown_slots"] == ["cached slot"]
    assert result["initializer_cache"]["hit"] is True
    assert llm.calls == 0
    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_rag_initializer_miss_schedules_persistent_upsert(monkeypatch) -> None:
    """A cacheable miss should schedule write-through after memory store."""
    runtime = RAGCache2Runtime(max_entries=10)
    created_tasks: list = []
    llm = _CountingAsyncLLM({"unknown_slots": ["fresh slot"]})
    monkeypatch.setattr(supervisor2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(supervisor2_module, "_initializer_llm", llm)
    monkeypatch.setattr(
        supervisor2_module.asyncio,
        "create_task",
        _capture_create_task(created_tasks),
    )
    state = {
        "original_query": "what did current user say?",
        "character_name": "<active character>",
        "context": {
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "user_name": "<current user>",
            "prompt_message_context": _prompt_context(),
        },
    }

    result = await supervisor2_module.rag_initializer(state)

    assert result["unknown_slots"] == ["fresh slot"]
    assert result["initializer_cache"]["hit"] is False
    assert llm.calls == 1
    assert len(created_tasks) == 1
    assert runtime.get_stats()["size"] == 1


@pytest.mark.asyncio
async def test_rag_initializer_payload_uses_prompt_context_for_large_image(
    monkeypatch,
) -> None:
    """Initializer human payload should carry image description, not bytes."""

    runtime = RAGCache2Runtime(max_entries=10)
    created_tasks: list = []
    llm = _CountingAsyncLLM({"unknown_slots": []})
    base64_payload = "a" * (1024 * 1024 + 1)
    description = "image shows a desk and handwritten notes"
    prompt_message_context = project_prompt_message_context(
        message_envelope={
            "body_text": "",
            "raw_wire_text": "[CQ:image,url=https://example.test/image.jpg]",
            "mentions": [],
            "attachments": [{
                "media_type": "image/jpeg",
                "base64_data": base64_payload,
                "description": "",
                "storage_shape": "inline",
            }],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        multimedia_input=[{
            "content_type": "image/jpeg",
            "base64_data": base64_payload,
            "description": description,
        }],
    )
    state = {
        "original_query": "",
        "character_name": "<active character>",
        "context": {
            "platform": "discord",
            "platform_channel_id": "chan-1",
            "user_name": "<current user>",
            "prompt_message_context": prompt_message_context,
        },
    }
    monkeypatch.setattr(supervisor2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(supervisor2_module, "_initializer_llm", llm)
    monkeypatch.setattr(
        supervisor2_module.asyncio,
        "create_task",
        _capture_create_task(created_tasks),
    )

    await supervisor2_module.rag_initializer(state)

    human_payload = llm.messages[0][1].content
    assert description in human_payload
    assert "base64_data" not in human_payload
    assert "raw_wire_text" not in human_payload
    assert base64_payload not in human_payload


def test_initializer_prompt_version_remains_v6_for_prompt_context() -> None:
    """Prompt-safe context cutover should not bump initializer prompt version."""

    assert supervisor2_module.INITIALIZER_PROMPT_VERSION == "initializer_prompt:v6"
