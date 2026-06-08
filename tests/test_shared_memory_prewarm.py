"""Deterministic tests for first-cycle shared-memory prewarm helpers."""

from __future__ import annotations

from typing import Any

import pytest
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_resolver.state import build_empty_rag_result
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _minimal_persona_state() -> dict[str, Any]:
    """Build the smallest persona state accepted by the RAG intake boundary."""

    turn_clock = build_turn_clock("2026-06-08 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="prewarm-episode-1",
        percept_id="prewarm-percept-1",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Need a memory-backed stance.",
        platform="debug",
        platform_channel_id="prewarm-channel",
        channel_type="private",
        platform_message_id="prewarm-message",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="Test User",
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    state = {
        "cognitive_episode": episode,
        "decontexualized_input": "Need a memory-backed stance.",
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-1",
        },
        "user_profile": {"affinity": 500},
        "prompt_message_context": {
            "body_text": "Need a memory-backed stance.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "prewarm test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": None,
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "global_user_id": "user-1",
        "platform": "debug",
        "platform_channel_id": "prewarm-channel",
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
    }
    return state


def _rag_request() -> dict[str, Any]:
    """Return the RAG intake output used by helper-contract tests."""

    request = {
        "original_query": "Need a memory-backed stance.",
        "character_name": "Kazusa",
        "context": {
            "platform": "debug",
            "platform_channel_id": "prewarm-channel",
            "prompt_message_context": {
                "body_text": "Need a memory-backed stance.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": ["character-1"],
                "broadcast": False,
            },
        },
        "current_user_id": "user-1",
        "character_user_id": "character-1",
    }
    return request


def _empty_result() -> dict[str, Any]:
    """Build the standard empty RAG result for merge helper tests."""

    rag_result = build_empty_rag_result(
        current_user_id="user-1",
        character_user_id="character-1",
    )
    return rag_result


class _FakePersistentMemorySearchAgent:
    """Patch target that records the persistent-memory worker call."""

    next_result: dict[str, Any] | BaseException = {
        "resolved": False,
        "result": [],
    }
    calls: list[dict[str, Any]] = []

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        max_attempts: int = 0,
    ) -> dict[str, Any]:
        """Record the worker invocation and return or raise the configured result."""

        call = {
            "task": task,
            "context": context,
            "max_attempts": max_attempts,
        }
        self.calls.append(call)
        if isinstance(self.next_result, BaseException):
            raise self.next_result
        return_value = self.next_result
        return return_value


def _patch_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fake persistent-memory worker into the capabilities module."""

    _FakePersistentMemorySearchAgent.calls = []
    _FakePersistentMemorySearchAgent.next_result = {
        "resolved": False,
        "result": [],
    }
    monkeypatch.setattr(
        capabilities,
        "PersistentMemorySearchAgent",
        _FakePersistentMemorySearchAgent,
        raising=False,
    )


def _patch_rag_intake(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:
    """Patch the existing RAG intake builder and capture its call shape."""

    def build_rag_request(**kwargs: Any) -> dict[str, Any]:
        captured["kwargs"] = kwargs
        request = _rag_request()
        return request

    monkeypatch.setattr(
        capabilities,
        "build_text_chat_rag_request",
        build_rag_request,
    )


@pytest.mark.asyncio
async def test_first_cycle_prewarm_uses_existing_rag_intake_and_persistent_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prewarm should reuse persona RAG intake and only call the shared worker."""

    captured: dict[str, Any] = {}
    _patch_rag_intake(monkeypatch, captured)
    _patch_worker(monkeypatch)
    _FakePersistentMemorySearchAgent.next_result = {
        "resolved": True,
        "result": [
            {
                "content": "Shared policy: respond lightly to image-only turns.",
                "source_system": "memory",
                "timestamp": "2026-05-24T07:41:21+00:00",
            }
        ],
    }

    rag_result = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert captured["kwargs"]["decontexualized_input"] == (
        "Need a memory-backed stance."
    )
    assert captured["kwargs"]["prompt_message_context"]["body_text"] == (
        "Need a memory-backed stance."
    )
    assert len(_FakePersistentMemorySearchAgent.calls) == 1
    worker_call = _FakePersistentMemorySearchAgent.calls[0]
    assert worker_call["task"] == "Need a memory-backed stance."
    assert worker_call["context"] == _rag_request()["context"]
    assert worker_call["max_attempts"] == 1
    assert rag_result["answer"] == ""
    assert rag_result["user_memory_unit_candidates"] == []
    assert rag_result["memory_evidence"]


@pytest.mark.asyncio
async def test_first_cycle_prewarm_projects_memory_without_answer_or_user_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only accepted shared-memory rows should reach projected memory evidence."""

    captured: dict[str, Any] = {}
    _patch_rag_intake(monkeypatch, captured)
    _patch_worker(monkeypatch)
    _FakePersistentMemorySearchAgent.next_result = {
        "resolved": True,
        "result": [
            {
                "content": "Shared nonverbal input policy.",
                "source_system": "memory",
                "timestamp": "2026-05-24T07:41:21+00:00",
            },
            {
                "content": "Private current-user continuity must not prewarm.",
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
            },
        ],
    }

    rag_result = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    rendered = repr(rag_result)
    assert rag_result["answer"] == ""
    assert rag_result["user_memory_unit_candidates"] == []
    assert len(rag_result["memory_evidence"]) == 1
    assert "Shared nonverbal input policy." in rendered
    assert "Private current-user continuity must not prewarm." not in rendered
    assert "user_memory_units" not in rendered
    assert "memory_evidence_agent" not in rendered


def test_merge_shared_memory_prewarm_result_filters_user_memory_source() -> None:
    """Merge should append shared memory evidence without changing base fields."""

    base_rag_result = _empty_result()
    base_rag_result["answer"] = "base answer"
    base_rag_result["memory_evidence"] = [{"summary": "base memory"}]
    base_rag_result["user_memory_unit_candidates"] = [{"unit_id": "base-unit"}]
    base_trace = base_rag_result["supervisor_trace"]

    prewarm_rag_result = _empty_result()
    prewarm_rag_result["answer"] = "prewarm answer must not copy"
    prewarm_rag_result["memory_evidence"] = [
        {"summary": "shared prewarm memory"},
        {
            "summary": "scoped memory must not copy",
            "source_system": "user_memory_units",
        },
    ]
    prewarm_rag_result["user_memory_unit_candidates"] = [
        {"unit_id": "private-unit"},
    ]
    prewarm_rag_result["supervisor_trace"] = {
        "loop_count": 99,
        "unknown_slots": ["prewarm"],
        "dispatched": [{"agent": "persistent_memory_search_agent"}],
    }

    merged = capabilities.merge_shared_memory_prewarm_result(
        base_rag_result,
        prewarm_rag_result,
    )

    assert merged["answer"] == "base answer"
    assert merged["user_memory_unit_candidates"] == [{"unit_id": "base-unit"}]
    assert merged["supervisor_trace"] is base_trace
    assert merged["memory_evidence"] == [
        {"summary": "base memory"},
        {"summary": "shared prewarm memory"},
    ]


@pytest.mark.asyncio
async def test_first_cycle_prewarm_returns_empty_on_unresolved_or_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unresolved and failed worker calls should degrade to the empty RAG shape."""

    captured: dict[str, Any] = {}
    _patch_rag_intake(monkeypatch, captured)
    _patch_worker(monkeypatch)
    unresolved = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert unresolved == _empty_result()

    _FakePersistentMemorySearchAgent.next_result = OpenAIError(
        "worker unavailable",
    )
    failed = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert failed == _empty_result()
