"""Deterministic tests for first-cycle shared-memory prewarm helpers."""

from __future__ import annotations

from typing import Any

import pytest
from openai import OpenAIError

from tests.cognition_core_v2_test_helpers import canonical_user_message_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _minimal_persona_state() -> dict[str, Any]:
    """Build the smallest persona state accepted by the prewarm boundary."""

    turn_clock = build_turn_clock("2026-06-08 09:00:00")
    episode = canonical_user_message_episode(
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
        "decontextualized_input": "Need a memory-backed stance.",
        "referents": [],
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-1",
        },
        "user_profile": {"relationship_state": 500},
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
        "user_name": "Test User",
        "platform": "debug",
        "platform_channel_id": "prewarm-channel",
        "platform_message_id": "prewarm-message",
        "platform_bot_id": "bot-1",
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
    }
    return state


def _empty_result() -> dict[str, Any]:
    """Build the expected empty prewarm RAG result."""

    rag_result = {
        "answer": "",
        "user_image": {
            "user_memory_context": empty_user_memory_context(),
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "resolver": "local_context_resolver",
            "iterations": 0,
            "node_count": 0,
            "resolved_node_count": 0,
            "blocked_node_count": 0,
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return rag_result


def _patch_persistent_memory_worker(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any] | BaseException,
) -> list[dict[str, Any]]:
    """Patch the shared persistent-memory worker and return captured calls."""

    calls: list[dict[str, Any]] = []

    class FakePersistentMemorySearchAgent:
        """Capture prewarm calls without touching live retrieval backends."""

        async def run(
            self,
            task: str,
            context: dict[str, Any],
            max_attempts: int = 3,
        ) -> dict[str, Any]:
            """Record the worker invocation and return the configured result."""

            calls.append({
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            })
            if isinstance(result, BaseException):
                raise result
            return_value = result
            return return_value

    monkeypatch.setattr(
        capabilities,
        "PersistentMemorySearchAgent",
        FakePersistentMemorySearchAgent,
        raising=False,
    )
    return calls


def _forbid_full_local_context_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail the test if prewarm enters the full RAG3 resolver."""

    async def resolve_local_context(
        _request: dict[str, Any],
        _context: dict[str, Any],
        _options: dict[str, Any],
    ) -> dict[str, Any]:
        """Reject full local-context resolver calls from prewarm."""

        raise AssertionError("prewarm must not call resolve_local_context")

    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        resolve_local_context,
    )


@pytest.mark.asyncio
async def test_first_cycle_prewarm_uses_shared_persistent_memory_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prewarm should call only the shared persistent-memory worker."""

    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [{
                "content": "Shared policy: respond lightly to image-only turns.",
                "source_system": "memory",
            }],
            "attempts": 1,
        },
    )
    _forbid_full_local_context_resolver(monkeypatch)

    rag_result = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert len(calls) == 1
    worker_call = calls[0]
    context = worker_call["context"]
    assert worker_call["task"] == "Need a memory-backed stance."
    assert worker_call["max_attempts"] == 1
    assert context["prompt_message_context"]["body_text"] == (
        "Need a memory-backed stance."
    )
    assert context["character_profile"]["name"] == "Kazusa"
    assert rag_result["answer"] == ""
    assert rag_result["user_memory_unit_candidates"] == []
    assert "Shared policy: respond lightly to image-only turns." in repr(
        rag_result["memory_evidence"]
    )


@pytest.mark.asyncio
async def test_first_cycle_prewarm_projects_memory_without_answer_or_user_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only accepted shared-memory rows should reach projected memory evidence."""

    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
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
            "attempts": 1,
        },
    )
    _forbid_full_local_context_resolver(monkeypatch)

    rag_result = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    rendered = repr(rag_result)
    assert len(calls) == 1
    assert rag_result["answer"] == ""
    assert rag_result["user_memory_unit_candidates"] == []
    assert len(rag_result["memory_evidence"]) == 1
    assert "Shared nonverbal input policy." in rendered
    assert "Private current-user continuity must not prewarm." not in rendered
    assert "Candidate must not prewarm." not in rendered
    assert "user_memory_units" not in rendered


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
async def test_first_cycle_prewarm_returns_empty_without_shared_evidence_or_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing shared evidence and resolver failures should degrade to empty."""

    _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [{
                "content": "Private current-user continuity must not prewarm.",
                "source_system": "user_memory_units",
            }],
            "attempts": 1,
        },
    )
    _forbid_full_local_context_resolver(monkeypatch)
    unresolved = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert unresolved == _empty_result()

    _patch_persistent_memory_worker(
        monkeypatch,
        OpenAIError("memory worker unavailable"),
    )
    failed = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert failed == _empty_result()
