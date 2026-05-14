"""Tests for persona supervisor RAG2 stage wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as rag2_module
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.time_context import build_character_time_context


@pytest.fixture(autouse=True)
def _stub_rag_event_logging(monkeypatch):
    """Keep deterministic RAG tests away from event-log persistence."""

    for recorder_name in (
        "record_rag_stage_event",
        "record_llm_stage_event",
        "record_model_contract_event",
    ):
        monkeypatch.setattr(
            rag2_module.event_logging,
            recorder_name,
            AsyncMock(),
        )


class _DummyResponse:
    """Small response object for patched RAG2 LLM calls."""

    def __init__(self, content: str) -> None:
        """Store model-compatible content."""
        self.content = content


class _InitializerLLM:
    """Static RAG initializer fake."""

    async def ainvoke(self, _messages: list) -> _DummyResponse:
        """Return one memory-evidence slot."""
        payload = {
            "unknown_slots": [
                "Memory-evidence: retrieve durable evidence about source policy",
            ]
        }
        response = _DummyResponse(json.dumps(payload))
        return response


class _ContinuationLLM:
    """Static continuation refiner fake."""

    def __init__(self, decision: dict) -> None:
        """Store the decision emitted by this fake."""
        self.decision = decision

    async def ainvoke(self, _messages: list) -> _DummyResponse:
        """Return the configured continuation decision."""
        response = _DummyResponse(json.dumps(self.decision))
        return response


class _SummaryLLM:
    """Evaluator summary fake."""

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Echo selected summary from capability payloads."""
        payload = json.loads(messages[1].content)
        raw_result = payload["raw_result"]
        summary = ""
        if isinstance(raw_result, dict):
            summary = str(raw_result.get("selected_summary", ""))
        response = _DummyResponse(summary)
        return response


class _FinalizerLLM:
    """Finalizer fake for public-shape tests."""

    async def ainvoke(self, _messages: list) -> _DummyResponse:
        """Return a stable final answer."""
        response = _DummyResponse("final")
        return response


class _FakeWorker:
    """Async RAG worker fake."""

    def __init__(self, result: dict) -> None:
        """Store the worker result."""
        self.result = result

    async def run(
        self,
        task: str,
        context: dict,
        max_attempts: int = 3,
    ) -> dict:
        """Return the configured worker result."""
        del task, context, max_attempts
        return_value = self.result
        return return_value


async def _noop_async(*args, **kwargs) -> None:
    """Accept cache write hooks without external persistence."""
    del args, kwargs


def _stage_1_research_snapshot_state() -> dict:
    """Build a full text-chat state for RAG request-shape snapshots.

    Returns:
        Persona graph state subset with a valid text-chat cognitive episode.
    """
    timestamp = "2026-04-27T00:00:00+12:00"
    time_context = build_character_time_context(timestamp)
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-rag-snapshot",
        percept_id="percept-rag-snapshot",
        timestamp=timestamp,
        time_context=time_context,
        user_input="Need current evidence.",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id="msg-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["msg-1", "msg-2"],
        active_turn_conversation_row_ids=["row-1", "row-2"],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    state = {
        "decontexualized_input": "Need current evidence.",
        "referents": [],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": timestamp,
        "time_context": time_context,
        "message_envelope": {
            "body_text": "Need current evidence.",
            "raw_wire_text": "Need current evidence.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "Need current evidence.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [{"role": "user", "content": "previous turn"}],
        "chat_history_wide": [{"role": "assistant", "content": "older turn"}],
        "reply_context": {"platform_message_id": "reply-1"},
        "indirect_speech_context": "No indirect speech.",
        "conversation_progress": {
            "status": "active",
            "continuity": "same_episode",
            "current_thread": "Pickup plan is active.",
        },
        "conversation_episode_state": {
            "updated_at": "2026-04-26T23:00:00+00:00",
            "turn_count": 7,
        },
        "promoted_reflection_context": {"summary": "recent reflection"},
        "active_turn_platform_message_ids": ["msg-1", "msg-2"],
        "active_turn_conversation_row_ids": ["row-1", "row-2"],
        "cognitive_episode": episode,
    }
    return state


def _minimal_text_chat_episode() -> dict:
    """Build a minimal text-chat cognitive episode for direct graph tests.

    Returns:
        Valid user-message cognitive episode with no active-turn collapse ids.
    """
    timestamp = "2026-04-27T00:00:00+12:00"
    time_context = build_character_time_context(timestamp)
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-rag-direct",
        percept_id="percept-rag-direct",
        timestamp=timestamp,
        time_context=time_context,
        user_input="clean body",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id="msg-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=[],
        active_turn_conversation_row_ids=[],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    return episode


def _memory_observation_result() -> dict:
    """Build one unresolved observation payload for public-shape tests."""
    result = {
        "resolved": False,
        "result": {
            "capability": "memory_evidence",
            "primary_worker": "persistent_memory_search_agent",
            "supporting_workers": [],
            "source_policy": "semantic durable memory evidence",
            "selected_summary": "",
            "resolved_refs": [],
            "projection_payload": {"memory_rows": []},
            "worker_payloads": {},
            "evidence": [],
            "missing_context": ["memory_evidence"],
            "conflicts": [],
            "observation_candidates": [
                {
                    "content": "Fresh retrieval is needed for current facts.",
                }
            ],
        },
        "attempts": 1,
        "cache": {"enabled": False, "hit": False, "reason": "patched"},
    }
    return result


def _live_result() -> dict:
    """Build one resolved live-context payload for public-shape tests."""
    result = {
        "resolved": True,
        "result": {
            "capability": "live_context",
            "primary_worker": "web_search_agent2",
            "supporting_workers": [],
            "source_policy": "fresh external retrieval",
            "selected_summary": "Fresh evidence found.",
            "resolved_refs": [],
            "projection_payload": {
                "external_text": "Fresh evidence found.",
                "url": "https://example.test/fresh",
            },
            "worker_payloads": {},
            "evidence": ["Fresh evidence found."],
            "missing_context": [],
            "conflicts": [],
        },
        "attempts": 1,
        "cache": {"enabled": False, "hit": False, "reason": "patched"},
    }
    return result


async def _run_patched_rag2_public_shape_case(
    monkeypatch,
    *,
    decision: dict,
) -> dict:
    """Run the real RAG2 entrypoint with patched external dependencies."""
    runtime = RAGCache2Runtime(max_entries=10)
    memory_worker = _FakeWorker(_memory_observation_result())
    live_worker = _FakeWorker(_live_result())

    memory_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["memory_evidence_agent"]
    )
    memory_entry["agent"] = memory_worker.run
    live_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["live_context_agent"]
    )
    live_entry["agent"] = live_worker.run

    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "memory_evidence_agent",
        memory_entry,
    )
    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "live_context_agent",
        live_entry,
    )
    monkeypatch.setattr(rag2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(rag2_module, "_initializer_llm", _InitializerLLM())
    monkeypatch.setattr(rag2_module, "_continuation_assessor_llm", _ContinuationLLM(decision))
    monkeypatch.setattr(rag2_module, "_evaluator_summarizer_llm", _SummaryLLM())
    monkeypatch.setattr(rag2_module, "_finalizer_llm", _FinalizerLLM())
    monkeypatch.setattr(rag2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(rag2_module, "record_initializer_hit", _noop_async)

    result = await rag2_module.call_rag_supervisor(
        original_query="Need current evidence.",
        character_name="<active character>",
        context={
            "platform": "qq",
            "platform_channel_id": "public-shape-test",
            "global_user_id": "user-1",
            "prompt_message_context": {
                "body_text": "Need current evidence.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
        },
    )
    return result


@pytest.mark.asyncio
async def test_call_rag_supervisor_public_keys_unchanged(monkeypatch) -> None:
    """Continuation metadata must not change the public RAG2 return keys."""
    continue_decision = {
        "should_continue": True,
        "refined_query": (
            "Need current evidence. Prior memory only provided a source "
            "strategy, so retrieve fresh public evidence."
        ),
        "reason": "fresh evidence is available",
    }
    stop_decision = {
        "should_continue": False,
        "refined_query": "",
        "reason": "no next source",
    }

    continued_result = await _run_patched_rag2_public_shape_case(
        monkeypatch,
        decision=continue_decision,
    )
    stop_result = await _run_patched_rag2_public_shape_case(
        monkeypatch,
        decision=stop_decision,
    )

    expected_keys = {"answer", "known_facts", "unknown_slots", "loop_count"}
    assert set(continued_result.keys()) == expected_keys
    assert set(stop_result.keys()) == expected_keys
    assert (
        continued_result["known_facts"][0]["continuation"]["should_continue"]
        is True
    )
    assert (
        stop_result["known_facts"][0]["continuation"]["should_continue"]
        is False
    )


@pytest.mark.asyncio
async def test_stage_1_research_calls_rag2_and_projects_payload(monkeypatch) -> None:
    captured: dict = {}

    async def _call_quote_aware_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict:
        captured["fresh_query"] = fresh_query
        captured["reply_context"] = reply_context
        captured["character_name"] = character_name
        captured["context"] = context
        return {
            "answer": "resolved",
            "known_facts": [
                {
                    "slot": "profile",
                    "agent": "user_profile_agent",
                    "resolved": True,
                    "summary": "profile",
                    "raw_result": {
                        "global_user_id": "user-1",
                        "user_memory_context": {
                            "objective_facts": [
                                {
                                    "fact": "User likes tea",
                                    "subjective_appraisal": "Kazusa sees this as a stable preference.",
                                    "relationship_signal": "Offer tea-related continuity.",
                                }
                            ]
                        },
                    },
                }
            ],
            "unknown_slots": [],
            "loop_count": 1,
        }

    monkeypatch.setattr(
        supervisor_module,
        "call_quote_aware_rag_supervisor",
        _call_quote_aware_rag_supervisor,
    )

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "你记得我喜欢什么吗？",
        "referents": [],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "time_context": build_character_time_context("2026-04-27T00:00:00+12:00"),
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {
            "platform_message_id": "reply-1",
            "reply_excerpt": "BYD Shark 6 uses a 1.5T hybrid system.",
        },
        "indirect_speech_context": "",
        "cognitive_episode": _minimal_text_chat_episode(),
        "conversation_progress": {
            "status": "active",
            "continuity": "same_episode",
            "current_thread": "Pickup plan is active.",
        },
        "conversation_episode_state": {
            "updated_at": "2026-04-26T23:00:00+00:00",
            "turn_count": 7,
        },
    })

    assert captured["fresh_query"] == "你记得我喜欢什么吗？"
    assert captured["reply_context"] == {
        "platform_message_id": "reply-1",
        "reply_excerpt": "BYD Shark 6 uses a 1.5T hybrid system.",
    }
    assert captured["character_name"] == "Kazusa"
    assert captured["context"]["channel_type"] == "group"
    assert "message_envelope" not in captured["context"]
    assert captured["context"]["character_profile"] == {
        "global_user_id": "character-1",
        "name": "Kazusa",
    }
    assert captured["context"]["prompt_message_context"]["body_text"] == "clean body"
    assert captured["context"]["chat_history_recent"] == []
    assert captured["context"]["conversation_progress"]["current_thread"] == "Pickup plan is active."
    assert captured["context"]["conversation_episode_state"]["turn_count"] == 7
    assert result["rag_result"]["answer"] == "resolved"
    assert result["rag_result"]["user_image"]["user_memory_context"]["objective_facts"][0]["fact"] == "User likes tea"


@pytest.mark.asyncio
async def test_stage_1_research_pre_stage_04_request_shape_snapshot(monkeypatch) -> None:
    """Current RAG request shape should stay stable across adapter extraction."""
    captured: dict = {}

    async def _call_quote_aware_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict:
        captured["request"] = {
            "fresh_query": fresh_query,
            "reply_context": reply_context,
            "character_name": character_name,
            "context": context,
        }
        return_value = {
            "answer": "snapshot answer",
            "known_facts": [],
            "unknown_slots": [],
            "loop_count": 1,
        }
        return return_value

    monkeypatch.setattr(
        supervisor_module,
        "call_quote_aware_rag_supervisor",
        _call_quote_aware_rag_supervisor,
    )

    result = await supervisor_module.stage_1_research(
        _stage_1_research_snapshot_state()
    )

    expected_request = {
        "fresh_query": "Need current evidence.",
        "reply_context": {"platform_message_id": "reply-1"},
        "character_name": "Kazusa",
        "context": {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "character_profile": {
                "global_user_id": "character-1",
                "name": "Kazusa",
            },
            "active_turn_platform_message_ids": ["msg-1", "msg-2"],
            "active_turn_conversation_row_ids": ["row-1", "row-2"],
            "global_user_id": "user-1",
            "user_name": "User",
            "user_profile": {"affinity": 500},
            "current_timestamp": "2026-04-27T00:00:00+12:00",
            "time_context": build_character_time_context(
                "2026-04-27T00:00:00+12:00"
            ),
            "prompt_message_context": {
                "body_text": "Need current evidence.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": ["character-1"],
                "broadcast": False,
            },
            "channel_topic": "test",
            "chat_history_recent": [{"role": "user", "content": "previous turn"}],
            "chat_history_wide": [{"role": "assistant", "content": "older turn"}],
            "reply_context": {"platform_message_id": "reply-1"},
            "indirect_speech_context": "No indirect speech.",
            "conversation_progress": {
                "status": "active",
                "continuity": "same_episode",
                "current_thread": "Pickup plan is active.",
            },
            "conversation_episode_state": {
                "updated_at": "2026-04-26T23:00:00+00:00",
                "turn_count": 7,
            },
            "promoted_reflection_context": {"summary": "recent reflection"},
        },
    }
    assert captured["request"] == expected_request
    assert result["rag_result"]["answer"] == "snapshot answer"


@pytest.mark.asyncio
async def test_stage_1_research_passes_empty_reply_context_to_wrapper(
    monkeypatch,
) -> None:
    """Empty reply metadata should reach the quote-aware wrapper unchanged."""
    captured: dict = {}

    async def _call_quote_aware_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict:
        del fresh_query, character_name, context
        captured["reply_context"] = reply_context
        return_value = {
            "answer": "empty reply answer",
            "known_facts": [],
            "unknown_slots": [],
            "loop_count": 1,
        }
        return return_value

    monkeypatch.setattr(
        supervisor_module,
        "call_quote_aware_rag_supervisor",
        _call_quote_aware_rag_supervisor,
    )
    state = _stage_1_research_snapshot_state()
    state["reply_context"] = {}

    result = await supervisor_module.stage_1_research(state)

    assert captured["reply_context"] == {}
    assert result["rag_result"]["answer"] == "empty reply answer"


@pytest.mark.asyncio
async def test_stage_1_research_skips_rag_for_unresolved_referents(monkeypatch) -> None:
    """Unresolved required references should skip RAG and preserve payload shape."""
    called = False

    async def _call_quote_aware_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict:
        del fresh_query, reply_context, character_name, context
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        supervisor_module,
        "call_quote_aware_rag_supervisor",
        _call_quote_aware_rag_supervisor,
    )

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "time_context": build_character_time_context("2026-04-27T00:00:00+12:00"),
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "cognitive_episode": _minimal_text_chat_episode(),
    })

    rag_result = result["rag_result"]
    assert called is False
    assert rag_result["answer"] == ""
    assert rag_result["user_image"]["user_memory_context"]["stable_patterns"] == []
    assert rag_result["user_image"]["user_memory_context"]["active_commitments"] == []
    assert rag_result["character_image"] == {}
    assert rag_result["memory_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["unknown_slots"] == []


@pytest.mark.asyncio
async def test_stage_1_research_runs_rag_for_mixed_referents(monkeypatch) -> None:
    """Mixed referents should not trigger the old binary RAG skip cliff."""
    called = False

    async def _call_quote_aware_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict:
        del fresh_query, reply_context, character_name, context
        nonlocal called
        called = True
        return {
            "answer": "partial evidence",
            "known_facts": [],
            "unknown_slots": [],
            "loop_count": 1,
        }

    monkeypatch.setattr(
        supervisor_module,
        "call_quote_aware_rag_supervisor",
        _call_quote_aware_rag_supervisor,
    )

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "他上次说的那些关于X的话是什么意思？",
        "referents": [
            {"phrase": "他", "referent_role": "subject", "status": "resolved"},
            {"phrase": "那些话", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "time_context": build_character_time_context("2026-04-27T00:00:00+12:00"),
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "cognitive_episode": _minimal_text_chat_episode(),
    })

    assert called is True
    assert result["rag_result"]["answer"] == "partial evidence"


@pytest.mark.asyncio
async def test_stage_1_research_skips_when_referents_are_all_unresolved(monkeypatch) -> None:
    """Structured unresolved referents should be authoritative."""
    called = False

    async def _call_quote_aware_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict:
        del fresh_query, reply_context, character_name, context
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        supervisor_module,
        "call_quote_aware_rag_supervisor",
        _call_quote_aware_rag_supervisor,
    )

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "time_context": build_character_time_context("2026-04-27T00:00:00+12:00"),
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    })

    assert called is False
    assert result["rag_result"]["answer"] == ""
