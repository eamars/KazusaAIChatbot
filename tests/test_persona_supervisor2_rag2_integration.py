"""Tests for persona supervisor RAG2 stage wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from tests.cognition_core_v2_test_helpers import canonical_user_message_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as rag2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_initializer as rag_initializer_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_evaluator as rag_evaluator_module
from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
)
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


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

    async def ainvoke(self, _messages: list, *, config=None) -> _DummyResponse:
        """Return one memory-evidence slot."""
        payload = {
            "unknown_slots": [
                "Memory-evidence: retrieve durable evidence about source policy",
            ]
        }
        response = _DummyResponse(json.dumps(payload))
        return response


class _MultiSlotInitializerLLM:
    """Static RAG initializer fake with caller-provided slots."""

    def __init__(self, slots: list[str]) -> None:
        """Store the slot queue emitted by this fake."""
        self.slots = slots

    async def ainvoke(self, _messages: list, *, config=None) -> _DummyResponse:
        """Return the configured slot queue."""
        payload = {"unknown_slots": self.slots}
        response = _DummyResponse(json.dumps(payload))
        return response


def test_normalize_initializer_slots_drops_invalid_person_slot_reference() -> None:
    """Person-resolved speaker slots must reference a person-producing slot."""

    slots = rag2_module._normalize_initializer_slots(
        [
            "Conversation-evidence: retrieve messages containing 'opus'",
            (
                "Conversation-evidence: retrieve messages "
                "speaker=person resolved in slot 1"
            ),
            "Conversation-evidence: retrieve messages speaker=any_speaker",
        ]
    )

    assert slots == [
        "Conversation-evidence: retrieve messages containing 'opus'",
        "Conversation-evidence: retrieve messages speaker=any_speaker",
    ]


def test_normalize_initializer_slots_keeps_valid_person_slot_reference() -> None:
    """Person references are valid when earlier slots can resolve a person."""

    slots = rag2_module._normalize_initializer_slots(
        [
            "Person-context: resolve display name 小钳子",
            (
                "Conversation-evidence: retrieve messages "
                "speaker=person resolved in slot 1"
            ),
            (
                "Conversation-evidence: retrieve phrase to identify the speaker"
            ),
            (
                "Conversation-evidence: retrieve profile comment "
                "speaker=person resolved in slot 3"
            ),
        ]
    )

    assert slots == [
        "Person-context: resolve display name 小钳子",
        "Conversation-evidence: retrieve messages speaker=person resolved in slot 1",
        "Conversation-evidence: retrieve phrase to identify the speaker",
        "Conversation-evidence: retrieve profile comment speaker=person resolved in slot 3",
    ]


def test_normalize_initializer_slots_removes_source_ids() -> None:
    """Initializer slots are prompt-facing and must not carry raw source ids."""

    slots = rag2_module._normalize_initializer_slots(
        [
            (
                "Conversation-evidence: retrieve messages from speaker 小钳子 "
                "(global_user_id: 263c883d-aeff-4e0b-a758-6f69186ae8ec)"
            ),
            (
                "Conversation-evidence: retrieve message ID 529487488 "
                "containing product image"
            ),
        ]
    )

    assert slots == [
        "Conversation-evidence: retrieve messages from speaker 小钳子",
        "Conversation-evidence: retrieve 消息记录 containing product image",
    ]


def test_route_after_evaluator_stops_at_loop_count_four() -> None:
    """RAG2 should stop once the universal loop budget is exhausted."""

    state = {
        "unknown_slots": ["Memory-evidence: retrieve another fact"],
        "known_facts": [],
        "loop_count": 4,
    }

    route = rag2_module._route_after_evaluator(state)

    assert route == "finalize"


class _ContinuationLLM:
    """Static continuation refiner fake."""

    def __init__(self, decision: dict) -> None:
        """Store the decision emitted by this fake."""
        self.decision = decision
        self.calls: list[list] = []

    async def ainvoke(self, messages: list, *, config=None) -> _DummyResponse:
        """Return the configured continuation decision."""
        self.calls.append(messages)
        response = _DummyResponse(json.dumps(self.decision))
        return response


class _SummaryLLM:
    """Evaluator summary fake."""

    async def ainvoke(self, messages: list, *, config=None) -> _DummyResponse:
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

    async def ainvoke(self, _messages: list, *, config=None) -> _DummyResponse:
        """Return a stable final answer."""
        response = _DummyResponse("final")
        return response


class _FakeWorker:
    """Async RAG worker fake."""

    def __init__(self, result: dict) -> None:
        """Store the worker result."""
        self.result = result
        self.calls: list[dict] = []

    async def run(
        self,
        task: str,
        context: dict,
        max_attempts: int = 3,
    ) -> dict:
        """Return the configured worker result."""
        self.calls.append(
            {
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            }
        )
        return_value = self.result
        return return_value


async def _noop_async(*args, **kwargs) -> None:
    """Accept cache write hooks without external persistence."""
    del args, kwargs


def _rag_evidence_snapshot_state() -> dict:
    """Build a full text-chat state for RAG request-shape snapshots.

    Returns:
        Persona graph state subset with a valid text-chat cognitive episode.
    """
    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    episode = canonical_user_message_episode(
        episode_id="episode-rag-snapshot",
        percept_id="percept-rag-snapshot",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
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
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    episode = canonical_user_message_episode(
        episode_id="episode-rag-direct",
        percept_id="percept-rag-direct",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
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


def _local_context_node(
    *,
    node_id: str,
    node_kind: str,
    parent_id: str | None,
    children: list[str],
) -> dict:
    """Build a minimal valid local-context resolver graph node."""

    node = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": node_id,
        "node_kind": node_kind,
        "objective": "Resolve production RAG evidence.",
        "parent_id": parent_id,
        "children": children,
        "depends_on": [],
        "consumes": {},
        "produces": [],
        "status": "resolved",
        "investigation_summary": [],
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "attempts": [],
        "collapsed_into": None,
    }
    return node


def _local_context_packet(
    *,
    answer: str,
    user_image: dict | None = None,
    memory_evidence: list[dict] | None = None,
) -> dict:
    """Build a minimal valid RAG3 packet for production-helper tests."""

    root = _local_context_node(
        node_id="root",
        node_kind="synthesis",
        parent_id=None,
        children=["task_1"],
    )
    task = _local_context_node(
        node_id="task_1",
        node_kind="memory_evidence",
        parent_id="root",
        children=[],
    )
    if user_image is None:
        user_image = {
            "user_memory_context": empty_user_memory_context(),
        }
    packet = {
        "schema_version": LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
        "investigation_summary": ["RAG3 resolved production context."],
        "knowledge_we_know_so_far": ["RAG3 resolved production context."],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "rag_result": {
            "answer": answer,
            "user_image": user_image,
            "user_memory_unit_candidates": [],
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": list(memory_evidence or []),
            "recall_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {
                "resolver": "local_context_resolver",
                "node_count": 2,
            },
        },
        "graph": {
            "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "task_1",
            "nodes": {
                "root": root,
                "task_1": task,
            },
            "traversal_order": ["root", "task_1"],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        },
        "trace_summary": {
            "iterations": 1,
            "node_count": 2,
        },
    }
    return packet


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
            "primary_worker": "web_agent3",
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
    monkeypatch.setattr(rag_initializer_module, "_initializer_llm", _InitializerLLM())
    monkeypatch.setattr(rag_evaluator_module, "_continuation_assessor_llm", _ContinuationLLM(decision))
    monkeypatch.setattr(rag_evaluator_module, "_evaluator_summarizer_llm", _SummaryLLM())
    monkeypatch.setattr(rag_evaluator_module, "_finalizer_llm", _FinalizerLLM())
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
async def test_assess_continuation_waits_for_pending_evidence_slot(
    monkeypatch,
) -> None:
    """Queued evidence slots should run before refined-query re-entry."""

    continuation_llm = _ContinuationLLM(
        {
            "should_continue": True,
            "refined_query": "Search exact conversation evidence again.",
            "reason": "the fake would continue if called",
        }
    )
    monkeypatch.setattr(
        rag_evaluator_module,
        "_continuation_assessor_llm",
        continuation_llm,
    )

    decision = await rag2_module._assess_continuation(
        observation_payload={
            "original_query": "What did I forget?",
            "current_slot": "Recall: retrieve active_episode_agreement",
            "agent": "recall_agent",
            "resolved": False,
            "source_policy": "",
            "missing_context": ["recall_evidence"],
            "conflicts": [],
            "observation_candidates": [
                {"content": "A durable memory candidate was inconclusive."}
            ],
            "source_hints": [
                {"kind": "recall", "source": "conversation evidence needed"}
            ],
            "user_resolution_hints": [],
            "known_facts": [],
            "pending_slots": [
                "Conversation-evidence: retrieve exact agreement messages",
            ],
        },
        original_query="What did I forget?",
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""
    assert "待处理证据槽位" in decision["reason"]
    assert continuation_llm.calls == []


@pytest.mark.asyncio
async def test_assess_continuation_waits_for_pending_evidence_slot_after_memory(
    monkeypatch,
) -> None:
    """Queued evidence slots should also stop non-Recall re-entry."""

    continuation_llm = _ContinuationLLM(
        {
            "should_continue": True,
            "refined_query": "Search a broader memory trail.",
            "reason": "the fake would continue if called",
        }
    )
    monkeypatch.setattr(
        rag_evaluator_module,
        "_continuation_assessor_llm",
        continuation_llm,
    )

    decision = await rag2_module._assess_continuation(
        observation_payload={
            "original_query": "Did we discuss this?",
            "current_slot": "Memory-evidence: retrieve durable evidence",
            "agent": "memory_evidence_agent",
            "resolved": False,
            "source_policy": "",
            "missing_context": ["memory_evidence"],
            "conflicts": [],
            "observation_candidates": [
                {"content": "Nearby memory was not direct evidence."}
            ],
            "source_hints": [
                {"kind": "memory", "source": "conversation evidence needed"}
            ],
            "user_resolution_hints": [],
            "known_facts": [],
            "pending_slots": [
                "Conversation-evidence: retrieve exact messages",
            ],
        },
        original_query="Did we discuss this?",
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""
    assert "待处理证据槽位" in decision["reason"]
    assert continuation_llm.calls == []


@pytest.mark.asyncio
async def test_assess_continuation_finalizes_memory_miss_after_recall(
    monkeypatch,
) -> None:
    """Recall plus memory miss should not fan out into broad absence search."""

    continuation_llm = _ContinuationLLM(
        {
            "should_continue": True,
            "refined_query": "Search every broad historical source.",
            "reason": "the fake would continue if called",
        }
    )
    monkeypatch.setattr(
        rag_evaluator_module,
        "_continuation_assessor_llm",
        continuation_llm,
    )

    decision = await rag2_module._assess_continuation(
        observation_payload={
            "original_query": "Did this old agreement exist?",
            "current_slot": "Memory-evidence: retrieve durable evidence",
            "agent": "memory_evidence_agent",
            "resolved": False,
            "source_policy": "",
            "missing_context": ["memory_evidence"],
            "conflicts": [],
            "observation_candidates": [
                {"content": "Nearby memory was not direct evidence."}
            ],
            "source_hints": [
                {"kind": "memory", "source": "related durable memory"}
            ],
            "user_resolution_hints": [],
            "known_facts": [
                {
                    "slot": "Recall: retrieve active agreement",
                    "agent": "recall_agent",
                    "resolved": False,
                    "summary": "Recall found nearby but unconfirmed rows.",
                }
            ],
            "pending_slots": [],
        },
        original_query="Did this old agreement exist?",
        previous_refined_queries=[],
        continuation_count=0,
    )

    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""
    assert "Recall 和 memory evidence 都未解决" in decision["reason"]
    assert continuation_llm.calls == []


@pytest.mark.asyncio
async def test_assess_continuation_records_trace_with_supplied_trace_id(
    monkeypatch,
) -> None:
    """Continuation assessor tracing should use caller-provided trace id."""

    continuation_llm = _ContinuationLLM(
        {
            "should_continue": True,
            "refined_query": "Search current evidence for the unresolved target.",
            "reason": "candidate evidence is plausible but incomplete",
        }
    )
    monkeypatch.setattr(
        rag_evaluator_module,
        "_continuation_assessor_llm",
        continuation_llm,
    )
    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        rag_evaluator_module.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )

    decision = await rag2_module._assess_continuation(
        observation_payload={
            "original_query": "Need current evidence.",
            "current_slot": "Live-context: retrieve current public evidence",
            "agent": "live_context_agent",
            "resolved": False,
            "source_policy": "",
            "missing_context": ["live_context"],
            "conflicts": [],
            "observation_candidates": [
                {"content": "A current-source candidate needs confirmation."}
            ],
            "source_hints": [],
            "user_resolution_hints": [],
            "known_facts": [],
            "pending_slots": [],
        },
        original_query="Need current evidence.",
        previous_refined_queries=[],
        continuation_count=0,
        llm_trace_id="trace-continuation",
    )

    assert decision["should_continue"] is True
    assert len(continuation_llm.calls) == 1
    trace_recorder.assert_awaited_once()
    assert trace_recorder.await_args.kwargs["trace_id"] == "trace-continuation"


@pytest.mark.asyncio
async def test_call_rag_supervisor_does_not_expand_after_resolved_evidence(
    monkeypatch,
) -> None:
    """Resolved evidence should cap later unresolved secondary expansion."""

    runtime = RAGCache2Runtime(max_entries=10)
    memory_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "capability": "memory_evidence",
                "selected_summary": "Primary evidence found.",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "patched"},
        }
    )
    conversation_worker = _FakeWorker(
        {
            "resolved": False,
            "result": {
                "capability": "conversation_evidence",
                "selected_summary": "",
                "missing_context": ["conversation_evidence"],
                "observation_candidates": [
                    {"content": "Nearby but insufficient conversation row."}
                ],
                "source_hints": [
                    {"kind": "conversation", "source": "search nearby rows"}
                ],
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "patched"},
        }
    )
    continuation_llm = _ContinuationLLM(
        {
            "should_continue": True,
            "refined_query": "Search yet another query.",
            "reason": "the fake would continue if called",
        }
    )

    memory_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["memory_evidence_agent"]
    )
    memory_entry["agent"] = memory_worker.run
    conversation_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY[
            "conversation_evidence_agent"
        ]
    )
    conversation_entry["agent"] = conversation_worker.run

    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "memory_evidence_agent",
        memory_entry,
    )
    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "conversation_evidence_agent",
        conversation_entry,
    )
    monkeypatch.setattr(rag2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(
        rag_initializer_module,
        "_initializer_llm",
        _MultiSlotInitializerLLM(
            [
                "Memory-evidence: retrieve primary evidence",
                "Conversation-evidence: retrieve secondary evidence",
            ]
        ),
    )
    monkeypatch.setattr(
        rag_evaluator_module,
        "_continuation_assessor_llm",
        continuation_llm,
    )
    monkeypatch.setattr(rag_evaluator_module, "_evaluator_summarizer_llm", _SummaryLLM())
    monkeypatch.setattr(rag_evaluator_module, "_finalizer_llm", _FinalizerLLM())
    monkeypatch.setattr(rag2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(rag2_module, "record_initializer_hit", _noop_async)

    result = await rag2_module.call_rag_supervisor(
        original_query="Need primary and secondary evidence.",
        character_name="<active character>",
        context={
            "platform": "qq",
            "platform_channel_id": "resolved-then-unresolved-test",
            "global_user_id": "user-1",
            "prompt_message_context": {
                "body_text": "Need primary and secondary evidence.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
        },
    )

    assert len(memory_worker.calls) == 1
    assert len(conversation_worker.calls) == 1
    assert continuation_llm.calls == []
    assert result["known_facts"][0]["resolved"] is True
    assert result["known_facts"][1]["continuation"]["should_continue"] is False
    assert "已有已解决证据" in (
        result["known_facts"][1]["continuation"]["reason"]
    )


@pytest.mark.asyncio
async def test_call_rag_supervisor_continues_remaining_slots_after_unresolved_stop_decision(
    monkeypatch,
) -> None:
    """One unresolved source should not block queued independent slots."""
    runtime = RAGCache2Runtime(max_entries=10)
    conversation_worker = _FakeWorker(
        {
            "resolved": False,
            "result": {
                "capability": "conversation_evidence",
                "selected_summary": "",
                "projection_payload": {"summaries": [], "rows": []},
                "missing_context": ["conversation_evidence"],
                "observation_candidates": [],
                "source_hints": [],
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "patched"},
        }
    )
    memory_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "capability": "memory_evidence",
                "selected_summary": "Scoped preference found.",
                "projection_payload": {
                    "memory_rows": [
                        {
                            "content": "Scoped preference found.",
                            "source_system": "user_memory_units",
                        }
                    ]
                },
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "patched"},
        }
    )
    conversation_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["conversation_evidence_agent"]
    )
    conversation_entry["agent"] = conversation_worker.run
    memory_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["memory_evidence_agent"]
    )
    memory_entry["agent"] = memory_worker.run

    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "conversation_evidence_agent",
        conversation_entry,
    )
    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "memory_evidence_agent",
        memory_entry,
    )
    monkeypatch.setattr(rag2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(
        rag_initializer_module,
        "_initializer_llm",
        _MultiSlotInitializerLLM(
            [
                "Conversation-evidence: retrieve missing conversation evidence",
                "Memory-evidence: retrieve durable current-user preference",
            ]
        ),
    )
    monkeypatch.setattr(
        rag_evaluator_module,
        "_continuation_assessor_llm",
        _ContinuationLLM(
            {
                "should_continue": False,
                "refined_query": "",
                "reason": "no better conversation source",
            }
        ),
    )
    monkeypatch.setattr(rag_evaluator_module, "_evaluator_summarizer_llm", _SummaryLLM())
    monkeypatch.setattr(rag_evaluator_module, "_finalizer_llm", _FinalizerLLM())
    monkeypatch.setattr(rag2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(rag2_module, "record_initializer_hit", _noop_async)

    result = await rag2_module.call_rag_supervisor(
        original_query="Need conversation and memory evidence.",
        character_name="<active character>",
        context={
            "platform": "qq",
            "platform_channel_id": "multi-slot-test",
            "global_user_id": "user-1",
            "prompt_message_context": {
                "body_text": "Need conversation and memory evidence.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
        },
    )

    assert len(conversation_worker.calls) == 1
    assert len(memory_worker.calls) == 1
    assert result["unknown_slots"] == []
    assert [
        fact["agent"]
        for fact in result["known_facts"]
    ] == ["conversation_evidence_agent", "memory_evidence_agent"]
    assert result["known_facts"][0]["resolved"] is False
    assert result["known_facts"][1]["resolved"] is True


@pytest.mark.asyncio
async def test_rag_evidence_helper_calls_rag3_and_projects_payload(monkeypatch) -> None:
    captured: dict = {}

    async def _resolve_local_context(
        request: dict,
        context: dict,
        options: dict,
    ) -> dict:
        captured["request"] = request
        captured["context"] = context
        captured["options"] = options
        user_image = {
            "user_memory_context": empty_user_memory_context(),
        }
        user_image["user_memory_context"]["objective_facts"] = [{
            "fact": "User likes tea",
            "subjective_appraisal": (
                "Kazusa sees this as a stable preference."
            ),
            "relationship_signal": "Offer tea-related continuity.",
        }]
        return _local_context_packet(
            answer="resolved",
            user_image=user_image,
        )

    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        _resolve_local_context,
    )

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    rag_result = await supervisor_module.run_rag_evidence_for_persona_state({
        "decontexualized_input": "你记得我喜欢什么吗？",
        "referents": [],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
    }, agent_name="resolver_rag_evidence")

    assert captured["request"]["objective"] == "你记得我喜欢什么吗？"
    assert captured["request"]["source"] == "l2d"
    assert captured["context"]["character_name"] == "Kazusa"
    assert captured["context"]["platform"] == "qq"
    assert captured["context"]["platform_channel_id"] == "chan-1"
    assert captured["context"]["global_user_id"] == "user-1"
    assert captured["context"]["user_name"] == "User"
    assert captured["context"]["original_user_request"] == "你记得我喜欢什么吗？"
    assert captured["context"]["prompt_message_context"]["body_text"] == "clean body"
    assert captured["context"]["chat_history_recent"] == []
    assert captured["context"]["conversation_progress"]["current_thread"] == "Pickup plan is active."
    assert rag_result["answer"] == "resolved"
    objective_facts = rag_result["user_image"]["user_memory_context"][
        "objective_facts"
    ]
    assert objective_facts[0]["fact"] == "User likes tea"


@pytest.mark.asyncio
async def test_rag_evidence_request_shape_snapshot(monkeypatch) -> None:
    """Current RAG3 request shape should stay stable across adapter extraction."""
    captured: dict = {}

    async def _resolve_local_context(
        request: dict,
        context: dict,
        options: dict,
    ) -> dict:
        captured["request"] = {
            "resolver_request": request,
            "context": context,
            "options": options,
        }
        return_value = _local_context_packet(answer="snapshot answer")
        return return_value

    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        _resolve_local_context,
    )

    state = _rag_evidence_snapshot_state()
    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_rag_evidence",
    )

    expected_time_context = dict(state["local_time_context"])
    expected_time_context.update({
        "local_date": "2026-04-27",
        "local_time": "2026-04-27 00:00:00",
        "local_weekday": "Monday",
    })
    expected_request = {
        "resolver_request": {
            "schema_version": "local_context_resolver_request.v1",
            "objective": "Need current evidence.",
            "source": "l2d",
            "reason": "Cognition requested local context evidence.",
            "priority": "normal",
        },
        "context": {
            "schema_version": "local_context_resolver_context.v1",
            "character_name": "Kazusa",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "global_user_id": "user-1",
            "user_name": "User",
            "local_time_context": expected_time_context,
            "prompt_message_context": {
                "body_text": "Need current evidence.",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": ["character-1"],
                "broadcast": False,
            },
            "chat_history_recent": [{"role": "user", "content": "previous turn"}],
            "chat_history_wide": [{"role": "assistant", "content": "older turn"}],
            "conversation_progress": {
                "status": "active",
                "continuity": "same_episode",
                "current_thread": "Pickup plan is active.",
            },
            "original_user_request": "Need current evidence.",
            "current_timestamp_utc": "2026-04-26T12:00:00+00:00",
            "current_platform_message_id": "",
            "active_turn_platform_message_ids": ["msg-1", "msg-2"],
            "active_turn_conversation_row_ids": ["row-1", "row-2"],
            "session_media_refs": [],
        },
        "options": {
            "schema_version": "local_context_resolver_options.v1",
            "max_iterations": 3,
            "max_nodes": 8,
            "max_depth": 3,
            "max_node_attempts": 2,
            "max_subagent_attempts": 1,
        },
    }
    assert captured["request"] == expected_request
    assert rag_result["answer"] == "snapshot answer"


@pytest.mark.asyncio
async def test_rag_evidence_passes_empty_reply_context_to_wrapper(
    monkeypatch,
) -> None:
    """Empty reply metadata should not be required by the RAG3 boundary."""
    captured: dict = {}

    async def _resolve_local_context(
        request: dict,
        context: dict,
        options: dict,
    ) -> dict:
        del request, options
        captured["context"] = context
        return_value = _local_context_packet(answer="empty reply answer")
        return return_value

    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        _resolve_local_context,
    )
    state = _rag_evidence_snapshot_state()
    state["reply_context"] = {}

    rag_result = await supervisor_module.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_rag_evidence",
    )

    assert "reply_context" not in captured["context"]
    assert rag_result["answer"] == "empty reply answer"


@pytest.mark.asyncio
async def test_rag_evidence_skips_for_unresolved_referents(monkeypatch) -> None:
    """Unresolved required references should skip RAG and preserve payload shape."""
    resolve_local_context = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        resolve_local_context,
    )

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    rag_result = await supervisor_module.run_rag_evidence_for_persona_state({
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
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
    }, agent_name="resolver_rag_evidence")

    resolve_local_context.assert_not_awaited()
    assert rag_result["answer"] == ""
    assert rag_result["user_image"]["user_memory_context"]["stable_patterns"] == []
    assert rag_result["user_image"]["user_memory_context"]["active_commitments"] == []
    assert rag_result["character_image"] == {}
    assert rag_result["memory_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["unknown_slots"] == []


@pytest.mark.asyncio
async def test_rag_evidence_runs_for_mixed_referents(monkeypatch) -> None:
    """Mixed referents should not trigger the old binary RAG skip cliff."""

    resolve_local_context = AsyncMock(
        return_value=_local_context_packet(answer="partial evidence"),
    )
    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        resolve_local_context,
    )

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    rag_result = await supervisor_module.run_rag_evidence_for_persona_state({
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
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
    }, agent_name="resolver_rag_evidence")

    resolve_local_context.assert_awaited_once()
    assert rag_result["answer"] == "partial evidence"


@pytest.mark.asyncio
async def test_rag_evidence_skips_when_referents_are_all_unresolved(monkeypatch) -> None:
    """Structured unresolved referents should be authoritative."""
    resolve_local_context = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "resolve_local_context",
        resolve_local_context,
    )

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    rag_result = await supervisor_module.run_rag_evidence_for_persona_state({
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
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
    }, agent_name="resolver_rag_evidence")

    resolve_local_context.assert_not_awaited()
    assert rag_result["answer"] == ""
