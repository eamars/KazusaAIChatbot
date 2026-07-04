"""Production integration checks for the local-context resolver cutover."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
)
from kazusa_ai_chatbot.local_context_resolver import (
    DEFAULT_OPTION_LIMITS,
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock

pytestmark = pytest.mark.asyncio


def _resolver_request() -> dict[str, Any]:
    """Build a local-context recall capability request."""

    request = {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": "local_context_recall",
        "objective": "Find the local evidence for the user's request.",
        "reason": "The current cognition cycle needs local context.",
        "priority": "now",
    }
    return request


def _persona_state() -> dict[str, Any]:
    """Build a production-like persona state for resolver integration tests."""

    turn_clock = build_turn_clock("2026-07-04 09:30:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="local-context-cutover-episode",
        percept_id="local-context-cutover-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Please check the local evidence.",
        platform="debug",
        platform_channel_id="channel-123",
        channel_type="private",
        platform_message_id="message-123",
        platform_user_id="platform-user-123",
        global_user_id="global-user-123",
        user_name="Test User",
        active_turn_platform_message_ids=["message-123"],
        active_turn_conversation_row_ids=["row-123"],
        debug_modes={},
        target_addressed_user_ids=["character-123"],
        target_broadcast=False,
    )
    state = {
        "cognitive_episode": episode,
        "decontexualized_input": "Please check the local evidence.",
        "referents": [],
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-123",
        },
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "platform_message_id": "message-123",
        "platform_bot_id": "bot-123",
        "global_user_id": "global-user-123",
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "Please check the local evidence.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "channel_topic": "debug",
        "chat_history_recent": [
            {
                "speaker": "Test User",
                "text": "Earlier local evidence row.",
            },
        ],
        "chat_history_wide": [
            {
                "speaker": "Test User",
                "text": "Wider local evidence row.",
            },
        ],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {
            "current_thread": "local evidence check",
        },
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
    }
    return state


def _packet(*, answer: str = "") -> dict[str, Any]:
    """Build a minimal valid local-context packet for patched integration."""

    root = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": "root",
        "node_kind": "synthesis",
        "objective": "Find local evidence.",
        "parent_id": None,
        "children": ["memory_1"],
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
    memory_node = dict(root)
    memory_node.update({
        "node_id": "memory_1",
        "node_kind": "memory_evidence",
        "objective": "Use memory evidence.",
        "parent_id": "root",
        "children": [],
    })
    packet = {
        "schema_version": LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
        "investigation_summary": ["RAG3 found local evidence."],
        "knowledge_we_know_so_far": ["RAG3 found local evidence."],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "rag_result": {
            "answer": answer,
            "user_image": {},
            "user_memory_unit_candidates": [
                {
                    "summary": "Scoped user memory must not enter prewarm.",
                    "source_system": "user_memory_units",
                },
            ],
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [
                {
                    "summary": "Shared memory evidence from RAG3.",
                    "source_system": "memory",
                },
            ],
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
            "active_node_id": "memory_1",
            "nodes": {
                "root": root,
                "memory_1": memory_node,
            },
            "traversal_order": ["root", "memory_1"],
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


def _assert_common_local_context_io(
    request: dict[str, Any],
    context: dict[str, Any],
    options: dict[str, Any],
) -> None:
    """Assert production callers use the stable RAG3 public IO."""

    assert request["schema_version"] == LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION
    assert context["schema_version"] == LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION
    assert options["schema_version"] == LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION
    assert context["character_name"] == "Kazusa"
    assert context["platform"] == "debug"
    assert context["platform_channel_id"] == "channel-123"
    assert context["global_user_id"] == "global-user-123"
    assert context["user_name"] == "Test User"
    assert context["local_time_context"]["local_time"] == "2026-07-04 09:30:00"
    assert context["prompt_message_context"]["body_text"] == (
        "Please check the local evidence."
    )
    for option_name, option_value in DEFAULT_OPTION_LIMITS.items():
        assert options[option_name] == option_value


async def test_local_context_recall_uses_rag3_public_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2d-selected local-context recall should no longer call RAG2 supervisor."""

    calls: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    async def resolve_local_context(
        request: dict[str, Any],
        context: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture the production RAG3 request and return one packet."""

        calls.append((request, context, options))
        return_value = _packet(answer="RAG3 projected answer.")
        return return_value

    forbidden_rag2 = AsyncMock(side_effect=AssertionError("RAG2 was called"))
    record_rag_stage_event = AsyncMock()
    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        resolve_local_context,
        raising=False,
    )
    monkeypatch.setattr(
        capabilities,
        "call_quote_aware_rag_supervisor",
        forbidden_rag2,
        raising=False,
    )
    monkeypatch.setattr(
        capabilities.event_logging,
        "record_rag_stage_event",
        record_rag_stage_event,
    )

    observation = await capabilities.execute_resolver_capability_request(
        _resolver_request(),
        _persona_state(),
    )

    assert observation["capability_kind"] == "local_context_recall"
    assert observation["status"] == "succeeded"
    assert observation["rag_result"]["answer"] == "RAG3 projected answer."
    assert observation["rag_result"]["memory_evidence"]
    assert len(calls) == 1
    request, context, options = calls[0]
    assert request["source"] == "l2d"
    assert request["objective"] == "Find the local evidence for the user's request."
    assert request["reason"] == "The current cognition cycle needs local context."
    _assert_common_local_context_io(request, context, options)
    forbidden_rag2.assert_not_awaited()
    record_rag_stage_event.assert_awaited_once()
    event_kwargs = record_rag_stage_event.await_args.kwargs
    assert event_kwargs["agent_name"] == "resolver_local_context_recall"
    assert event_kwargs["status"] == "succeeded"
    assert event_kwargs["retrieval_count"] == 2


async def test_first_cycle_prewarm_uses_memory_worker_without_rag3_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-cycle prewarm should stay outside the full RAG3 resolver."""

    calls: list[dict[str, Any]] = []

    async def resolve_local_context(
        _request: dict[str, Any],
        _context: dict[str, Any],
        _options: dict[str, Any],
    ) -> dict[str, Any]:
        """Fail when prewarm enters the full local-context resolver."""

        raise AssertionError("prewarm must not call resolve_local_context")

    class FakePersistentMemorySearchAgent:
        """Capture the direct shared-memory prewarm worker call."""

        async def run(
            self,
            task: str,
            context: dict[str, Any],
            max_attempts: int = 3,
        ) -> dict[str, Any]:
            """Return one shared memory row and one forbidden scoped row."""

            calls.append({
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            })
            result = {
                "resolved": True,
                "result": [
                    {
                        "content": "Shared memory evidence from prewarm.",
                        "source_system": "memory",
                    },
                    {
                        "content": "Scoped user memory must not enter prewarm.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "global-user-123",
                    },
                ],
                "attempts": 1,
            }
            return result

    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        resolve_local_context,
        raising=False,
    )
    monkeypatch.setattr(
        capabilities,
        "PersistentMemorySearchAgent",
        FakePersistentMemorySearchAgent,
        raising=False,
    )

    rag_result = await capabilities.run_first_cycle_shared_memory_prewarm(
        _persona_state(),
    )

    assert rag_result["answer"] == ""
    assert "Shared memory evidence from prewarm." in repr(
        rag_result["memory_evidence"]
    )
    assert rag_result["user_memory_unit_candidates"] == []
    assert "Scoped user memory" not in repr(rag_result)
    assert len(calls) == 1
    worker_call = calls[0]
    assert worker_call["task"] == "Please check the local evidence."
    assert worker_call["max_attempts"] == 1
    assert worker_call["context"]["prompt_message_context"]["body_text"] == (
        "Please check the local evidence."
    )
