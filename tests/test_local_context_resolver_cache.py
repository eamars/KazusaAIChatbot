"""Cache2 integration tests for the local-context resolver."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    project_local_context_packet,
    resolve_local_context,
    validate_local_context_resolver_context,
    validate_local_context_resolver_request,
)
from kazusa_ai_chatbot.local_context_resolver import service as resolver_service
from kazusa_ai_chatbot.local_context_resolver.cache import (
    EXTERNAL_EVIDENCE_NODE_TTL_SECONDS,
    LIVE_CONTEXT_NODE_TTL_SECONDS,
    active_node_cache_ttl_seconds,
    build_active_node_cache_dependencies,
    build_active_node_cache_key,
    build_planner_cache_key,
)
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_runtime import dependency_matches_event


class _StageInvoker:
    """Return queued stage responses and retain payloads for inspection."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def __call__(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        if not self._responses:
            raise AssertionError("unexpected stage invocation")
        response = self._responses.pop(0)
        return response


@pytest.mark.asyncio
async def test_rag3_planner_and_active_node_use_cache2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated identical RAG3 calls should reuse planner and node cache."""

    runtime = RAGCache2Runtime(max_entries=10)
    planner = _StageInvoker([_planner_response()])
    node_resolver = _StageInvoker([_node_response()])

    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: runtime,
        raising=False,
    )
    monkeypatch.setattr(resolver_service, "_planner_stage_handler", planner)
    monkeypatch.setattr(resolver_service, "_node_stage_handler", node_resolver)

    first_packet = await resolve_local_context(
        _request(),
        _context(),
        _options(),
    )
    second_packet = await resolve_local_context(
        _request(),
        _context(),
        _options(),
    )
    second_rag_result = project_local_context_packet(second_packet)

    assert first_packet["trace_summary"]["planner_calls"] == 1
    assert first_packet["trace_summary"]["active_node_calls"] == 1
    assert second_packet["trace_summary"]["planner_calls"] == 0
    assert second_packet["trace_summary"]["active_node_calls"] == 0
    assert second_packet["trace_summary"]["planner_cache_hits"] == 1
    assert second_packet["trace_summary"]["active_node_cache_hits"] == 1
    assert len(planner.calls) == 1
    assert len(node_resolver.calls) == 1
    assert second_rag_result["memory_evidence"] == [{
        "summary": "#napcat is a playful local command anchor.",
        "source_policy": "shared_memory",
    }]


def test_planner_key_ignores_live_time_and_history_text_but_node_key_does_not(
) -> None:
    """Planner cache is aggressive while node cache remains evidence-exact."""

    request = _request()
    first_context = _context()
    first_context["chat_history_recent"] = [{
        "speaker": "operator",
        "text": "First history row for a repeated planning shape.",
    }]
    second_context = _context()
    second_context["local_time_context"] = {"local_date": "2026-07-05"}
    second_context["chat_history_recent"] = [{
        "speaker": "operator",
        "text": "Changed history row with the same source-domain shape.",
    }]
    options = _options()
    stage_identity = {"prompt_digest": "stage-v1", "model": "test-model"}
    active_node = {
        "node_kind": "live_context",
        "objective": "Read the supplied current local date.",
        "consumes": {},
        "produces": [],
    }

    first_planner_key = build_planner_cache_key(
        request=request,
        context=first_context,
        options=options,
        stage_identity=stage_identity,
    )
    second_planner_key = build_planner_cache_key(
        request=request,
        context=second_context,
        options=options,
        stage_identity=stage_identity,
    )
    first_node_key = build_active_node_cache_key(
        request=request,
        context=first_context,
        compact_context=resolver_service._compact_context(first_context),
        active_node=active_node,
        dependency_context=[],
        options=options,
        stage_identity=stage_identity,
    )
    second_node_key = build_active_node_cache_key(
        request=request,
        context=second_context,
        compact_context=resolver_service._compact_context(second_context),
        active_node=active_node,
        dependency_context=[],
        options=options,
        stage_identity=stage_identity,
    )

    assert first_planner_key == second_planner_key
    assert first_node_key != second_node_key


def test_active_node_cache_policy_uses_long_lived_stable_entries() -> None:
    """Stable local evidence nodes rely on key versioning and invalidation."""

    memory_dependencies = build_active_node_cache_dependencies(
        node_kind="memory_evidence",
        context=_context(),
    )
    conversation_dependencies = build_active_node_cache_dependencies(
        node_kind="conversation_evidence",
        context=_context(),
    )

    assert active_node_cache_ttl_seconds("memory_evidence") is None
    assert active_node_cache_ttl_seconds("person_context") is None
    assert active_node_cache_ttl_seconds("live_context") == (
        LIVE_CONTEXT_NODE_TTL_SECONDS
    )
    assert active_node_cache_ttl_seconds("external_evidence") == (
        EXTERNAL_EVIDENCE_NODE_TTL_SECONDS
    )
    assert memory_dependencies[0].source == "memory"
    assert conversation_dependencies[0].source == "conversation_history"
    assert conversation_dependencies[0].platform == "debug"
    assert conversation_dependencies[0].platform_channel_id == "group-1"


def test_active_node_cache_dependencies_match_user_global_invalidations() -> None:
    """User-global profile updates must invalidate stable RAG3 node caches."""

    scoped_dependencies = build_active_node_cache_dependencies(
        node_kind="scoped_memory",
        context=_context(),
    )
    person_dependencies = build_active_node_cache_dependencies(
        node_kind="person_context",
        context=_context(),
    )
    recall_dependencies = build_active_node_cache_dependencies(
        node_kind="recall_evidence",
        context=_context(),
    )
    cross_channel_user_event = CacheInvalidationEvent(
        source="user_profile",
        platform="debug",
        platform_channel_id="another-group",
        global_user_id="user-1",
    )
    other_user_event = CacheInvalidationEvent(
        source="user_profile",
        platform="debug",
        platform_channel_id="group-1",
        global_user_id="user-2",
    )

    assert dependency_matches_event(
        scoped_dependencies[0],
        cross_channel_user_event,
    )
    assert not dependency_matches_event(scoped_dependencies[0], other_user_event)
    assert any(
        dependency_matches_event(dependency, cross_channel_user_event)
        for dependency in recall_dependencies
    )
    assert dependency_matches_event(
        person_dependencies[0],
        other_user_event,
    )


def _request() -> dict[str, Any]:
    """Build a stable local-context request fixture."""

    request = validate_local_context_resolver_request({
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve what #napcat means in this group.",
        "source": "standalone_eval",
        "reason": "cache test",
        "priority": "normal",
    })
    return request


def _context() -> dict[str, Any]:
    """Build a stable prompt-safe local-context fixture."""

    context = validate_local_context_resolver_context({
        "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        "character_name": "active character",
        "platform": "debug",
        "platform_channel_id": "group-1",
        "global_user_id": "user-1",
        "user_name": "operator",
        "local_time_context": {"local_date": "2026-07-04"},
        "prompt_message_context": {
            "message_text": "@active character #napcat",
            "addressed_to_active_character": True,
        },
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
    })
    return context


def _options() -> dict[str, Any]:
    """Build default-sized local-context options for cache tests."""

    options = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        "max_iterations": 2,
        "max_nodes": 8,
        "max_depth": 3,
        "max_node_attempts": 2,
        "max_subagent_attempts": 1,
    }
    return options


def _planner_response() -> dict[str, object]:
    """Return one planner stage response for a memory lookup."""

    response = {
        "tasks": [{
            "objective": "Retrieve durable memory for the #napcat command.",
            "node_kind": "memory_evidence",
        }],
    }
    return response


def _node_response() -> dict[str, object]:
    """Return one active-node stage response for a memory lookup."""

    response = {
        "node_update": {
            "status": "resolved",
            "investigation_summary": [
                "Memory evidence resolved the #napcat command anchor.",
            ],
            "knowledge_we_know_so_far": [
                "#napcat is a playful local command anchor.",
            ],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        "artifacts": [{
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": "artifact_1",
            "artifact_type": "memory_ref",
            "producer_node_id": "active_node",
            "summary": "#napcat is a playful local command anchor.",
            "projection_payload": {
                "memory_evidence": [{
                    "summary": "#napcat is a playful local command anchor.",
                    "source_policy": "shared_memory",
                }],
            },
            "source_policy": "shared_memory",
            "prompt_visible": True,
        }],
    }
    return response
