"""Source-hydration tests for the local-context resolver."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    project_local_context_packet,
    resolve_local_context,
)
from kazusa_ai_chatbot.local_context_resolver import service as resolver_service
from kazusa_ai_chatbot.local_context_resolver import source_hydration
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime


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


class _FakeMemoryEvidenceAgent:
    """Fake source agent that returns scoped user-memory evidence."""

    calls: list[dict[str, object]] = []

    def __init__(self, *, cache_runtime: object | None = None) -> None:
        del cache_runtime

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        self.calls.append({
            "task": task,
            "context": dict(context),
            "max_attempts": max_attempts,
        })
        result = {
            "resolved": True,
            "result": {
                "selected_summary": "The user likes compact RAG traces.",
                "primary_worker": "user_memory_evidence_agent",
                "source_policy": "scoped current-user continuity evidence",
                "resolved_refs": [],
                "projection_payload": {
                    "memory_rows": [{
                        "content": "The user likes compact RAG traces.",
                        "source_system": "user_memory_units",
                        "scope_global_user_id": "user-1",
                    }],
                },
                "evidence": ["The user likes compact RAG traces."],
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False},
        }
        return result


@pytest.mark.asyncio
async def test_source_hydration_projects_source_agent_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production-enabled RAG3 active nodes hydrate specialist evidence."""

    runtime = RAGCache2Runtime(max_entries=10)
    planner = _StageInvoker([_planner_response()])
    node_resolver = _StageInvoker([_blocked_node_response()])
    _FakeMemoryEvidenceAgent.calls = []

    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: runtime,
        raising=False,
    )
    monkeypatch.setattr(resolver_service, "_planner_stage_handler", planner)
    monkeypatch.setattr(resolver_service, "_node_stage_handler", node_resolver)
    monkeypatch.setattr(
        source_hydration,
        "MemoryEvidenceAgent",
        _FakeMemoryEvidenceAgent,
    )

    packet = await resolve_local_context(
        _request(),
        _context(source_hydration_enabled=True),
        _options(),
    )
    rag_result = project_local_context_packet(packet)
    source_context = node_resolver.calls[0]["context"]["source_context"]
    source_projection = source_context[0]["projection_payload"]
    source_memory_row = source_projection["memory_evidence"][0]

    assert packet["graph"]["nodes"]["task_1"]["status"] == "resolved"
    assert packet["trace_summary"]["subagent_calls"] == 1
    assert len(_FakeMemoryEvidenceAgent.calls) == 1
    assert _FakeMemoryEvidenceAgent.calls[0]["task"].startswith("Memory-evidence:")
    assert "current user's private continuity evidence" in (
        _FakeMemoryEvidenceAgent.calls[0]["task"]
    )
    assert "scope_global_user_id" not in source_memory_row
    assert rag_result["user_memory_unit_candidates"] == [{
        "content": "The user likes compact RAG traces.",
        "source_system": "user_memory_units",
    }]


@pytest.mark.asyncio
async def test_source_hydration_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone RAG3 calls keep the supplied-context-only contract."""

    runtime = RAGCache2Runtime(max_entries=10)
    planner = _StageInvoker([_planner_response()])
    node_resolver = _StageInvoker([_resolved_node_response()])
    _FakeMemoryEvidenceAgent.calls = []

    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: runtime,
        raising=False,
    )
    monkeypatch.setattr(resolver_service, "_planner_stage_handler", planner)
    monkeypatch.setattr(resolver_service, "_node_stage_handler", node_resolver)
    monkeypatch.setattr(
        source_hydration,
        "MemoryEvidenceAgent",
        _FakeMemoryEvidenceAgent,
    )

    packet = await resolve_local_context(
        _request(),
        _context(source_hydration_enabled=False),
        _options(),
    )

    assert packet["trace_summary"]["subagent_calls"] == 0
    assert len(_FakeMemoryEvidenceAgent.calls) == 0
    assert "source_context" not in node_resolver.calls[0]["context"]


def _request() -> dict[str, Any]:
    """Build a stable local-context request fixture."""

    request = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": "Recall current-user continuity about RAG traces.",
        "source": "l2d",
        "reason": "source hydration test",
        "priority": "normal",
    }
    return request


def _context(*, source_hydration_enabled: bool) -> dict[str, Any]:
    """Build a production-shaped local-context context fixture."""

    context = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        "character_name": "active character",
        "platform": "debug",
        "platform_channel_id": "group-1",
        "global_user_id": "user-1",
        "user_name": "operator",
        "local_time_context": {"local_date": "2026-07-04"},
        "prompt_message_context": {
            "message_text": "@active character what do you remember?",
            "addressed_to_active_character": True,
        },
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "original_user_request": "what do you remember about RAG traces?",
        "current_timestamp_utc": "2026-07-04T00:00:00+00:00",
        "current_platform_message_id": "message-1",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": ["conversation-1"],
        "source_hydration_enabled": source_hydration_enabled,
    }
    return context


def _options() -> dict[str, Any]:
    """Build default-sized local-context options for source-hydration tests."""

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
    """Return one planner stage response for a scoped memory lookup."""

    response = {
        "tasks": [{
            "objective": "Retrieve current-user continuity about RAG traces.",
            "node_kind": "scoped_memory",
        }],
    }
    return response


def _blocked_node_response() -> dict[str, object]:
    """Return a node response that source hydration must recover."""

    response = {
        "node_update": {
            "status": "blocked",
            "investigation_summary": ["No supplied rows were enough."],
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": ["scoped memory evidence"],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        "artifacts": [],
    }
    return response


def _resolved_node_response() -> dict[str, object]:
    """Return a normal supplied-context-only active-node response."""

    response = {
        "node_update": {
            "status": "resolved",
            "investigation_summary": ["Supplied rows were enough."],
            "knowledge_we_know_so_far": ["Supplied memory was found."],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        "artifacts": [],
    }
    return response
