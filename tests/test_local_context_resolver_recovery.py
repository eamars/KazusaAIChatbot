"""Deterministic recovery tests for local-context stage ownership."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    resolve_local_context,
    validate_local_context_resolver_context,
    validate_local_context_resolver_request,
)
from kazusa_ai_chatbot.local_context_resolver import service as resolver_service
from kazusa_ai_chatbot.local_context_resolver import stages as resolver_stages
from tests.task_resolution_test_helpers import _scene_context


class _QueuedStageLLM:
    """Return queued stage responses and retain every prompt payload."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages, *, config):
        del config
        self.calls.append(json.loads(messages[1].content))
        if not self.responses:
            raise AssertionError("unexpected stage invocation")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(content=response)


class _EmptyCacheRuntime:
    """Avoid cross-test Cache2 state while exercising the service boundary."""

    async def get(self, *args, **kwargs):
        del args, kwargs

    async def store(self, *args, **kwargs):
        del args, kwargs


def _request(objective: str) -> dict[str, object]:
    return validate_local_context_resolver_request({
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": objective,
        "source": "test",
        "reason": "deterministic recovery test",
        "priority": "normal",
    })


def _context() -> dict[str, object]:
    return validate_local_context_resolver_context({
        "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        "character_name": "active character",
        "platform": "debug",
        "platform_channel_id": "recovery-channel",
        "global_user_id": "recovery-user",
        "user_name": "operator",
        "scene_context": _scene_context(),
        "local_time_context": {"local_date": "2026-07-04"},
        "prompt_message_context": {
            "message_text": "recover local context",
            "addressed_to_active_character": True,
        },
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
    })


def _options() -> dict[str, object]:
    return {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        "max_iterations": 4,
        "max_nodes": 8,
        "max_depth": 3,
        "max_node_attempts": 2,
        "max_subagent_attempts": 1,
    }


def _resolved_node_response(summary: str) -> dict[str, object]:
    return {
        "node_update": {
            "status": "resolved",
            "investigation_summary": [summary],
            "knowledge_we_know_so_far": [summary],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": ["bounded recovery test evidence"],
        },
        "artifacts": [],
    }


def _blocked_node_response() -> dict[str, object]:
    return {
        "node_update": {
            "status": "blocked",
            "investigation_summary": ["The source was unavailable."],
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": ["The source result is unavailable."],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": ["bounded recovery test evidence"],
        },
        "artifacts": [],
    }


def _synthesis_response() -> dict[str, object]:
    return {
        "investigation_summary": ["The resolved local evidence was retained."],
        "knowledge_we_know_so_far": ["A bounded local fact was observed."],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": ["bounded recovery test evidence"],
    }


def _accept_candidate(candidate: dict[str, object]) -> None:
    """Accept a parsed candidate for direct stage-runner tests."""

    del candidate


@pytest.mark.asyncio
async def test_stage_repair_prompt_carries_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local stage retry carries exactly the bounded contract block."""

    fake_llm = _QueuedStageLLM([
        json.dumps({
            "tasks": [{
                "objective": "retrieve one local fact",
                "node_kind": "unsupported_node_kind",
            }],
        }),
        json.dumps({
            "tasks": [{
                "objective": "retrieve one local fact",
                "node_kind": "memory_evidence",
            }],
        }),
    ])
    monkeypatch.setattr(resolver_stages, "_planner_llm", fake_llm)

    async def record_trace(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        resolver_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    resolver_stages.drain_stage_trace_records()

    request = _request("retrieve one local fact")
    options = _options()

    def validate_planner_candidate(
        candidate: dict[str, object],
    ) -> None:
        resolver_service._graph_from_planner_response(
            request,
            candidate,
            options,
        )

    result = await resolver_stages.plan_local_context_graph({
        "stage": "graph_planner",
        "request": {"objective": "retrieve one local fact"},
    }, candidate_validator=validate_planner_candidate)

    assert result["tasks"][0]["node_kind"] == "memory_evidence"
    assert len(fake_llm.calls) == 2
    repair = fake_llm.calls[1]["contract_repair"]
    assert set(repair) == {
        "repair_instruction",
        "reason",
        "contract_error",
        "invalid_candidate",
    }
    assert "expected known node kind" in repair["contract_error"]
    assert "unsupported_node_kind" in repair["invalid_candidate"]
    assert "JSON syntax" not in repair["repair_instruction"]
    assert fake_llm.calls[1]["stage"] == "graph_planner"


@pytest.mark.asyncio
async def test_stage_provider_failure_is_contained_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider failures consume bounded local-stage attempts."""

    fake_llm = _QueuedStageLLM([
        RuntimeError("provider unavailable"),
        RuntimeError("provider unavailable"),
        json.dumps(_resolved_node_response("node recovered")),
    ])
    monkeypatch.setattr(resolver_stages, "_node_llm", fake_llm)

    async def record_trace(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        resolver_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    resolver_stages.drain_stage_trace_records()

    result = await resolver_stages.resolve_local_context_node({
        "stage": "active_node_resolver",
        "active_node": {"node_id": "task_1"},
    }, candidate_validator=_accept_candidate)

    assert result["node_update"]["status"] == "resolved"
    assert len(fake_llm.calls) == 3
    assert all(
        set(call["contract_repair"]) == {
            "repair_instruction",
            "reason",
            "contract_error",
            "invalid_candidate",
        }
        for call in fake_llm.calls[1:]
    )
    assert all(call["contract_repair"]["contract_error"] == ""
               for call in fake_llm.calls[1:])


@pytest.mark.asyncio
async def test_planner_provider_exhaustion_returns_blocked_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner exhaustion returns the existing bounded blocked packet."""

    planner_llm = _QueuedStageLLM([
        RuntimeError("planner provider unavailable"),
        RuntimeError("planner provider unavailable"),
        RuntimeError("planner provider unavailable"),
    ])
    monkeypatch.setattr(resolver_stages, "_planner_llm", planner_llm)
    async def record_trace(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        resolver_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    resolver_stages.drain_stage_trace_records()
    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: _EmptyCacheRuntime(),
    )

    packet = await resolve_local_context(
        _request("planner recovery"),
        _context(),
        _options(),
    )

    assert len(planner_llm.calls) == 3
    assert packet["graph"]["nodes"]["root"]["status"] == "blocked"
    assert packet["trace_summary"]["failure_stage"] == "planner"
    assert packet["trace_summary"]["attempt_diagnostics"] == [{
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "local_context_resolver.graph_planner",
        "error_code": "local_context_planner_blocked",
        "attempt_count": 3,
        "safe_checkpoint": "pre_state_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }]


@pytest.mark.asyncio
async def test_active_node_exhaustion_blocks_one_node_and_continues_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One blocked active node does not discard later evidence nodes."""

    planner_llm = _QueuedStageLLM([
        json.dumps({
            "tasks": [
                {
                    "objective": "first local fact",
                    "node_kind": "memory_evidence",
                },
                {
                    "objective": "second local fact",
                    "node_kind": "conversation_evidence",
                },
            ],
        }),
    ])
    invalid_node_response = _resolved_node_response("invalid node status")
    invalid_node_response["node_update"]["status"] = "unsupported_status"
    node_llm = _QueuedStageLLM([
        json.dumps(invalid_node_response),
        json.dumps(invalid_node_response),
        json.dumps(invalid_node_response),
        json.dumps(_resolved_node_response("second local fact resolved")),
    ])
    synthesizer_llm = _QueuedStageLLM([json.dumps(_synthesis_response())])
    monkeypatch.setattr(resolver_stages, "_planner_llm", planner_llm)
    monkeypatch.setattr(resolver_stages, "_node_llm", node_llm)
    monkeypatch.setattr(resolver_stages, "_synthesizer_llm", synthesizer_llm)

    async def record_trace(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        resolver_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    resolver_stages.drain_stage_trace_records()
    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: _EmptyCacheRuntime(),
    )

    packet = await resolve_local_context(
        _request("active node recovery"),
        _context(),
        _options(),
    )

    assert len(planner_llm.calls) == 1
    assert len(node_llm.calls) == 4
    assert len(synthesizer_llm.calls) == 1
    assert all(
        call["contract_repair"]["contract_error"]
        for call in node_llm.calls[1:3]
    )
    assert all(
        "unsupported_status" in call["contract_repair"]["invalid_candidate"]
        for call in node_llm.calls[1:3]
    )
    assert packet["graph"]["nodes"]["task_1"]["status"] == "blocked"
    assert packet["graph"]["nodes"]["task_2"]["status"] == "resolved"
    diagnostics = packet["trace_summary"]["attempt_diagnostics"]
    assert [row["error_code"] for row in diagnostics] == [
        "local_context_node_blocked",
    ]
    assert diagnostics[0]["final_status"] == "accepted_degraded"


@pytest.mark.asyncio
async def test_collapse_exhaustion_defaults_to_no_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collapse review exhaustion leaves resolved nodes uncollapsed."""

    async def planner(
        payload: dict[str, object],
        *,
        candidate_validator,
    ) -> dict[str, object]:
        del payload
        candidate = {
            "tasks": [
                {
                    "objective": "first duplicate fact",
                    "node_kind": "memory_evidence",
                },
                {
                    "objective": "second duplicate fact",
                    "node_kind": "memory_evidence",
                },
            ],
        }
        candidate_validator(candidate)
        return candidate

    async def node_resolver(
        payload: dict[str, object],
        *,
        candidate_validator,
    ) -> dict[str, object]:
        del payload
        candidate = _resolved_node_response("duplicate fact resolved")
        candidate_validator(candidate)
        return candidate

    collapse_llm = _QueuedStageLLM([
        json.dumps({"collapse_decision": {"should_collapse": True}}),
        json.dumps({
            "collapse_decision": {
                "should_collapse": False,
                "target_candidate_ref": "",
            },
        }),
        json.dumps({
            "collapse_decision": {
                "should_collapse": "false",
                "target_candidate_ref": "",
                "reason": "none",
            },
        }),
    ])

    monkeypatch.setattr(resolver_service, "_planner_stage_handler", planner)
    monkeypatch.setattr(resolver_service, "_node_stage_handler", node_resolver)
    monkeypatch.setattr(resolver_stages, "_collapse_llm", collapse_llm)
    async def record_trace(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        resolver_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    resolver_stages.drain_stage_trace_records()
    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: _EmptyCacheRuntime(),
    )

    packet = await resolve_local_context(
        _request("collapse recovery"),
        _context(),
        _options(),
    )

    assert len(collapse_llm.calls) == 3
    assert all(
        set(call["contract_repair"]) == {
            "repair_instruction",
            "reason",
            "contract_error",
            "invalid_candidate",
        }
        for call in collapse_llm.calls[1:]
    )
    assert all(call["contract_repair"]["contract_error"]
               for call in collapse_llm.calls[1:])
    assert packet["graph"]["nodes"]["task_1"]["status"] == "resolved"
    assert packet["graph"]["nodes"]["task_2"]["status"] == "resolved"
    assert packet["graph"]["collapse_events"] == []
    assert packet["trace_summary"]["collapse_calls"] == 1
    diagnostics = packet["trace_summary"]["attempt_diagnostics"]
    assert [row["error_code"] for row in diagnostics] == [
        "local_context_collapse_skipped",
    ]
    assert diagnostics[0]["final_status"] == "skipped"


@pytest.mark.asyncio
async def test_synthesis_exhaustion_uses_deterministic_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis exhaustion retains the service-owned deterministic product."""

    async def planner(
        payload: dict[str, object],
        *,
        candidate_validator,
    ) -> dict[str, object]:
        del payload
        candidate = {
            "tasks": [{
                "objective": "unavailable local fact",
                "node_kind": "memory_evidence",
            }],
        }
        candidate_validator(candidate)
        return candidate

    async def node_resolver(
        payload: dict[str, object],
        *,
        candidate_validator,
    ) -> dict[str, object]:
        del payload
        candidate = _blocked_node_response()
        candidate_validator(candidate)
        return candidate

    synthesizer_llm = _QueuedStageLLM([
        json.dumps({"investigation_summary": []}),
        json.dumps({
            "investigation_summary": [],
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
        }),
        json.dumps({
            "investigation_summary": [],
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [""],
        }),
    ])

    monkeypatch.setattr(resolver_service, "_planner_stage_handler", planner)
    monkeypatch.setattr(resolver_service, "_node_stage_handler", node_resolver)
    monkeypatch.setattr(resolver_stages, "_synthesizer_llm", synthesizer_llm)
    async def record_trace(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {}

    monkeypatch.setattr(
        resolver_stages.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    resolver_stages.drain_stage_trace_records()
    monkeypatch.setattr(
        resolver_service,
        "get_rag_cache2_runtime",
        lambda: _EmptyCacheRuntime(),
    )

    packet = await resolve_local_context(
        _request("synthesis recovery"),
        _context(),
        _options(),
    )

    assert len(synthesizer_llm.calls) == 3
    assert all(
        set(call["contract_repair"]) == {
            "repair_instruction",
            "reason",
            "contract_error",
            "invalid_candidate",
        }
        for call in synthesizer_llm.calls[1:]
    )
    assert all(call["contract_repair"]["contract_error"]
               for call in synthesizer_llm.calls[1:])
    assert packet["trace_summary"]["synthesis_calls"] == 1
    deterministic = resolver_service._deterministic_synthesis_response(
        packet["graph"]
    )
    assert packet["investigation_summary"] == (
        deterministic["investigation_summary"]
    )
    assert packet["knowledge_we_know_so_far"] == (
        deterministic["knowledge_we_know_so_far"]
    )
    diagnostics = packet["trace_summary"]["attempt_diagnostics"]
    assert [row["error_code"] for row in diagnostics] == [
        "local_context_synthesis_degraded",
    ]
    assert diagnostics[0]["final_status"] == "accepted_degraded"
