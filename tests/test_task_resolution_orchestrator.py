"""Deterministic tests for bounded task-orchestrator dispatch behavior."""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from typing import Any

import pytest


def _context() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_execution_context.v1",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "debug:user:test-user",
        "channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "source_message_id": "message-1",
        "local_time_context": {"local_time": "2026-08-01 10:00"},
        "prompt_message_context": {"text": "Resolve this public source."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A bounded test persona.",
        "conversation_summary": "A bounded test conversation.",
        "current_timestamp_utc": "2026-07-31T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": [],
        "session_media_refs": [],
        "coding_workspace_root": "C:/workspace/test",
        "max_output_chars": 3000,
    }


def _checkpoint() -> dict[str, object]:
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    return state.create_task_resolution_checkpoint(
        {
            "capability": "task_resolution_request",
            "semantic_goal": "Analyze a public documentation page.",
            "reason": "The answer requires source evidence.",
            "evidence_handles": ["e1"],
        },
        _context(),
    )


def _result(
    *,
    specialist: str,
    status: str,
    evidence: list[dict[str, object]] | None = None,
    remaining_needs: list[str] | None = None,
    retryable: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "task_specialist_result.v1",
        "specialist": specialist,
        "status": status,
        "evidence": list(evidence or []),
        "completed_subgoals": [],
        "remaining_needs": list(remaining_needs or []),
        "reason": f"{specialist} returned {status}.",
        "retryable": retryable,
    }


@pytest.mark.asyncio
async def test_wrong_text_selection_reroutes_to_public_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incompatible specialist consumes budget without failing the task."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    selections = iter((
        {
            "specialist": "text_computation",
            "subgoal": "Read the public page without repository work.",
            "coding_objective_mode": "none",
        },
        {
            "specialist": "public_research",
            "subgoal": "Retrieve evidence from the public page.",
            "coding_objective_mode": "none",
        },
    ))
    objectives: list[str] = []

    async def select_next(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        return next(selections)

    async def text_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        objectives.append(str(request["objective"]))
        return _result(
            specialist="text_computation",
            status="incompatible",
            remaining_needs=["Public page evidence is still required."],
        )

    async def public_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        objectives.append(str(request["objective"]))
        return _result(
            specialist="public_research",
            status="resolved",
            evidence=[{
                "schema_version": "task_resolution_evidence.v1",
                "evidence_id": "public-evidence-1",
                "task_node_id": "node-1",
                "specialist": "public_research",
                "summary": "The public page supplied the required facts.",
                "provenance_refs": ["https://example.test/docs"],
                "limitations": [],
            }],
        )

    handlers: dict[
        str,
        Callable[[dict[str, object], dict[str, object]], Awaitable[dict[str, object]]],
    ] = {
        "text_computation": text_handler,
        "public_research": public_handler,
    }
    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)
    monkeypatch.setattr(
        orchestrator,
        "specialist_handler",
        lambda specialist: handlers[specialist],
    )

    result = await orchestrator.run_task_orchestrator(
        _checkpoint(),
        _context(),
        inline_deadline=None,
    )

    assert result["status"] == "resolved"
    assert result["evidence"][0]["specialist"] == "public_research"
    assert objectives == [
        "Read the public page without repository work.",
        "Retrieve evidence from the public page.",
    ]


@pytest.mark.asyncio
async def test_inline_coding_selection_defers_handover_without_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coding selection becomes durable handover before any coding call."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )

    async def select_next(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        return {
            "specialist": "coding",
            "subgoal": "Prepare a repository patch proposal.",
            "coding_objective_mode": "propose_patch",
        }

    def unexpected_handler(_specialist: str) -> Any:
        raise AssertionError("coding handler must not run inline")

    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)
    monkeypatch.setattr(orchestrator, "specialist_handler", unexpected_handler)

    result = await orchestrator.run_task_orchestrator(
        _checkpoint(),
        _context(),
        inline_deadline=orchestrator.monotonic() + 30.0,
    )

    assert result["status"] == "deferred"
    assert result["checkpoint"]["pending_dispatch"] == {
        "schema_version": "task_pending_dispatch.v1",
        "task_node_id": "node-1",
        "specialist": "coding",
        "subgoal": "Prepare a repository patch proposal.",
        "coding_objective_mode": "propose_patch",
        "phase": "selected",
    }


@pytest.mark.asyncio
async def test_structural_replacement_consumes_persisted_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid selection structure consumes one of four durable LLM calls."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    seen_counts: list[int] = []

    async def select_next(
        checkpoint: dict[str, object],
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, str]:
        seen_counts.append(int(checkpoint["orchestrator_call_count"]))
        if len(seen_counts) == 1:
            raise orchestrator.TaskResolutionContractError(
                "orchestrator selection: malformed structure"
            )
        return {
            "specialist": "local_context",
            "subgoal": "Retrieve the relevant local context.",
            "coding_objective_mode": "none",
        }

    async def local_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        return _result(
            specialist="local_context",
            status="resolved",
            evidence=[{
                "schema_version": "task_resolution_evidence.v1",
                "evidence_id": "local-evidence-1",
                "task_node_id": request["task_node_id"],
                "specialist": "local_context",
                "summary": "The local context resolved the task.",
                "provenance_refs": ["conversation:row-1"],
                "limitations": [],
            }],
        )

    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)
    monkeypatch.setattr(orchestrator, "specialist_handler", lambda _name: local_handler)

    result = await orchestrator.run_task_orchestrator(
        _checkpoint(),
        _context(),
        inline_deadline=None,
    )

    assert result["status"] == "resolved"
    assert seen_counts == [1, 2]


@pytest.mark.asyncio
async def test_started_dispatch_is_not_relaunched_after_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after durable start becomes unavailable without redispatch."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    checkpoint = _checkpoint()
    checkpoint["orchestrator_call_count"] = 1
    checkpoint["pending_dispatch"] = {
        "schema_version": "task_pending_dispatch.v1",
        "task_node_id": "node-1",
        "specialist": "public_research",
        "subgoal": "Read the selected public source.",
        "coding_objective_mode": "none",
        "phase": "started",
    }

    def unexpected_handler(_specialist: str) -> Any:
        raise AssertionError("started dispatch must not be relaunched")

    monkeypatch.setattr(orchestrator, "specialist_handler", unexpected_handler)

    result = await orchestrator.run_task_orchestrator(
        checkpoint,
        _context(),
        inline_deadline=None,
    )

    assert result["status"] == "unavailable"


@pytest.mark.asyncio
async def test_partial_remaining_need_runs_local_then_public_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated remaining need activates a child node inside one loop."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    selections = iter((
        {
            "specialist": "local_context",
            "subgoal": "Retrieve local context for the request.",
            "coding_objective_mode": "none",
        },
        {
            "specialist": "public_research",
            "subgoal": "Verify the remaining public fact.",
            "coding_objective_mode": "none",
        },
    ))
    visited_nodes: list[str] = []

    async def select_next(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        return next(selections)

    async def local_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        node_id = str(request["task_node_id"])
        visited_nodes.append(node_id)
        return _result(
            specialist="local_context",
            status="partial",
            evidence=[{
                "schema_version": "task_resolution_evidence.v1",
                "evidence_id": "local-evidence-1",
                "task_node_id": node_id,
                "specialist": "local_context",
                "summary": "Local context identified the subject.",
                "provenance_refs": ["conversation:row-1"],
                "limitations": ["A current public fact remains."],
            }],
            remaining_needs=["Verify the current public fact."],
        )

    async def public_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        node_id = str(request["task_node_id"])
        visited_nodes.append(node_id)
        return _result(
            specialist="public_research",
            status="resolved",
            evidence=[{
                "schema_version": "task_resolution_evidence.v1",
                "evidence_id": "public-evidence-1",
                "task_node_id": node_id,
                "specialist": "public_research",
                "summary": "The public fact was verified.",
                "provenance_refs": ["https://example.test/source"],
                "limitations": [],
            }],
        )

    handlers = {
        "local_context": local_handler,
        "public_research": public_handler,
    }
    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)
    monkeypatch.setattr(
        orchestrator,
        "specialist_handler",
        lambda specialist: handlers[specialist],
    )

    result = await orchestrator.run_task_orchestrator(
        _checkpoint(),
        _context(),
        inline_deadline=None,
    )

    assert result["status"] == "resolved"
    assert visited_nodes == ["node-1", "node-2"]
    assert [row["specialist"] for row in result["evidence"]] == [
        "local_context",
        "public_research",
    ]


@pytest.mark.asyncio
async def test_public_dependency_stops_at_coding_handover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public evidence may lead to coding handover without coding execution."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    selections = iter((
        {
            "specialist": "public_research",
            "subgoal": "Identify the documented repository requirement.",
            "coding_objective_mode": "none",
        },
        {
            "specialist": "coding",
            "subgoal": "Prepare a patch proposal for the documented change.",
            "coding_objective_mode": "propose_patch",
        },
    ))

    async def select_next(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        return next(selections)

    async def public_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        node_id = str(request["task_node_id"])
        return _result(
            specialist="public_research",
            status="partial",
            evidence=[{
                "schema_version": "task_resolution_evidence.v1",
                "evidence_id": "public-requirement-1",
                "task_node_id": node_id,
                "specialist": "public_research",
                "summary": "The public documentation defines the change.",
                "provenance_refs": ["https://example.test/docs"],
                "limitations": ["A repository proposal remains."],
            }],
            remaining_needs=["Prepare the repository proposal."],
        )

    def handler_for(specialist: str) -> Any:
        if specialist == "coding":
            raise AssertionError("coding handler must remain outside this test")
        return public_handler

    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)
    monkeypatch.setattr(orchestrator, "specialist_handler", handler_for)

    result = await orchestrator.run_task_orchestrator(
        _checkpoint(),
        _context(),
        inline_deadline=orchestrator.monotonic() + 30.0,
    )

    assert result["status"] == "deferred"
    assert result["checkpoint"]["active_node_id"] == "node-2"
    assert result["checkpoint"]["pending_dispatch"]["specialist"] == "coding"
    assert result["checkpoint"]["pending_dispatch"][
        "coding_objective_mode"
    ] == "propose_patch"


def test_retryable_temporary_failure_allows_one_second_invocation() -> None:
    """A typed transient failure is the only blind-evidence retry exception."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    checkpoint = state.record_specialist_result(
        _checkpoint(),
        _result(
            specialist="public_research",
            status="temporarily_unavailable",
            remaining_needs=["Retry the temporarily unavailable public source."],
            retryable=True,
        ),
    )

    candidates = orchestrator._eligible_specialists(checkpoint)

    assert "public_research" in candidates


@pytest.mark.asyncio
async def test_inline_budget_below_dispatch_floor_defers_without_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreground budget exhaustion returns the same untouched checkpoint."""

    orchestrator = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.orchestrator"
    )
    checkpoint = _checkpoint()

    def unexpected_handler(_specialist: str) -> Any:
        raise AssertionError("specialist dispatch should not start")

    monkeypatch.setattr(orchestrator, "specialist_handler", unexpected_handler)

    result = await orchestrator.run_task_orchestrator(
        checkpoint,
        _context(),
        inline_deadline=0.0,
    )

    assert result["status"] == "deferred"
    assert result["checkpoint"]["session_id"] == checkpoint["session_id"]
    assert result["checkpoint"]["dispatch_count"] == 0
