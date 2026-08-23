"""Deterministic adapters over the four retained Kazusa specialists."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import pytest

from agentic_resolver.integrations import kazusa_tools
from kazusa_ai_chatbot.task_resolution.contracts import (
    TASK_RESOLUTION_EVIDENCE_VERSION,
    TASK_SPECIALIST_RESULT_VERSION,
)
from kazusa_ai_chatbot.task_resolution.specialists import coding, public_research
from tests.test_task_resolution_orchestrator import _context


def _specialist_result(
    request: Mapping[str, object],
    *,
    specialist: str,
    status: str = "resolved",
) -> dict[str, object]:
    """Return one strict retained specialist result fixture."""

    evidence: list[dict[str, object]] = []
    completed_subgoals: list[str] = []
    remaining_needs: list[str] = []
    if status == "resolved":
        evidence = [{
            "schema_version": TASK_RESOLUTION_EVIDENCE_VERSION,
            "evidence_id": f"{specialist}-evidence",
            "task_node_id": request["task_node_id"],
            "specialist": specialist,
            "summary": f"{specialist} returned bounded evidence.",
            "provenance_refs": [f"fixture:{specialist}"],
            "limitations": [],
        }]
        completed_subgoals = [str(request["objective"])]
    else:
        remaining_needs = [str(request["objective"])]
    reason = f"{specialist} retained its existing lifecycle."
    if status == "approval_required":
        reason = "coding retained its existing approval lifecycle."
    result = {
        "schema_version": TASK_SPECIALIST_RESULT_VERSION,
        "specialist": specialist,
        "status": status,
        "evidence": evidence,
        "completed_subgoals": completed_subgoals,
        "remaining_needs": remaining_needs,
        "reason": reason,
        "retryable": False,
    }
    return result


def test_kazusa_registry_exposes_four_existing_specialists() -> None:
    """The integration roster contains exactly the four approved handlers."""

    registry = kazusa_tools.build_kazusa_tool_registry(_context())

    assert registry.names == (
        "coding",
        "local_context",
        "public_research",
        "text_computation",
    )
    assert registry.get("coding").side_effect_class == "approval_gated"
    assert all(
        registry.get(name).side_effect_class == "read"
        for name in (
            "local_context",
            "public_research",
            "text_computation",
        )
    )


@pytest.mark.asyncio
async def test_kazusa_adapters_call_existing_handlers_without_modifying_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each native adapter passes the exact retained specialist request shape."""

    calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def _handler_for(specialist: str):
        async def _handler(
            request: dict[str, object],
            execution_context: dict[str, object],
        ) -> dict[str, object]:
            calls.append((specialist, request, execution_context))
            result = _specialist_result(request, specialist=specialist)
            return result

        return _handler

    for specialist in (
        "local_context",
        "public_research",
        "coding",
        "text_computation",
    ):
        monkeypatch.setattr(
            kazusa_tools,
            f"resolve_with_{specialist}",
            _handler_for(specialist),
        )
    context = _context()
    registry = kazusa_tools.build_kazusa_tool_registry(context)

    for specialist in registry.names:
        arguments: dict[str, object] = {
            "objective": f"Resolve through {specialist}.",
        }
        if specialist == "coding":
            arguments["coding_objective_mode"] = "read_only"
        result = await registry.execute_tool(
            specialist,
            arguments,
            permission_scope={},
            timeout_seconds=5,
            maximum_result_characters=8_000,
        )
        assert result.status == "success"
        assert result.output["specialist"] == specialist

    assert [call[0] for call in calls] == list(registry.names)
    for specialist, request, execution_context in calls:
        assert set(request) == {
            "schema_version",
            "task_node_id",
            "objective",
            "available_evidence",
            "remaining_needs",
            "trusted_scope",
            "coding_objective_mode",
        }
        assert request["schema_version"] == "task_specialist_request.v1"
        expected_mode = "read_only" if specialist == "coding" else "none"
        assert request["coding_objective_mode"] == expected_mode
        assert execution_context == context
        assert request["trusted_scope"] == {
            "trigger_source": "agentic_resolver",
            "platform": context["platform"],
            "channel_id": context["channel_id"],
            "channel_type": context["channel_type"],
            "source_message_id": context["source_message_id"],
            "requester_global_user_id": context["requester_global_user_id"],
            "requester_platform_user_id": (
                context["requester_platform_user_id"]
            ),
        }


def test_public_research_tool_retains_existing_web_agent_ownership() -> None:
    """Public research remains the complex resolver handler, not a web bypass."""

    adapter_source = inspect.getsource(kazusa_tools)
    handler_source = inspect.getsource(public_research)

    assert (
        kazusa_tools.resolve_with_public_research
        is public_research.resolve_with_public_research
    )
    assert "resolve_complex_task" in handler_source
    assert "web_agent3" not in adapter_source
    assert "resolve_complex_task" not in adapter_source


@pytest.mark.asyncio
async def test_coding_tool_retains_existing_approval_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coding mode and approval-required status pass through unchanged."""

    captured_request: dict[str, object] = {}

    async def _coding_handler(
        request: dict[str, object],
        execution_context: dict[str, object],
    ) -> dict[str, object]:
        del execution_context
        captured_request.update(request)
        result = _specialist_result(
            request,
            specialist="coding",
            status="approval_required",
        )
        return result

    monkeypatch.setattr(kazusa_tools, "resolve_with_coding", _coding_handler)
    registry = kazusa_tools.build_kazusa_tool_registry(_context())

    result = await registry.execute_tool(
        "coding",
        {
            "objective": "Prepare a bounded patch proposal.",
            "coding_objective_mode": "propose_patch",
        },
        permission_scope={},
        timeout_seconds=5,
        maximum_result_characters=8_000,
    )

    assert kazusa_tools.resolve_with_coding is _coding_handler
    assert coding.resolve_with_coding is not _coding_handler
    assert captured_request["coding_objective_mode"] == "propose_patch"
    assert result.status == "success"
    assert result.output["status"] == "approval_required"
    assert "approval" in str(result.output["reason"])
