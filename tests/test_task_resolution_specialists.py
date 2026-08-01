"""Tests for the four task-resolution specialist public adapters."""

from __future__ import annotations

import importlib
from copy import deepcopy

import pytest

from tests.test_task_resolution_orchestrator import _checkpoint, _context


def _request() -> dict[str, object]:
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    return state.build_specialist_request(_checkpoint())


@pytest.mark.asyncio
async def test_local_context_maps_only_canonical_public_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local adapter returns prompt-safe evidence from the public packet."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.specialists.local_context"
    )
    calls: list[tuple[dict[str, object], ...]] = []

    async def resolve_local_context(
        resolver_request: dict[str, object],
        resolver_context: dict[str, object],
        resolver_options: dict[str, object],
    ) -> dict[str, object]:
        calls.append((resolver_request, resolver_context, resolver_options))
        return {"knowledge_still_lacking": []}

    monkeypatch.setattr(module, "resolve_local_context", resolve_local_context)
    monkeypatch.setattr(
        module,
        "project_local_context_packet",
        lambda _packet: {"answer": "A relevant prior commitment was found."},
    )

    result = await module.resolve_with_local_context(_request(), _context())

    assert result["status"] == "resolved"
    assert result["evidence"][0]["specialist"] == "local_context"
    assert calls[0][0]["objective"] == _request()["objective"]
    assert "workspace_root" not in calls[0][1]


@pytest.mark.asyncio
async def test_public_research_no_evidence_is_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong public route stays recoverable when it yields no evidence."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.specialists.public_research"
    )

    async def resolve_complex_task(
        resolver_request: dict[str, object],
        resolver_context: dict[str, object],
        resolver_options: dict[str, object],
    ) -> dict[str, object]:
        del resolver_request, resolver_context, resolver_options
        return {"evidence_refs": []}

    monkeypatch.setattr(module, "resolve_complex_task", resolve_complex_task)
    monkeypatch.setattr(
        module,
        "project_complex_task_packet",
        lambda _packet: {
            "investigation_summary": "",
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": ["A different specialist is required."],
        },
    )

    result = await module.resolve_with_public_research(_request(), _context())

    assert result["status"] == "incompatible"
    assert result["evidence"] == []


@pytest.mark.asyncio
async def test_public_research_projects_graph_evidence_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public research retains canonical source URLs from resolver nodes."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.specialists.public_research"
    )
    source_url = "https://example.test/rtx5090"
    packet = {
        "graph": {
            "nodes": {
                "node-1": {
                    "evidence_refs": [{
                        "evidence_id": source_url,
                    }],
                },
            },
        },
    }

    async def resolve_complex_task(
        resolver_request: dict[str, object],
        resolver_context: dict[str, object],
        resolver_options: dict[str, object],
    ) -> dict[str, object]:
        del resolver_request, resolver_context, resolver_options
        return packet

    monkeypatch.setattr(module, "resolve_complex_task", resolve_complex_task)
    monkeypatch.setattr(
        module,
        "project_complex_task_packet",
        lambda _packet: {
            "investigation_summary": "Current retail evidence was found.",
            "knowledge_we_know_so_far": [
                "RTX 5090 is listed at $1,999 USD by Example Retailer.",
            ],
            "knowledge_still_lacking": [],
        },
    )

    result = await module.resolve_with_public_research(_request(), _context())

    assert result["status"] == "resolved"
    assert result["evidence"][0]["provenance_refs"] == [source_url]
    assert "$1,999 USD" in result["evidence"][0]["summary"]


@pytest.mark.asyncio
async def test_text_computation_refuses_unsupported_domain_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The text handler returns a typed refusal instead of fabricating output."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.specialists.text_computation"
    )

    async def route_text_task(**_kwargs: object) -> dict[str, str]:
        return {
            "task_type": "unsupported",
            "reason": "The task requires public web evidence.",
        }

    async def unexpected_generation(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("unsupported text work must not reach generation")

    monkeypatch.setattr(module, "_route_text_task", route_text_task)
    monkeypatch.setattr(module, "_generate_text_artifact", unexpected_generation)

    result = await module.resolve_with_text_computation(_request(), _context())

    assert result["status"] == "incompatible"
    assert result["evidence"] == []


@pytest.mark.asyncio
async def test_text_computation_evaluates_structured_expression_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-supplied arithmetic uses the retained deterministic evaluator."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.task_resolution.specialists.text_computation"
    )
    context = deepcopy(_context())
    context["prompt_message_context"] = {
        "text": "Calculate the supplied expression.",
        "numeric_expression": "(12 + 8) / 4",
    }

    async def unexpected_llm(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("structured arithmetic must not call an LLM")

    monkeypatch.setattr(module._task_router_llm, "ainvoke", unexpected_llm)
    monkeypatch.setattr(module._generator_llm, "ainvoke", unexpected_llm)

    result = await module.resolve_with_text_computation(_request(), context)

    assert result["status"] == "resolved"
    assert "5" in result["evidence"][0]["summary"]
