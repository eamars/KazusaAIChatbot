"""Prompt contract tests for complex-task resolver LLM stages."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.complex_task_resolver import stages
from kazusa_ai_chatbot.complex_task_resolver import service
from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
    validate_complex_task_resolver_context,
    validate_complex_task_resolver_options,
    validate_complex_task_resolver_request,
)
from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    COMPLEX_TASK_GRAPH_VERSION,
    ComplexTaskValidationError,
)


def test_llm_stage_prompts_project_semantic_content_only() -> None:
    """Reject deterministic graph and transport vocabulary in stage prompts."""

    prompt_text = "\n".join((
        stages._PLANNER_PROMPT,
        stages._NODE_RESOLVER_PROMPT,
        stages._COLLAPSE_PROMPT,
        stages._SYNTHESIZER_PROMPT,
    ))
    forbidden_fragments = (
        "schema_version",
        "node_id",
        "parent_id",
        "source_node_id",
        "target_node_id",
        "attempt_index",
        "trace",
        "cache",
        "status",
    )

    for fragment in forbidden_fragments:
        assert fragment not in prompt_text


def test_node_prompt_projects_registered_subagents() -> None:
    """Require prompt-facing capabilities to come from the registry."""

    assert "algorithmic:" in stages._NODE_RESOLVER_PROMPT
    assert "evidence:" in stages._NODE_RESOLVER_PROMPT
    assert "Available Resolver-Local Subagents" in stages._NODE_RESOLVER_PROMPT


def test_node_prompt_requires_clock_time_normalization() -> None:
    """Keep clock-time conversion with the caller before calculator IO."""

    assert "shorthand numeric units" in stages._NODE_RESOLVER_PROMPT
    assert '"6 * 1000"' in stages._NODE_RESOLVER_PROMPT
    assert '"1000"' in stages._NODE_RESOLVER_PROMPT
    assert "clock-time arithmetic" in stages._NODE_RESOLVER_PROMPT
    assert "numeric minutes" in stages._NODE_RESOLVER_PROMPT
    assert "midnight" in stages._NODE_RESOLVER_PROMPT
    assert "7:00 PM becomes" in stages._NODE_RESOLVER_PROMPT
    assert '"1140 + 140"' in stages._NODE_RESOLVER_PROMPT


def test_semantic_output_rejects_deterministic_keys() -> None:
    """Reject deterministic fields when they appear in semantic LLM output."""

    with pytest.raises(ComplexTaskValidationError):
        service._reject_forbidden_semantic_output_keys(
            {
                "decision": "record_knowledge",
                "knowledge_we_know_so_far": [{
                    "schema_version": "not semantic",
                }],
            },
            "node_decision",
        )


def test_production_semantic_normalizers_reject_internal_envelopes() -> None:
    """Reject internal control envelopes from production LLM stage output."""

    graph = _two_node_graph()
    active_node = graph["nodes"]["task_1"]

    with pytest.raises(ComplexTaskValidationError):
        service._normalize_node_stage_response(
            {
                "node_update": {
                    "status": "resolved",
                    "investigation_summary": "internal shape",
                },
            },
            graph=graph,
            active_node=active_node,
            allow_internal_envelope=False,
        )

    with pytest.raises(ComplexTaskValidationError):
        service._normalize_synthesis_stage_response(
            {
                "followup_tasks": [{
                    "schema_version": "complex_task_followup_task.v1",
                    "objective": "Internal follow-up.",
                    "kind": "subtask",
                    "reason": "Internal envelope.",
                }],
            },
            allow_internal_envelope=False,
        )

    with pytest.raises(ComplexTaskValidationError):
        service._apply_collapse_response(
            graph,
            "task_1",
            {
                "collapse_decision": {
                    "should_collapse": True,
                    "target_node_id": "task_2",
                    "reason": "internal graph target",
                },
            },
            {"collapse_count": 0},
        )


@pytest.mark.asyncio
async def test_no_candidate_collapse_fallback_uses_semantic_shape() -> None:
    """Keep service-authored no-candidate collapse output prompt-semantic."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare purchase options.",
        "reason": "prompt contract test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    options = validate_complex_task_resolver_options({
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": {"max_iterations": 1},
    })
    graph = _single_pending_graph()

    response = await service._review_collapse(
        request,
        context,
        options,
        graph,
        "task_1",
    )

    assert response["collapse_decision"]["matching_candidate"] == ""
    assert "target_node_id" not in response["collapse_decision"]
    service._apply_collapse_response(
        graph,
        "task_1",
        response,
        {"collapse_count": 0},
    )


def test_semantic_repair_and_subagent_notes_avoid_transport_terms() -> None:
    """Avoid feeding internal response and subagent transport terms to prompts."""

    repair_context = service._semantic_repair_context({
        "node_update": {
            "status": "resolved",
            "investigation_summary": "Prose answer was attempted.",
            "knowledge_still_lacking": [
                "deterministic subagent request"
            ],
        },
    })
    subagent_notes = service._subagent_boundary_notes({
        "schema_version": "complex_task_subagent_result.v1",
        "resolved": False,
        "status": "unavailable",
        "result": {},
        "attempts": 1,
        "cache": {
            "enabled": True,
            "hit": False,
            "cache_name": "web_agent3",
            "reason": "cache miss",
        },
        "trace": {"internal": True},
        "unresolved_items": ["web unavailable"],
    })
    projection_text = str({
        "repair_context": repair_context,
        "subagent_notes": subagent_notes,
    })

    for fragment in ("node_update", "status", "cache", "trace"):
        assert fragment not in projection_text


def test_prompt_input_projections_are_semantic_only() -> None:
    """Reject deterministic graph fields from prompt-facing projections."""

    node = service._make_graph_node(
        node_id="task_1",
        parent_id="root",
        depth=1,
        objective="Collect public evidence.",
        node_kind="evidence_need",
        status="pending",
        children=[],
    )
    context_projection = service._compact_context({
        "schema_version": "complex_task_resolver_context.v1",
        "conversation_summary": "Conversation summary.",
        "persona_context_summary": "Persona summary.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [{
            "schema_version": "evidence_ref.v1",
            "evidence_kind": "external_document",
            "evidence_id": "https://example.test/source",
            "owner": "test",
            "excerpt": "Source excerpt.",
            "observed_at": "2026-06-30T00:00:00Z",
        }],
    })
    prompt_projection = {
        "active_node": service._compact_node(node),
        "context": context_projection,
    }
    forbidden_fragments = (
        "schema_version",
        "node_id",
        "parent_id",
        "attempt_index",
        "status",
        "trace_summary",
    )

    projection_text = str(prompt_projection)
    for fragment in forbidden_fragments:
        assert fragment not in projection_text


def _two_node_graph() -> dict[str, object]:
    """Return a small graph for production-normalizer tests."""

    root = service._make_graph_node(
        node_id="root",
        parent_id=None,
        depth=0,
        objective="Root question.",
        node_kind="root",
        status="expanded",
        children=["task_1", "task_2"],
    )
    task_1 = service._make_graph_node(
        node_id="task_1",
        parent_id="root",
        depth=1,
        objective="Active task.",
        node_kind="subtask",
        status="pending",
        children=[],
    )
    task_2 = service._make_graph_node(
        node_id="task_2",
        parent_id="root",
        depth=1,
        objective="Resolved task.",
        node_kind="subtask",
        status="resolved",
        children=[],
    )
    task_2["investigation_summary"] = "Resolved task summary."
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "task_1",
        "nodes": {
            "root": root,
            "task_1": task_1,
            "task_2": task_2,
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _single_pending_graph() -> dict[str, object]:
    """Return a graph whose active node has no collapse candidates."""

    root = service._make_graph_node(
        node_id="root",
        parent_id=None,
        depth=0,
        objective="Root question.",
        node_kind="root",
        status="expanded",
        children=["task_1"],
    )
    task_1 = service._make_graph_node(
        node_id="task_1",
        parent_id="root",
        depth=1,
        objective="Active task.",
        node_kind="evidence_need",
        status="pending",
        children=[],
    )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "task_1",
        "nodes": {
            "root": root,
            "task_1": task_1,
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph
