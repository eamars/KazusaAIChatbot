"""Contract tests for the standalone complex-task resolver package."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.action_spec.models import EVIDENCE_REF_VERSION
from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_GRAPH_VERSION,
    COMPLEX_TASK_FOLLOWUP_TASK_VERSION,
    COMPLEX_TASK_NODE_ATTEMPT_VERSION,
    COMPLEX_TASK_NODE_VERSION,
    COMPLEX_TASK_RESOLUTION_PACKET_VERSION,
    COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
    COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
    validate_complex_task_graph,
    validate_complex_task_followup_task,
    validate_complex_task_node,
    validate_complex_task_node_attempt,
    validate_complex_task_resolution_packet,
    validate_complex_task_resolver_context,
    validate_complex_task_resolver_options,
    validate_complex_task_resolver_request,
    validate_complex_task_subagent_result,
)
from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    ComplexTaskValidationError,
)


def test_request_context_and_packet_contracts_validate() -> None:
    """Validate the public standalone request, context, and packet shapes."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare two local inference options with sourced facts.",
        "reason": "Standalone review case",
        "source": "review_harness",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "The user is reviewing resolver design.",
        "persona_context_summary": "Structural answer, not final dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    packet = validate_complex_task_resolution_packet({
        "schema_version": COMPLEX_TASK_RESOLUTION_PACKET_VERSION,
        "root_question": request["objective"],
        "investigation_summary": "Comparison evidence was partially collected.",
        "knowledge_we_know_so_far": ["Fact with source."],
        "knowledge_still_lacking": ["Missing current benchmark."],
        "recommended_next_iteration": [
            "Check whether a newer benchmark source is available.",
        ],
        "evidence_boundary_notes": ["Inference clearly marked."],
        "graph": {
            "schema_version": COMPLEX_TASK_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "root",
            "nodes": {
                "root": {
                    "schema_version": COMPLEX_TASK_NODE_VERSION,
                    "node_id": "root",
                    "parent_id": None,
                    "depth": 0,
                    "objective": request["objective"],
                    "node_kind": "root",
                    "status": "resolved",
                    "children": [],
                    "investigation_summary": (
                        "Root resolved from child semantic projections."
                    ),
                    "knowledge_we_know_so_far": ["Fact with source."],
                    "knowledge_still_lacking": [],
                    "recommended_next_iteration": [],
                    "evidence_boundary_notes": ["Inference clearly marked."],
                    "evidence_refs": [],
                    "source_observation_ids": [],
                    "collapsed_into": None,
                    "attempts": [],
                },
            },
            "traversal_order": ["root"],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        },
        "trace_summary": {
            "iterations": 1,
            "collapse_count": 0,
        },
    })

    assert request["source"] == "review_harness"
    assert context["available_evidence"] == []
    assert packet["knowledge_still_lacking"] == ["Missing current benchmark."]


def test_graph_contract_reuses_evidence_ref_shape() -> None:
    """Validate graph nodes can carry existing action-spec evidence refs."""

    node = validate_complex_task_node({
        "schema_version": COMPLEX_TASK_NODE_VERSION,
        "node_id": "evidence_1",
        "parent_id": "root",
        "depth": 1,
        "objective": "Fetch the current official documentation.",
        "node_kind": "evidence_need",
        "status": "resolved",
        "children": [],
        "investigation_summary": "Official documentation found.",
        "knowledge_we_know_so_far": [
            "Official page says the CLI can edit files.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "evidence_refs": [{
            "schema_version": EVIDENCE_REF_VERSION,
            "evidence_kind": "external_document",
            "evidence_id": "https://example.test/docs",
            "owner": "complex_task_resolver",
            "excerpt": "CLI can edit files.",
            "observed_at": "2026-06-30T00:00:00Z",
        }],
        "source_observation_ids": ["obs_1"],
        "collapsed_into": None,
        "attempts": [],
    })

    assert node["evidence_refs"][0]["schema_version"] == EVIDENCE_REF_VERSION


def test_node_attempt_contract_records_prompt_safe_loop_state() -> None:
    """Validate one active-node attempt observation."""

    attempt = validate_complex_task_node_attempt({
        "schema_version": COMPLEX_TASK_NODE_ATTEMPT_VERSION,
        "attempt_index": 1,
        "action": "refine_search",
        "status": "partial",
        "input_summary": "evidence_need: Fetch current source facts.",
        "result_summary": "Initial evidence dependency was unavailable.",
        "blockers": ["web_search unavailable"],
        "next_action": "try supplied structured evidence before blocking",
    })

    assert attempt["action"] == "refine_search"
    assert attempt["blockers"] == ["web_search unavailable"]


def test_followup_task_contract_validates_executable_control_shape() -> None:
    """Validate structured follow-up tasks without using prose recommendations."""

    task = validate_complex_task_followup_task({
        "schema_version": COMPLEX_TASK_FOLLOWUP_TASK_VERSION,
        "objective": "Find public benchmark evidence for model throughput.",
        "kind": "evidence_need",
        "reason": "Parent node lacks source-backed throughput numbers.",
    })

    assert task["kind"] == "evidence_need"
    assert task["reason"] == (
        "Parent node lacks source-backed throughput numbers."
    )

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_followup_task({
            "schema_version": COMPLEX_TASK_FOLLOWUP_TASK_VERSION,
            "objective": "Search public benchmark evidence.",
            "kind": "web_search",
            "reason": "Unsupported follow-up kind must fail closed.",
        })


def test_invalid_contract_values_fail_closed() -> None:
    """Reject wrong schema versions, invalid enums, and unsafe graph shape."""

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_resolver_request({
            "schema_version": "wrong",
            "objective": "x",
            "reason": "x",
            "source": "review_harness",
            "priority": "review",
        })

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_node({
            "schema_version": COMPLEX_TASK_NODE_VERSION,
            "node_id": "root",
            "parent_id": None,
            "depth": 0,
            "objective": "Resolve the task.",
            "node_kind": "unsupported",
            "status": "resolved",
            "children": [],
            "investigation_summary": "Resolved.",
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
            "evidence_refs": [],
            "source_observation_ids": [],
            "collapsed_into": None,
            "attempts": [],
        })

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_graph({
            "schema_version": COMPLEX_TASK_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "missing",
            "nodes": {},
            "traversal_order": [],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        })


def test_options_contract_rejects_external_stage_configuration() -> None:
    """Keep LLM stages and subagents out of public resolver IO."""

    async def invoke_stage(payload: dict[str, object]) -> dict[str, object]:
        return payload

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_resolver_options({
            "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
            "planner_llm": invoke_stage,
            "node_resolver_llm": invoke_stage,
            "collapse_llm": invoke_stage,
            "synthesizer_llm": invoke_stage,
            "subagents": {},
            "limits": {},
        })

    options = validate_complex_task_resolver_options({
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": {
            "max_iterations": 3,
            "max_nodes": 8,
            "max_depth": 3,
            "max_node_attempts": 2,
            "max_subagent_attempts": 1,
        },
    })

    assert options["limits"]["max_iterations"] == 3

    for limit_name, limit_value in (
        ("max_iterations", 9),
        ("max_nodes", 9),
        ("max_depth", 4),
        ("max_node_attempts", 4),
        ("max_subagent_attempts", 2),
    ):
        with pytest.raises(ComplexTaskValidationError):
            validate_complex_task_resolver_options({
                "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
                "limits": {
                    limit_name: limit_value,
                },
            })


def test_subagent_result_allows_raw_trace_not_hidden_semantic_hints() -> None:
    """Keep anti-cheat checks on semantic output, not read-only traces."""

    result = validate_complex_task_subagent_result({
        "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
        "resolved": False,
        "status": "partial",
        "result": {"summary": "No comparable public evidence was found."},
        "attempts": 1,
        "cache": {"enabled": False},
        "trace": {
            "web_agent3": {
                "task": (
                    "Compare expected final answer phrasing with retrieved "
                    "public docs."
                ),
            },
        },
        "unresolved_items": ["No comparable public evidence was found."],
    })

    assert result["trace"]["web_agent3"]["task"].startswith("Compare")

    with pytest.raises(ComplexTaskValidationError):
        validate_complex_task_subagent_result({
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": False,
            "status": "partial",
            "result": {"summary": "expected final answer leaked"},
            "attempts": 1,
            "cache": {"enabled": False},
            "trace": {},
            "unresolved_items": ["No comparable public evidence was found."],
        })
