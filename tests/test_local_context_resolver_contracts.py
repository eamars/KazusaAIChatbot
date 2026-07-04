"""Contract tests for the standalone local-context resolver package."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    LocalContextValidationError,
    validate_local_context_artifact,
    validate_local_context_graph,
    validate_local_context_node,
    validate_local_context_resolution_packet,
    validate_local_context_resolver_context,
    validate_local_context_resolver_options,
    validate_local_context_resolver_request,
)


def test_request_context_and_packet_contracts_validate() -> None:
    """Validate the shared standalone and production-facing IO shapes."""

    request = validate_local_context_resolver_request({
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve what #napcat means in this group.",
        "source": "standalone_eval",
        "reason": "Standalone public IO contract test.",
        "priority": "normal",
    })
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
    graph = _graph()
    packet = validate_local_context_resolution_packet({
        "schema_version": LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
        "investigation_summary": [
            "Checked durable memory for the #napcat command anchor.",
        ],
        "knowledge_we_know_so_far": [
            "#napcat is a playful local command anchor.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [
            "No live NapCat runtime status was queried.",
        ],
        "rag_result": _rag_result(),
        "graph": graph,
        "trace_summary": {
            "iterations": 1,
            "node_count": 2,
            "subagent_calls": 0,
        },
    })

    assert request["source"] == "standalone_eval"
    assert context["prompt_message_context"]["message_text"] == (
        "@active character #napcat"
    )
    assert packet["rag_result"]["memory_evidence"][0]["summary"] == (
        "#napcat is a playful local command anchor."
    )


def test_node_and_artifact_contracts_validate_source_owned_evidence() -> None:
    """Validate node and artifact shapes used by local evidence subagents."""

    node = validate_local_context_node({
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": "memory_1",
        "node_kind": "memory_evidence",
        "objective": "Retrieve durable memory for #napcat.",
        "parent_id": "root",
        "children": [],
        "depends_on": [],
        "consumes": {},
        "produces": ["napcat_command_memory"],
        "status": "resolved",
        "investigation_summary": [
            "Memory evidence resolved the command anchor.",
        ],
        "knowledge_we_know_so_far": [
            "#napcat is a playful local command anchor.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "attempts": [],
        "collapsed_into": None,
    })
    artifact = validate_local_context_artifact({
        "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
        "artifact_id": "artifact_1",
        "artifact_type": "memory_ref",
        "producer_node_id": node["node_id"],
        "summary": "#napcat is a playful local command anchor.",
        "projection_payload": {
            "memory_evidence": [{
                "summary": "#napcat is a playful local command anchor.",
                "source_policy": "shared_memory",
            }],
        },
        "source_policy": "shared_memory",
        "prompt_visible": True,
    })

    assert node["produces"] == ["napcat_command_memory"]
    assert artifact["projection_payload"]["memory_evidence"][0]["summary"] == (
        "#napcat is a playful local command anchor."
    )


def test_invalid_contract_values_fail_closed() -> None:
    """Reject wrong versions, invalid enum values, and unsafe graph shape."""

    with pytest.raises(LocalContextValidationError):
        validate_local_context_resolver_request({
            "schema_version": "wrong",
            "objective": "Resolve local context.",
            "source": "standalone_eval",
            "reason": "contract test",
            "priority": "normal",
        })

    with pytest.raises(LocalContextValidationError):
        validate_local_context_node({
            "schema_version": LOCAL_CONTEXT_NODE_VERSION,
            "node_id": "root",
            "node_kind": "unsupported",
            "objective": "Resolve local context.",
            "parent_id": None,
            "children": [],
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
        })

    with pytest.raises(LocalContextValidationError):
        validate_local_context_graph({
            "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
            "root_node_id": "root",
            "active_node_id": "missing",
            "nodes": {},
            "traversal_order": [],
            "collapse_events": [],
            "max_nodes": 8,
            "max_depth": 3,
        })


def test_options_contract_rejects_behavior_injection() -> None:
    """Keep LLM stages and subagents out of public resolver options."""

    async def invoke_stage(payload: dict[str, object]) -> dict[str, object]:
        return_value = payload
        return return_value

    with pytest.raises(LocalContextValidationError):
        validate_local_context_resolver_options({
            "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
            "planner_llm": invoke_stage,
            "subagents": {},
            "max_iterations": 3,
            "max_nodes": 8,
            "max_depth": 3,
            "max_node_attempts": 2,
            "max_subagent_attempts": 1,
        })

    options = validate_local_context_resolver_options({
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        "max_iterations": 3,
        "max_nodes": 8,
        "max_depth": 3,
        "max_node_attempts": 2,
        "max_subagent_attempts": 1,
    })

    assert options["max_iterations"] == 3

    for limit_name, limit_value in (
        ("max_iterations", 5),
        ("max_nodes", 9),
        ("max_depth", 4),
        ("max_node_attempts", 3),
        ("max_subagent_attempts", 2),
    ):
        invalid_options = {
            "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
            "max_iterations": 3,
            "max_nodes": 8,
            "max_depth": 3,
            "max_node_attempts": 2,
            "max_subagent_attempts": 1,
        }
        invalid_options[limit_name] = limit_value
        with pytest.raises(LocalContextValidationError):
            validate_local_context_resolver_options(invalid_options)


def _rag_result() -> dict[str, object]:
    """Build a prompt-facing RAG result used by contract tests."""

    rag_result = {
        "answer": "",
        "user_image": {},
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [{
            "summary": "#napcat is a playful local command anchor.",
            "source_policy": "shared_memory",
        }],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "resolver": "local_context_resolver",
            "node_count": 2,
        },
    }
    return rag_result


def _node(
    node_id: str,
    *,
    parent_id: str | None,
    status: str,
    children: list[str] | None = None,
    depends_on: list[str] | None = None,
) -> dict[str, object]:
    """Build a valid local-context graph node for contract tests."""

    node = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": node_id,
        "node_kind": "synthesis" if parent_id is None else "memory_evidence",
        "objective": f"Objective for {node_id}",
        "parent_id": parent_id,
        "children": children or [],
        "depends_on": depends_on or [],
        "consumes": {},
        "produces": [],
        "status": status,
        "investigation_summary": [],
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "attempts": [],
        "collapsed_into": None,
    }
    return node


def _graph() -> dict[str, object]:
    """Build a valid local-context graph for packet tests."""

    graph = {
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "memory_1",
        "nodes": {
            "root": _node(
                "root",
                parent_id=None,
                status="resolved",
                children=["memory_1"],
            ),
            "memory_1": _node(
                "memory_1",
                parent_id="root",
                status="resolved",
            ),
        },
        "traversal_order": ["root", "memory_1"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph
