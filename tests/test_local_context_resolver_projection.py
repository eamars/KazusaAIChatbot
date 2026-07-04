"""Projection tests for local-context resolver packets."""

from __future__ import annotations

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
    project_local_context_packet,
)


def test_packet_projection_returns_prompt_facing_rag_result_only() -> None:
    """Project the final packet into the existing cognition evidence surface."""

    packet = {
        "schema_version": LOCAL_CONTEXT_RESOLUTION_PACKET_VERSION,
        "investigation_summary": [
            "Resolved #napcat from durable memory.",
        ],
        "knowledge_we_know_so_far": [
            "#napcat is a playful local command anchor.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
        "rag_result": {
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
        },
        "graph": _graph(),
        "trace_summary": {
            "iterations": 1,
            "node_count": 2,
        },
    }

    rag_result = project_local_context_packet(packet)

    assert rag_result["memory_evidence"][0]["summary"] == (
        "#napcat is a playful local command anchor."
    )
    assert "graph" not in rag_result
    assert "trace_summary" not in rag_result
    assert "memory_1" not in str(rag_result)


def _graph() -> dict[str, object]:
    """Build graph material that must remain outside prompt-facing evidence."""

    root = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": "root",
        "node_kind": "synthesis",
        "objective": "Resolve #napcat.",
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
        "objective": "Retrieve #napcat memory.",
        "parent_id": "root",
        "children": [],
    })
    graph = {
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
    }
    return graph
