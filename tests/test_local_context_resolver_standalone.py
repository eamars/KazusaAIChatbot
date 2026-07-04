"""Standalone public-IO service tests for the local-context resolver."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LOCAL_CONTEXT_GRAPH_VERSION,
    LOCAL_CONTEXT_NODE_VERSION,
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    LocalContextValidationError,
    project_local_context_packet,
    resolve_local_context,
    validate_local_context_resolver_context,
    validate_local_context_resolver_request,
)
from kazusa_ai_chatbot.local_context_resolver import service as resolver_service
from kazusa_ai_chatbot.local_context_resolver import stages as resolver_stages


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
async def test_resolve_local_context_runs_standalone_public_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve one memory node without touching production callers."""

    request = validate_local_context_resolver_request({
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve what #napcat means in this group.",
        "source": "standalone_eval",
        "reason": "standalone test",
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
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Retrieve durable memory for the #napcat command.",
            "node_kind": "memory_evidence",
        }],
    }])
    node_resolver = _StageInvoker([{
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
            "evidence_boundary_notes": [
                "No live NapCat runtime status was queried.",
            ],
        },
        "artifacts": [{
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": "artifact_1",
            "artifact_type": "memory_ref",
            "producer_node_id": "task_1",
            "summary": "#napcat is a playful local command anchor.",
            "projection_payload": {
                "memory_evidence": [
                    {
                        "summary": "#napcat is a playful local command anchor.",
                        "source_policy": "shared_memory",
                        "message_id": "raw-message-id",
                        "scope_global_user_id": "user-1",
                        "timestamp": "2026-07-04T09:30:00Z",
                    },
                    {
                        "summary": "#napcat is a playful local command anchor.",
                        "source_policy": "shared_memory",
                        "message_id": "raw-message-id",
                        "scope_global_user_id": "user-1",
                        "timestamp": "2026-07-04T09:30:00Z",
                    },
                ],
            },
            "source_policy": "shared_memory",
            "prompt_visible": True,
        }],
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_candidate_ref": "",
            "reason": "no duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": [
            "Resolved #napcat from durable memory.",
        ],
        "knowledge_we_know_so_far": [
            "#napcat is a playful local command anchor.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [
            "No live NapCat runtime status was queried.",
        ],
    }])
    monkeypatch.setattr(
        resolver_service,
        "_planner_stage_handler",
        planner,
    )
    monkeypatch.setattr(
        resolver_service,
        "_node_stage_handler",
        node_resolver,
    )
    monkeypatch.setattr(
        resolver_service,
        "_collapse_stage_handler",
        collapse,
    )
    monkeypatch.setattr(
        resolver_service,
        "_synthesizer_stage_handler",
        synthesizer,
    )
    options = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        "max_iterations": 2,
        "max_nodes": 8,
        "max_depth": 3,
        "max_node_attempts": 2,
        "max_subagent_attempts": 1,
    }

    packet = await resolve_local_context(request, context, options)
    rag_result = project_local_context_packet(packet)

    assert planner.calls[0]["request"]["source"] == "standalone_eval"
    assert packet["schema_version"] == "local_context_resolution_packet.v1"
    assert packet["graph"]["nodes"]["task_1"]["status"] == "resolved"
    assert packet["trace_summary"]["iterations"] == 1
    assert packet["trace_summary"]["node_count"] == 2
    assert rag_result["memory_evidence"][0]["summary"] == (
        "#napcat is a playful local command anchor."
    )
    assert len(rag_result["memory_evidence"]) == 1
    assert "raw-message-id" not in str(rag_result)
    assert "scope_global_user_id" not in str(rag_result)
    assert "2026-07-04T09:30:00Z" not in str(rag_result)
    assert "local_context_recall" not in str(packet["trace_summary"])


@pytest.mark.asyncio
async def test_resolve_local_context_collapses_duplicate_candidate_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collapse duplicate nodes through prompt-safe candidate references."""

    request = validate_local_context_resolver_request({
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve duplicate memory tasks for #napcat.",
        "source": "standalone_eval",
        "reason": "collapse test",
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
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Retrieve durable memory for #napcat.",
                "node_kind": "memory_evidence",
            },
            {
                "objective": "Check duplicate #napcat memory context.",
                "node_kind": "memory_evidence",
            },
        ],
    }])
    node_resolver = _StageInvoker([
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": ["Resolved first memory node."],
                "knowledge_we_know_so_far": [
                    "#napcat is a playful local command anchor.",
                ],
                "knowledge_still_lacking": [],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
            "artifacts": [],
        },
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": ["Resolved duplicate memory node."],
                "knowledge_we_know_so_far": [
                    "#napcat is a playful local command anchor.",
                ],
                "knowledge_still_lacking": [],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
            "artifacts": [],
        },
    ])
    collapse = _StageInvoker([
        {
            "collapse_decision": {
                "should_collapse": True,
                "target_candidate_ref": "candidate_1",
                "reason": "same #napcat memory anchor",
            },
        },
    ])
    synthesizer = _StageInvoker([{
        "investigation_summary": ["Resolved and collapsed duplicate nodes."],
        "knowledge_we_know_so_far": [
            "#napcat is a playful local command anchor.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    monkeypatch.setattr(
        resolver_service,
        "_planner_stage_handler",
        planner,
    )
    monkeypatch.setattr(
        resolver_service,
        "_node_stage_handler",
        node_resolver,
    )
    monkeypatch.setattr(
        resolver_service,
        "_collapse_stage_handler",
        collapse,
    )
    monkeypatch.setattr(
        resolver_service,
        "_synthesizer_stage_handler",
        synthesizer,
    )
    options = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        "max_iterations": 3,
        "max_nodes": 8,
        "max_depth": 3,
        "max_node_attempts": 2,
        "max_subagent_attempts": 1,
    }

    packet = await resolve_local_context(request, context, options)

    candidate = collapse.calls[0]["candidates"][0]
    assert candidate["candidate_ref"] == "candidate_1"
    assert "node_id" not in candidate
    assert packet["graph"]["nodes"]["task_2"]["status"] == "collapsed"
    assert packet["graph"]["nodes"]["task_2"]["collapsed_into"] == "task_1"
    assert packet["trace_summary"]["collapse_count"] == 1
    assert packet["trace_summary"]["collapse_calls"] == 1


def test_planner_synthesis_rows_do_not_become_evidence_nodes() -> None:
    """Keep final synthesis owned by the service, not planner child nodes."""

    tasks = resolver_service._planner_tasks({
        "tasks": [
            {
                "objective": "Synthesize the final packet.",
                "node_kind": "synthesis",
            },
            {
                "objective": "Retrieve durable memory for #napcat.",
                "node_kind": "memory_evidence",
            },
        ],
    })

    assert tasks == [{
        "objective": "Retrieve durable memory for #napcat.",
        "node_kind": "memory_evidence",
    }]

    with pytest.raises(LocalContextValidationError):
        resolver_service._planner_tasks({
            "tasks": [{
                "objective": "Synthesize the final packet.",
                "node_kind": "synthesis",
            }],
        })


def test_collapse_candidates_exclude_root_node() -> None:
    """Do not expose graph root bookkeeping as a prompt collapse candidate."""

    root = _node(
        node_id="root",
        parent_id=None,
        node_kind="synthesis",
        status="resolved",
        children=["synthesis_1"],
    )
    synthesis_node = _node(
        node_id="synthesis_1",
        parent_id="root",
        node_kind="synthesis",
        status="resolved",
        children=[],
    )
    graph = {
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "synthesis_1",
        "nodes": {
            "root": root,
            "synthesis_1": synthesis_node,
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }

    candidates = resolver_service._collapse_candidates(graph, "synthesis_1")

    assert candidates == []


def test_compact_context_strips_model_input_identifiers() -> None:
    """Sanitize prompt input before any LLM stage sees caller context."""

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
            "message_id": "raw-message-id",
        },
        "chat_history_recent": [{
            "speaker": "operator",
            "text": "Keep this URL: https://example.test/napcat",
            "source_message_id": "raw-source-id",
            "timestamp": "2026-07-04T09:20:00Z",
        }],
        "chat_history_wide": [{
            "source": "user_memory_units",
            "scope_global_user_id": "user-1",
            "summary": "The current user prefers jasmine tea.",
        }],
        "conversation_progress": {
            "platform_channel_id": "group-1",
            "summary": "progress summary",
        },
    })

    compact = resolver_service._compact_context(context)
    compact_text = str(compact)

    assert "@active character #napcat" in compact_text
    assert "https://example.test/napcat" in compact_text
    assert "jasmine tea" in compact_text
    assert "raw-message-id" not in compact_text
    assert "raw-source-id" not in compact_text
    assert "scope_global_user_id" not in compact_text
    assert "platform_channel_id" not in compact_text
    assert "2026-07-04T09:20:00Z" not in compact_text


def test_collapse_response_does_not_hide_blocked_active_node() -> None:
    """A blocked node must stay blocked even if collapse review says duplicate."""

    root = _node(
        node_id="root",
        parent_id=None,
        node_kind="synthesis",
        status="resolved",
        children=["memory_1", "memory_2"],
    )
    resolved_node = _node(
        node_id="memory_1",
        parent_id="root",
        node_kind="memory_evidence",
        status="resolved",
        children=[],
    )
    blocked_node = _node(
        node_id="memory_2",
        parent_id="root",
        node_kind="memory_evidence",
        status="blocked",
        children=[],
    )
    graph = {
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "memory_2",
        "nodes": {
            "root": root,
            "memory_1": resolved_node,
            "memory_2": blocked_node,
        },
        "traversal_order": ["root", "memory_1", "memory_2"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    trace_summary = {"collapse_count": 0}

    resolver_service._apply_collapse_response(
        graph=graph,
        active_node_id="memory_2",
        response={
            "collapse_decision": {
                "should_collapse": True,
                "target_candidate_ref": "candidate_1",
                "reason": "looks duplicate",
            },
        },
        trace_summary=trace_summary,
    )

    assert graph["nodes"]["memory_2"]["status"] == "blocked"
    assert graph["nodes"]["memory_2"]["collapsed_into"] is None
    assert graph["collapse_events"] == []
    assert trace_summary["collapse_count"] == 0


def test_stage_json_parser_escapes_control_characters_inside_strings() -> None:
    """Parse model JSON with raw newline characters inside string literals."""

    parsed = resolver_stages._parse_stage_json_output(
        '{"knowledge_we_know_so_far": ["first line\nsecond line"]}',
        "unit test stage",
    )

    assert parsed["knowledge_we_know_so_far"] == ["first line\nsecond line"]


def test_stage_prompts_keep_source_field_and_time_boundaries() -> None:
    """Keep prompt rules that protect local evidence classification."""

    node_prompt = resolver_stages._NODE_PROMPT
    planner_prompt = resolver_stages._PLANNER_PROMPT
    synthesizer_prompt = resolver_stages._SYNTHESIZER_PROMPT

    assert "Prefer one task when one source domain can satisfy" in planner_prompt
    assert "Do not add recall_evidence for recent chat events" in planner_prompt
    assert "Do not add scoped_memory merely to double-check" in planner_prompt
    assert "For current time/date/weekday questions" in planner_prompt
    assert "Treat chat row local_time values as message timestamps only" in node_prompt
    assert "Match artifact_type and projection_payload field ownership" in node_prompt
    assert "user_memory_units source rows write" in node_prompt
    assert "retained rag_result surface has no" in node_prompt
    assert "person_ref is for named-person profile" in node_prompt
    assert "recall_ref is for active agreements" in node_prompt
    assert "Do not use recall_ref for exact quoted phrases" in node_prompt
    assert "confirmation, provenance, quote, URL" in node_prompt
    assert "named-person profile or impression objectives" in node_prompt
    assert "Do not infer\n  current time" in synthesizer_prompt
    assert "leave knowledge_still_lacking empty" in synthesizer_prompt
    assert "profile/impression evidence is enough" in synthesizer_prompt


def test_node_update_accepts_single_semantic_string_lists() -> None:
    """Normalize local-LLM single-string semantic list fields."""

    produces = resolver_service._node_update_value(
        "produces",
        "continuity_setting_matcha_ice_cream_shop",
    )

    assert produces == ["continuity_setting_matcha_ice_cream_shop"]
    with pytest.raises(LocalContextValidationError):
        resolver_service._node_update_value("attempts", "one attempt")


def test_prompt_payload_sanitizes_string_values_and_local_time() -> None:
    """Strip embedded raw metadata from final prompt-facing payload strings."""

    rag_result = resolver_service._rag_result_from_artifacts(
        artifacts=[{
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": "artifact_1",
            "artifact_type": "conversation_ref",
            "producer_node_id": "task_1",
            "summary": "Sanitize conversation payload.",
            "projection_payload": {
                "answer": (
                    "#napcat anchor, message_id=raw-message-1, "
                    "scope_global_user_id=user-1"
                ),
                "conversation_evidence": [{
                    "message_text": "@active character #napcat",
                    "local_time": "2026-07-04T13:59:00Z",
                    "source": "platform_user_id=user-1",
                    "content": (
                        "Keep #napcat while removing "
                        "source_message_id=raw-message-2"
                    ),
                }],
            },
            "source_policy": "chat_history_recent",
            "prompt_visible": True,
        }],
        synthesis={
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "investigation_summary": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        trace_summary=_trace_summary(),
    )

    result_text = str(rag_result)
    assert "#napcat" in result_text
    assert "raw-message" not in result_text
    assert "scope_global_user_id" not in result_text
    assert "platform_user_id" not in result_text
    assert "source_message_id" not in result_text
    assert "user-1" not in result_text
    assert "local_time" not in result_text
    assert "2026-07-04T13:59:00Z" not in result_text


def test_rag_result_does_not_fallback_synthesis_into_memory() -> None:
    """Do not treat generic synthesis rows as source-owned memory evidence."""

    rag_result = resolver_service._rag_result_from_artifacts(
        artifacts=[],
        synthesis={
            "knowledge_we_know_so_far": [
                "Mika said the blue comet marker phrase.",
            ],
            "knowledge_still_lacking": [],
            "investigation_summary": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        trace_summary=_trace_summary(),
    )

    assert rag_result["memory_evidence"] == []
    assert "blue comet marker" not in str(rag_result)


def test_rag_result_normalizes_scoped_memory_unit_projection() -> None:
    """Project user memory unit rows to the scoped candidate field."""

    rag_result = resolver_service._rag_result_from_artifacts(
        artifacts=[{
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": "scoped_memory_1",
            "artifact_type": "memory_ref",
            "producer_node_id": "task_1",
            "summary": "Scoped memory.",
            "projection_payload": {
                "memory_evidence": [{
                    "source": "user_memory_units",
                    "content": "The current user prefers jasmine tea.",
                }],
                "user_memory_unit_candidates": [],
            },
            "source_policy": "direct retrieval from user-scoped memory",
            "prompt_visible": True,
        }],
        synthesis={
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "investigation_summary": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        trace_summary=_trace_summary(),
    )

    assert rag_result["memory_evidence"] == []
    assert rag_result["user_memory_unit_candidates"] == [{
        "source": "user_memory_units",
        "content": "The current user prefers jasmine tea.",
    }]


def test_rag_result_normalizes_conversation_recall_projection() -> None:
    """Keep chat-source artifacts in conversation evidence."""

    rag_result = resolver_service._rag_result_from_artifacts(
        artifacts=[{
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": "conversation_1",
            "artifact_type": "conversation_ref",
            "producer_node_id": "task_1",
            "summary": "Chat follow-up.",
            "projection_payload": {
                "recall_evidence": [{
                    "speaker": "rana",
                    "text": "NapCat 信息",
                    "conversation_row_id": "row-rag3-1",
                }],
                "conversation_evidence": [],
            },
            "source_policy": "chat_history_recent",
            "prompt_visible": True,
        }],
        synthesis={
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "investigation_summary": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        trace_summary=_trace_summary(),
    )

    assert rag_result["recall_evidence"] == []
    assert rag_result["conversation_evidence"] == [{
        "speaker": "rana",
        "text": "NapCat 信息",
    }]
    assert rag_result["supervisor_trace"]["dispatched"] == [{
        "agent": "conversation_evidence_agent",
        "source_refs": [{
            "conversation_row_id": "row-rag3-1",
        }],
    }]


def test_rag_result_keeps_recall_artifact_recall_only() -> None:
    """Avoid duplicating active agreements into conversation evidence."""

    rag_result = resolver_service._rag_result_from_artifacts(
        artifacts=[{
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": "recall_1",
            "artifact_type": "recall_ref",
            "producer_node_id": "task_1",
            "summary": "Agreement.",
            "projection_payload": {
                "recall_evidence": [{
                    "content": "Check NapCat at 09:30.",
                }],
                "conversation_evidence": [{
                    "text": "我们今天九点半约好一起检查 NapCat 状态。",
                }],
            },
            "source_policy": "active agreement recall",
            "prompt_visible": True,
        }],
        synthesis={
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "investigation_summary": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        trace_summary=_trace_summary(),
    )

    assert rag_result["recall_evidence"] == [{
        "content": "Check NapCat at 09:30.",
    }]
    assert rag_result["conversation_evidence"] == []


def test_artifact_type_accepts_stage_semantic_aliases() -> None:
    """Normalize source-artifact aliases without weakening final contracts."""

    artifact = resolver_service._validated_artifact_for_node(
        {
            "artifact_id": "person_profile_1",
            "artifact_type": "third_party_profiles",
            "producer_node_id": "active_node",
            "summary": "小明 profile evidence.",
            "projection_payload": {
                "third_party_profiles": [{
                    "name": "小明",
                    "impression": "可靠",
                }],
            },
            "source_policy": "user_profile",
            "prompt_visible": True,
        },
        "task_1",
    )

    assert artifact["artifact_type"] == "person_ref"
    assert artifact["producer_node_id"] == "task_1"

    scoped_artifact = resolver_service._validated_artifact_for_node(
        {
            "artifact_id": "scoped_memory_1",
            "artifact_type": "user_memory_unit_recall",
            "producer_node_id": "model_generated_node_name",
            "summary": "Scoped user-memory evidence.",
            "projection_payload": {
                "user_memory_unit_candidates": [{
                    "content": "Scoped continuity row.",
                }],
            },
            "source_policy": "user_memory_units",
            "prompt_visible": True,
        },
        "task_2",
    )

    assert scoped_artifact["artifact_type"] == "memory_ref"
    assert scoped_artifact["producer_node_id"] == "task_2"

    singular_candidate = resolver_service._validated_artifact_for_node(
        {
            "artifact_id": "scoped_memory_2",
            "artifact_type": "user_memory_unit_candidate",
            "summary": "Scoped user-memory candidate.",
            "projection_payload": {
                "user_memory_unit_candidates": [{
                    "content": "Another scoped row.",
                }],
            },
            "source_policy": "user memory units",
            "prompt_visible": True,
        },
        "task_3",
    )

    assert singular_candidate["artifact_type"] == "memory_ref"
    assert singular_candidate["producer_node_id"] == "task_3"


def _trace_summary() -> dict[str, object]:
    """Build a minimal trace summary for prompt-facing RAG result tests."""

    return {
        "iterations": 1,
        "node_count": 1,
        "resolved_node_count": 1,
        "blocked_node_count": 0,
    }


def _graph_with_memory_node() -> dict[str, object]:
    """Build a standalone graph fixture for future production parity tests."""

    root = _node(
        node_id="root",
        parent_id=None,
        node_kind="synthesis",
        status="resolved",
        children=["memory_1"],
    )
    memory_node = _node(
        node_id="memory_1",
        parent_id="root",
        node_kind="memory_evidence",
        status="pending",
        children=[],
    )
    graph = {
        "schema_version": LOCAL_CONTEXT_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "memory_1",
        "nodes": {
            "root": root,
            "memory_1": memory_node,
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _node(
    *,
    node_id: str,
    parent_id: str | None,
    node_kind: str,
    status: str,
    children: list[str],
) -> dict[str, object]:
    """Build a graph node fixture for standalone service tests."""

    node = {
        "schema_version": LOCAL_CONTEXT_NODE_VERSION,
        "node_id": node_id,
        "node_kind": node_kind,
        "objective": f"Objective for {node_id}",
        "parent_id": parent_id,
        "children": children,
        "depends_on": [],
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
