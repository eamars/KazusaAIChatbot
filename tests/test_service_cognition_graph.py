"""Deterministic semantic cognition-graph builder tests."""

from __future__ import annotations

import pytest

def _response_state(*, visual_directives: dict[str, object]) -> dict[str, object]:
    """Build a response graph state with meaningful semantic artifacts."""

    long_input = "input-start\n" + ("turn-" * 220) + "input-end <&> \"quoted\""
    return {
        "user_input": long_input,
        "reply_context": {
            "reply_to_display_name": "Ari",
            "reply_excerpt": "the earlier message being answered",
        },
        "platform": "debug",
        "debug_modes": {},
        "cognitive_episode": {
            "origin_metadata": {"debug_modes": {}},
        },
        "internal_monologue": "The request is grounded in the current scene.",
        "logical_stance": "respond",
        "character_intent": "provide useful information",
        "judgment_note": "A direct response is appropriate.",
        "conversation_progress": {
            "current_goal": "answer the operator",
            "progress_note": "the requested inspection is in progress",
        },
        "rag_result": {
            "answer": "retrieval conclusion",
            "memory_evidence": [
                {
                    "fact": "the operator prefers complete detail",
                    "summary": "operator preference",
                    "source": "memory",
                    "relevance": 0.91,
                    "prompt": "exclude this nested prompt",
                },
            ],
            "conversation_evidence": [
                {"content": "conversation evidence", "title": "prior turn"},
            ],
            "external_evidence": [
                {"content": "external evidence", "source": "document"},
            ],
            "recall_evidence": [
                {"content": "recalled evidence", "recency": "recent"},
            ],
            "media_evidence": [
                {
                    "visual_observation": "a calm scene",
                    "evidence_boundary_notes": "observation only",
                },
            ],
            "user_image": {
                "user_memory_context": {
                    "active_commitments": [
                        {"fact": "show the important information"},
                    ],
                    "stable_patterns": [
                        {"fact": "prefers direct inspection"},
                    ],
                },
            },
            "supervisor_trace": ["process detail must stay out"],
        },
        "action_specs": [
            {
                "kind": "speak",
                "reason": "A visible answer is needed.",
                "urgency": "now",
                "visibility": "user_visible",
                "deadline": None,
                "continuation": {"mode": "none"},
                "params": {"secret_target": "exclude this raw parameter"},
            },
        ],
        "action_results": [
            {
                "action_kind": "speak",
                "status": "executed",
                "visibility": "user_visible",
                "result_summary": "visible answer rendered",
                "action_attempt_id": "exclude-this-attempt-id",
                "handler_owner": "exclude-this-handler",
            },
        ],
        "action_directives": {"visual_directives": visual_directives},
    }


def _response_graph_result() -> dict[str, object]:
    """Build the graph-level response result."""

    return {
        "should_respond": True,
        "reason_to_respond": (
            "reason-start " + ("The operator supplied a concrete inspection request. " * 40)
            + "reason-end"
        ),
        "final_dialog": [
            "first visible line\nsecond visible line",
            "final message <&> \"safe text\"",
        ],
        "future_promises": [],
    }


def _node(graph: dict[str, object], node_id: str) -> dict[str, object]:
    """Return one graph node by id."""

    nodes = graph["nodes"]
    assert isinstance(nodes, list)
    node = next(item for item in nodes if item["id"] == node_id)
    assert isinstance(node, dict)
    return node


def test_response_graph_contains_semantic_details_and_visual_directive(
    monkeypatch,
) -> None:
    """Response graph should expose actual L1/L2/L3 semantic values."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "COGNITION_VISUAL_DIRECTIVES_ENABLED", True)
    visual_directives = {
        "facial_expression": ["focused"],
        "body_language": ["hands relaxed"],
        "gaze_direction": ["toward the screen"],
        "visual_vibe": ["attentive", "attentive"],
    }
    graph = service._build_response_cognition_graph(
        graph_result=_response_graph_result(),
        consolidation_state=_response_state(visual_directives=visual_directives),
        run_id="response-run",
        cognitive_episode={
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
        },
    )

    intake = _node(graph, "intake")
    assert intake["detail"]["input"].startswith("input-start")
    assert intake["detail"]["input"].endswith("input-end <&> \"quoted\"")
    assert "decontextualized_input" not in intake["detail"]
    assert "summary" not in intake["detail"]
    assert _node(graph, "l1.relevance")["detail"]["reasoning"].endswith(
        "reason-end"
    )

    memory = _node(graph, "l2.memory")["detail"]
    assert memory["retrieval_answer"] == "retrieval conclusion"
    assert memory["memory_evidence"][0]["fact"] == (
        "the operator prefers complete detail"
    )
    assert memory["conversation_evidence"][0]["content"] == (
        "conversation evidence"
    )
    assert memory["active_commitments"][0]["fact"] == (
        "show the important information"
    )

    actions = _node(graph, "l2.actions")["detail"]
    assert actions["selected_actions"][0]["reason"] == (
        "A visible answer is needed."
    )
    assert actions["action_results"][0]["result_summary"] == (
        "visible answer rendered"
    )
    assert "secret_target" not in repr(actions)
    assert "exclude-this-attempt-id" not in repr(actions)

    visual = _node(graph, "l3.visual_directives")
    assert visual["status"] == "completed"
    assert visual["detail"] == visual_directives

    surface = _node(graph, "l3.surface")
    assert surface["detail"]["messages"] == _response_graph_result()[
        "final_dialog"
    ]
    assert "summary" not in surface["detail"]
    assert "status" not in surface["detail"]
    assert graph["trigger_source"] == "user_message"
    assert graph["input_sources"] == ["dialog_text"]


def test_response_graph_distinguishes_enabled_empty_and_disabled_visual(
    monkeypatch,
) -> None:
    """Empty enabled output and disabled output use different node states."""

    from kazusa_ai_chatbot import service

    empty_directives = {
        "facial_expression": [],
        "body_language": [],
        "gaze_direction": [],
        "visual_vibe": [],
    }
    monkeypatch.setattr(service, "COGNITION_VISUAL_DIRECTIVES_ENABLED", True)
    enabled_graph = service._build_response_cognition_graph(
        graph_result=_response_graph_result(),
        consolidation_state=_response_state(visual_directives=empty_directives),
        run_id="enabled-empty",
    )
    enabled_visual = _node(enabled_graph, "l3.visual_directives")
    assert enabled_visual["status"] == "completed"
    assert "empty" in repr(enabled_visual["detail"]).lower()

    monkeypatch.setattr(service, "COGNITION_VISUAL_DIRECTIVES_ENABLED", False)
    disabled_graph = service._build_response_cognition_graph(
        graph_result=_response_graph_result(),
        consolidation_state=_response_state(visual_directives={
            "facial_expression": ["must not fabricate"],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        }),
        run_id="disabled",
    )
    disabled_visual = _node(disabled_graph, "l3.visual_directives")
    assert disabled_visual["status"] == "skipped"
    assert "must not fabricate" not in repr(disabled_visual["detail"])
    assert "disabled" in repr(disabled_visual["detail"]).lower()


def test_response_graph_handles_malformed_semantic_artifacts(monkeypatch) -> None:
    """Malformed optional artifacts should not crash graph construction."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "COGNITION_VISUAL_DIRECTIVES_ENABLED", True)
    state = _response_state(visual_directives={
        "facial_expression": "wrong type",
        "body_language": None,
        "gaze_direction": {"unexpected": "mapping"},
        "visual_vibe": ["valid"],
    })
    state["rag_result"] = {
        "answer": 42,
        "memory_evidence": {"not": "a list"},
    }
    state["action_specs"] = {"not": "a list"}
    state["action_results"] = [None, "wrong row"]

    graph = service._build_response_cognition_graph(
        graph_result=_response_graph_result(),
        consolidation_state=state,
        run_id="malformed",
    )

    visual = _node(graph, "l3.visual_directives")
    assert visual["status"] in {"completed", "failed"}
    assert "wrong type" not in repr(visual["detail"])
    assert _node(graph, "l2.memory")["detail"]


def test_self_cognition_graph_uses_shared_semantic_vocabulary(monkeypatch) -> None:
    """Self-cognition graph should expose equivalent semantic artifacts."""

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.self_cognition import models

    monkeypatch.setattr(service, "COGNITION_VISUAL_DIRECTIVES_ENABLED", True)

    artifacts = {
        models.ARTIFACT_RUN_RECORD: {
            "run_id": "self-run-1",
            "trigger_kind": "group_chat_review",
            "selected_route": "action_candidate",
            "output_mode": "scheduled_action_request",
            "status": "completed",
        },
        models.ARTIFACT_COGNITION_INPUT: {
            "source_packet": {
                "visible_context": [
                    {"role": "user", "body_text": "actual self input"},
                ],
                "conversation_progress": {"current_goal": "follow up"},
            },
        },
        models.ARTIFACT_COGNITION_OUTPUT: {
            "internal_monologue": "Review the recent conversation.",
            "logical_stance": "follow up carefully",
            "character_intent": "maintain continuity",
            "judgment_note": "The source is relevant.",
            "rag_result": {
                "answer": "self retrieval answer",
                "memory_evidence": [
                    {"fact": "the source case needs continuity"},
                ],
            },
            "action_specs": [{"kind": "speak", "reason": "continue"}],
            "action_directives": {"visual_directives": {
                "facial_expression": [],
                "body_language": [],
                "gaze_direction": [],
                "visual_vibe": [],
            }},
        },
        models.ARTIFACT_ROUTE_EFFECT: {
            "route": "action_candidate",
            "effect_summary": "candidate selected",
        },
        models.ARTIFACT_ACTION_ATTEMPT: {
            "status": "candidate",
            "action_kind": "send_message",
        },
        models.ARTIFACT_ACTION_CANDIDATE: {
            "text": "actual self visible message",
        },
    }

    graph = service._build_self_cognition_cognition_graph(artifacts)

    assert graph is not None
    source = _node(graph, "self.source")
    assert "actual self input" in repr(source["detail"])
    assert "summary" not in source["detail"]
    reasoning = _node(graph, "self.reasoning")
    assert reasoning["detail"]["judgment_note"] == "The source is relevant."
    memory = _node(graph, "l2.memory")
    assert memory["detail"]["retrieval_answer"] == "self retrieval answer"
    assert memory["detail"]["memory_evidence"][0]["fact"] == (
        "the source case needs continuity"
    )
    visual = _node(graph, "l3.visual_directives")
    assert visual["status"] == "skipped"
    assert "disabled" in repr(visual["detail"]).lower()
    enabled_artifacts = dict(artifacts)
    enabled_artifacts[models.ARTIFACT_COGNITION_OUTPUT] = {
        **artifacts[models.ARTIFACT_COGNITION_OUTPUT],
        "debug_modes": {},
    }
    pending_graph = service._build_self_cognition_cognition_graph(
        enabled_artifacts,
        visual_stage_reached=False,
    )
    assert pending_graph is not None
    assert _node(pending_graph, "l3.visual_directives")["status"] == "pending"
    failed_graph = service._build_self_cognition_cognition_graph(
        enabled_artifacts,
        visual_stage_failed=True,
        visual_stage_reached=True,
    )
    assert failed_graph is not None
    assert _node(failed_graph, "l3.visual_directives")["status"] == "failed"
    surface = _node(graph, "self.surface")
    assert surface["detail"]["messages"] == [
        "actual self visible message",
    ]


@pytest.mark.asyncio
async def test_latest_cognition_publication_uses_one_source_neutral_snapshot() -> None:
    """Every admitted source replaces one canonical latest graph value."""

    from kazusa_ai_chatbot import service

    service._clear_latest_cognition_graph()
    service._publish_latest_cognition_graph(
        {
            "run_id": "chat-run",
            "status": "completed",
            "nodes": [],
            "edges": [],
        },
        cognitive_episode={
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
        },
    )
    service._publish_latest_cognition_graph(
        {
            "run_id": "self-run",
            "status": "completed",
            "nodes": [],
            "edges": [],
        },
        cognitive_episode={
            "trigger_source": "internal_thought",
            "input_sources": ["internal_monologue"],
        },
    )

    latest = await service.ops_latest_cognition_graph()

    assert latest.cognition_graph is not None
    assert latest.cognition_graph["run_id"] == "self-run"
    assert latest.cognition_graph["trigger_source"] == "internal_thought"
    assert latest.cognition_graph["input_sources"] == ["internal_monologue"]
    assert "self_cognition_graph" not in latest.model_dump()


@pytest.mark.asyncio
async def test_self_cognition_failure_publishes_bounded_partial_snapshot() -> None:
    """Self-cognition failures replace stale latest state without raw errors."""

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.self_cognition import models

    service._clear_latest_cognition_graph()
    await service._publish_self_cognition_latest_graph(
        {
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-failed-run",
            },
        },
        cognitive_episode={
            "trigger_source": "internal_thought",
            "input_sources": ["internal_monologue"],
        },
        status="failed",
        reason="self_cognition_case_failure",
    )

    latest = await service.ops_latest_cognition_graph()

    assert latest.cognition_graph is not None
    assert latest.cognition_graph["run_id"] == "self-failed-run"
    assert latest.cognition_graph["status"] == "failed"
    assert latest.cognition_graph["trigger_source"] == "internal_thought"
    assert latest.cognition_graph["input_sources"] == [
        "internal_monologue",
    ]
    assert latest.cognition_graph["nodes"] == []
    assert "cognition backend unavailable" not in repr(
        latest.cognition_graph
    )


def test_latest_cognition_publication_fails_closed_for_malformed_source() -> None:
    """Malformed source metadata cannot relabel a run as a user message."""

    from kazusa_ai_chatbot import service

    service._clear_latest_cognition_graph()
    service._publish_latest_cognition_graph(
        {
            "run_id": "malformed-source-run",
            "status": "completed",
            "nodes": [{"id": "must-be-removed"}],
            "edges": [],
        },
        cognitive_episode={
            "trigger_source": "<script>user_message</script>" * 40,
            "input_sources": ["dialog_text", {"bad": "source"}],
        },
    )

    latest_graph = service._latest_cognition_graph

    assert latest_graph is not None
    assert latest_graph["status"] == "partial"
    assert latest_graph["trigger_source"] == "not_reported"
    assert latest_graph["input_sources"] == []
    assert latest_graph["nodes"] == []
    assert "user_message" not in repr(latest_graph)
    assert "<script>" not in repr(latest_graph)
    assert latest_graph["redaction"]["reason"] == "trigger_source_invalid"


def test_response_graph_records_visual_stage_failure(monkeypatch) -> None:
    """A recorded enabled visual-stage failure remains visible in telemetry."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "COGNITION_VISUAL_DIRECTIVES_ENABLED", True)
    graph = service._build_response_cognition_graph(
        graph_result=_response_graph_result(),
        consolidation_state=_response_state(visual_directives={}),
        run_id="visual-failed",
        graph_status="failed",
        visual_stage_failed=True,
        visual_stage_reached=True,
    )

    visual = _node(graph, "l3.visual_directives")
    assert graph["status"] == "failed"
    assert visual["status"] == "failed"
    assert "failed" in visual["detail"]["empty_state"].lower()


@pytest.mark.asyncio
async def test_self_cognition_publisher_uses_canonical_latest_graph() -> None:
    """Self-cognition metadata reaches the same latest storage seam."""

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.self_cognition import models

    service._clear_latest_cognition_graph()
    artifacts = {
        models.ARTIFACT_RUN_RECORD: {
            "run_id": "self-run-published",
            "selected_route": "silent_no_write",
            "status": "completed",
        },
        models.ARTIFACT_COGNITION_INPUT: {
            "source_packet": {
                "visible_context": [
                    {"role": "user", "body_text": "self input"},
                ],
            },
        },
        models.ARTIFACT_COGNITION_OUTPUT: {
            "cognitive_episode": {
                "trigger_source": "internal_thought",
                "input_sources": ["internal_monologue"],
            },
            "internal_monologue": "review the source",
        },
    }

    await service._publish_self_cognition_latest_graph(artifacts)
    latest = await service.ops_latest_cognition_graph()

    assert latest.cognition_graph is not None
    assert latest.cognition_graph["run_id"] == "self-run-published"
    assert latest.cognition_graph["trigger_source"] == "internal_thought"
    assert latest.cognition_graph["input_sources"] == ["internal_monologue"]
