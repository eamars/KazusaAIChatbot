"""Deterministic semantic cognition-graph builder tests."""

from __future__ import annotations


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


def _native_v2_cognition_output() -> dict[str, object]:
    """Build the safe native V2 semantic projection used by graph tests."""

    return {
        'intention': {
            'route': 'speech',
            'intention': '回应当前关系中的受伤感受',
            'reason': '当前事件足以支持直接回应',
        },
        'selected_bid_reason': '她先承认这次伤害，再决定如何回应。',
        'private_monologue': '我确实被这句话刺痛了，但我想先把感受说清楚。',
        'affect_projection': [
            {
                'emotion': '悲伤',
                'phase': '激活',
                'intensity': '高',
                'trend': '上升',
                'cause_summary': '重要关系中的持续贬低带来了失落感。',
            },
        ],
        'cognition_observability': {
            'execution': {
                'selected_question_count': 2,
                'dispatched_question_count': 2,
                'selected_branch_count': 2,
                'dispatched_branch_count': 2,
                'completed_branch_count': 2,
                'failed_branch_count': 0,
                'maximum_concurrency': 2,
                'overlap_ms': 42,
                'dependency_wait_ms': 0,
                'total_ms': 188,
            },
            'appraisals': [
                {
                    'question_kind': 'relationship_social',
                    'semantic_question': '这次行为怎样改变了关系中的安全感？',
                    'status': 'completed',
                    'explanation': '持续贬低削弱了关系安全感。',
                    'propositions': [
                        {
                            'proposition_kind': 'relationship_shift',
                            'semantic_value': '亲近关系中的信任受到伤害。',
                        },
                    ],
                    'deltas': [
                        {'delta': -20, 'reason': '关系安全感下降。'},
                    ],
                },
            ],
            'branches': [
                {
                    'phase': 'preliminary',
                    'branch_index': 1,
                    'goal_kind': 'bond_protection',
                    'status': 'completed',
                    'selection': 'primary',
                    'intention': '保护重要关系中的边界',
                    'desired_outcome': '让对方停止贬低并理解伤害',
                    'concrete_detail': '先明确说明这句话造成的伤害',
                    'reason': '关系价值使这次伤害不能被轻轻带过。',
                    'private_monologue': '我不想把这份受伤假装成没事。',
                    'expected_consequences': ['对方知道边界已经被触碰'],
                    'confidence': '高',
                },
                {
                    'phase': 'preliminary',
                    'branch_index': 2,
                    'goal_kind': 'autonomy_boundary',
                    'status': 'completed',
                    'selection': 'suppressed',
                    'intention': '立即反击',
                    'desired_outcome': '结束当前攻击',
                    'concrete_detail': '用更强硬的话顶回去',
                    'reason': '被冒犯会自然地产生反击冲动。',
                    'private_monologue': '我很想马上反击，但这会让关系更糟。',
                    'expected_consequences': ['冲突可能进一步升级'],
                    'confidence': '中',
                },
            ],
            'collapse': {
                'primary_branch_index': 1,
                'supporting_branch_indices': [],
                'suppressed_branch_indices': [2],
                'selection_reason': '主目标保留了受伤事实，反击目标被压下。',
            },
        },
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


def test_response_graph_exposes_native_v2_parallel_results(monkeypatch) -> None:
    """Native V2 branches, collapse, affect, and monologue reach the graph."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, 'COGNITION_VISUAL_DIRECTIVES_ENABLED', True)
    state = _response_state(visual_directives={})
    state['cognition_core_output'] = _native_v2_cognition_output()
    graph = service._build_response_cognition_graph(
        graph_result=_response_graph_result(),
        consolidation_state=state,
        run_id='native-v2-run',
    )

    node_ids = {node['id'] for node in graph['nodes']}
    assert {
        'v2.parallel',
        'v2.appraisal',
        'v2.branch.1',
        'v2.branch.2',
        'v2.collapse',
        'v2.affect',
    } <= node_ids
    parallel = _node(graph, 'v2.parallel')
    assert parallel['detail']['parallel_execution']['maximum_concurrency'] == 2
    assert parallel['detail']['parallel_execution']['completed_branch_count'] == 2
    appraisal = _node(graph, 'v2.appraisal')
    assert appraisal['detail']['appraisal_results'][0]['explanation'] == (
        '持续贬低削弱了关系安全感。'
    )
    primary = _node(graph, 'v2.branch.1')
    assert primary['detail']['selection'] == 'primary'
    assert primary['detail']['intention'] == '保护重要关系中的边界'
    assert primary['detail']['private_monologue'] == (
        '我不想把这份受伤假装成没事。'
    )
    suppressed = _node(graph, 'v2.branch.2')
    assert suppressed['detail']['selection'] == 'suppressed'
    collapse = _node(graph, 'v2.collapse')
    assert collapse['detail']['selected_bid_reason'] == (
        '她先承认这次伤害，再决定如何回应。'
    )
    affect = _node(graph, 'v2.affect')
    assert affect['detail']['affect_projection'][0]['emotion'] == '悲伤'
    detail_text = repr([node['detail'] for node in graph['nodes']])
    assert 'branch_id' not in detail_text
    assert 'evidence_handles' not in detail_text
    assert 'prompt' not in detail_text
    assert any(
        edge['source'] == 'v2.branch.1'
        and edge['target'] == 'v2.collapse'
        and edge['kind'] == 'join'
        for edge in graph['edges']
    )


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
