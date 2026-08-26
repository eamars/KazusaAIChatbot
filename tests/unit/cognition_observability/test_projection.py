"""Deterministic tests for the closed cognition-observation projection."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from kazusa_ai_chatbot.brain_service.cognition_observation_projection import (
    build_live_cognition_observation,
    build_self_cognition_observation,
)

_NOW = datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc)


def _empty_rag() -> dict[str, object]:
    """Build the canonical prewarm-only RAG shape."""

    return {
        "answer": "",
        "user_image": {},
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {},
    }


def _outcome(reason_code: str) -> dict[str, object]:
    """Build one valid prewarm outcome for a fixed reason disposition."""

    status_by_reason = {
        "worker_unresolved": "empty",
        "worker_contract_invalid": "failed",
        "projection_failed": "failed",
        "no_shared_memory": "empty",
        "worker_error": "failed",
        "shared_memory_ready": "completed",
        "shared_memory_merged": "completed",
        "empty_query_after_character_mention": "skipped",
        "not_first_cycle": "skipped",
        "unsupported_episode": "skipped",
    }
    status = status_by_reason[reason_code]
    evidence = []
    merged = 0
    attempted = status != "skipped"
    latency = 1 if attempted else 0
    if reason_code in {"shared_memory_ready", "shared_memory_merged"}:
        evidence = [{"summary": "shared evidence", "content": "fact"}]
        merged = 1 if reason_code == "shared_memory_merged" else 0
    rag = _empty_rag()
    rag["memory_evidence"] = evidence
    return {
        "schema_version": "shared_memory_prewarm_outcome.v1",
        "status": status,
        "reason_code": reason_code,
        "attempted": attempted,
        "latency_ms": latency,
        "retrieved_shared_count": len(evidence),
        "merged_shared_count": merged,
        "rag_result": rag,
    }


def _core() -> dict[str, object]:
    """Build a small canonical cognition output."""

    return {
        "schema_version": "cognition_output.v3",
        "appraisals": [],
        "active_character_goal": {
            "goal_kind": "answer",
            "intent": "answer the question",
            "reason": "the question is grounded",
            "cause_summary": "a user question arrived",
        },
        "private_monologue": "Keep the answer grounded.",
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "answer directly",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": "Do not invent unknown facts.",
        },
        "affect_projection": [],
    }


def _live_state() -> dict[str, object]:
    """Build a live persona state with an explicit typed prewarm outcome."""

    return {
        "user_input": "What happened?",
        "cognition_core_output": _core(),
        "logical_stance": "direct",
        "character_intent": "answer",
        "judgment_note": "grounded",
        "cognitive_episode": {
            "target_scope": {"channel_type": "private"},
        },
        "shared_memory_prewarm_outcome": _outcome("shared_memory_ready"),
    }


def _live_graph() -> dict[str, object]:
    """Build the settled graph facts needed by live projection."""

    return {
        "should_respond": True,
        "reason_to_respond": "the question is answerable",
        "final_dialog": ["A grounded answer."],
        "llm_trace_id": "trace-1",
    }


def _section(observation, section_id: str):
    """Find a canonical section by its producer-owned identifier."""

    return next(section for section in observation.sections if section.section_id == section_id)


def test_live_projection_reports_all_shared_memory_prewarm_dispositions() -> None:
    """Every fixed worker reason remains visible with its exact disposition."""

    for reason_code in (
        "worker_unresolved",
        "worker_contract_invalid",
        "projection_failed",
        "no_shared_memory",
        "worker_error",
        "shared_memory_ready",
        "shared_memory_merged",
        "empty_query_after_character_mention",
        "not_first_cycle",
        "unsupported_episode",
    ):
        state = _live_state()
        state["shared_memory_prewarm_outcome"] = _outcome(reason_code)
        observation = build_live_cognition_observation(
            graph_result=_live_graph(),
            persona_state=state,
            run_id="run-1",
            cognition_invocation_id="inv-1",
            terminal_status="completed_visible",
            visual_stage_failed=False,
            visual_stage_reached=False,
            failure_code="",
            generated_at=_NOW,
        )
        assert observation is not None
        section = _section(observation, "evidence.shared_memory_prewarm")
        assert section.fields[1].value == reason_code
        assert section.status == _outcome(reason_code)["status"]


def test_context_sources_share_one_detail_shape_and_budget() -> None:
    """Context sources use the exact closed detail shape and aggregation."""

    relationship_axes = [
        "familiarity=known",
        "positive_regard=warm",
        "trust=steady",
        "attachment=present",
        "desired_closeness=near",
        "perceived_closeness=near",
        "care=mutual",
        "boundary_safety=safe",
        "exclusivity=none",
        "unresolved_injury=none",
        "salience=high",
    ]
    character_details = [
        "affect.emotion_id=calm",
        "affect.intensity=0.5",
        "affect.phase=steady",
        "affect.trend=flat",
        "affect.root_kind=state",
        "affect.cause_class=rest",
        "affect.freshness=fresh",
        "pressures.kind=deadline",
        "pressures.salience=0.4",
        "pressures.lifecycle=active",
        "pressures.cause_class=task",
        "pressures.freshness=fresh",
    ]
    relationship_details = relationship_axes + [
        "causal_context.entity_kind=conversation",
        "causal_context.semantic_summary=recent care",
        "causal_context.salience=high",
        "causal_context.lifecycle=active",
        "causal_context.freshness=fresh",
        "affect.emotion_id=calm",
        "affect.intensity=0.4",
        "affect.phase=steady",
        "affect.trend=flat",
        "affect.freshness=fresh",
        "relationship_freshness=fresh",
        "evidence_freshness=current",
    ]
    state = _live_state()
    state.update({
        "settled_relevance_context_consumption": {
            "character_operational_context": {
                "status": "active",
                "summary": "character context",
                "semantic_summary": "hidden semantic summary",
                "character_name": "hidden character name",
                "state": "hidden state",
                "affect": [{
                    "emotion_id": "calm",
                    "intensity": 0.5,
                    "phase": "steady",
                    "trend": "flat",
                    "root_kind": "state",
                    "cause_class": "rest",
                    "freshness": "fresh",
                }],
                "pressures": [{
                    "kind": "deadline",
                    "salience": 0.4,
                    "lifecycle": "active",
                    "cause_class": "task",
                    "freshness": "fresh",
                }],
            },
            "relationship_context": {
                "status": "active",
                "summary": "relationship summary",
                "axes": {
                    "familiarity": "known",
                    "positive_regard": "warm",
                    "trust": "steady",
                    "attachment": "present",
                    "desired_closeness": "near",
                    "perceived_closeness": "near",
                    "care": "mutual",
                    "boundary_safety": "safe",
                    "exclusivity": "none",
                    "unresolved_injury": "none",
                    "salience": "high",
                },
                "causal_context": [
                    {
                        "entity_kind": "conversation",
                        "semantic_summary": "recent care",
                        "salience": "high",
                        "lifecycle": "active",
                        "freshness": "fresh",
                    },
                ],
                "affect": [
                    {
                        "emotion_id": "calm",
                        "intensity": 0.4,
                        "phase": "steady",
                        "trend": "flat",
                        "freshness": "fresh",
                    },
                ],
                "relationship_freshness": "fresh",
                "evidence_freshness": "current",
            },
            "style": {
                "status": "active",
                "relevance": {
                    "user": {
                        "status": "active",
                        "revision": 2,
                        "confidence": 0.8,
                        "engagement_guidelines": ["user relevance"],
                    },
                    "group_channel": {
                        "status": "active",
                        "revision": 3,
                        "confidence": 0.7,
                        "engagement_guidelines": ["group relevance"],
                    },
                },
            },
        },
        "text_surface_input": {},
        "interaction_style_context": {
            "surface": {
                "user": {
                    "status": "active",
                    "revision": 4,
                    "confidence": 0.9,
                    "speech_guidelines": ["user speech"],
                    "social_guidelines": ["user social"],
                    "pacing_guidelines": ["user pacing"],
                    "engagement_guidelines": ["user engagement"],
                },
                "group_channel": {
                    "status": "active",
                    "revision": 5,
                    "confidence": 0.6,
                    "speech_guidelines": ["group speech"],
                    "social_guidelines": ["group social"],
                    "pacing_guidelines": ["group pacing"],
                    "engagement_guidelines": ["group engagement"],
                },
            },
        },
    })
    observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=state,
        run_id="run-1",
        cognition_invocation_id="inv-1",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    section = _section(observation, "reasoning.context_consumption")
    assert all(
        [field.key for field in record.fields][:3]
        == ["stage", "source_kind", "status"]
        for record in section.records
    )
    character_record = next(
        record
        for record in section.records
        if record.fields[1].value == "character_operational_context"
        and record.fields[0].value == "settled_relevance"
    )
    character_fields = {
        field.key: field.value
        for field in character_record.fields
    }
    assert character_fields["summary"] == "character context"
    assert character_fields["details"] == character_details
    assert all(
        value not in character_fields["details"]
        for value in (
            "summary=banned",
            "semantic_summary=hidden semantic summary",
            "character_name=hidden character name",
            "state=hidden state",
        )
    )
    relationship_record = next(
        record
        for record in section.records
        if record.fields[1].value == "relationship_context"
        and record.fields[0].value == "settled_relevance"
    )
    relationship_fields = {
        field.key: field.value
        for field in relationship_record.fields
    }
    assert relationship_fields["summary"] == "relationship summary"
    assert relationship_fields["details"] == relationship_details
    assert "summary=relationship summary" not in relationship_fields["details"]
    assert "semantic_summary=hidden relationship summary" not in (
        relationship_fields["details"]
    )
    style_relevance_record = next(
        record
        for record in section.records
        if record.fields[1].value == "style_relevance"
    )
    assert style_relevance_record.fields[-1].value == [
        "consumer_role=relevance",
        "source=user",
        "status=active",
        "revision=2",
        "confidence=0.8",
        "engagement_guidelines=user relevance",
        "source=group_channel",
        "status=active",
        "revision=3",
        "confidence=0.7",
        "engagement_guidelines=group relevance",
    ]
    style_surface_record = next(
        record
        for record in section.records
        if record.fields[0].value == "surface"
        and record.fields[1].value == "style"
    )
    assert style_surface_record.fields[-1].value == [
        "consumer_role=surface",
        "source=user",
        "status=active",
        "revision=4",
        "confidence=0.9",
        "speech_guidelines=user speech",
        "social_guidelines=user social",
        "pacing_guidelines=user pacing",
        "engagement_guidelines=user engagement",
        "source=group_channel",
        "status=active",
        "revision=5",
        "confidence=0.6",
        "speech_guidelines=group speech",
        "social_guidelines=group social",
        "pacing_guidelines=group pacing",
        "engagement_guidelines=group engagement",
    ]
    assert section.displayed_record_count == len(section.records)
    assert section.reported_record_count == section.displayed_record_count

    def project_context(
        *,
        settled: object = None,
        cognition: object = None,
        include_settled: bool = False,
        include_cognition: bool = False,
    ):
        case_state = _live_state()
        if include_settled:
            case_state["settled_relevance_context_consumption"] = settled
        if include_cognition:
            case_state["cognition_input"] = cognition
        case_observation = build_live_cognition_observation(
            graph_result=_live_graph(),
            persona_state=case_state,
            run_id="context-case",
            cognition_invocation_id="context-case-invocation",
            terminal_status="completed_visible",
            visual_stage_failed=False,
            visual_stage_reached=False,
            failure_code="",
            generated_at=_NOW,
        )
        assert case_observation is not None
        return _section(case_observation, "reasoning.context_consumption")

    no_reported = project_context()
    assert no_reported.status == "not_reported"
    assert no_reported.fields[0].value == "not_reported"
    assert no_reported.fields[1].value == 10
    assert all(
        record.fields[2].value == "not_reported"
        for record in no_reported.records
    )

    only_invalid = project_context(
        settled={
            "character_operational_context": {
                "affect": "invalid",
            },
        },
        include_settled=True,
    )
    assert only_invalid.status == "failed"
    assert only_invalid.fields[0].value == "failed"
    assert only_invalid.fields[1].value == 10

    only_empty = project_context(
        settled={
            "character_operational_context": {
                "status": "empty",
            },
        },
        include_settled=True,
    )
    assert only_empty.status == "empty"
    assert only_empty.fields[0].value == "empty"
    assert only_empty.fields[1].value == 10

    valid_and_invalid = project_context(
        settled={
            "character_operational_context": {
                "status": "active",
                "affect": [{"emotion_id": "calm"}],
            },
            "relationship_context": {
                "axes": [],
            },
        },
        include_settled=True,
    )
    assert valid_and_invalid.status == "partial"
    assert valid_and_invalid.fields[0].value == "partial"
    assert valid_and_invalid.fields[1].value == 10

    valid_detail = project_context(
        settled={
            "character_operational_context": {
                "status": "active",
                "affect": [{"emotion_id": "calm"}],
            },
        },
        include_settled=True,
    )
    assert valid_detail.status == "completed"
    assert valid_detail.fields[0].value == "completed"
    assert valid_detail.fields[1].value == 10


def test_public_group_scene_projects_discriminator_headers_and_status() -> None:
    """Group-scene status and headers preserve valid versus invalid content."""

    valid_turn = {
        "role": "user",
        "speaker_name": "Alice",
        "text": "hello",
        "addressed_names": ["Character"],
        "reply_to_name": "",
        "scene_position": "trigger",
        "anchor_kind": "message",
    }

    valid_state = _live_state()
    valid_state.update({
        "public_group_scene_projection_status": "completed",
        "public_group_scene_context": {
            "schema_version": "group_scene_context.v1",
            "visible_participants": ["Alice", "Bob"],
            "omitted_turn_count": 2,
            "turns": [valid_turn],
        },
    })
    valid_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=valid_state,
        run_id="group-scene-valid",
        cognition_invocation_id="group-scene-valid-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert valid_observation is not None
    valid_section = _section(
        valid_observation,
        "context.public_group_scene",
    )
    assert [field.key for field in valid_section.fields] == [
        "status",
        "visible_participants",
        "visible_participant_count",
        "omitted_turn_count",
    ]
    assert [field.value for field in valid_section.fields] == [
        "completed",
        ["Alice", "Bob"],
        2,
        2,
    ]
    assert valid_section.status == "completed"

    mixed_state = _live_state()
    mixed_state.update({
        "public_group_scene_projection_status": "completed",
        "public_group_scene_context": {
            "schema_version": "group_scene_context.v1",
            "visible_participants": "Alice",
            "omitted_turn_count": 1,
            "turns": [valid_turn],
        },
    })
    mixed_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=mixed_state,
        run_id="group-scene-mixed",
        cognition_invocation_id="group-scene-mixed-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert mixed_observation is not None
    mixed_section = _section(
        mixed_observation,
        "context.public_group_scene",
    )
    assert mixed_section.status == "partial"
    assert [field.key for field in mixed_section.fields] == [
        "status",
        "omitted_turn_count",
    ]

    invalid_state = _live_state()
    invalid_state.update({
        "public_group_scene_projection_status": "completed",
        "public_group_scene_context": {
            "schema_version": "group_scene_context.v1",
            "visible_participants": "Alice",
            "omitted_turn_count": -1,
            "turns": [],
        },
    })
    invalid_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=invalid_state,
        run_id="group-scene-invalid",
        cognition_invocation_id="group-scene-invalid-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert invalid_observation is not None
    assert _section(
        invalid_observation,
        "context.public_group_scene",
    ).status == "failed"

    skipped_state = _live_state()
    skipped_state["public_group_scene_projection_status"] = "skipped"
    skipped_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=skipped_state,
        run_id="group-scene-skipped",
        cognition_invocation_id="group-scene-skipped-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert skipped_observation is not None
    assert _section(
        skipped_observation,
        "context.public_group_scene",
    ).status == "skipped"

    failed_state = _live_state()
    failed_state["public_group_scene_projection_status"] = "failed"
    failed_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=failed_state,
        run_id="group-scene-failed",
        cognition_invocation_id="group-scene-failed-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert failed_observation is not None
    assert _section(
        failed_observation,
        "context.public_group_scene",
    ).status == "failed"


def test_conversation_progress_invalid_headers_affect_section_status() -> None:
    """Progress headers omit invalid values and classify the section."""

    valid_event = {
        "semantic_summary": "a grounded event",
        "state": "active",
        "actor": "user",
        "action": "asked",
        "object": "question",
        "beneficiary": "character",
        "precondition": "context available",
    }
    valid_progress = {
        "schema_version": "conversation_progress_prompt.v2",
        "status": "active",
        "continuity": "steady",
        "turn_count": 4,
        "current_thread": "grounded thread",
        "character_stance": "helpful",
        "user_goal": "understand",
        "current_blocker": "",
        "emotional_trajectory": "stable",
        "episode_narrative": "the exchange continues",
        "overused_moves": ["reassurance"],
        "events": [valid_event],
    }
    state = _live_state()
    state["conversation_progress"] = valid_progress
    observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=state,
        run_id="progress-valid",
        cognition_invocation_id="progress-valid-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    section = _section(observation, "context.conversation_progress")
    assert section.status == "completed"
    assert [field.key for field in section.fields] == [
        "status",
        "continuity",
        "turn_count",
        "current_thread",
        "character_stance",
        "user_goal",
        "current_blocker",
        "emotional_trajectory",
        "episode_narrative",
        "overused_moves",
    ]

    mixed_progress = dict(valid_progress)
    mixed_progress["overused_moves"] = {"invalid": True}
    mixed_state = _live_state()
    mixed_state["conversation_progress"] = mixed_progress
    mixed_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=mixed_state,
        run_id="progress-mixed",
        cognition_invocation_id="progress-mixed-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert mixed_observation is not None
    mixed_section = _section(
        mixed_observation,
        "context.conversation_progress",
    )
    assert mixed_section.status == "partial"
    assert "overused_moves" not in {
        field.key for field in mixed_section.fields
    }

    invalid_progress = dict(valid_progress)
    invalid_progress["status"] = {"invalid": True}
    invalid_progress["overused_moves"] = {"invalid": True}
    invalid_progress["events"] = []
    invalid_state = _live_state()
    invalid_state["conversation_progress"] = invalid_progress
    invalid_observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=invalid_state,
        run_id="progress-invalid",
        cognition_invocation_id="progress-invalid-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert invalid_observation is not None
    assert _section(
        invalid_observation,
        "context.conversation_progress",
    ).status == "failed"


def test_self_source_uses_stable_wire_field_keys_and_order() -> None:
    """Self source packets map to the four stable v1 field keys."""

    from kazusa_ai_chatbot.self_cognition import models

    observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-source-run",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_INPUT: {
                "source_packet": {
                    "case_name": "scheduled_reflection",
                    "instruction": "inspect the current context",
                    "actionability": "consider a bounded follow-up",
                    "semantic_due_state": "due_now",
                },
            },
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert observation is not None
    section = _section(observation, "self.source")
    assert [field.key for field in section.fields] == [
        "source_kind",
        "summary",
        "reason",
        "due_state",
    ]
    assert [field.value for field in section.fields] == [
        "scheduled_reflection",
        "inspect the current context",
        "consider a bounded follow-up",
        "due_now",
    ]


def test_self_action_results_fallback_requires_a_valid_empty_result_list() -> None:
    """Self action attempts are fallback evidence only for valid empty lists."""

    from kazusa_ai_chatbot.self_cognition import models

    action_attempt = {
        "kind": "reminder",
        "status": "scheduled",
        "visibility": "private",
        "result_summary": "attempt fallback",
        "reason": "bounded",
        "due_at": "2026-08-27T00:00:00Z",
    }

    empty_output = _core()
    empty_output["action_results"] = []
    empty_observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-action-empty",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "cognition_core_output": empty_output,
            },
            models.ARTIFACT_ACTION_ATTEMPT: action_attempt,
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert empty_observation is not None
    empty_section = _section(empty_observation, "action.results")
    assert empty_section.status == "completed"
    assert {
        field.key: field.value
        for field in empty_section.records[0].fields
    }["outcome"] == "attempt fallback"

    missing_output = _core()
    missing_observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-action-missing",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "cognition_core_output": missing_output,
            },
            models.ARTIFACT_ACTION_ATTEMPT: action_attempt,
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert missing_observation is not None
    missing_section = _section(missing_observation, "action.results")
    assert missing_section.status == "not_reported"
    assert missing_section.records == []

    wrong_type_output = _core()
    wrong_type_output["action_results"] = {"invalid": True}
    wrong_type_observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-action-wrong-type",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "cognition_core_output": wrong_type_output,
            },
            models.ARTIFACT_ACTION_ATTEMPT: action_attempt,
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert wrong_type_observation is not None
    wrong_type_section = _section(
        wrong_type_observation,
        "action.results",
    )
    assert wrong_type_section.status == "failed"
    assert wrong_type_section.records == []

    non_empty_output = _core()
    non_empty_output["action_results"] = [{
        "action_kind": "real_action",
        "status": "sent",
        "visibility": "private",
        "result_summary": "real result",
        "reason": "selected",
        "due_at": "2026-08-28T00:00:00Z",
    }]
    non_empty_observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-action-non-empty",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "cognition_core_output": non_empty_output,
            },
            models.ARTIFACT_ACTION_ATTEMPT: action_attempt,
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert non_empty_observation is not None
    non_empty_section = _section(
        non_empty_observation,
        "action.results",
    )
    assert non_empty_section.status == "completed"
    assert {
        field.key: field.value
        for field in non_empty_section.records[0].fields
    }["outcome"] == "real result"


def test_self_visible_message_precedence_fails_closed_and_counts_source_rows() -> None:
    """Preferred self message sources do not silently fall through on errors."""

    from kazusa_ai_chatbot.self_cognition import models

    preferred_output = _core()
    preferred_output["final_dialog"] = ["cognition fallback"]
    preferred_observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-message-preferred",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "cognition_core_output": preferred_output,
            },
            models.ARTIFACT_ACTION_CANDIDATE: {
                "text": "preferred message",
                "messages": ["lower-precedence message"],
            },
            models.ARTIFACT_ROUTE_EFFECT: {
                "visible_dialog": ["route fallback"],
            },
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert preferred_observation is not None
    preferred_section = _section(
        preferred_observation,
        "surface.visible_messages",
    )
    assert preferred_section.status == "completed"
    assert preferred_section.records[0].fields[1].value == "preferred message"

    invalid_output = _core()
    invalid_output["final_dialog"] = ["cognition fallback"]
    invalid_observation = build_self_cognition_observation(
        artifact_payloads={
            models.ARTIFACT_RUN_RECORD: {
                "run_id": "self-message-invalid",
                "status": "completed",
            },
            models.ARTIFACT_COGNITION_OUTPUT: {
                "cognition_core_output": invalid_output,
            },
            models.ARTIFACT_ACTION_CANDIDATE: {
                "text": {"invalid": True},
                "messages": ["lower-precedence message"],
            },
            models.ARTIFACT_ROUTE_EFFECT: {
                "visible_dialog": ["route fallback"],
            },
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert invalid_observation is not None
    invalid_section = _section(
        invalid_observation,
        "surface.visible_messages",
    )
    assert invalid_section.status == "failed"
    assert invalid_section.records == []

    live_observation = build_live_cognition_observation(
        graph_result={
            **_live_graph(),
            "final_dialog": ["first", 7, "third"],
        },
        persona_state=_live_state(),
        run_id="live-message-counts",
        cognition_invocation_id="live-message-counts-invocation",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert live_observation is not None
    live_section = _section(
        live_observation,
        "surface.visible_messages",
    )
    assert live_section.status == "partial"
    assert live_section.reported_record_count == 3
    assert live_section.displayed_record_count == 2
    assert {
        field.key: field.value
        for field in live_section.fields
    }["message_count"] == 3


def test_live_and_self_projections_share_exact_section_catalog() -> None:
    """Live and self runs expose the exact ordered producer catalog."""

    live = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=_live_state(),
        run_id="run-1",
        cognition_invocation_id="inv-1",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    self_observation = build_self_cognition_observation(
        artifact_payloads={
            "self_cognition_run_record.json": {
                "run_id": "self-run-1",
                "status": "completed",
            },
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert live is not None
    assert self_observation is not None
    assert [section.section_id for section in live.sections] == [
        "input.turn",
        "decision.response",
        "cognition.appraisals",
        "cognition.goal",
        "cognition.response_plan",
        "cognition.affect",
        "reasoning.subjective",
        "reasoning.context_consumption",
        "evidence.memory",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "action.requests",
        "action.results",
        "action.continuation",
        "surface.visual_directives",
        "surface.visible_messages",
    ]
    assert [section.section_id for section in self_observation.sections] == [
        "cognition.appraisals",
        "cognition.goal",
        "cognition.response_plan",
        "cognition.affect",
        "reasoning.subjective",
        "reasoning.context_consumption",
        "evidence.memory",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "action.requests",
        "action.results",
        "action.continuation",
        "surface.visual_directives",
        "surface.visible_messages",
        "self.source",
        "self.route",
        "self.consolidation",
    ]
    assert [
        (node.node_id, tuple(node.section_refs))
        for node in live.nodes
    ] == [
        ("input.turn", ("input.turn",)),
        ("decision.response", ("decision.response",)),
        ("cognition.meaning", ("cognition.appraisals",)),
        ("cognition.goal", ("cognition.goal",)),
        ("cognition.response", ("cognition.response_plan",)),
        ("cognition.affect", ("cognition.affect",)),
        (
            "reasoning.context",
            ("reasoning.subjective", "reasoning.context_consumption"),
        ),
        (
            "evidence.memory",
            (
                "evidence.shared_memory_prewarm",
                "evidence.memory",
                "context.conversation_progress",
                "context.public_group_scene",
            ),
        ),
        (
            "action.results",
            ("action.requests", "action.results", "action.continuation"),
        ),
        ("surface.visual", ("surface.visual_directives",)),
        ("surface.visible", ("surface.visible_messages",)),
    ]
    assert [
        (node.node_id, tuple(node.section_refs))
        for node in self_observation.nodes
    ] == [
        ("self.source", ("self.source",)),
        ("cognition.meaning", ("cognition.appraisals",)),
        ("cognition.goal", ("cognition.goal",)),
        ("cognition.response", ("cognition.response_plan",)),
        ("cognition.affect", ("cognition.affect",)),
        (
            "reasoning.context",
            ("reasoning.subjective", "reasoning.context_consumption"),
        ),
        (
            "evidence.memory",
            (
                "evidence.shared_memory_prewarm",
                "evidence.memory",
                "context.conversation_progress",
                "context.public_group_scene",
            ),
        ),
        ("self.route", ("self.route",)),
        (
            "action.results",
            ("action.requests", "action.results", "action.continuation"),
        ),
        ("surface.visual", ("surface.visual_directives",)),
        ("surface.visible", ("surface.visible_messages",)),
        ("self.consolidation", ("self.consolidation",)),
    ]


def test_projection_uses_closed_source_field_mapping_and_invalid_row_counts() -> None:
    """Invalid evidence rows are omitted while source and display counts stay true."""

    state = _live_state()
    state["rag_result"] = {
        "memory_evidence": [
            {
                "summary": "kept",
                "fact": "fact body",
                "excerpt": "ignored excerpt",
                "content": "ignored content",
                "title": "source title",
                "relevance": 0.9,
                "recency": "recent",
                "due_state": "ready",
                "evidence_boundary_notes": "bounded evidence",
                "database_identifier": "hidden",
            },
            7,
        ],
    }
    observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=state,
        run_id="run-1",
        cognition_invocation_id="inv-1",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    section = _section(observation, "evidence.memory")
    assert section.status == "partial"
    assert section.reported_record_count == 2
    assert section.displayed_record_count == 1
    assert section.truncated is True
    assert [field.key for field in section.records[0].fields] == [
        "source_kind",
        "summary",
        "content",
        "title",
        "relevance",
        "recency",
        "due_state",
        "evidence_boundary_notes",
    ]
    assert [field.value for field in section.records[0].fields] == [
        "memory_evidence",
        "kept",
        "fact body",
        "source title",
        0.9,
        "recent",
        "ready",
        "bounded evidence",
    ]


def test_projection_excludes_protected_and_operational_fields() -> None:
    """Protected source keys never appear in the canonical observation payload."""

    state = _live_state()
    protected_values = {
        "prompt": "sentinel prompt",
        "raw_model_output": "sentinel raw model output",
        "embedding": "sentinel embedding",
        "raw_message": "sentinel raw message",
        "message_envelope": "sentinel message envelope",
        "database_identifier": "sentinel database identifier",
        "adapter_identifier": "sentinel adapter identifier",
        "action_parameter": "sentinel action parameter",
        "handler_metadata": "sentinel handler metadata",
        "worker_error_text": "sentinel worker error text",
    }
    state["rag_result"] = {
        "memory_evidence": [{
            "summary": "safe",
            **protected_values,
        }],
    }
    observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=state,
        run_id="run-1",
        cognition_invocation_id="inv-1",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    serialized = json.dumps(
        observation.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )
    for value in protected_values.values():
        assert value not in serialized
    projected_field_keys = {
        field.key
        for section in observation.sections
        for field in section.fields
    }
    projected_field_keys.update(
        field.key
        for section in observation.sections
        for record in section.records
        for field in record.fields
    )
    assert projected_field_keys.isdisjoint(protected_values)
    assert observation.disclosure.excluded == list(protected_values)


def test_projection_emits_only_canonical_sequence_and_reference_edges() -> None:
    """Edges use only the two wire kinds and the fixed producer topology."""

    observation = build_live_cognition_observation(
        graph_result=_live_graph(),
        persona_state=_live_state(),
        run_id="run-1",
        cognition_invocation_id="inv-1",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    assert all(edge.kind in {"sequence", "reference"} for edge in observation.edges)
    assert all(edge.label == "" for edge in observation.edges)
    assert (
        observation.edges[0].source,
        observation.edges[0].target,
        observation.edges[0].kind,
    ) == ("input.turn", "decision.response", "sequence")
