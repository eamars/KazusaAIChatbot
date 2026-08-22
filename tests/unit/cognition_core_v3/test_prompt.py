"""Deterministic tests for Cognition V3 dynamic question packets."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import (
    anchor,
    goal_cognition,
    prompt,
    workspace,
)
from kazusa_ai_chatbot.cognition_core_v3.budget import estimate_message_tokens
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    _appraisal_role_assignment_handles_by_evidence,
    _build_serial_initial_context,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import APPRAISAL_STAGE_FAMILIES
from kazusa_ai_chatbot.cognition_core_v3.semantic_source_planner import (
    question_proposition_kind_semantics,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    GOAL_RESOLUTION_VALUES,
    SEMANTIC_QUESTION_KINDS,
    validate_cognition_core_input,
)

_LIVE_CASE_MANIFEST = Path(
    "tests/fixtures/cognition_core_v3_live_case_manifest.json"
)


def _manifest_input(case_id: str) -> dict[str, object]:
    """Load and validate one frozen V3 input without a comparison runner."""

    manifest = json.loads(_LIVE_CASE_MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["cases"]
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise AssertionError(f"expected one manifest row for {case_id}")
    return deepcopy(validate_cognition_core_input(matches[0]["canonical_input"]))


def _question() -> prompt.ChainQuestion:
    """Build one bounded appraisal question using a registered contract."""

    question = prompt.build_appraisal_stage_question(
        planned_questions=[{
            "question_kind": "event_agency",
            "evidence_handles": ["e1"],
            "permitted_role_handles": ["ce1", "current_user"],
            "permitted_role_assignment_handles": ["self", "current_user"],
            "permitted_delta_paths": [
                "active_events.ce1.responsibility",
            ],
            "semantic_question": "What happened and who had agency?",
        }],
        stage_name="A1",
    )
    return question


def _first_message(
    question: prompt.ChainQuestion,
    *,
    scene_marker: str,
) -> str:
    """Render one complete volatility-ordered first user packet."""

    observation_context = _observation_context(scene_marker)
    first_message = prompt.build_first_user_message(
        observation_context=observation_context,
        question=question,
    )
    return first_message


def _observation_context(
    scene_marker: str,
)-> dict[str, object]:
    """Build one exact prompt-safe observation carrier."""

    return {
        "conversation_frame": {
            "channel_scope": "private",
            "character_role": "current character",
            "conversation_continuity": "",
            "current_user_role": "current user",
            "dialogue_role_bindings": [],
            "participant_bindings": [],
            "public_group_scene": "",
            "semantic_temporal_context": "current turn",
        },
        "direct_facts": [],
        "entity_index": [],
        "evidence": [{
            "handle": "e1",
            "source_kind": "episode",
            "semantic_text": scene_marker,
            "authority": "current_episode",
            "provenance_role": "current_episode",
        }],
        "supplemental_context": {
            "dialogue_observation": [],
            "local_time_context": [],
            "non_dialog_percepts": [],
            "trigger_source": "user_message",
        },
    }


def test_prompt_questions_are_bounded_contract_oriented_and_dynamic() -> None:
    """Question packets keep fixed structure while current-run values vary."""

    question = _question()
    first_message = _first_message(question, scene_marker="scene-one")
    changed_message = _first_message(question, scene_marker="scene-two")

    assert first_message != changed_message
    decoded = json.loads(first_message)
    assert [next(iter(section)) for section in decoded] == [
        "observation_context",
        "question",
    ]
    assert decoded[-1]["question"] == {
        "contract_name": question.contract_name,
        "instruction": prompt.CHAIN_QUESTION_POINTERS[
            question.contract_name
        ],
        "payload": dict(question.payload),
    }
    assert "output_contract" not in decoded[-1]["question"]
    assert "scene-one" in first_message
    assert "scene-two" not in first_message
    observation = decoded[0]["observation_context"]
    assert observation["evidence"][0]["semantic_text"] == "scene-one"
    assert "visible_percepts" not in first_message
    assert "semantic_scene" not in first_message
    for private_field in (
        "relationship_id",
        "episode_id",
        "source_id",
        "platform_message_id",
        "current_global_user_id",
    ):
        assert private_field not in first_message

    later_message = prompt.build_question_message(
        question,
        interludes=[
            {
                "notice_kind": "state_transition",
                "accepted_count": 2,
                "rejected_count": 1,
            }
        ],
    )
    later_decoded = json.loads(later_message)
    assert [next(iter(section)) for section in later_decoded] == [
        "interludes",
        "question",
    ]

    extra_sections = _observation_context("scene-extra")
    extra_sections["unexpected"] = True
    with pytest.raises(prompt.PromptContractError, match="exact fields"):
        prompt.build_first_user_message(
            observation_context=extra_sections,
            question=question,
        )

    nested_extra_sections = _observation_context("scene-nested-extra")
    scene_context = nested_extra_sections["conversation_frame"]
    assert isinstance(scene_context, dict)
    scene_context["unexpected"] = True
    with pytest.raises(prompt.PromptContractError, match="exact fields"):
        prompt.build_first_user_message(
            observation_context=nested_extra_sections,
            question=question,
        )

    relationship_id_sections = _observation_context("scene-private")
    relationship = relationship_id_sections["conversation_frame"]
    assert isinstance(relationship, dict)
    relationship["relationship_id"] = "durable-relationship-id"
    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.build_first_user_message(
            observation_context=relationship_id_sections,
            question=question,
        )

    source_id_sections = _observation_context("scene-source-id")
    source_id_sections["evidence"][0]["source_id"] = "platform-source-id"
    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.build_first_user_message(
            observation_context=source_id_sections,
            question=question,
        )

    sleep_sections = _observation_context("scene-sleep")
    sleep_scene_context = sleep_sections["conversation_frame"]
    assert isinstance(sleep_scene_context, dict)
    sleep_scene_context["character_sleep_phase"] = "goal-only-marker"
    with pytest.raises(prompt.PromptContractError, match="goal-only"):
        prompt.build_first_user_message(
            observation_context=sleep_sections,
            question=question,
        )
    with pytest.raises(prompt.PromptContractError, match="registered"):
        prompt.ChainQuestion(
            contract_name="unregistered_contract.v1",
            payload={},
        )

    assert tuple(prompt.CHAIN_QUESTION_POINTERS) == (
        prompt.CHAIN_CONTRACT_NAMES
    )
    assert prompt.RUNTIME_PROMPT_TEXTS == tuple(
        prompt.CHAIN_QUESTION_POINTERS.values()
    )
    assert all(
        pointer and pointer.count("\n") <= 2 and len(pointer) <= 500
        for pointer in prompt.RUNTIME_PROMPT_TEXTS
    )


def test_first_a1_packet_uses_compact_static_head_and_local_contract() -> None:
    """The first appraisal request stays below the deterministic size proxy."""

    input_payload = _manifest_input("ordinary_neutral_response")
    context = _build_serial_initial_context(input_payload)
    role_domains = _appraisal_role_assignment_handles_by_evidence(
        context["observation_context"],
        input_payload["evidence"],
    )
    a1_families = dict(APPRAISAL_STAGE_FAMILIES)["A1"]
    planned_questions = [
        question
        for question in context["questions"]
        if question["question_kind"] in a1_families
    ]
    question = prompt.build_appraisal_stage_question(
        planned_questions=planned_questions,
        stage_name="A1",
        role_assignment_handles_by_evidence=role_domains,
    )
    first_user_message = prompt.build_first_user_message(
        observation_context=context["observation_context"],
        question=question,
    )

    request_chars = len(context["system_content"]) + len(first_user_message)
    request_tokens = estimate_message_tokens(
        [context["system_content"], first_user_message]
    )
    assert len(context["system_content"]) <= 4_000
    assert request_chars <= 12_000
    assert request_tokens <= 4_300
    assert question.payload["families"]
    assert question.payload["output_contract"]["json_schema"][
        "family_value_schema"
    ]


def test_first_packet_preserves_independent_prompt_safe_participant_bindings() -> None:
    """Canonical scene bindings render with handles and no persistent ids."""

    participant_bindings = [
        {
            "handle": "p1",
            "display_name": "Team lead",
            "entity_kind": "third_party",
        },
        {
            "handle": "p2",
            "display_name": "Junior colleague",
            "entity_kind": "third_party",
        },
    ]
    scene_context = {
        "channel_scope": "group",
        "character_role": "current character",
        "current_user_role": "current user",
        "semantic_scene": "A group work discussion.",
        "public_group_scene": "The team is discussing the next step.",
        "conversation_continuity": "",
        "semantic_temporal_context": "current turn",
        "participant_bindings": participant_bindings,
    }
    projection_payload = {
        "character_constraints": {},
        "character_operational_context": {},
        "relationship": {},
        "goals": [],
        "threats": [],
        "events": [],
        "knowledge_gaps": [],
        "affect": [],
        "causal_candidates": [],
        "evidence": [],
    }
    canonical_episode = {
        "schema_version": "cognitive_episode.v1",
        "episode_id": "episode:participant-bindings",
        "trigger_source": "user_message",
        "origin_metadata": {},
        "target_scope": {},
        "percepts": [
            {
                "schema_version": "percept.v1",
                "percept_kind": "dialog",
                "source_kind": "dialog",
                "source_id": "message:participant-bindings",
                "content": {"semantic_text": "Hello team."},
                "observed_at": "2026-08-21T00:00:00Z",
            }
        ],
        "evidence_refs": [],
        "created_at": "2026-08-21T00:00:00Z",
        "privacy_scope": "conversation",
        "continuation_depth": 0,
    }
    observation_context = prompt.build_observation_context(
        projection_payload=projection_payload,
        scene_context=scene_context,
        episode=canonical_episode,
        evidence=[],
        direct_facts=[],
    )

    rendered = prompt.build_first_user_message(
        observation_context=observation_context,
        question=_question(),
    )
    scene_section = observation_context["conversation_frame"]
    assert scene_section["dialogue_role_bindings"] == [{
        "speaker_handle": "current_user",
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
    }]
    assert "speaker_role" not in rendered
    assert "addressee_role" not in rendered
    assert scene_section["participant_bindings"] == participant_bindings
    assert scene_section["participant_bindings"] is not participant_bindings
    assert "entity_id" not in rendered
    assert "Team lead" in rendered
    assert "Junior colleague" in rendered

    participant_bindings[0]["display_name"] = "Changed outside packet"
    assert observation_context["conversation_frame"]["participant_bindings"][0][
        "display_name"
    ] == "Team lead"

    invalid_scene = dict(scene_context)
    invalid_scene["participant_bindings"] = [
        {
            **participant_bindings[1],
            "entity_id": "persistent-user-id",
        }
    ]
    with pytest.raises(prompt.PromptContractError, match="fields are not exact"):
        prompt.build_observation_context(
            projection_payload=projection_payload,
            scene_context=invalid_scene,
            episode=canonical_episode,
            evidence=[],
            direct_facts=[],
        )


def test_observation_context_binds_current_dialog_evidence_only() -> None:
    """Current dialog evidence carries typed deictic ownership locally."""

    projection_payload = {
        "character_constraints": {},
        "character_operational_context": {},
        "relationship": {},
        "goals": [],
        "threats": [],
        "events": [],
        "knowledge_gaps": [],
        "affect": [],
        "causal_candidates": [],
        "evidence": [],
    }
    scene_context = {
        "channel_scope": "private",
        "character_role": "current character",
        "current_user_role": "current user",
        "public_group_scene": "",
        "conversation_continuity": "",
        "semantic_temporal_context": "current turn",
        "participant_bindings": [],
    }
    episode = {
        "schema_version": "cognitive_episode.v1",
        "episode_id": "episode:evidence-binding",
        "trigger_source": "user_message",
        "origin_metadata": {},
        "target_scope": {},
        "percepts": [{
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "dialog-source",
            "content": {"semantic_text": "The current dialog."},
            "observed_at": "2026-08-21T00:00:00Z",
        }],
        "evidence_refs": [],
        "created_at": "2026-08-21T00:00:00Z",
        "privacy_scope": "conversation",
        "continuation_depth": 0,
    }
    evidence = [
        {
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "dialog-source",
                "occurred_at": "2026-08-21T00:00:00Z",
                "semantic_summary": "The current dialog.",
            },
            "semantic_text": "The current dialog.",
            "authority": "current_event",
            "visible_to": ["q:event_agency"],
        },
        {
            "evidence_handle": "e2",
            "evidence_ref": {
                "source_kind": "promoted_memory",
                "source_id": "memory-source",
                "occurred_at": "2026-08-20T00:00:00Z",
                "semantic_summary": "Retrieved context.",
            },
            "semantic_text": "Retrieved context.",
            "authority": "retrieved",
            "memory_scope": "current_user_continuity",
            "visible_to": ["q:event_agency"],
        },
        {
            "evidence_handle": "e3",
            "evidence_ref": {
                "source_kind": "tool_result",
                "source_id": "dialog-source",
                "occurred_at": "2026-08-20T00:00:00Z",
                "semantic_summary": "A non-dialog result.",
            },
            "semantic_text": "A non-dialog result.",
            "authority": "current_event",
            "visible_to": ["q:event_agency"],
        },
    ]

    observation_context = prompt.build_observation_context(
        projection_payload=projection_payload,
        scene_context=scene_context,
        episode=episode,
        evidence=evidence,
        direct_facts=[],
    )

    evidence_by_handle = {
        row["handle"]: row for row in observation_context["evidence"]
    }
    assert evidence_by_handle["e1"]["dialogue_role_binding"] == {
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
        "speaker_handle": "current_user",
    }
    assert "dialogue_role_binding" not in evidence_by_handle["e2"]
    assert "dialogue_role_binding" not in evidence_by_handle["e3"]
    rendered = prompt.build_first_user_message(
        observation_context=observation_context,
        question=_question(),
    )
    assert "dialog-source" not in rendered
    assert "memory-source" not in rendered


def test_observation_context_rejects_private_evidence_binding_fields() -> None:
    """Evidence-local role bindings remain a closed prompt-safe object."""

    observation_context = _observation_context("current dialog")
    observation_context["evidence"][0]["dialogue_role_binding"] = {
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
        "speaker_handle": "current_user",
        "entity_id": "private-event-id",
    }
    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.build_first_user_message(
            observation_context=observation_context,
            question=_question(),
        )
    del observation_context["evidence"][0]["dialogue_role_binding"][
        "entity_id"
    ]
    observation_context["evidence"][0]["dialogue_role_binding"][
        "unexpected_field"
    ] = "value"
    with pytest.raises(prompt.PromptContractError, match="fields are not exact"):
        prompt.build_first_user_message(
            observation_context=observation_context,
            question=_question(),
        )


def test_observation_context_preserves_unmatched_fact_provenance_and_entity_state():
    """Prompt projection keeps typed fact provenance and candidate semantics."""

    projection_payload = {
        "character_constraints": {},
        "character_operational_context": {},
        "relationship": {},
        "goals": [{
            "handle": "g1",
            "description": "Protect the current boundary.",
            "lifecycle": "pursuing",
            "salience": "high",
            "duration": "recent",
            "importance": "high",
            "progress": "early",
            "obstruction": "low",
            "urgency": "high",
        }],
        "threats": [],
        "events": [],
        "knowledge_gaps": [],
        "affect": [],
        "causal_candidates": [
            {
                "handle": "ce1",
                "candidate_kind": "event",
                "evidence_handle": "e1",
                "description": "An event candidate.",
                "lifecycle": "candidate",
                "responsibility": "uncertain",
            },
            {
                "handle": "ct1",
                "candidate_kind": "threat",
                "evidence_handle": "e1",
                "description": "A threat candidate.",
                "lifecycle": "active",
                "residual_pressure": "high",
                "harm": "medium",
            },
            {
                "handle": "ck1",
                "candidate_kind": "knowledge_gap",
                "evidence_handle": "e1",
                "description": "A knowledge gap candidate.",
                "lifecycle": "open",
                "uncertainty": "high",
                "relevance": "medium",
            },
        ],
        "evidence": [],
    }
    scene_context = {
        "channel_scope": "private",
        "character_role": "current character",
        "current_user_role": "current user",
        "semantic_scene": "A bounded scene.",
        "public_group_scene": "",
        "conversation_continuity": "",
        "semantic_temporal_context": "current turn",
        "participant_bindings": [],
    }
    episode = {
        "schema_version": "cognitive_episode.v1",
        "episode_id": "episode:direct-fact",
        "trigger_source": "user_message",
        "origin_metadata": {},
        "target_scope": {},
        "percepts": [{
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "message:direct-fact",
            "content": {"semantic_text": "Observed message."},
            "observed_at": "2026-08-21T00:00:00Z",
        }],
        "evidence_refs": [],
        "created_at": "2026-08-21T00:00:00Z",
        "privacy_scope": "conversation",
        "continuation_depth": 0,
    }
    direct_fact = {
        "fact_id": "fact:private",
        "producer": "action_result",
        "fact_kind": "threat_resolved",
        "target_refs": [{
            "scope": "user",
            "kind": "threat",
            "entity_id": "threat:1",
        }],
        "evidence_ref": {
            "source_kind": "action_result",
            "source_id": "action-result:private",
            "occurred_at": "2026-08-21T00:00:00Z",
            "semantic_summary": "The action resolved the threat.",
        },
        "observed_progress": 4,
    }
    observation_context = prompt.build_observation_context(
        projection_payload=projection_payload,
        scene_context=scene_context,
        episode=episode,
        evidence=[],
        direct_facts=[direct_fact],
        handle_to_ref={
            "t1": {"entity_id": "threat:1", "kind": "threat"},
        },
    )

    direct_row = observation_context["direct_facts"][0]
    assert "evidence_handle" not in direct_row
    assert direct_row["evidence_ref"] == {
        "source_kind": "action_result",
        "occurred_at": "2026-08-21T00:00:00Z",
        "semantic_summary": "The action resolved the threat.",
    }
    assert direct_row["target_handles"] == ["t1"]
    assert "source_id" not in json.dumps(observation_context)
    assert "fact_id" not in json.dumps(observation_context)
    assert "entity_id" not in json.dumps(observation_context)

    entities = {
        row["entity_kind"]: row
        for row in observation_context["entity_index"]
    }
    assert entities["goal"]["semantic_state"] == {
        key: value
        for key, value in projection_payload["goals"][0].items()
        if key != "handle"
    }
    assert entities["event"]["semantic_state"]["responsibility"] == "uncertain"
    assert entities["threat"]["semantic_state"]["residual_pressure"] == "high"
    assert entities["knowledge_gap"]["semantic_state"]["uncertainty"] == "high"
    assert entities["event"]["evidence_handles"] == ["e1"]


def test_observation_context_rejects_unsupported_dialogue_role_tokens(
    monkeypatch,
):
    """Dialogue role mapping fails closed instead of inventing an actor handle."""

    monkeypatch.setattr(
        prompt,
        "project_model_visible_percepts",
        lambda _episode: [{
            "input_source": "dialog",
            "content": {"semantic_text": "A message."},
            "speaker_role": "unbound third party",
            "addressee_role": "当前角色",
            "first_person_role": "当前用户",
            "implicit_imperative_subject_role": "当前角色",
        }],
    )
    with pytest.raises(prompt.PromptContractError, match="unsupported"):
        prompt.build_observation_context(
            projection_payload={
                "goals": [],
                "threats": [],
                "events": [],
                "knowledge_gaps": [],
                "causal_candidates": [],
            },
            scene_context={
                "channel_scope": "private",
                "character_role": "current character",
                "current_user_role": "current user",
                "public_group_scene": "",
                "conversation_continuity": "",
                "semantic_temporal_context": "current turn",
                "participant_bindings": [],
            },
            episode={},
            evidence=[],
            direct_facts=[],
        )


def test_appraisal_payload_preserves_planner_role_domains() -> None:
    """Appraisal questions carry exact subject and assignment handle domains."""

    payload = prompt.build_appraisal_question_payload(
        stage_name="A1",
        questions=[
            {
                "question_kind": "event_agency",
                "evidence_handles": ["e1"],
                "permitted_role_handles": ["ce1", "current_user"],
                "permitted_role_assignment_handles": [
                    "self",
                    "current_user",
                    "p1",
                ],
                "permitted_delta_paths": ["active_events.ce1.responsibility"],
                "semantic_question": "Who had agency in the event?",
            }
        ]
    )
    row = payload["families"][0]
    assert row == {
        "family": "event_agency",
        "evidence_handles": ["e1"],
        "permitted_subject_handles": ["ce1", "current_user"],
        "permitted_object_handles": ["ce1", "current_user"],
        "permitted_role_assignment_handles": [
            "self",
            "current_user",
            "p1",
        ],
        "semantic_question": "Who had agency in the event?",
        "proposition_kinds": ["responsibility", "intentionality"],
        "proposition_kind_semantics": {
            "responsibility": "事件主体对结果负有责任",
            "intentionality": "事件主体有意促成该结果",
        },
        "writable_delta_paths": ["active_events.ce1.responsibility"],
        "delta_bounds": [{
            "path": "active_events.ce1.responsibility",
            "minimum": -40,
            "maximum": 40,
        }],
    }
    contract = payload["output_contract"]
    role_values = [
        "actor",
        "experiencer",
        "target",
        "object",
        "affected_goal",
        "affected_relationship",
    ]
    schema = contract["json_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["event_agency"]
    assert schema["additionalProperties"] is False
    assert "properties" not in schema
    family_schema = schema["family_value_schema"]
    assert family_schema["required"] == ["propositions", "deltas"]
    proposition_schema = family_schema["properties"]["propositions"]["items"]
    assert proposition_schema["required"] == [
        "proposition_kind",
        "subject_handle",
        "evidence_handles",
        "role_assignments",
        "semantic_value",
    ]
    assert proposition_schema["optional"] == ["object_handle"]
    role_schema = proposition_schema["properties"]["role_assignments"]["items"]
    assert role_schema["properties"]["role"]["enum"] == role_values
    delta_schema = family_schema["properties"]["deltas"]["items"]
    assert delta_schema["required"] == [
        "target_path",
        "delta",
        "evidence_handles",
        "reason",
    ]
    rendered = prompt.build_question_message(
        prompt.ChainQuestion(
            contract_name="semantic_appraisal_group.v1",
            payload=payload,
        )
    )
    assert "permitted_subject_handles" in rendered
    assert "permitted_role_assignment_handles" in rendered
    assert "entity_id" not in rendered


def test_appraisal_payload_carries_evidence_specific_role_authority() -> None:
    """The question exposes role authority per cited evidence handle."""

    payload = prompt.build_appraisal_question_payload(
        stage_name="A2",
        questions=[{
            "question_kind": "relationship_social",
            "evidence_handles": ["e1", "e2"],
            "permitted_role_handles": ["self", "current_user"],
            "permitted_role_assignment_handles": ["self", "current_user"],
            "permitted_delta_paths": [],
            "semantic_question": "判断关系含义。",
        }],
        role_assignment_handles_by_evidence={
            "e1": ["current_user", "self"],
            "e2": [],
        },
        relation_context={
            "character_constraints": {},
            "character_operational_context": {},
            "relationship_projection": {},
            "current_affect": [],
        },
    )
    assert payload["families"][0][
        "permitted_role_assignment_handles_by_evidence"
    ] == {
        "e1": ["current_user", "self"],
        "e2": [],
    }
    proposition_schema = (
        payload["output_contract"]["json_schema"]["family_value_schema"]
        ["properties"]["propositions"]["items"]
    )
    assert proposition_schema["properties"]["object_handle"]["type"] == [
        "string",
        "null",
    ]


def test_appraisal_payload_renders_all_fixed_family_vocabularies() -> None:
    """Every fixed A1/A2 family carries its canonical kinds and meanings."""

    family_rows = []
    for family in (
        "event_agency",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
        "relationship_social",
        "moral_identity",
        "existential_drive",
    ):
        family_rows.append({
            "question_kind": family,
            "evidence_handles": ["e1"],
            "permitted_role_handles": ["ce1"],
            "permitted_role_assignment_handles": ["self"],
            "permitted_delta_paths": ["meaning_state.ce1.value"],
            "semantic_question": f"Question for {family}.",
        })

    for stage_name, families in (
        ("A1", family_rows[:3]),
        ("A2", family_rows[3:]),
    ):
        payload = prompt.build_appraisal_question_payload(
            stage_name=stage_name,
            questions=families,
            relation_context=(
                {
                    "character_constraints": {},
                    "character_operational_context": {},
                    "relationship_projection": {},
                    "current_affect": [],
                }
                if stage_name == "A2"
                else None
            ),
        )
        assert [row["family"] for row in payload["families"]] == [
            row["question_kind"] for row in families
        ]
        for row in payload["families"]:
            family = row["family"]
            assert row["proposition_kinds"] == list(
                question_proposition_kinds(family)
            )
            assert row["proposition_kind_semantics"] == (
                question_proposition_kind_semantics(family)
            )
            assert row["writable_delta_paths"] == [
                "meaning_state.ce1.value"
            ]


def test_appraisal_first_consumer_domains_are_stage_specific() -> None:
    """A1 carries world facts while A2 alone carries relationship context."""

    question = {
        "question_kind": "event_agency",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["ce1"],
        "permitted_role_assignment_handles": ["self"],
        "permitted_delta_paths": [],
        "semantic_question": "Who had agency?",
    }
    a1 = prompt.build_appraisal_question_payload(
        stage_name="A1",
        questions=[question],
    )
    a2 = prompt.build_appraisal_question_payload(
        stage_name="A2",
        questions=[{
            **question,
            "question_kind": "relationship_social",
        }],
        relation_context={
            "character_constraints": {},
            "character_operational_context": {},
            "relationship_projection": {},
            "current_affect": [],
            "character_sleep_phase": "awake",
        },
    )
    assert "relation_context" not in a1
    assert "available_actions" not in a1
    assert "available_resolver_capabilities" not in a1
    assert set(a2["relation_context"]) == {
        "character_constraints",
        "character_operational_context",
        "relationship_projection",
        "current_affect",
        "character_sleep_phase",
    }
    with pytest.raises(prompt.PromptContractError, match="A1 appraisal"):
        prompt.build_appraisal_question_payload(
            stage_name="A1",
            questions=[question],
            relation_context=a2["relation_context"],
        )


def test_goal_payload_distinguishes_current_and_retrieved_evidence() -> None:
    """The goal contract exposes current evidence independently of retrieval."""

    dialogue_bindings = [{
        "speaker_handle": "current_user",
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
    }]
    payload = prompt.build_goal_question_payload(
        goal_kind="ordinary_response",
        goal_projection={},
        evidence_handles={"e1", "e2"},
        action_tendencies=[],
        branch_intent_guidance="",
        role_bindings={},
        selection_operations=[],
        progress_evidence=[],
        authoritative_state={},
        continuity_context={},
        current_episode_evidence_handles={"e1"},
        dialogue_role_bindings=dialogue_bindings,
    )

    contract = payload["goal_output_contract"]
    assert contract["allowed_evidence_handles"] == ["e1", "e2"]
    assert contract["current_episode_evidence_handles"] == ["e1"]
    assert contract["relational_willingness_contract"][
        "minimum_current_episode_evidence_handles"
    ] == 1
    assert contract["required_evidence_handles"] == []
    assert payload["dialogue_role_bindings"] == dialogue_bindings
    assert "second_person_handle" in (
        prompt.ORDINARY_GOAL_BID_QUESTION_GUIDANCE
    )
    assert "second_person_handle" in prompt.APPRAISAL_QUESTION_GUIDANCE

    draft = {
        "intention": "Respond to the current message.",
        "desired_outcome": "The conversation continues.",
        "concrete_detail": "A grounded reply is prepared.",
        "reason": "The current message supports a response.",
        "private_monologue": "Keep the reply grounded.",
        "target_role_handles": [],
        "evidence_handles": ["e1", "e2"],
        "expected_consequences": ["The conversation continues."],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "relationship_sensitive",
            "stance": "conditional_accept",
            "current_user_relationship_state": "developing_or_uncertain",
            "reason": "检索证据不是当前回合的依据。",
            "evidence_handles": ["e2"],
        },
    }
    with pytest.raises(ValueError, match="current episode evidence"):
        goal_cognition.validate_goal_bid_draft(
            draft,
            evidence_handles={"e1", "e2"},
            role_handles=set(),
            require_relational_willingness=True,
            episode_handles={"e1"},
        )


def test_goal_payload_rejects_noncanonical_dialogue_binding_fields() -> None:
    """Goal prompts reject role rows with private or unknown fields."""

    invalid_bindings = [{
        "speaker_handle": "current_user",
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
        "entity_id": "private-id",
    }]
    with pytest.raises(prompt.PromptContractError, match="not exact"):
        prompt.build_goal_question_payload(
            goal_kind="ordinary_response",
            goal_projection={},
            evidence_handles={"e1"},
            action_tendencies=[],
            branch_intent_guidance="",
            role_bindings={},
            selection_operations=[],
            progress_evidence=[],
            authoritative_state={},
            continuity_context={},
            current_episode_evidence_handles={"e1"},
            dialogue_role_bindings=invalid_bindings,
        )


def test_dialogue_binding_requires_second_person_to_match_addressee() -> None:
    """Observation and goal carriers reject contradictory second-person rows."""

    inconsistent_binding = {
        "speaker_handle": "current_user",
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "current_user",
    }
    observation_context = _observation_context("dialogue")
    observation_context["conversation_frame"]["dialogue_role_bindings"] = [
        inconsistent_binding,
    ]
    with pytest.raises(
        prompt.PromptContractError,
        match="second_person_handle must match addressee_handle",
    ):
        prompt.build_first_user_message(
            observation_context=observation_context,
            question=_question(),
        )

    with pytest.raises(
        prompt.PromptContractError,
        match="second_person_handle must match addressee_handle",
    ):
        prompt.build_goal_question_payload(
            goal_kind="ordinary_response",
            goal_projection={},
            evidence_handles={"e1"},
            action_tendencies=[],
            branch_intent_guidance="",
            role_bindings={},
            selection_operations=[],
            progress_evidence=[],
            authoritative_state={},
            continuity_context={},
            current_episode_evidence_handles={"e1"},
            dialogue_role_bindings=[inconsistent_binding],
        )


def test_action_plan_payload_carries_only_required_self_cognition_context() -> None:
    """Targetless group P1 receives an exact response contract carrier."""

    context = {
        "required_fields": [
            "decision",
            "evidence_handles",
            "semantic_target_handle",
            "participation_basis",
            "response_goal",
            "reason",
        ],
        "allowed_decisions": ["stay_silent", "propose_visible_reply"],
        "allowed_evidence_handles": ["e1"],
        "allowed_semantic_target_handles": ["self", "current_group_scene"],
        "allowed_participation_basis_values": [
            "direct_address",
            "explicit_character_reference",
            "grounded_scene_intervention",
        ],
        "response_goal_max_chars": 300,
        "reason_max_chars": 300,
    }
    common = {
        "primary_bid_handle": "b1",
        "supporting_bid_handles": [],
        "bid_index": {"b1": {"branch_id": "ordinary_response"}},
        "action_index": {},
        "resolver_index": {},
        "resolver_context": "",
        "runtime_capability_limits": [],
        "current_goal_progress": None,
        "required_resolver_evidence_dependency": None,
    }
    payload = prompt.build_action_plan_question_payload(
        **common,
        self_cognition_response_context=context,
    )
    assert payload["self_cognition_response_context"] == context
    output_contract = payload["action_plan_output_contract"]
    assert output_contract["required_fields"] == [
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
        "self_cognition_response",
    ]
    assert output_contract["additionalProperties"] is False
    assert output_contract["properties"]["goal_resolution"] == {
        "type": "string",
        "enum": sorted(GOAL_RESOLUTION_VALUES),
    }
    assert output_contract["properties"]["self_cognition_response"] == {
        "type": "object",
        "required_fields": context["required_fields"],
        "additionalProperties": False,
    }

    ordinary_payload = prompt.build_action_plan_question_payload(**common)
    assert "self_cognition_response_context" not in ordinary_payload
    ordinary_contract = ordinary_payload["action_plan_output_contract"]
    assert ordinary_contract["required_fields"] == [
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
    ]
    assert ordinary_contract["properties"] == {
        "action_requests": {
            "type": "array",
            "items": {"type": "object"},
        },
        "resolver_requests": {
            "type": "array",
            "items": {"type": "object"},
        },
        "goal_resolution": {
            "type": "string",
            "enum": sorted(GOAL_RESOLUTION_VALUES),
        },
        "resolver_pending_resolution": {
            "type": ["object", "null"],
        },
        "resolver_goal_progress": {
            "type": ["object", "null"],
        },
    }
    assert "self_cognition_response" not in ordinary_contract["properties"]


def test_action_plan_self_cognition_context_rejects_ambiguous_fields() -> None:
    """The P1 carrier rejects the former target/evidence vocabulary."""

    context = {
        "required": True,
        "target_handles": ["self", "current_group_scene"],
        "current_episode_evidence_handles": ["e1"],
        "allowed_decisions": ["stay_silent", "propose_visible_reply"],
        "participation_basis_values": ["direct_address"],
        "response_goal_max_chars": 300,
        "reason_max_chars": 300,
    }
    with pytest.raises(prompt.PromptContractError):
        prompt.build_action_plan_question_payload(
            primary_bid_handle="b1",
            supporting_bid_handles=[],
            bid_index={"b1": {"branch_id": "ordinary_response"}},
            action_index={},
            resolver_index={},
            resolver_context="",
            runtime_capability_limits=[],
            current_goal_progress=None,
            required_resolver_evidence_dependency=None,
            self_cognition_response_context=context,
        )


def test_action_plan_future_speak_contract_projects_canonical_authority_refs() -> None:
    """Future-speak receives a closed schema and authority-only detail domain."""

    common = {
        "primary_bid_handle": "b1",
        "supporting_bid_handles": [],
        "bid_index": {"b1": {"branch_id": "ordinary_response"}},
        "action_index": {"a1": {"action_kind": "future_speak"}},
        "resolver_index": {},
        "resolver_context": "",
        "runtime_capability_limits": [],
        "current_goal_progress": None,
        "required_resolver_evidence_dependency": None,
        "evidence": [
            {
                "evidence_handle": "e1",
                "authority": "current_event",
                "source_id": "private-source-id",
            },
            {
                "evidence_handle": "e2",
                "authority": "contextual_fact",
                "source_id": "private-source-id-2",
            },
        ],
    }

    payload = prompt.build_action_plan_question_payload(**common)
    contract = payload["scheduled_authority_contract"]
    assert contract["schema_version"] == "scheduled_authority_proposal.v1"
    assert contract["required_for_action_handles"] == ["a1"]
    assert contract["proposal_fields"] == [
        "schema_version",
        "temporal_alignment",
        "authorized_content_summary",
        "authorized_detail_refs",
    ]
    assert contract["detail_ref_fields"] == [
        "evidence_handle",
        "semantic_summary",
        "provenance_role",
    ]
    assert contract["required_temporal_alignment"] == "aligned"
    assert contract["temporal_alignment_rule"]
    assert contract["allowed_detail_refs"] == [{
        "evidence_handle": "e1",
        "provenance_role": "current_event",
    }]
    assert "source_id" not in json.dumps(payload)

    ordinary_payload = prompt.build_action_plan_question_payload(
        **{**common, "action_index": {}, "evidence": []}
    )
    assert "scheduled_authority_contract" not in ordinary_payload


def test_runtime_prompts_exclude_test_fixture_rubric_and_expected_answer_metadata() -> None:
    """Static prompts and structural packets contain no evaluation metadata."""

    manifest_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "cognition_core_v3_live_case_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    static_prompt_text = "\n".join(
        (anchor.ENGINE_MANUAL, *prompt.RUNTIME_PROMPT_TEXTS)
    )
    normalized_static_text = static_prompt_text.casefold()

    for forbidden_phrase in (
        "pytest",
        "fixture",
        "rubric",
        "expected answer",
        "case id",
        "development plan",
        "migration",
    ):
        assert forbidden_phrase not in normalized_static_text

    for case in manifest["cases"]:
        assert case["fixture_id"] not in static_prompt_text
        assert case["pytest_node_id"] not in static_prompt_text
        if case["case_id"] in SEMANTIC_QUESTION_KINDS:
            continue
        case_token = re.compile(
            rf"(?<![a-z0-9_]){re.escape(case['case_id'])}(?![a-z0-9_])",
            re.IGNORECASE,
        )
        assert case_token.search(static_prompt_text) is None

    question = _question()
    metadata_sections = _observation_context("metadata-scene")
    metadata_sections["case_id"] = "hidden-case"
    with pytest.raises(prompt.PromptContractError, match="evaluation metadata"):
        prompt.build_first_user_message(
            observation_context=metadata_sections,
            question=question,
        )

    legitimate_user_text = (
        "The user asks how pytest fixtures and relationship_id fields work."
    )
    legitimate_message = _first_message(
        question,
        scene_marker=legitimate_user_text,
    )
    assert legitimate_user_text in legitimate_message


def test_first_message_allows_goal_fields_only_in_registered_question() -> None:
    """A first goal question retains its fields outside the cold carriers."""

    sections = _observation_context("goal-first-scene")
    question = prompt.ChainQuestion(
        contract_name="ordinary_goal_bid.v1",
        payload={
            "branch_intent_guidance": "advance the current ordinary goal",
            "private_continuity_context": "prompt-safe semantic continuity",
        },
    )
    first_message = prompt.build_first_user_message(
        observation_context=sections,
        question=question,
    )
    first_packet = json.loads(first_message)

    assert first_packet[-1]["question"]["payload"] == dict(
        question.payload
    )

    carrier_sections = _observation_context("goal-carrier-scene")
    carrier_sections["branch_intent_guidance"] = (
        "misplaced goal-only carrier"
    )
    with pytest.raises(prompt.PromptContractError, match="goal-only"):
        prompt.build_first_user_message(
            observation_context=carrier_sections,
            question=question,
        )

    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"run_id": "private-run"},
        )
    with pytest.raises(
        prompt.PromptContractError,
        match="evaluation metadata",
    ):
        prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"case_id": "evaluation-case"},
        )


def _complete_workspace_bid(
    branch_id: str,
    entity_id: str,
    target_entity_id: str,
) -> dict[str, object]:
    """Build one complete bid with private entity references for W1 testing."""

    bid: dict[str, object] = {
        "branch_id": branch_id,
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": entity_id,
        },
        "intention": f"Advance {branch_id}.",
        "desired_outcome": f"The {branch_id} matter advances.",
        "concrete_detail": f"Use the {branch_id} next step.",
        "reason": f"The {branch_id} evidence supports this step.",
        "private_monologue": "I have a grounded reason to proceed.",
        "target_roles": [
            {
                "role": "target",
                "entity_kind": "user",
                "entity_id": target_entity_id,
            }
        ],
        "evidence_handles": ["e1"],
        "expected_consequences": ["The current matter changes."],
        "confidence": "medium",
    }
    if branch_id == "ordinary_response":
        bid["relational_willingness"] = {
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "The current matter permits a response.",
            "evidence_handles": ["e1"],
        }
    return bid


def test_workspace_question_uses_stable_handles_for_complete_bids() -> None:
    """W1 receives partition fields while private bid metadata stays local."""

    bids = [
        _complete_workspace_bid(
            "bond_protection",
            "goal-bond",
            "user-bond",
        ),
        _complete_workspace_bid(
            "ordinary_response",
            "goal-ordinary",
            "user-current",
        ),
        _complete_workspace_bid(
            "loss_recovery",
            "goal-loss",
            "user-loss",
        ),
    ]
    current_event = [
        {
            "handle": "e1",
            "source_kind": "episode",
            "semantic_text": "The current event supports partitioning.",
        }
    ]
    goal_contexts = {
        "goal-bond": {
            "goal_kind": "bond_protection",
            "description": "The bond matter remains active.",
            "status": "pursuing",
        },
        "goal-loss": {
            "goal_kind": "loss_recovery",
            "description": "The loss matter remains active.",
            "status": "pursuing",
        },
    }

    payload = prompt.build_workspace_question_payload(
        bids=bids,
        current_event=current_event,
        goal_contexts=goal_contexts,
    )
    question = prompt.ChainQuestion(
        contract_name="workspace_partition.v1",
        payload=payload,
    )
    rendered = prompt.build_question_message(question)

    assert set(payload) == {"bid_index"}
    assert set(payload["bid_index"]) == {"b1", "b2", "b3"}
    assert payload["bid_index"]["b1"] == {
        "branch_id": "ordinary_response",
        "goal_kind": "ordinary_response",
        "persistent_goal": None,
    }
    assert payload["bid_index"]["b2"] == {
        "branch_id": "bond_protection",
        "goal_kind": "bond_protection",
        "persistent_goal": {
            "goal_kind": "bond_protection",
            "lifecycle": "pursuing",
        },
    }
    assert all(
        "intention" not in row for row in payload["bid_index"].values()
    )
    assert "entity_id" not in rendered

    partition_request = workspace.prepare_partition(
        bids,
        current_event,
        goal_contexts,
    )
    partition = {
        "primary_bid_handle": "b1",
        "supporting_bid_handles": ["b2"],
        "suppressed_bid_handles": ["b3"],
    }
    assert payload == partition_request.prompt_payload
    assert workspace.validate_workspace_partition(
        partition,
        set(partition_request.handles),
    ) == partition
