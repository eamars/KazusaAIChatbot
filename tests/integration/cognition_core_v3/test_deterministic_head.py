"""Deterministic engine head and terminal-stage behavior without model calls.

These cases replicate the engine's deterministic head over the canonical
connector-mapping fixture and inspect question selection, branch selection,
and the terminal-outcome stage skip directly, so no scripted or live model is
involved for those cases; one case runs the full engine twice with scripted
content to pin cross-run stage-order stability.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    SEMANTIC_QUESTION_KINDS,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _cognition_elapsed_seconds,
    _episode_updated_at,
    _fact_without_producer,
    _native_relationship_context,
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.cognition_core_v3 import run_cognition
from tests.integration.cognition_core_v3.conftest import (
    ScriptedLLMInvoker,
    default_scripted_responses,
    episode_evidence_handle,
    make_v3_services,
    ordinary_goal_draft,
)

EXPECTED_STAGE_CALL_SEQUENCE = (
    "A1",
    "A2",
    "G1a",
    "P1",
)


def _deterministic_head(payload):
    """Run the engine's deterministic head over one validated payload.

    The sequence mirrors the facade exactly: input validation, state
    reduction with direct facts and elapsed time, deterministic goal creation,
    revalidation, prompt projection, and semantic question selection. No model
    call is part of this head.
    """
    validated = validate_cognition_core_input(payload)
    previous_state = validate_cognition_state(validated["mutable_state"])
    updated_at = _episode_updated_at(validated["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(
        previous_state, updated_at
    )

    fact_pairs = [
        (fact["producer"], _fact_without_producer(fact))
        for fact in validated["direct_facts"]
    ]
    reducer_relationship_context = _native_relationship_context(
        validated.get("relationship_context")
    )
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=validated["character_constraints"],
        relationship_context=reducer_relationship_context,
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=validated["character_constraints"],
        relationship_context=reducer_relationship_context,
        evidence=validated["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)

    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=validated["character_constraints"],
        character_identity_context=validated["character_identity_context"],
        relationship_context=validated.get("relationship_context"),
        character_operational_context=validated.get(
            "character_operational_context"
        ),
        scene_context=validated["scene_context"],
        evidence=validated["evidence"],
    )
    questions = plan_semantic_questions(
        validated["evidence"], preliminary_state, projection.handle_to_ref
    )
    return preliminary_state, questions


def _active_joy_event(evidence_ref):
    """Build one complete retained event that deterministically derives joy."""

    return {
        "entity_id": "event:joy",
        "description": "A meaningful success remains active.",
        "salience": 80,
        "role_refs": [],
        "evidence_refs": [dict(evidence_ref)],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "status": "active",
        "outcome_impact": 80,
        "responsibility": 70,
        "intentionality": 0,
        "harm": 0,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 100,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
    }


def test_deterministic_head_plans_all_six_questions_for_episode_evidence(
    cognition_payload,
):
    """Semantic question selection covers every kind over episode evidence."""
    _preliminary_state, questions = _deterministic_head(cognition_payload)
    episode_handle = next(
        row["evidence_handle"]
        for row in cognition_payload["evidence"]
        if row["evidence_ref"]["source_kind"] == "episode"
    )

    question_kinds = [question["question_kind"] for question in questions]
    assert set(question_kinds) == set(SEMANTIC_QUESTION_KINDS)
    assert len(questions) == 6

    for question in questions:
        handles = question["evidence_handles"]
        assert len(handles) >= 1
        assert episode_handle in handles


def test_branch_selection_without_persistent_goals_selects_ordinary_response_only(
    cognition_payload,
):
    """Without active persistent goals only the ordinary branch is selected."""
    preliminary_state, questions = _deterministic_head(cognition_payload)
    question_ids = [question["question_id"] for question in questions]

    assert preliminary_state["goals"] == []
    preliminary = select_preliminary_branches(preliminary_state["goals"])
    final = select_final_branches(
        preliminary, preliminary_state["goals"], question_ids
    )

    assert [definition.branch_id for definition in preliminary] == (
        ["ordinary_response"]
    )
    assert [definition.branch_id for definition in final] == (
        ["ordinary_response"]
    )


@pytest.mark.asyncio
async def test_stage_call_sequence_is_stable_across_two_runs(
    cognition_payload,
):
    """Two independent scripted runs produce the identical stage sequence."""

    handle = episode_evidence_handle(cognition_payload)
    first_invoker = ScriptedLLMInvoker(
        defaults=default_scripted_responses(handle)
    )
    second_invoker = ScriptedLLMInvoker(
        defaults=default_scripted_responses(handle)
    )

    await run_cognition(cognition_payload, make_v3_services(first_invoker))
    await run_cognition(cognition_payload, make_v3_services(second_invoker))

    assert tuple(first_invoker.calls) == EXPECTED_STAGE_CALL_SEQUENCE
    assert tuple(second_invoker.calls) == EXPECTED_STAGE_CALL_SEQUENCE


@pytest.mark.asyncio
async def test_cold_serial_chain_preserves_complete_v2_state_emotion_relationship_goal_and_action_output(
    cognition_payload,
):
    """One cold chain emits the complete V2 state-to-action surface."""

    payload = deepcopy(cognition_payload)
    evidence_ref = payload["evidence"][0]["evidence_ref"]
    payload["mutable_state"]["active_events"] = [
        _active_joy_event(evidence_ref)
    ]
    episode_handle = episode_evidence_handle(payload)
    defaults = default_scripted_responses(episode_handle)
    invoker = ScriptedLLMInvoker(defaults=defaults)
    output = await run_cognition(
        payload,
        replace(make_v3_services(invoker)),
    )
    validated = validate_cognition_core_output(output)
    replacement_state = validated["state_update"]["replacement_state"]

    assert tuple(invoker.calls[-2:]) == ("G1a", "P1")
    assert set(invoker.calls[:-2]) <= {"A1", "A2"}
    assert replacement_state["state_scope"] == "user"
    assert replacement_state["active_events"][0]["entity_id"] == "event:joy"
    assert replacement_state["affect_activations"][0]["emotion_id"] == "joy"
    assert validated["affect_projection"][0]["emotion"] == "joy"
    assert validated["relationship_projection"]["axis_summaries"]["care"]
    assert validated["admitted_bid"]["branch_id"] == "ordinary_response"
    assert validated["intention"]["selected_branch_id"] == "ordinary_response"
    assert validated["action_requests"] == []
    assert validated["resolver_requests"] == []
    assert validated["goal_resolution"] == "blocked"


@pytest.mark.asyncio
async def test_sensitive_ordinary_primary_collapse_records_ordered_g1b(
    cognition_payload,
):
    """A sensitive ordinary bid retains the ordered active-bid observation."""

    payload = deepcopy(cognition_payload)
    evidence_handle = episode_evidence_handle(payload)
    evidence_ref = payload["evidence"][0]["evidence_ref"]
    payload["mutable_state"]["goals"] = [{
        "entity_id": "goal:bond-protection",
        "description": "Protect a current boundary.",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [dict(evidence_ref)],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "status": "pursuing",
        "goal_kind": "bond_protection",
        "importance": 80,
        "progress": 10,
        "obstruction": 0,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
        "urgency": 60,
    }]
    ordinary_draft = json.loads(ordinary_goal_draft(evidence_handle))
    ordinary_draft["relational_willingness"] = {
        "applicability": "relationship_sensitive",
        "stance": "negotiate",
        "current_user_relationship_state": "developing_or_uncertain",
        "reason": "当前关系需要协商边界。",
        "evidence_handles": [evidence_handle],
    }
    active_group = {
        "bids": [{
            "branch_id": "bond_protection",
            "intention": "Protect the active boundary.",
            "desired_outcome": "The boundary remains clear.",
            "concrete_detail": "State the current boundary.",
            "reason": "The persistent boundary goal remains active.",
            "private_monologue": "Keep the boundary grounded.",
            "target_role_handles": [],
            "evidence_handles": [evidence_handle],
            "expected_consequences": ["The boundary remains visible."],
            "confidence": "medium",
        }],
    }
    responses = default_scripted_responses(evidence_handle)
    responses["G1a"] = json.dumps(ordinary_draft, ensure_ascii=False)
    responses["G1b"] = json.dumps(active_group)
    invoker = ScriptedLLMInvoker(defaults=responses)

    output = await run_cognition(payload, make_v3_services(invoker))

    assert tuple(invoker.calls) == ("A1", "A2", "G1a", "G1b", "P1")
    assert output["admitted_bid"]["branch_id"] == "ordinary_response"
    assert output["relational_willingness"]["stance"] == "negotiate"
    assert output["diagnostics"]["warnings"].count(
        "authoritative_relational_willingness"
    ) == 1
    branches = output["cognition_observability"]["branches"]
    assert [branch["goal_kind"] for branch in branches] == [
        "ordinary_response",
        "bond_protection",
    ]
    assert [branch["status"] for branch in branches] == [
        "completed",
        "completed",
    ]
    assert [branch["selection"] for branch in branches] == [
        "primary",
        "suppressed",
    ]

