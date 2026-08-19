"""Deterministic engine head and terminal-stage behavior without model calls.

These cases replicate the engine's deterministic head over the canonical
connector-mapping fixture and inspect question selection, branch selection,
and the terminal-outcome stage skip directly, so no scripted or live model is
involved for those cases; one case runs the full engine twice with scripted
content to pin cross-run stage-order stability.
"""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    SEMANTIC_QUESTION_KINDS,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _cognition_elapsed_seconds,
    _episode_updated_at,
    _fact_without_producer,
    _native_relationship_context,
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    _make_terminal_stage_producer,
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
)

# Registry-ordered ainvoke sequence for one canonical scripted run: the first
# wave chains in registry order, then the isolated ordinary goal chain and the
# terminal outcome chain, ending with the single accepted action plan.
EXPECTED_STAGE_CALL_SEQUENCE = (
    "event_agency",
    "moral_identity",
    "relationship_social",
    "epistemic_comparison_memory",
    "existential_drive",
    "goal_ordinary_response",
    "goal_threat_outcome",
    "action_planning",
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
async def test_terminal_outcome_stage_skips_model_call_without_planned_question(
    cognition_payload, v3_services
):
    """The terminal stage keeps a contentless accepted state with no call."""
    preliminary_state, _ = _deterministic_head(cognition_payload)
    producers = _make_terminal_stage_producer(
        v3_services, preliminary_state, (), {}
    )

    outcome = await producers["goal_threat_outcome"](None)

    assert outcome.accepted is True
    assert outcome.semantic_summary is None
    assert outcome.local_state == {
        "selected_evidence_handles": [],
        "propositions": [],
        "deltas": [],
    }
    assert v3_services.llm.calls == []


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

