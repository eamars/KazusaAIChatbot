"""Deterministic proof for persistent autonomy-goal branch creation."""

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    create_deterministic_goals,
)

NOW = "2026-07-14T00:00:00Z"


def test_boundary_pressure_creates_autonomy_goal_as_branch_two() -> None:
    """A grounded boundary condition creates the second selected branch."""

    state = build_acquaintance_user_state(
        global_user_id="goal-branch-two-user",
        updated_at=NOW,
    )
    condition_evidence = {
        "source_kind": "episode",
        "source_id": "episode:synthetic-boundary-pressure",
        "occurred_at": NOW,
        "semantic_summary": (
            "Synthetic pressure makes the current relationship boundary unsafe."
        ),
    }
    state["relationship"].update({
        "boundary_safety": -80,
        "salience": 80,
        "evidence_refs": [condition_evidence],
    })

    updated_state = create_deterministic_goals(
        state,
        character_constraints=build_character_production_state(
            updated_at=NOW,
        ),
        updated_at=NOW,
    )
    validate_cognition_state(updated_state)

    autonomy_goals = [
        goal
        for goal in updated_state["goals"]
        if goal["goal_kind"] == "autonomy_boundary"
    ]
    assert len(autonomy_goals) == 1
    assert autonomy_goals[0]["status"] == "pursuing"
    assert autonomy_goals[0]["evidence_refs"] == [condition_evidence]

    branches = select_preliminary_branches(updated_state["goals"])

    assert [branch.branch_id for branch in branches] == [
        "ordinary_response",
        "autonomy_boundary",
    ]
    assert branches[1].goal_kind == "autonomy_boundary"
    assert branches[1].branch_intent_guidance
