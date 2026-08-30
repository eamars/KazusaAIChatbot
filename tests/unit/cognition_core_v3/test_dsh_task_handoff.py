"""Executable tests for cognition-to-DSH task handoff."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v3.appraisal import bind_axis_changes
from kazusa_ai_chatbot.cognition_core_v3.contracts import CanonicalAppraisal
from tests.unit.cognition_core_v3.test_handleless_contract import _input


def test_task_control_preserves_continuation_goal_without_coding_special_case() -> None:
    """A generic DSH request creates one guarded continuation goal."""

    payload = _input()
    metadata: dict[str, object] = {}
    updated, transitions, receipts, provenance = bind_axis_changes(
        payload,
        (
            CanonicalAppraisal(
                family="goal_threat_outcome",
                applicable=True,
                semantic_summary="The task checkpoint changes the scene.",
                cause_summary="The DSH task returned a typed checkpoint.",
                axis_changes=(
                    {
                        "axis": "outcome_impact",
                        "shift": "strong_increase",
                        "reason": "The checkpoint creates a grounded event.",
                    },
                ),
            ),
        ),
        goal={
            "intent": "continue the accepted task with new evidence",
            "cause_summary": "the task returned a bounded checkpoint",
        },
        goal_resolution="requires_user_input",
        resolver_requests=[{
            "capability": "task_resolution_request",
            "semantic_goal": "continue the accepted task with new evidence",
            "reason": "the prior DSH result was deferred",
            "goal_continuation_ref": None,
        }],
        action_requests=[{
            "action_kind": "accepted_task_control",
            "decision": "continue",
            "context_ref": "accepted_task:task-1",
        }],
        binding_metadata=metadata,
    )

    current_goals = [
        row for row in updated["goals"]
        if row["goal_kind"] == "ordinary_response"
    ]
    assert len(current_goals) == 1
    continuation = metadata["continuation_goal_ref"]
    assert continuation == {
        "scope": "user",
        "kind": "goal",
        "entity_id": current_goals[0]["entity_id"],
    }
    assert transitions
    assert receipts
    assert provenance
    assert all("coding" not in repr(item).lower() for item in transitions)
