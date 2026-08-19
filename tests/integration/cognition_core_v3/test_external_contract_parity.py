"""V3 engine output parity against the V2 external contracts.

Each test runs the engine over the canonical connector-mapping fixture with
the scripted invoker and then validates one externally visible surface of the
result through the unchanged V2 contract entry points.
"""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    validate_action_bid,
    validate_cognition_core_output,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v3 import run_cognition

REQUIRED_OUTPUT_KEYS = frozenset(
    {
        "schema_version",
        "intention",
        "goal_continuation_ref",
        "supporting_bids",
        "state_update",
        "affect_projection",
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
        "resolver_progress",
        "selected_bid_reason",
        "private_monologue",
        "expression_policy",
        "diagnostics",
    }
)

STATE_UPDATE_KEYS = frozenset(
    {
        "state_scope",
        "owner_key",
        "expected_previous_state",
        "replacement_state",
        "comparison_results",
        "changed_paths",
    }
)

DIAGNOSTICS_KEYS = frozenset(
    {
        "run_id",
        "stage_status",
        "selected_question_count",
        "dispatched_question_count",
        "selected_branch_count",
        "dispatched_branch_count",
        "completed_branch_count",
        "failed_branch_count",
        "overlap_ms",
        "dependency_wait_ms",
        "total_ms",
        "warnings",
    }
)

ENGINE_STAGE_NAMES = (
    "input_validation",
    "deterministic_preliminary",
    "semantic_appraisal",
    "final_reduction",
    "branch_cognition",
    "workspace_collapse",
    "action_planning",
)


@pytest.mark.asyncio
async def test_run_cognition_output_satisfies_v2_core_validator(
    cognition_payload, v3_services
):
    """The assembled output passes the V2 core output validator."""
    output = await run_cognition(cognition_payload, v3_services)
    validated = validate_cognition_core_output(output)

    assert validated["schema_version"] == "cognition_core_output.v2"
    assert REQUIRED_OUTPUT_KEYS.issubset(validated.keys())


@pytest.mark.asyncio
async def test_admitted_bid_passes_v2_action_bid_contract(
    cognition_payload, v3_services
):
    """The admitted bid and its relational decision pass V2 validators."""
    output = await run_cognition(cognition_payload, v3_services)
    episode_handle = next(
        row["evidence_handle"]
        for row in cognition_payload["evidence"]
        if row["evidence_ref"]["source_kind"] == "episode"
    )

    admitted_bid = output["admitted_bid"]
    validate_action_bid(admitted_bid)

    assert admitted_bid["branch_id"] == "ordinary_response"
    assert admitted_bid["target_roles"] == []
    assert admitted_bid["evidence_handles"] == [episode_handle]
    validate_relational_willingness(
        admitted_bid["relational_willingness"],
        evidence_handles={episode_handle},
    )


@pytest.mark.asyncio
async def test_state_update_and_diagnostics_carry_v2_shapes(
    cognition_payload, v3_services
):
    """State update and diagnostics expose the exact V2 field sets."""
    output = await run_cognition(cognition_payload, v3_services)

    state_update = output["state_update"]
    assert set(state_update.keys()) == STATE_UPDATE_KEYS
    assert state_update["state_scope"] == "user"

    diagnostics = output["diagnostics"]
    assert set(diagnostics.keys()) == DIAGNOSTICS_KEYS
    stage_status = diagnostics["stage_status"]
    assert tuple(stage_status[name] for name in ENGINE_STAGE_NAMES) == (
        "completed",
    ) * len(ENGINE_STAGE_NAMES)
