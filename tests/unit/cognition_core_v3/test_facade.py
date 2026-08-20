"""Deterministic engine facade behavior over the canonical fixture.

Each case runs the full engine with a scripted invoker answering from fixed
per-stage content, so every assertion targets stable runtime outcomes: stage
completion, call order, protected chain records, state carriers, and collapse
behavior. The local fixtures below build their own invoker and services pair;
shared building blocks come from the integration conftest helpers.
"""

from __future__ import annotations

import copy
import json

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _episode_interaction_date_utc,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    FAMILIARITY_DAILY_BONUS_INCREMENT,
    FAMILIARITY_DATE_INCREMENT,
)
from kazusa_ai_chatbot.cognition_core_v3 import (
    run_cognition,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from tests.integration.cognition_core_v3.conftest import (
    ScriptedLLMInvoker,
    default_scripted_responses,
    episode_evidence_handle,
    make_v3_services,
)
from tests.test_cognition_chain_connector_mapping import _global_state

# Registry order of protected semantic owner stage ids for one canonical run.
EXPECTED_PROTECTED_STAGE_SEQUENCE = (
    "causal_normative:event_agency",
    "causal_normative:moral_identity",
    "relationship:relationship_social",
    "epistemic_meaning:epistemic_comparison_memory",
    "epistemic_meaning:existential_drive",
    "ordinary_response:ordinary_response",
    "terminal_outcome:goal_threat_outcome",
)

# Exact ainvoke order for one canonical run: wave chains in registry order,
# then the isolated ordinary goal chain, the terminal outcome chain, and a
# single accepted action-planning call. The scripted invoker introduces no
# internal await points, so these waves complete without interleaving.
EXPECTED_STAGE_CALL_SEQUENCE = (
    "A1",
    "A2",
    "G1a",
    "P1",
)


@pytest.fixture()
def facade_payload():
    """Canonical V2-shaped input built from the connector-mapping state."""
    payload = build_cognition_input_from_global_state(_global_state())
    return payload


def make_facade_bundle(payload, responses=None):
    """Build a scripted invoker and engine services for one content set.

    The optional ``responses`` mapping overrides per-stage scripted content;
    every other stage keeps the canonical default over the episode evidence
    handle. Returns the (invoker, services) pair so a test can inspect call
    order while running the engine.
    """
    handle = episode_evidence_handle(payload)
    invoker = ScriptedLLMInvoker(
        responses=responses,
        defaults=default_scripted_responses(handle),
    )
    services = make_v3_services(invoker)
    return (invoker, services)


@pytest.fixture()
def facade_bundle(facade_payload):
    """Scripted invoker and engine services with canonical content."""
    bundle = make_facade_bundle(facade_payload)
    return bundle


@pytest.mark.asyncio
async def test_run_cognition_completes_all_stages_without_diagnostics_warnings(
    facade_payload, facade_bundle
):
    """Every engine stage completes and the run reports no warnings."""
    _, services = facade_bundle
    output = await run_cognition(facade_payload, services)

    diagnostics = output["diagnostics"]
    assert tuple(diagnostics["stage_status"].values()) == ("completed",) * 7
    assert diagnostics["warnings"] == []
    assert diagnostics["selected_question_count"] == 6
    assert diagnostics["dispatched_question_count"] == 6
    assert diagnostics["selected_branch_count"] == 1
    assert diagnostics["completed_branch_count"] == 1
    assert diagnostics["failed_branch_count"] == 0


@pytest.mark.asyncio
async def test_stage_calls_follow_registry_order_with_single_action_planning_call(
    facade_payload, facade_bundle
):
    """Model calls arrive in registry order with one action-planning call."""
    invoker, services = facade_bundle
    await run_cognition(facade_payload, services)

    assert tuple(invoker.calls) == EXPECTED_STAGE_CALL_SEQUENCE


@pytest.mark.asyncio
async def test_cold_first_packet_reaches_llm_and_enters_accepted_transcript(
    facade_payload,
):
    """The first cold request carries all sections into the accepted prefix."""

    invoker, services = make_facade_bundle(facade_payload)
    captured_calls: list[tuple[str, tuple[object, ...]]] = []
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Retain exact messages before delegating to the scripted invoker."""

        stage_name = config.stage_name.split(".")[0]
        captured_calls.append((stage_name, tuple(messages)))
        response = await scripted_ainvoke(messages, config=config)
        return response

    invoker.ainvoke = recording_ainvoke
    await run_cognition(facade_payload, services)

    first_stage, first_messages = captured_calls[0]
    assert first_stage == "A1"
    first_payload = first_messages[-1].content
    first_packet = json.loads(first_payload)
    assert [next(iter(section)) for section in first_packet] == [
        "constraints_and_operational_state",
        "relationship_and_mutable_state",
        "episode_and_scene",
        "evidence_and_affordances",
        "question",
    ]
    assert first_packet[2]["episode_and_scene"]["episode"][
        "visible_percepts"
    ]
    assert first_packet[3]["evidence_and_affordances"]["evidence"]

    second_stage, second_messages = captured_calls[1]
    assert second_stage == "A2"
    assert second_messages[1].content == first_payload
    assert second_messages[2].content == default_scripted_responses(
        episode_evidence_handle(facade_payload)
    )["A1"]
    later_packet = json.loads(second_messages[-1].content)
    assert [next(iter(section)) for section in later_packet] == ["question"]


@pytest.mark.asyncio
async def test_next_cold_question_keeps_first_packet_after_appraisal_exhaustion(
    facade_payload,
):
    """The first accepted owner gets all cold carriers after early failures."""

    invoker, services = make_facade_bundle(
        facade_payload,
        responses={"A1": "{invalid", "A2": "{invalid"},
    )
    captured_calls: list[tuple[str, tuple[object, ...]]] = []
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Retain exact messages before delegating to the scripted invoker."""

        stage_name = config.stage_name.split(".")[0]
        captured_calls.append((stage_name, tuple(messages)))
        response = await scripted_ainvoke(messages, config=config)
        return response

    invoker.ainvoke = recording_ainvoke
    await run_cognition(facade_payload, services)

    g1a_messages = next(
        messages
        for stage_name, messages in captured_calls
        if stage_name == "G1a"
    )
    assert len(g1a_messages) == 2
    g1a_packet = json.loads(g1a_messages[-1].content)
    assert [next(iter(section)) for section in g1a_packet] == [
        "constraints_and_operational_state",
        "relationship_and_mutable_state",
        "episode_and_scene",
        "evidence_and_affordances",
        "question",
    ]
    assert g1a_packet[-1]["question"]["contract_name"] == (
        "ordinary_goal_bid.v1"
    )


@pytest.mark.asyncio
async def test_state_update_preserves_v2_carriers_and_applies_relationship_maintenance_once(
    facade_payload, facade_bundle
):
    """Maintenance applies once and the unique care receipt applies exactly once.

    The canonical relationship content carries a single integer care delta,
    so its bounded change lands on the carrier a single time; the unique
    relationship receipt also grants the daily familiarity bonus alongside
    the new-interaction date increment. Other carriers stay intact.
    """
    _, services = facade_bundle
    output = await run_cognition(facade_payload, services)

    state_update = output["state_update"]
    previous = state_update["expected_previous_state"]
    replacement = state_update["replacement_state"]

    interaction_date_utc = _episode_interaction_date_utc(
        facade_payload["episode"]
    )
    source_id = f"episode:{facade_payload['episode']['episode_id']}"

    previous_relationship = previous["relationship"]
    replacement_relationship = replacement["relationship"]
    care_delta_value = 2
    familiarity_increase = (
        FAMILIARITY_DATE_INCREMENT + FAMILIARITY_DAILY_BONUS_INCREMENT
    )

    assert (
        replacement_relationship["care"]
        == previous_relationship["care"] + care_delta_value
    )
    assert (
        replacement_relationship["familiarity"]
        == previous_relationship["familiarity"] + familiarity_increase
    )
    assert (
        replacement_relationship["salience"]
        == previous_relationship["salience"] + care_delta_value
    )
    assert replacement_relationship["trust"] == previous_relationship["trust"]
    assert (
        replacement_relationship["attachment"]
        == previous_relationship["attachment"]
    )

    previous_maintenance = previous_relationship["relationship_maintenance"]
    replacement_maintenance = (
        replacement_relationship["relationship_maintenance"]
    )

    assert previous_maintenance["last_interaction_date_utc"] is None
    assert (
        replacement_maintenance["last_interaction_date_utc"]
        == interaction_date_utc
    )
    assert previous_maintenance["last_source_id"] is None
    assert replacement_maintenance["last_source_id"] == source_id
    assert replacement_maintenance["processed_source_ids"] == [source_id]

    assert previous_maintenance["last_bonus_date_utc"] is None
    assert (
        replacement_maintenance["last_bonus_date_utc"] == interaction_date_utc
    )

    assert previous["goals"] == replacement["goals"]
    assert previous["affect_activations"] == replacement["affect_activations"]
    assert len(state_update["changed_paths"]) >= 1


@pytest.mark.asyncio
async def test_admitted_bid_carries_ordinary_branch_and_scripted_relational_decision(
    facade_payload, facade_bundle
):
    """The admitted bid keeps the scripted relational decision verbatim."""
    _, services = facade_bundle
    handle = episode_evidence_handle(facade_payload)
    output = await run_cognition(facade_payload, services)

    admitted_bid = output["admitted_bid"]
    assert admitted_bid["branch_id"] == "ordinary_response"
    assert admitted_bid["target_roles"] == []
    assert admitted_bid["evidence_handles"] == [handle]

    willingness = admitted_bid["relational_willingness"]
    assert willingness["applicability"] == "not_relationship_sensitive"
    assert willingness["stance"] == "not_applicable"
    assert output["relational_willingness"] == willingness

    assert output["intention"]["route"] == "speech"
    assert output["goal_resolution"] == "blocked"
    assert output["action_requests"] == []
    assert output["resolver_requests"] == []


@pytest.mark.asyncio
async def test_authoritative_relational_decision_collapses_without_partition_call(
    facade_payload,
):
    """A sensitive relational decision collapses authoritatively.

    The ordinary draft answers with a relationship-sensitive decision; the
    engine then collapses from that authoritative decision without a
    workspace-partition model call and records exactly one diagnostics warning
    for the override.
    """
    handle = episode_evidence_handle(facade_payload)

    sensitive_draft = {
        "intention": "Reply to the user's greeting in character.",
        "desired_outcome": "The user receives an in-character reply.",
        "concrete_detail": "Greet the user and open a topic of interest.",
        "reason": "The user opened the conversation with a simple greeting.",
        "private_monologue": "A quiet hello is an invitation, not a demand.",
        "target_role_handles": [],
        "evidence_handles": [handle],
        "expected_consequences": ["The conversation continues from the greeting."],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "relationship_sensitive",
            "stance": "accept",
            "current_user_relationship_state": "developing_or_uncertain",
            "reason": '愿意回应用户的问候并延续对话',
            "evidence_handles": [handle],
        },
    }

    override_bundle = make_facade_bundle(
        facade_payload,
        responses={"G1a": json.dumps(sensitive_draft)},
    )
    override_invoker, override_services = override_bundle
    output = await run_cognition(facade_payload, override_services)

    warnings = output["diagnostics"]["warnings"]
    assert warnings.count("authoritative_relational_willingness") == 1

    willingness = output["relational_willingness"]
    assert willingness["applicability"] == "relationship_sensitive"
    assert willingness["stance"] == "accept"
    assert (
        output["admitted_bid"]["branch_id"] == "ordinary_response"
    )
    assert "workspace_collapse" not in override_invoker.calls


# Every evidence source kind plans the outcome question, so an evidence-free
# payload is the only full-run input where no semantic questions are planned.
ZERO_EVIDENCE_ORDINARY_DRAFT = {
    "intention": "Reply to the user in character.",
    "desired_outcome": "The user receives an in-character reply.",
    "concrete_detail": "Greet the user and open a topic of interest.",
    "reason": "The conversation needs a grounded opening response.",
    "private_monologue": "A quiet hello is an invitation, not a demand.",
    "target_role_handles": [],
    "evidence_handles": [],
    "expected_consequences": ["The conversation continues from the greeting."],
    "confidence": "medium",
    "relational_willingness": {
        "applicability": "not_relationship_sensitive",
        "stance": "not_applicable",
        "current_user_relationship_state": "not_applicable",
        "reason": '关系状态稳定',
        "evidence_handles": [],
    },
}


@pytest.mark.asyncio
async def test_evidence_free_run_fails_closed_after_bounded_ordinary_attempts(
    facade_payload,
):
    """A required ordinary branch that exhausts its attempts escalates.

    The evidence-free payload plans no semantic questions, so appraisal stages
    skip their model calls; the ordinary goal chain runs and rejects every
    attempt because no authorized evidence handle exists. With no complete
    sibling bid the run raises instead of committing state.
    """
    variant = copy.deepcopy(facade_payload)
    variant["evidence"] = []

    invoker = ScriptedLLMInvoker(
        responses={
            "G1a": json.dumps(
                ZERO_EVIDENCE_ORDINARY_DRAFT
            ),
            "P1": json.dumps({"goal_resolution": "blocked"}),
        }
    )
    services = make_v3_services(invoker)

    with pytest.raises(CognitionExecutionError, match="ordinary_response"):
        await run_cognition(variant, services)

    assert tuple(invoker.calls) == ("G1a",) * 3


@pytest.mark.asyncio
async def test_rejected_goal_draft_attempts_are_captured_before_branch_escalation(
    facade_payload,
):
    """Bounded rejected attempts stay in the trace before escalation.

    A structurally invalid ordinary draft is rejected on every attempt; the
    protected chain records keep one rejected entry per attempt while the
    required-branch failure still escalates out of the run. Downstream stages
    never start after the bounded budget runs out.
    """
    invoker, services = make_facade_bundle(
        facade_payload,
        responses={"G1a": "{invalid json"},
    )
    with pytest.raises(CognitionExecutionError):
        await run_cognition(facade_payload, services)

    expected_calls = (
        "A1",
        "A2",
    ) + ("G1a",) * 3
    assert tuple(invoker.calls) == expected_calls

