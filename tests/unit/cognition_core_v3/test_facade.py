"""Deterministic engine facade behavior over the canonical fixture.

Each case runs the full engine with a scripted invoker answering from fixed
per-stage content, so every assertion targets stable runtime outcomes: stage
completion, call order, protected chain records, state carriers, and collapse
behavior. The local fixtures below build their own invoker and services pair;
shared building blocks come from the integration conftest helpers.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3 import run_cognition
from kazusa_ai_chatbot.cognition_core_v3.budget import (
    CognitionContextLimitError,
    ContextBudgetLedger,
    ContextBudgetPlan,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    SerialChainHarness,
)
from kazusa_ai_chatbot.cognition_core_v3.facade_helpers import (
    _episode_interaction_date_utc,
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import ChainQuestion
from kazusa_ai_chatbot.cognition_core_v3.transcript import ChainTranscriptV1
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    FAMILIARITY_DAILY_BONUS_INCREMENT,
    FAMILIARITY_DATE_INCREMENT,
    apply_state_update,
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


@pytest.fixture(autouse=True)
def _disable_chain_observability_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scripted facade tests deterministic and outside persistence."""

    monkeypatch.setattr(
        facade_module.llm_tracing,
        "record_cognition_chain_transcript",
        AsyncMock(return_value={"status": "skipped"}),
    )
    monkeypatch.setattr(
        facade_module.event_logging,
        "record_cognition_chain_event",
        AsyncMock(return_value={"status": "skipped"}),
    )
    monkeypatch.setattr(
        facade_module.db,
        "save_cognition_chain_run",
        AsyncMock(return_value=True),
    )

_WORLD_APPRAISAL_FAMILIES = (
    "event_agency",
    "goal_threat_outcome",
    "epistemic_comparison_memory",
)
_RELATION_APPRAISAL_FAMILIES = (
    "relationship_social",
    "moral_identity",
    "existential_drive",
)


def _empty_group_response(families):
    """Build the smallest valid fixed-stage appraisal response."""

    return json.dumps({
        family: {"propositions": [], "deltas": []}
        for family in families
    })


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


def _payload_with_active_bond_goal(payload):
    """Add one persistent active sibling goal to the canonical fixture."""

    variant = copy.deepcopy(payload)
    evidence_ref = variant["evidence"][0]["evidence_ref"]
    variant["mutable_state"]["goals"] = [{
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
        "urgency": 60,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
    }]
    return variant


def _targetless_group_self_cognition_payload(payload):
    """Convert the canonical episode into a targetless group turn."""

    variant = copy.deepcopy(payload)
    episode = variant["episode"]
    episode["trigger_source"] = "self_cognition"
    target_scope = episode["target_scope"]
    target_scope["channel_type"] = "group"
    target_scope["current_global_user_id"] = None
    target_scope["current_platform_user_id"] = None
    variant["scene_context"]["channel_scope"] = "group"
    return variant


def _active_bond_goal_draft(evidence_handle):
    """Build one complete active sibling draft for G1b recovery tests."""

    return {
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
    }


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
async def test_cold_p1_calls_canonical_v2_action_plan_validator(
    facade_payload,
    monkeypatch,
):
    """The active P1 seam delegates exact validation to the V2 owner."""

    validator_calls: list[dict[str, object]] = []
    validator_kwargs: list[dict[str, object]] = []
    original_validator = facade_module.validate_action_plan_decision

    def recording_validator(parsed, **kwargs):
        validator_calls.append(dict(parsed))
        validator_kwargs.append(dict(kwargs))
        return original_validator(parsed, **kwargs)

    monkeypatch.setattr(
        facade_module,
        "validate_action_plan_decision",
        recording_validator,
    )
    _, services = make_facade_bundle(facade_payload)

    await run_cognition(facade_payload, services)

    assert len(validator_calls) == 1
    assert set(validator_calls[0]) == {
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
    }
    assert validator_kwargs[0]["accepted_at_utc"] == (
        "2026-07-14T00:00:00+00:00"
    )


@pytest.mark.asyncio
async def test_targetless_self_cognition_p1_receives_required_context(
    facade_payload,
):
    """Targetless group P1 exposes the exact required response domain."""

    payload = _targetless_group_self_cognition_payload(facade_payload)
    handle = episode_evidence_handle(payload)
    p1_response = json.dumps({
        "action_requests": [],
        "resolver_requests": [],
        "goal_resolution": "blocked",
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "self_cognition_response": {
            "decision": "stay_silent",
            "evidence_handles": [],
            "semantic_target_handle": "current_group_scene",
            "participation_basis": "",
            "response_goal": "",
            "reason": "当前场景没有需要公开回应的依据。",
        },
    }, ensure_ascii=False)
    invoker, services = make_facade_bundle(
        payload,
        responses={"P1": p1_response},
    )
    captured: list[dict[str, object]] = []
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Capture the action-plan packet before scripted validation."""

        if config.stage_name == "P1":
            packet = json.loads(messages[-1].content)
            captured.append(packet[-1]["question"]["payload"])
        return await scripted_ainvoke(messages, config=config)

    invoker.ainvoke = recording_ainvoke
    output = await run_cognition(payload, services)

    assert captured
    context = captured[0]["self_cognition_response_context"]
    assert context == {
        "required_fields": [
            "decision",
            "evidence_handles",
            "semantic_target_handle",
            "participation_basis",
            "response_goal",
            "reason",
        ],
        "allowed_decisions": ["stay_silent", "propose_visible_reply"],
        "allowed_evidence_handles": [handle],
        "allowed_semantic_target_handles": ["self", "current_group_scene"],
        "allowed_participation_basis_values": [
            "direct_address",
            "explicit_character_reference",
            "grounded_scene_intervention",
        ],
        "response_goal_max_chars": 300,
        "reason_max_chars": 300,
    }
    assert output["self_cognition_response"]["decision"] == "stay_silent"


@pytest.mark.asyncio
async def test_targetless_self_cognition_p1_exhaustion_is_typed_failure(
    facade_payload,
):
    """Required self-cognition exhaustion remains an action-stage failure."""

    payload = _targetless_group_self_cognition_payload(facade_payload)
    _invoker, services = make_facade_bundle(
        payload,
        responses={"P1": "{invalid", "P1.repair1": "{invalid"},
    )

    with pytest.raises(CognitionExecutionError) as error_info:
        await run_cognition(payload, services)

    assert error_info.value.stage == "action_planning"
    assert error_info.value.error_code in {
        "self_cognition_response_unavailable",
        "serial_attempts_exhausted",
    }


@pytest.mark.asyncio
async def test_primary_stage_configs_use_code_owned_completion_caps(
    facade_payload,
):
    """Cold primary stages retain route identity while applying bounded caps."""

    invoker, services = make_facade_bundle(facade_payload)
    captured_caps: list[tuple[str, int | None]] = []
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Capture stage caps before returning the scripted response."""

        captured_caps.append((config.stage_name, config.max_completion_tokens))
        return await scripted_ainvoke(messages, config=config)

    invoker.ainvoke = recording_ainvoke
    await run_cognition(facade_payload, services)

    assert captured_caps == [
        ("A1", 4_096),
        ("A2", 4_096),
        ("G1a", 8_192),
        ("P1", 8_192),
    ]


@pytest.mark.asyncio
async def test_cold_first_packet_reaches_llm_and_enters_accepted_transcript(
    facade_payload,
    monkeypatch,
):
    """The first cold request carries all sections into the accepted prefix."""

    invoker, services = make_facade_bundle(facade_payload)
    l1_residue = {
        "schema_version": "l1_residue.v1",
        "emotional_appraisal": "bounded pressure",
        "interaction_subtext": "the user awaits a reply",
        "salience_hints": [episode_evidence_handle(facade_payload)],
        "risk_flags": [],
    }
    l1_take_count = 0

    def ready_l1(_task, _warnings):
        nonlocal l1_take_count
        l1_take_count += 1
        return l1_residue, True

    monkeypatch.setattr(
        facade_module,
        "_take_ready_l1_residue",
        ready_l1,
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

    first_stage, first_messages = captured_calls[0]
    assert first_stage == "A1"
    first_payload = first_messages[-1].content
    first_packet = json.loads(first_payload)
    assert [next(iter(section)) for section in first_packet] == [
        "observation_context",
        "question",
    ]
    observation_context = first_packet[0]["observation_context"]
    assert observation_context["evidence"]
    assert "visible_percepts" not in first_payload
    assert "semantic_scene" not in first_payload

    second_stage, second_messages = captured_calls[1]
    assert second_stage == "A2"
    assert second_messages[1].content == first_payload
    assert second_messages[2].content == default_scripted_responses(
        episode_evidence_handle(facade_payload)
    )["A1"]
    later_packet = json.loads(second_messages[-1].content)
    assert [next(iter(section)) for section in later_packet] == ["question"]
    assert "l1_residue" not in later_packet[-1]["question"]["payload"]
    assert l1_take_count == 1


@pytest.mark.asyncio
async def test_next_cold_question_keeps_first_packet_after_appraisal_exhaustion(
    facade_payload,
    monkeypatch,
):
    """The first accepted owner gets all cold carriers after early failures."""

    invoker, services = make_facade_bundle(
        facade_payload,
        responses={"A1": "{invalid", "A2": "{invalid"},
    )
    l1_residue = {
        "schema_version": "l1_residue.v1",
        "emotional_appraisal": "bounded pressure",
        "interaction_subtext": "the user awaits a reply",
        "salience_hints": [episode_evidence_handle(facade_payload)],
        "risk_flags": [],
    }
    l1_take_count = 0

    def ready_l1(_task, _warnings):
        nonlocal l1_take_count
        l1_take_count += 1
        return l1_residue, True

    monkeypatch.setattr(
        facade_module,
        "_take_ready_l1_residue",
        ready_l1,
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
        "observation_context",
        "question",
    ]
    assert g1a_packet[-1]["question"]["contract_name"] == (
        "ordinary_goal_bid.v1"
    )
    assert g1a_packet[-1]["question"]["payload"]["l1_residue"] == l1_residue
    assert g1a_messages[-1].content.count('"l1_residue"') == 1
    assert l1_take_count == 1


@pytest.mark.asyncio
async def test_appraisal_structural_failure_recovers_each_family_directly(
    facade_payload,
):
    """A failed grouped request recovers each family as a singleton."""

    invoker, services = make_facade_bundle(facade_payload)
    original_ainvoke = invoker.ainvoke

    async def scripted_recovery(messages, *, config):
        if config.stage_name == "A1":
            invoker.calls.append("A1")
            return SimpleNamespace(content="{invalid")
        if config.stage_name.startswith("A1."):
            family = config.stage_name.split(".", 1)[1]
            invoker.calls.append("A1")
            return SimpleNamespace(content=_empty_group_response((family,)))
        return await original_ainvoke(messages, config=config)

    invoker.ainvoke = scripted_recovery
    output = await run_cognition(facade_payload, services)

    assert tuple(invoker.calls) == (
        "A1",
        "A1",
        "A1",
        "A1",
        "A2",
        "G1a",
        "P1",
    )
    assert output["diagnostics"]["warnings"] == []


@pytest.mark.asyncio
async def test_appraisal_provider_failure_recovers_in_frozen_family_order(
    facade_payload,
):
    """Provider exhaustion recovers each affected family without regrouping."""

    invoker, services = make_facade_bundle(facade_payload)
    original_ainvoke = invoker.ainvoke
    provider_calls = []

    async def provider_failure(messages, *, config):
        if config.stage_name == "A1":
            provider_calls.append(config.stage_name)
            invoker.calls.append("A1")
            raise ConnectionError("scripted provider failure")
        if config.stage_name.startswith("A1."):
            family = config.stage_name.split(".", 1)[1]
            invoker.calls.append("A1")
            return SimpleNamespace(content=_empty_group_response((family,)))
        return await original_ainvoke(messages, config=config)

    invoker.ainvoke = provider_failure
    output = await run_cognition(facade_payload, services)

    assert provider_calls == ["A1"]
    assert tuple(invoker.calls) == (
        "A1",
        "A1",
        "A1",
        "A1",
        "A2",
        "G1a",
        "P1",
    )
    assert output["diagnostics"]["warnings"] == []


@pytest.mark.asyncio
async def test_appraisal_recovery_carriers_last_until_first_singleton_accepts(
    facade_payload,
    monkeypatch,
):
    """Rejected singleton tails retain carriers until one result is accepted."""

    _, services = make_facade_bundle(facade_payload)
    families = (
        "event_agency",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
    )
    questions_by_family = {
        family: {
            "question_kind": family,
            "question_id": f"q:{family}",
            "evidence_handles": ["e1"],
            "permitted_role_handles": ["ce1"],
            "permitted_role_assignment_handles": ["self"],
            "permitted_delta_paths": ["active_events.ce1.responsibility"],
            "semantic_question": f"Question for {family}.",
        }
        for family in families
    }
    calls: list[dict[str, object]] = []
    invocation_index = 0

    async def scripted_invocation(**kwargs):
        nonlocal invocation_index
        calls.append(dict(kwargs))
        invocation_index += 1
        if invocation_index in {1, 2}:
            return SimpleNamespace(
                validated=None,
                disposition=SimpleNamespace(kind="structural_exhausted"),
            )
        accepted_family = families[invocation_index - 2]
        return SimpleNamespace(
            validated=[{"question_id": f"q:{accepted_family}"}],
            disposition=SimpleNamespace(kind="accepted"),
        )

    monkeypatch.setattr(
        facade_module,
        "invoke_serial_question_with_repair",
        scripted_invocation,
    )
    observation_context = {
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
        "evidence": [],
        "supplemental_context": {
            "dialogue_observation": [],
            "local_time_context": [],
            "non_dialog_percepts": [],
            "trigger_source": "user_message",
        },
    }
    rows, failures, context_limited, _l1_observed, carriers_pending = (
        await facade_module._run_appraisal_stages(
            questions_by_family=questions_by_family,
            evidence_handles=("e1",),
            handle_to_ref={
                "ce1": {"entity_id": "candidate:event:e1", "kind": "event"},
                "self": {"entity_id": "character-1", "kind": "self"},
            },
            harness=object(),
            system_content="system",
            services=services,
            warnings=[],
            observation_context=observation_context,
            relation_context={"character_constraints": {}},
            l1_residue={"salience_handles": ["ce1"]},
            l1_observed=False,
            interludes=(
                {"notice_kind": "resolver_observation", "evidence_handle": "e1"},
            ),
            attempt_owner="serial_appraisal",
            branch_prefix="test_appraisal",
            stage_prefix="",
            deterministic_only=True,
            json_repair_callback=None,
            deadline_monotonic=None,
        )
    )

    assert [row["question_id"] for row in rows] == [
        "q:goal_threat_outcome",
        "q:epistemic_comparison_memory",
    ]
    assert failures == {
        "q:event_agency": "semantic_appraisal_contract_exhausted",
    }
    assert context_limited is False
    assert carriers_pending is False
    assert len(calls) == 4
    grouped, failed_singleton, accepted_singleton, later_singleton = calls
    assert grouped["interludes"]
    assert grouped["observation_context"] == observation_context
    assert grouped["question"].payload["l1_residue"] == {
        "salience_handles": ["ce1"]
    }
    assert failed_singleton["interludes"]
    assert failed_singleton["observation_context"] == observation_context
    assert failed_singleton["question"].payload["l1_residue"] == {
        "salience_handles": ["ce1"]
    }
    assert accepted_singleton["interludes"]
    assert accepted_singleton["observation_context"] == observation_context
    assert accepted_singleton["question"].payload["l1_residue"] == {
        "salience_handles": ["ce1"]
    }
    assert later_singleton["interludes"] == ()
    assert later_singleton["observation_context"] is None
    assert "l1_residue" not in later_singleton["question"].payload


@pytest.mark.asyncio
async def test_appraisal_observation_carrier_survives_failed_a1_until_a2_accepts(
    facade_payload,
    monkeypatch,
):
    """All A1 failures keep first-consumer carriers for the A2 question."""

    _, services = make_facade_bundle(facade_payload)
    families = (
        "event_agency",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
        "relationship_social",
        "moral_identity",
        "existential_drive",
    )
    questions_by_family = {
        family: {
            "question_kind": family,
            "question_id": f"q:{family}",
            "evidence_handles": ["e1"],
            "permitted_role_handles": ["ce1"],
            "permitted_role_assignment_handles": ["self"],
            "permitted_delta_paths": [],
            "semantic_question": f"Question for {family}.",
        }
        for family in families
    }
    observation_context = {
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
        "evidence": [],
        "supplemental_context": {
            "dialogue_observation": [],
            "local_time_context": [],
            "non_dialog_percepts": [],
            "trigger_source": "user_message",
        },
    }
    calls = []
    invocation_index = 0

    async def scripted_invocation(**kwargs):
        nonlocal invocation_index
        invocation_index += 1
        calls.append(dict(kwargs))
        if invocation_index <= 4:
            return SimpleNamespace(
                validated=None,
                disposition=SimpleNamespace(kind="structural_exhausted"),
            )
        return SimpleNamespace(
            validated=[{"question_id": f"q:{family}"} for family in families[3:]],
            disposition=SimpleNamespace(kind="accepted"),
        )

    monkeypatch.setattr(
        facade_module,
        "invoke_serial_question_with_repair",
        scripted_invocation,
    )
    rows, _failures, _limited, _observed, carriers_pending = (
        await facade_module._run_appraisal_stages(
            questions_by_family=questions_by_family,
            evidence_handles=("e1",),
            handle_to_ref={},
            harness=object(),
            system_content="system",
            services=services,
            warnings=[],
            observation_context=observation_context,
            relation_context={
                "character_constraints": {},
                "character_operational_context": {},
                "relationship_projection": {},
                "current_affect": [],
            },
            l1_residue={"salience_handles": ["ce1"]},
            l1_observed=False,
            interludes=({"notice_kind": "resolver_observation"},),
            attempt_owner="serial_appraisal",
            branch_prefix="test_appraisal",
            stage_prefix="",
            deterministic_only=True,
            json_repair_callback=None,
            deadline_monotonic=None,
        )
    )
    family_rosters = [
        tuple(row["family"] for row in call["question"].payload["families"])
        for call in calls
        if call["question"].payload.get("families")
    ]
    assert family_rosters == [
        families[:3],
        (families[0],),
        (families[1],),
        (families[2],),
        families[3:],
    ]
    a2_call = calls[-1]
    assert a2_call["observation_context"] == observation_context
    assert a2_call["interludes"]
    assert a2_call["question"].payload["l1_residue"]
    assert "relation_context" in a2_call["question"].payload
    assert rows == [{"question_id": f"q:{family}"} for family in families[3:]]
    assert carriers_pending is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("context", "deadline"))
async def test_appraisal_boundary_failure_does_not_trigger_family_recovery(
    facade_payload,
    monkeypatch,
    failure_kind,
):
    """Context and deadline exhaustion stop before recovery calls."""

    _, services = make_facade_bundle(facade_payload)
    questions_by_family = {
        "event_agency": {
            "question_kind": "event_agency",
            "question_id": "q:event_agency",
            "evidence_handles": [],
            "permitted_role_handles": [],
            "permitted_role_assignment_handles": [],
            "permitted_delta_paths": [],
            "semantic_question": "One bounded event appraisal.",
        }
    }
    calls = []

    async def boundary_failure(**kwargs):
        calls.append(kwargs["question"])
        if failure_kind == "context":
            raise CognitionContextLimitError("scripted context limit")
        raise AssertionError("deadline should be checked before invocation")

    monkeypatch.setattr(
        facade_module,
        "invoke_serial_question_with_repair",
        boundary_failure,
    )
    warnings = []
    rows, failures, context_limited, l1_observed, _carriers_pending = (
        await facade_module._run_appraisal_stages(
            questions_by_family=questions_by_family,
            evidence_handles=(),
            handle_to_ref={},
            harness=object(),
            system_content="",
            services=services,
            warnings=warnings,
            observation_context=None,
            relation_context={"character_constraints": {}},
            l1_residue=None,
            l1_observed=False,
            interludes=(),
            attempt_owner="serial_appraisal",
            branch_prefix="test_appraisal",
            stage_prefix="",
            deterministic_only=True,
            json_repair_callback=None,
            deadline_monotonic=(
                0.0 if failure_kind == "deadline" else None
            ),
        )
    )

    assert rows == []
    assert failures == {
        "q:event_agency": "semantic_appraisal_contract_exhausted",
    }
    assert context_limited is (failure_kind == "context")
    assert l1_observed is False
    assert len(calls) == (1 if failure_kind == "context" else 0)

def test_exhausted_active_group_salvage_persists_only_canonical_rows() -> None:
    """Structural salvage appends no rejected raw candidate to the chain."""

    question = ChainQuestion(
        contract_name="active_goal_bid_group.v1",
        payload={
            "roster": [{"branch_id": "safety_coping", "goal_kind": "safety"}]
        },
    )
    harness = SerialChainHarness(
        transcript=ChainTranscriptV1(),
        ledger=AttemptLedger({"test": 1}),
        budget=ContextBudgetLedger(
            ContextBudgetPlan(serving_window_tokens=50_000),
        ),
    )
    valid_bid = {
        "branch_id": "safety_coping",
        "intention": "核实边界",
        "desired_outcome": "保持安全",
        "concrete_detail": "先确认证据",
        "reason": "当前事件需要边界确认。",
        "private_monologue": "内部原始思路不进入持久投影。",
        "target_role_handles": ["self"],
        "evidence_handles": ["ev_1"],
        "expected_consequences": ["回应保持有依据"],
        "confidence": "medium",
    }
    rows, failures = facade_module._salvage_exhausted_active_group(
        raw_output=json.dumps({"bids": [valid_bid]}),
        question=question,
        roster=[{"branch_id": "safety_coping", "goal_kind": "safety"}],
        evidence_handles={"ev_1"},
        role_handles={"self"},
        harness=harness,
    )
    assert [row["branch_id"] for row in rows] == ["safety_coping"]
    assert failures == {}
    messages = harness.transcript.to_messages()
    assert messages[0][0] == "human"
    assert messages[1][0] == "assistant"
    assert messages[1][1] == json.dumps(
        {"bids": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "内部原始思路" in messages[1][1]
    assert "raw rejected provider prose" not in messages[1][1]


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


def test_v3_reduction_preserves_resolved_threat_affect_cause() -> None:
    """A grounded threat resolution keeps the canonical affect root receipt."""

    payload = build_cognition_input_from_global_state(_global_state())
    evidence_ref = dict(payload["evidence"][0]["evidence_ref"])
    threat = {
        "entity_id": "threat:fixture-resolution",
        "description": "A bounded fixture threat.",
        "salience": 80,
        "role_refs": [],
        "evidence_refs": [evidence_ref],
        "created_at": payload["episode"]["created_at"],
        "updated_at": payload["episode"]["created_at"],
        "status": "active",
        "likelihood": 80,
        "expected_harm": 70,
        "uncertainty": 10,
        "controllability": 20,
        "coping_potential": 20,
        "residual_pressure": 70,
    }
    reduction_state = copy.deepcopy(payload["mutable_state"])
    reduction_state["threats"] = [threat]
    reduction_state = validate_cognition_state(reduction_state)
    direct_fact = {
        "fact_id": "fact:threat-resolution",
        "fact_kind": "threat_resolved",
        "target_refs": [{
            "scope": "user",
            "kind": "threat",
            "entity_id": "threat:fixture-resolution",
        }],
        "evidence_ref": {
            "source_kind": "action_result",
            "source_id": "action-result:threat-resolution",
            "occurred_at": payload["episode"]["created_at"],
            "semantic_summary": "The action resolved the fixture threat.",
        },
    }
    payload["direct_facts"] = [{
        "producer": "action_result",
        **direct_fact,
    }]
    reduction_state = apply_state_update(
        reduction_state,
        direct_facts=[("action_result", direct_fact)],
        updated_at=payload["episode"]["created_at"],
    )
    reduction_state = validate_cognition_state(reduction_state)
    projection = SimpleNamespace(handle_to_ref={
        "t1": {
            "scope": "user",
            "kind": "threat",
            "entity_id": "threat:fixture-resolution",
        },
    })

    (
        final_state,
        _results,
        failures,
        _comparisons,
        _receipts,
    ) = facade_module._reduce_serial_appraisals(
        reduction_state=reduction_state,
        appraisal_rows=[],
        payload=payload,
        projection=projection,
        updated_at=payload["episode"]["created_at"],
        elapsed_seconds=0,
        reducer_relationship_context=None,
    )

    assert failures == {}
    resolved_root = {
        "scope": "user",
        "kind": "threat",
        "entity_id": "threat:fixture-resolution",
    }
    assert final_state["threats"][0]["status"] == "resolved"
    assert final_state["affect_activations"], final_state
    activation = next(
        activation
        for activation in final_state["affect_activations"]
        if activation["primary_root"] == resolved_root
    )
    assert activation["primary_root"] == resolved_root
    assert resolved_root in activation["root_refs"]
    assert activation["cause_status"] == "resolved"

    affect_projection = facade_module.project_affect(
        final_state["affect_activations"],
        final_state,
    )
    projected_activation = next(
        row for row in affect_projection
        if row["emotion"] == "relief"
    )
    assert "bounded fixture threat" in projected_activation["cause_summary"]
    authoritative_state = facade_module._goal_authoritative_state(
        final_state=final_state,
        current_goal_kind="ordinary_response",
        final_projection=SimpleNamespace(payload={
            "goals": [],
            "threats": [],
            "events": [],
            "knowledge_gaps": [],
        }),
    )
    assert any(
        "bounded fixture threat" in row["cause_summary"]
        for row in authoritative_state["affect"]
    )


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
async def test_cold_ordinary_goal_uses_v2_validator_and_episode_handles(
    facade_payload,
    monkeypatch,
):
    """Cold G1a binds relational evidence to the current episode domain."""

    canonical_validator = facade_module.validate_goal_bid_draft
    ordinary_calls: list[dict[str, object]] = []

    def recording_goal_validator(
        parsed: object,
        **kwargs: object,
    ) -> dict[str, object]:
        if kwargs.get("require_relational_willingness") is True:
            ordinary_calls.append(dict(kwargs))
        return canonical_validator(parsed, **kwargs)

    monkeypatch.setattr(
        facade_module,
        "validate_goal_bid_draft",
        recording_goal_validator,
    )
    _, services = make_facade_bundle(facade_payload)
    await run_cognition(facade_payload, services)

    episode_handle = episode_evidence_handle(facade_payload)
    assert ordinary_calls
    assert ordinary_calls[0]["episode_handles"] == {episode_handle}


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


@pytest.mark.asyncio
async def test_cold_workspace_question_uses_prompt_safe_partition_handles(
    facade_payload,
    monkeypatch,
):
    """A persistent sibling bid reaches W1 through stable prompt handles."""

    payload = copy.deepcopy(facade_payload)
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
        "urgency": 60,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
    }]
    active_draft = {
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
    }
    responses = default_scripted_responses(evidence_handle)
    responses["G1b"] = json.dumps(
        {"bids": [active_draft]},
        ensure_ascii=False,
    )
    responses["W1"] = json.dumps({
        "primary_bid_handle": "b1",
        "supporting_bid_handles": ["b2"],
        "suppressed_bid_handles": [],
    })
    invoker, services = make_facade_bundle(payload, responses=responses)
    canonical_partition_calls: list[set[str]] = []
    canonical_validator = facade_module.validate_workspace_partition

    def recording_partition_validator(
        parsed: object,
        handles: set[str],
    ) -> dict[str, object]:
        canonical_partition_calls.append(set(handles))
        return canonical_validator(parsed, handles)

    monkeypatch.setattr(
        facade_module,
        "validate_workspace_partition",
        recording_partition_validator,
    )
    captured_stages: list[str] = []
    captured_messages: dict[str, str] = {}
    captured_w1_messages: list[str] = []
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Capture the live W1 question before scripted model handling."""

        stage_name = config.stage_name.split(".")[0]
        captured_stages.append(stage_name)
        captured_messages[stage_name] = messages[-1].content
        if stage_name == "W1":
            captured_w1_messages.append(messages[-1].content)
        return await scripted_ainvoke(messages, config=config)

    invoker.ainvoke = recording_ainvoke
    output = await run_cognition(payload, services)

    assert tuple(captured_stages) == (
        "A1",
        "A2",
        "G1a",
        "G1b",
        "W1",
        "P1",
    )

    g1a_packet = json.loads(captured_messages["G1a"])
    assert g1a_packet[0]["interludes"][0]["notice_kind"] == (
        "state_transition"
    )
    assert g1a_packet[0]["interludes"][0]["stage"] == "I1"
    assert g1a_packet[0]["interludes"][0]["notice"]

    g1b_packet = json.loads(captured_messages["G1b"])
    g1b_payload = g1b_packet[-1]["question"]["payload"]
    assert g1b_payload["allowed_evidence_handles"] == [evidence_handle]
    assert g1b_payload["allowed_role_handles"]
    expected_dialogue_bindings = [{
        "speaker_handle": "current_user",
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
    }]
    assert g1a_packet[-1]["question"]["payload"][
        "dialogue_role_bindings"
    ] == expected_dialogue_bindings
    assert g1b_payload["dialogue_role_bindings"] == expected_dialogue_bindings
    assert g1b_payload["branch_roster"][0][
        "allowed_evidence_handles"
    ] == [evidence_handle]
    assert "semantic_context" not in g1b_payload

    expected_continuity = {
        "private_continuity_context": payload[
            "private_continuity_context"
        ],
        "past_dialog_cognition_context": payload[
            "past_dialog_cognition_context"
        ],
        "group_engagement_action_context": {
            "engagement_guidelines": payload[
                "group_engagement_action_context"
            ]["engagement_guidelines"],
            "confidence": payload["group_engagement_action_context"][
                "confidence"
            ],
        },
    }
    assert g1a_packet[-1]["question"]["payload"][
        "continuity_context"
    ] == expected_continuity
    assert g1b_payload["continuity_context"] == expected_continuity
    authoritative_state = g1a_packet[-1]["question"]["payload"][
        "authoritative_state"
    ]
    assert set(authoritative_state["matter_projections"]) == {
        "goals",
        "threats",
        "events",
        "knowledge_gaps",
    }
    assert authoritative_state["matter_projections"]["goals"]
    assert "entity_id" not in json.dumps(authoritative_state)

    assert len(captured_w1_messages) == 1
    assert "entity_id" not in captured_w1_messages[0]
    packet = json.loads(captured_w1_messages[0])
    question = next(
        section["question"]
        for section in packet
        if "question" in section
    )
    assert set(question["payload"]["bid_index"]) == {"b1", "b2"}
    assert "current_event" not in question["payload"]
    assert canonical_partition_calls == [{"b1", "b2"}]

    assert packet[0]["interludes"][0]["notice_kind"] == "I2"
    p1_packet = json.loads(captured_messages["P1"])
    p1_payload = p1_packet[-1]["question"]["payload"]
    assert p1_payload["primary_bid_handle"] == "b1"
    assert p1_payload["supporting_bid_handles"] == ["b2"]
    assert set(p1_payload["bid_index"]) == {"b1", "b2"}
    assert set(p1_payload["action_index"]) >= {"a1"}
    assert set(p1_payload["resolver_index"]) >= {"r1"}
    assert "primary_bid" not in p1_payload
    assert "supporting_bids" not in p1_payload
    assert "entity_id" not in json.dumps(p1_payload)
    assert output["diagnostics"]["warnings"] == []


def test_goal_authoritative_state_retains_all_canonical_matter_categories(
    facade_payload,
):
    """Post-I1 matter rows remain bounded and ID-free for goal judgment."""

    final_projection = SimpleNamespace(payload={
        "goals": [{
            "handle": "g1",
            "description": "A goal remains in progress.",
            "lifecycle": "进行中",
        }],
        "threats": [{
            "handle": "t1",
            "description": "A threat is resolved.",
            "lifecycle": "已解决",
        }],
        "events": [{
            "handle": "ev1",
            "description": "An event remains active.",
            "lifecycle": "进行中",
        }],
        "knowledge_gaps": [{
            "handle": "k1",
            "description": "A knowledge gap remains open.",
            "lifecycle": "开放",
        }],
    })

    authoritative_state = facade_module._goal_authoritative_state(
        final_state=facade_payload["mutable_state"],
        current_goal_kind="ordinary_response",
        final_projection=final_projection,
    )

    matters = authoritative_state["matter_projections"]
    assert [row["lifecycle"] for row in matters["goals"]] == ["进行中"]
    assert [row["lifecycle"] for row in matters["threats"]] == ["已解决"]
    assert [row["lifecycle"] for row in matters["events"]] == ["进行中"]
    assert [row["lifecycle"] for row in matters["knowledge_gaps"]] == [
        "开放"
    ]
    assert "entity_id" not in json.dumps(authoritative_state)


@pytest.mark.asyncio
async def test_cold_relationship_sensitive_run_still_queries_active_goal_group(
    facade_payload,
):
    """Relational sensitivity skips W1 after G1b has supplied active bids."""

    payload = copy.deepcopy(facade_payload)
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
        "urgency": 60,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
    }]
    ordinary_draft = {
        "intention": "Reply to the user's greeting in character.",
        "desired_outcome": "The user receives an in-character reply.",
        "concrete_detail": "Greet the user and open a topic of interest.",
        "reason": "The user opened the conversation with a simple greeting.",
        "private_monologue": "A quiet hello is an invitation, not a demand.",
        "target_role_handles": [],
        "evidence_handles": [evidence_handle],
        "expected_consequences": [
            "The conversation continues from the greeting."
        ],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "relationship_sensitive",
            "stance": "accept",
            "current_user_relationship_state": "developing_or_uncertain",
            "reason": '愿意回应用户的问候并延续对话',
            "evidence_handles": [evidence_handle],
        },
    }
    active_draft = {
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
    }
    responses = default_scripted_responses(evidence_handle)
    responses["G1a"] = json.dumps(ordinary_draft, ensure_ascii=False)
    responses["G1b"] = json.dumps(
        {"bids": [active_draft]},
        ensure_ascii=False,
    )
    invoker, services = make_facade_bundle(payload, responses=responses)

    output = await run_cognition(payload, services)

    assert tuple(invoker.calls) == ("A1", "A2", "G1a", "G1b", "P1")
    assert "G1b" in invoker.calls
    assert "W1" not in invoker.calls
    assert output["diagnostics"]["warnings"].count(
        "authoritative_relational_willingness"
    ) == 1


@pytest.mark.asyncio
async def test_cold_i1_terminal_goal_is_excluded_from_g1b(
    facade_payload,
):
    """I1-terminal goals do not create stale G1b branches."""

    payload = copy.deepcopy(facade_payload)
    evidence_handle = episode_evidence_handle(payload)
    evidence_ref = dict(payload["evidence"][0]["evidence_ref"])
    created_at = payload["episode"]["created_at"]
    payload["mutable_state"]["threats"] = [{
        "entity_id": "threat:fixture-route",
        "description": "A bounded route threat.",
        "salience": 80,
        "role_refs": [],
        "evidence_refs": [dict(evidence_ref)],
        "created_at": created_at,
        "updated_at": created_at,
        "status": "active",
        "likelihood": 80,
        "expected_harm": 70,
        "uncertainty": 10,
        "controllability": 20,
        "coping_potential": 20,
        "residual_pressure": 70,
    }]
    payload["mutable_state"] = validate_cognition_state(
        payload["mutable_state"]
    )
    responses = default_scripted_responses(evidence_handle)
    responses["A1"] = json.dumps({
        "event_agency": {"propositions": [], "deltas": []},
        "goal_threat_outcome": {
            "propositions": [
                {
                    "proposition_kind": "threat_resolved",
                    "subject_handle": "t1",
                    "evidence_handles": [evidence_handle],
                    "role_assignments": [],
                    "semantic_value": "The route threat is resolved.",
                },
                {
                    "proposition_kind": "goal_completed",
                    "subject_handle": "g1",
                    "evidence_handles": [evidence_handle],
                    "role_assignments": [{
                        "role": "affected_goal",
                        "entity_handle": "g1",
                    }],
                    "semantic_value": "The safety goal is complete.",
                },
            ],
            "deltas": [
                {
                    "target_path": "threats.t1.residual_pressure",
                    "delta": -40,
                    "evidence_handles": [evidence_handle],
                    "reason": "The route threat is resolved.",
                },
            ],
        },
        "epistemic_comparison_memory": {
            "propositions": [],
            "deltas": [],
        },
    })
    invoker, services = make_facade_bundle(payload, responses=responses)

    output = await run_cognition(payload, services)

    assert tuple(invoker.calls) == ("A1", "A2", "G1a", "P1")
    warnings = output["diagnostics"]["warnings"]
    assert "stale_goal_bid_dropped:safety_coping" not in warnings
    assert not any(
        row["goal_kind"] == "safety"
        for row in output["cognition_observability"]["branches"]
    )
    safety_goal = next(
        goal
        for goal in output["state_update"]["replacement_state"]["goals"]
        if goal["goal_kind"] == "safety"
    )
    assert safety_goal["status"] == "satisfied"
    assert output["state_update"]["replacement_state"]["threats"][0][
        "status"
    ] == "resolved"
    relief = next(
        activation
        for activation in output["state_update"]["replacement_state"][
            "affect_activations"
        ]
        if activation["emotion_id"] == "relief"
    )
    assert relief["cause_status"] == "resolved"
    assert relief["primary_root"]["kind"] == "threat"
    assert relief["primary_root"] in relief["root_refs"]
    projected_relief = next(
        row
        for row in output["affect_projection"]
        if row["emotion"] == "relief"
    )
    assert "A bounded route threat." in projected_relief["cause_summary"]


@pytest.mark.asyncio
async def test_cold_single_bid_action_plan_receives_i2_interlude(
    facade_payload,
):
    """A one-bid cold path attaches I2 directly to P1."""

    invoker, services = make_facade_bundle(facade_payload)
    captured_p1_message: list[str] = []
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Capture the P1 packet before the scripted response is returned."""

        if config.stage_name.split(".")[0] == "P1":
            captured_p1_message.append(messages[-1].content)
        return await scripted_ainvoke(messages, config=config)

    invoker.ainvoke = recording_ainvoke
    await run_cognition(facade_payload, services)

    assert len(captured_p1_message) == 1
    packet = json.loads(captured_p1_message[0])
    assert packet[0]["interludes"][0]["notice_kind"] == "I2"
    payload = packet[-1]["question"]["payload"]
    assert payload["primary_bid_handle"] == "b1"
    assert payload["supporting_bid_handles"] == []


@pytest.mark.asyncio
async def test_cold_state_reduction_does_not_repeat_after_action_planning(
    facade_payload,
    monkeypatch,
):
    """Cold appraisal reduction runs once before goals and never after P1."""

    reduction_calls: list[dict[str, object]] = []
    original_reduce = facade_module._reduce_serial_appraisals

    def recording_reduce(*args, **kwargs):
        """Record each cold reduction call before delegating to its owner."""

        reduction_calls.append(dict(kwargs))
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(
        facade_module,
        "_reduce_serial_appraisals",
        recording_reduce,
    )
    invoker, services = make_facade_bundle(facade_payload)

    await run_cognition(facade_payload, services)

    assert len(reduction_calls) == 1
    assert invoker.calls.index("P1") > invoker.calls.index("G1a")


@pytest.mark.asyncio
async def test_required_ordinary_failure_recovers_from_valid_active_sibling(
    facade_payload,
):
    """G1b permits a complete sibling to recover an exhausted G1a branch."""

    payload = _payload_with_active_bond_goal(facade_payload)
    handle = episode_evidence_handle(payload)
    defaults = default_scripted_responses(handle)
    active_draft = _active_bond_goal_draft(handle)

    def responses(stage_name, attempt_index):
        del attempt_index
        if stage_name == "G1a":
            return "{invalid json"
        if stage_name == "G1b":
            return json.dumps({"bids": [active_draft]})
        return defaults[stage_name]

    invoker, services = make_facade_bundle(payload, responses=responses)
    captured_goal_messages: dict[str, list[str]] = {"G1a": [], "G1b": []}
    scripted_ainvoke = invoker.ainvoke

    async def recording_ainvoke(messages, *, config):
        """Capture both goal packets through the required-branch recovery."""

        stage_name = config.stage_name.split(".")[0]
        if stage_name in captured_goal_messages:
            captured_goal_messages[stage_name].append(messages[-1].content)
        return await scripted_ainvoke(messages, config=config)

    invoker.ainvoke = recording_ainvoke
    output = await run_cognition(payload, services)

    assert tuple(invoker.calls) == (
        "A1",
        "A2",
        "G1a",
        "G1a",
        "G1b",
        "P1",
    )
    assert "required_branch_recovered_by_valid_bid:ordinary_response" in (
        output["diagnostics"]["warnings"]
    )
    branch_rows = output["cognition_observability"]["branches"]
    ordinary_row = next(
        row for row in branch_rows
        if row["goal_kind"] == "ordinary_response"
    )
    assert ordinary_row["status"] == "failed"
    assert ordinary_row["failure_code"] == "ordinary_response_unavailable"
    assert output["admitted_bid"]["branch_id"] == "bond_protection"
    assert captured_goal_messages["G1a"]
    assert captured_goal_messages["G1b"]
    g1a_payload = json.loads(captured_goal_messages["G1a"][0])[-1][
        "question"
    ]["payload"]
    g1b_payload = json.loads(captured_goal_messages["G1b"][0])[-1][
        "question"
    ]["payload"]
    assert g1a_payload["continuity_context"] == g1b_payload[
        "continuity_context"
    ]


@pytest.mark.asyncio
async def test_required_ordinary_failure_without_sibling_fails_before_commit(
    facade_payload,
    monkeypatch,
):
    """No complete G1b sibling keeps the required branch fail-closed."""

    payload = _payload_with_active_bond_goal(facade_payload)
    handle = episode_evidence_handle(payload)
    defaults = default_scripted_responses(handle)
    commit_calls: list[object] = []

    def responses(stage_name, attempt_index):
        del attempt_index
        if stage_name in {"G1a", "G1b"}:
            return "{invalid json"
        return defaults[stage_name]

    def record_commit(*args, **kwargs):
        del args, kwargs
        commit_calls.append(True)
        raise AssertionError("state commit must not follow required failure")

    monkeypatch.setattr(
        facade_module,
        "advance_session_after_output",
        record_commit,
    )
    invoker, services = make_facade_bundle(payload, responses=responses)

    with pytest.raises(CognitionExecutionError, match="ordinary_response"):
        await run_cognition(payload, services)

    assert tuple(invoker.calls) == (
        "A1",
        "A2",
        "G1a",
        "G1a",
        "G1b",
        "G1b",
    )
    assert commit_calls == []


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

    assert tuple(invoker.calls) == ("G1a",) * 2


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
    ) + ("G1a",) * 2
    assert tuple(invoker.calls) == expected_calls


@pytest.mark.asyncio
async def test_l1_failures_are_bounded_by_facade_residue_and_drain_paths() -> None:
    """L1 task failures become advisory warnings at both join boundaries."""

    async def fail_l1() -> None:
        raise ValueError("provider detail must stay out of primary flow")

    ready_task = asyncio.create_task(fail_l1())
    await asyncio.sleep(0)
    ready_warnings: list[str] = []
    residue, observed = facade_module._take_ready_l1_residue(
        ready_task,
        ready_warnings,
    )

    assert residue is None
    assert observed is True
    assert ready_warnings == ["sidecar_l1_unavailable"]

    drain_task = asyncio.create_task(fail_l1())
    await asyncio.sleep(0)
    drain_warnings: list[str] = []
    await facade_module._drain_l1_sidecar(
        drain_task,
        invocation_state=None,
        warnings=drain_warnings,
    )

    assert drain_warnings == ["sidecar_l1_unavailable"]


@pytest.mark.asyncio
async def test_l1_drain_propagates_outer_cancellation_after_owned_cleanup() -> None:
    """Facade cleanup drains L1 before returning an outer cancellation."""

    cancellation_seen = asyncio.Event()
    release_l1_cleanup = asyncio.Event()

    async def run_l1() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_l1_cleanup.wait()
            raise

    l1_task = asyncio.create_task(run_l1())
    warnings: list[str] = []
    drain_task = asyncio.create_task(
        facade_module._drain_l1_sidecar(
            l1_task,
            invocation_state=None,
            warnings=warnings,
        )
    )
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    drain_task.cancel()
    await asyncio.sleep(0)
    assert not drain_task.done()
    release_l1_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await drain_task
    assert l1_task.cancelled()
    assert warnings == ["sidecar_l1_dropped"]

