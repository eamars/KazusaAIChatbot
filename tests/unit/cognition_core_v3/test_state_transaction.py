"""Focused deterministic proofs for the canonical cognition transaction."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.appraisal import bind_axis_changes
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CanonicalAppraisal,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    _prepare_state_transaction,
    run_cognition,
)
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionExecutionError
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_relationship_maintenance,
    create_guarded_goal,
    materialize_causal_root,
)
from tests.unit.cognition_core_v3.test_handleless_contract import (
    _input,
    _services,
)


def _appraisal(
    family: str,
    axis: str | None = None,
    shift: str = "strong_increase",
) -> CanonicalAppraisal:
    return CanonicalAppraisal(
        family=family,
        applicable=True,
        semantic_summary="accepted semantic meaning",
        cause_summary="accepted concrete cause",
        axis_changes=(
            ({"axis": axis, "shift": shift, "reason": "the cause changes this axis"},)
            if axis is not None
            else ()
        ),
    )


def test_state_transaction_reclaims_terminal_capacity_and_preserves_active_causes() -> None:
    payload = _input()
    state = payload["mutable_state"]
    for index in range(32):
        evidence = {
            "source_kind": "episode",
            "source_id": f"episode:capacity-{index}",
            "occurred_at": state["updated_at"],
            "semantic_summary": f"capacity evidence {index}",
        }
        state, _root_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence=evidence,
            description=f"capacity event {index}",
        )
    active_ids = {
        row["entity_id"] for row in state["active_events"][26:]
    }
    for row in state["active_events"][:26]:
        row["status"] = "resolved"
    validate_cognition_state(state)
    payload["mutable_state"] = state
    payload["evidence"] = [{
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:capacity-new",
            "occurred_at": state["updated_at"],
            "semantic_summary": "the new causal observation",
        },
    }]
    updated, _transitions, receipts, _provenance = bind_axis_changes(
        payload,
        (_appraisal("event_agency", "responsibility"),),
    )
    validate_cognition_state(updated)
    assert len(updated["active_events"]) == 32
    assert active_ids.issubset({row["entity_id"] for row in updated["active_events"]})
    assert any(
        row["source_id"] == "episode:capacity-new"
        for entity in updated["active_events"]
        for row in entity["evidence_refs"]
    )
    assert receipts[0]["disposition"] in {"applied", "clamped"}


def test_repeated_answerable_turns_do_not_accumulate_transient_events_or_goals() -> None:
    payload = _input()
    appraisals = list(
        _appraisal(family)
        for family in (*CANONICAL_A1_FAMILIES, *CANONICAL_A2_FAMILIES)
    )
    appraisals[1] = _appraisal(
        "goal_threat_outcome",
        "urgency",
        shift="moderate_increase",
    )
    state = payload["mutable_state"]
    for _index in range(40):
        current_payload = dict(payload)
        current_payload["mutable_state"] = state
        state, _transitions, _receipts, _provenance = bind_axis_changes(
            current_payload,
            tuple(appraisals),
            goal={"intent": "answer the current request", "cause_summary": "the request"},
            goal_resolution="answerable_now",
        )
    assert state["active_events"] == []
    assert state["goals"] == []
    validate_cognition_state(state)


def test_zero_clamped_gap_decrease_does_not_terminalize_before_same_episode_recurrence() -> None:
    payload = _input()
    first_state, _transitions, first_receipts, _provenance = bind_axis_changes(
        payload,
        (
            _appraisal(
                "epistemic_comparison_memory",
                "uncertainty",
                shift="strong_decrease",
            ),
        ),
    )
    validate_cognition_state(first_state)
    first_gap = first_state["knowledge_gaps"]
    assert len(first_gap) == 1
    assert first_gap[0]["status"] == "open"
    assert first_gap[0]["uncertainty"] == 0
    assert first_receipts[0]["applied_targets"][0]["applied_delta"] == 0

    repeated_payload = dict(payload)
    repeated_payload["mutable_state"] = first_state
    second_state, _transitions, second_receipts, _provenance = bind_axis_changes(
        repeated_payload,
        (
            _appraisal(
                "epistemic_comparison_memory",
                "uncertainty",
                shift="strong_increase",
            ),
        ),
    )
    validate_cognition_state(second_state)
    second_gap = second_state["knowledge_gaps"]
    assert len(second_gap) == 1
    assert second_gap[0]["entity_id"] == first_gap[0]["entity_id"]
    assert second_gap[0]["status"] == "open"
    assert second_gap[0]["uncertainty"] == 40
    assert second_receipts[0]["applied_targets"][0]["applied_delta"] == 40


def test_stale_response_goal_cutover_preserves_active_causes_and_other_goals() -> None:
    """Canonical cutover removes only stale ordinary-response goals."""

    payload = _input()
    state = deepcopy(payload["mutable_state"])
    evidence = payload["evidence"][0]["evidence_ref"]
    state, event_id, _created = materialize_causal_root(
        state,
        kind="event",
        primary_evidence=evidence,
        description="an active emotional cause",
    )
    stale_evidence = {**evidence, "source_id": "pre-cutover-response"}
    durable_evidence = {**evidence, "source_id": "durable-bond-goal"}
    state["goals"] = [
        create_guarded_goal(
            state,
            goal_kind="ordinary_response",
            description="answer a completed prior turn",
            role_refs=[],
            evidence_refs=[stale_evidence],
            axes={},
        ),
        create_guarded_goal(
            state,
            goal_kind="bond_protection",
            description="protect an existing bond",
            role_refs=[{
                "role": "affected_relationship",
                "entity_kind": "relationship",
                "entity_id": state["relationship"]["relationship_id"],
            }],
            evidence_refs=[durable_evidence],
            axes={},
        ),
    ]
    state["affect_activations"] = [{
        "activation_id": "emotion:sadness",
        "emotion_id": "sadness",
        "primary_root": {
            "scope": "user",
            "kind": "event",
            "entity_id": event_id,
        },
        "root_refs": [{
            "scope": "user",
            "kind": "event",
            "entity_id": event_id,
        }],
        "phase": "active",
        "score": 40,
        "peak_score": 40,
        "trend": "stable",
        "cause_status": "active",
        "started_at": state["updated_at"],
        "updated_at": state["updated_at"],
        "last_reinforced_at": state["updated_at"],
    }]
    validate_cognition_state(state)
    payload["mutable_state"] = state

    _original, prepared, _transitions = _prepare_state_transaction(payload)

    assert [goal["goal_kind"] for goal in prepared["goals"]] == [
        "bond_protection",
    ]
    assert any(
        event["entity_id"] == event_id
        for event in prepared["active_events"]
    )
    assert prepared["affect_activations"][0]["primary_root"]["entity_id"] == (
        event_id
    )


def test_unresolved_continuation_replaces_prior_response_goal_exactly() -> None:
    """One unresolved turn owns one current ordinary-response goal."""

    payload = _input()
    state = deepcopy(payload["mutable_state"])
    evidence = payload["evidence"][0]["evidence_ref"]
    prior_goal = create_guarded_goal(
        state,
        goal_kind="ordinary_response",
        description="the prior unresolved wording goal",
        role_refs=[],
        evidence_refs=[{**evidence, "source_id": "prior-response-goal"}],
        axes={},
    )
    state["goals"] = [prior_goal]
    payload["mutable_state"] = state
    binding_metadata: dict[str, object] = {}

    updated, _transitions, _receipts, _provenance = bind_axis_changes(
        payload,
        (),
        goal={
            "intent": "ask for the current missing detail",
            "cause_summary": "the current observation",
        },
        goal_resolution="requires_user_input",
        binding_metadata=binding_metadata,
    )

    response_goals = [
        goal
        for goal in updated["goals"]
        if goal["goal_kind"] == "ordinary_response"
    ]
    assert len(response_goals) == 1
    assert response_goals[0]["description"] == (
        "ask for the current missing detail"
    )
    assert binding_metadata["continuation_goal_ref"]["entity_id"] == (
        response_goals[0]["entity_id"]
    )


def test_all_protected_capacity_defers_without_losing_semantic_state() -> None:
    payload = _input()
    state = payload["mutable_state"]
    for index in range(32):
        evidence = {
            "source_kind": "episode",
            "source_id": f"episode:protected-{index}",
            "occurred_at": state["updated_at"],
            "semantic_summary": f"protected cause {index}",
        }
        state, _root_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence=evidence,
            description=f"protected cause {index}",
        )
    root_refs = [
        {
            "scope": "user",
            "kind": "event",
            "entity_id": row["entity_id"],
        }
        for row in state["active_events"]
    ]
    activation_rows = []
    for index, emotion_id in enumerate(("sadness", "anger", "fear", "joy")):
        refs = root_refs[index * 8:(index + 1) * 8]
        activation_rows.append({
            "activation_id": f"emotion:{emotion_id}",
            "emotion_id": emotion_id,
            "primary_root": refs[0],
            "root_refs": refs,
            "phase": "active",
            "score": 30,
            "peak_score": 30,
            "trend": "stable",
            "cause_status": "active",
            "started_at": state["updated_at"],
            "updated_at": state["updated_at"],
            "last_reinforced_at": state["updated_at"],
        })
    state["affect_activations"] = activation_rows
    validate_cognition_state(state)
    payload["mutable_state"] = state
    payload["evidence"] = [{
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:protected-new",
            "occurred_at": state["updated_at"],
            "semantic_summary": "new protected observation",
        },
    }]
    updated, _transitions, receipts, _provenance = bind_axis_changes(
        payload,
        (_appraisal("event_agency", "responsibility"),),
    )
    validate_cognition_state(updated)
    assert len(updated["active_events"]) == 32
    assert not any(
        row["source_id"] == "episode:protected-new"
        for entity in updated["active_events"]
        for row in entity["evidence_refs"]
    )
    assert any(row.get("disposition") == "capacity_deferred" for row in receipts)


def test_relationship_maintenance_rotates_same_day_source_ids() -> None:
    payload = _input()
    state = deepcopy(payload["mutable_state"])
    maintenance = state["relationship"]["relationship_maintenance"]
    maintenance["last_interaction_date_utc"] = "2026-07-14"
    maintenance["processed_source_ids"] = [
        f"episode:old-{index}" for index in range(256)
    ]
    updated = apply_relationship_maintenance(
        state,
        source_episode_id="new-episode",
        interaction_date_utc="2026-07-14",
        elapsed_seconds=0,
    )
    processed = updated["relationship"]["relationship_maintenance"][
        "processed_source_ids"
    ]
    assert len(processed) == 256
    assert processed[-1] == "episode:new-episode"
    assert processed[0] == "episode:old-1"


def test_continuation_goal_admission_is_exact_for_coding_and_task_resolution() -> None:
    payload = _input()
    evidence = payload["evidence"]
    for action_requests, resolver_requests, expected_continuation in (
        ([{"action_kind": "accepted_coding_task_request", "decision": "status"}], [], True),
        ([{"action_kind": "accepted_coding_task_request", "decision": "approve_and_verify"}], [], True),
        ([{"action_kind": "accepted_coding_task_request", "decision": "cancel"}], [], True),
        ([], [{"capability": "task_resolution_request"}], True),
        ([], [{"capability": "human_clarification"}], False),
        ([], [{"capability": "approval_preparation"}], False),
        ([], [{"capability": "self_goal_resolution"}], False),
    ):
        metadata: dict[str, object] = {}
        updated, _transitions, _receipts, _provenance = bind_axis_changes(
            {
                "episode": payload["episode"],
                "mutable_state": deepcopy(payload["mutable_state"]),
                "state_scope": "user",
                "evidence": evidence,
            },
            (),
            goal={"intent": "continue the accepted work", "cause_summary": "the request"},
            action_requests=action_requests,
            resolver_requests=resolver_requests,
            binding_metadata=metadata,
        )
        validate_cognition_state(updated)
        if expected_continuation:
            assert len(updated["goals"]) == 1
            assert metadata["continuation_goal_ref"] == {
                "scope": "user",
                "kind": "goal",
                "entity_id": updated["goals"][0]["entity_id"],
            }
        else:
            assert updated["goals"] == []
            assert "continuation_goal_ref" not in metadata


class _CharacterTransactionInvoker:
    """Return one fixed valid four-stage product for timestamp coverage."""

    async def ainvoke(self, messages: object, *, config: object) -> object:
        stage = config.stage_name.rsplit(".", 1)[-1]
        if stage == "A1":
            value = {
                family: {
                    "applicable": True,
                    "semantic_summary": "the event has a clear effect",
                    "cause_summary": "the current observation is consequential",
                    "axis_changes": [],
                }
                for family in CANONICAL_A1_FAMILIES
            }
            value["event_agency"]["axis_changes"] = [{
                "axis": "responsibility",
                "shift": "strong_increase",
                "reason": "the observation assigns responsibility",
            }]
        elif stage == "A2":
            value = {
                family: {
                    "applicable": True,
                    "semantic_summary": "the character context remains grounded",
                    "cause_summary": "the same observation supplies the cause",
                    "axis_changes": [],
                }
                for family in CANONICAL_A2_FAMILIES
            }
        elif stage == "G":
            value = {
                "active_character_goal": {
                    "goal_kind": "clarify",
                    "intent": "understand the observation",
                    "reason": "the observation needs a grounded response",
                    "cause_summary": "the current observation",
                },
                "relational_willingness": {
                    "applicable": False,
                    "stance": "not applicable",
                    "reason": "no relationship judgment is needed",
                    "cause_summary": "the current observation",
                },
                "private_monologue": (
                    "I want to understand this because the observation matters."
                ),
            }
        else:
            value = {
                "goal_resolution": "answerable_now",
                "response_goal": "acknowledge the observation",
                "action_requests": [],
                "resolver_requests": [],
                "epistemic_boundary": (
                    "Assert only what the current observation supports."
                ),
            }
        return SimpleNamespace(content=json.dumps(value, ensure_ascii=False))


@pytest.mark.asyncio
async def test_character_state_transaction_advances_timestamp_and_validates_final_affect() -> None:
    payload = _input()
    character_state = build_character_production_state(
        updated_at=payload["mutable_state"]["updated_at"],
    )
    payload["mutable_state"] = character_state
    payload["state_scope"] = "character"
    output = await run_cognition(
        payload,
        _services(_CharacterTransactionInvoker()),
    )
    projection = output["state_projection"]
    replacement = projection["replacement_state"]
    validate_cognition_state(replacement)
    assert datetime.fromisoformat(
        replacement["updated_at"].replace("Z", "+00:00")
    ) > datetime.fromisoformat(
        character_state["updated_at"].replace("Z", "+00:00")
    )
    assert isinstance(replacement["affect_activations"], list)


@pytest.mark.asyncio
async def test_character_noop_transaction_advances_timestamp_strictly() -> None:
    class _NoopInvoker(_CharacterTransactionInvoker):
        async def ainvoke(self, messages: object, *, config: object) -> object:
            response = await super().ainvoke(messages, config=config)
            value = json.loads(response.content)
            if config.stage_name.endswith(".A1"):
                value["event_agency"]["axis_changes"] = []
            return SimpleNamespace(content=json.dumps(value, ensure_ascii=False))

    payload = _input()
    character_state = build_character_production_state(
        updated_at=payload["mutable_state"]["updated_at"],
    )
    payload["mutable_state"] = character_state
    payload["state_scope"] = "character"
    output = await run_cognition(payload, _services(_NoopInvoker()))
    replacement = output["state_projection"]["replacement_state"]
    assert datetime.fromisoformat(
        replacement["updated_at"].replace("Z", "+00:00")
    ) > datetime.fromisoformat(
        character_state["updated_at"].replace("Z", "+00:00")
    )


@pytest.mark.asyncio
async def test_cognition_turn_deadline_bounds_full_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _input()

    class _SlowInvoker:
        async def ainvoke(self, messages: object, *, config: object) -> object:
            await asyncio.sleep(1)
            return SimpleNamespace(content="{}")

    original_wait_for = facade_module.asyncio.wait_for
    seen_timeouts: list[float] = []

    async def bounded_wait(awaitable: object, timeout: float) -> object:
        seen_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=0.001)

    monkeypatch.setattr(facade_module.asyncio, "wait_for", bounded_wait)
    with pytest.raises(CognitionExecutionError) as error:
        await run_cognition(payload, _services(_SlowInvoker()))
    assert seen_timeouts == [240]
    assert error.value.error_code == "cognition_turn_deadline_exhausted"
    assert error.value.stage == "cognition_core_v3"
    assert error.value.safe_checkpoint == "pre_state_commit"
    assert error.value.retryable is False
