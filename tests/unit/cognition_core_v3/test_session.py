"""Deterministic tests for the Cognition V3 chain-session registry."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math

import pytest

from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
)
from kazusa_ai_chatbot.cognition_core_v3 import session
from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    validate_current_turn_relational_willingness,
)


def _payload(
    *,
    cycle_index: int,
    mutable_state: dict[str, object] | None = None,
    evidence: list[dict[str, object]] | None = None,
    willingness: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a minimal session-compatible input packet."""

    state = mutable_state if mutable_state is not None else {
        "state_scope": "user",
        "updated_at": "2026-08-20T00:00:00Z",
    }
    rows = evidence if evidence is not None else [
        {
            "evidence_handle": "e1",
            "evidence_ref": {"source_kind": "episode", "source_id": "s1"},
        }
    ]
    payload: dict[str, object] = {
        "schema_version": "cognition_core_input.v2",
        "episode": {"episode_id": "episode-1"},
        "state_scope": "user",
        "mutable_state": state,
        "character_constraints": {},
        "character_identity_context": {},
        "evidence": rows,
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "",
        "scene_context": {},
        "private_continuity_context": "",
    }
    if cycle_index != 0:
        payload["resolver_cycle_index"] = cycle_index
    if willingness is not None:
        payload["current_turn_relational_willingness"] = willingness
    return payload


def _resolver_row(
    handle: str = "e2",
    *,
    source_id: str | None = None,
) -> dict[str, object]:
    """Build one resolver-observation row for session matching."""

    resolved_source_id = (
        f"source-{handle}"
        if source_id is None
        else source_id
    )
    return {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": "resolver_observation",
            "source_id": resolved_source_id,
        },
    }


def _willingness() -> dict[str, object]:
    """Build one valid ordinary-turn relational carrier."""

    value = {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": "episode-1",
        "branch_id": "ordinary_response",
        "decision": {
            "schema_version": "relational_willingness.v2",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "evidence",
            "evidence_handles": ["e1"],
        },
    }
    return_value = dict(
        validate_current_turn_relational_willingness(
            value,
            episode_id="episode-1",
        )
    )
    return return_value


def _goal_progress() -> dict[str, object]:
    """Build one canonical resolver goal-progress carrier."""

    return {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "Find one bounded fact.",
        "current_focus": "Inspect the accepted observation.",
        "deliverables": [],
        "missing_user_inputs": [],
        "evidence_dependencies": [],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": [],
    }


def _continuation_ref() -> dict[str, object]:
    """Build one canonical task-resolution continuation reference."""

    return dict(
        build_goal_continuation_ref(
            source_episode_id="episode-1",
            source_message_id="message-1",
            branch_id="ordinary_response",
            goal_ref={
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal-1",
            },
        )
    )


def _resolver_request() -> dict[str, object]:
    """Build one canonical task-resolution request."""

    return {
        "schema_version": "resolver_capability_request.v1",
        "capability_kind": "task_resolution_request",
        "objective": "Find one bounded fact.",
        "reason": "The response needs grounded evidence.",
        "priority": "now",
        "goal_continuation_ref": _continuation_ref(),
    }


def _replacement_from(session_value: session.ChainSessionV1) -> dict[str, object]:
    """Return the authoritative replacement state from the last output."""

    assert session_value.last_output is not None
    state_update = session_value.last_output["state_update"]
    assert isinstance(state_update, dict)
    replacement = state_update["replacement_state"]
    assert isinstance(replacement, dict)
    return replacement


def _output(
    payload: dict[str, object],
    *,
    replacement: dict[str, object] | None = None,
    goal_progress: dict[str, object] | None = None,
    include_goal_progress: bool = False,
    resolver_requests: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a minimal accepted output with optional recurrence carriers."""

    try:
        state_update = build_state_update(
            payload["mutable_state"],
            replacement or payload["mutable_state"],
        )
    except KeyError:
        state_update = {
            "expected_previous_state": payload["mutable_state"],
            "replacement_state": replacement or payload["mutable_state"],
        }
    output: dict[str, object] = {"state_update": state_update}
    if include_goal_progress:
        output["resolver_goal_progress"] = goal_progress
    if resolver_requests is not None:
        output["resolver_requests"] = resolver_requests
    return output


def _multi_entity_state(*, reverse: bool = False) -> dict[str, object]:
    """Build a valid state whose entity and activation rows have order variance."""

    goals = [
        {
            "entity_id": "goal-2",
            "created_at": "2026-08-20T00:02:00Z",
            "description": "Second goal",
        },
        {
            "entity_id": "goal-1",
            "created_at": "2026-08-20T00:01:00Z",
            "description": "First goal",
        },
    ]
    activations = [
        {"emotion_id": "joy", "score": 2},
        {"emotion_id": "fear", "score": 1},
    ]
    if reverse:
        goals.reverse()
        activations.reverse()
    return {
        "state_scope": "user",
        "owner_user_id": "owner-1",
        "updated_at": "2026-08-20T00:00:00Z",
        "goals": goals,
        "threats": [],
        "active_events": [],
        "knowledge_gaps": [],
        "affect_activations": activations,
    }


def _identity_context_with_finite_scalars() -> dict[str, object]:
    """Build representative identity partitions containing finite scalars."""

    return {
        "moral_identity": {
            "core": {"coherence": 0.75},
            "personality": {"warmth": 0.6},
            "boundaries": {"firmness": 0.8},
            "self_image": {"stability": 0.9},
        },
        "existential_drive": {
            "core": {"meaning": 0.7},
            "personality": {"curiosity": 0.65},
            "self_image": {"agency": 0.55},
        },
        "relationship_social": {
            "personality": {"trust": 0.72},
            "boundaries": {"openness": 0.48},
        },
        "event_agency": {
            "personality": {"responsibility": 0.61},
            "boundaries": {"accountability": 0.83},
        },
        "goal_threat_outcome": {
            "personality": {"persistence": 0.77},
            "boundaries": {"risk": 0.42},
        },
        "goal_cognition": {
            "core": {"focus": 0.88},
            "personality": {"planning": 0.63},
            "boundaries": {"scope": 0.58},
            "self_image": {"confidence": 0.69},
        },
        "epistemic_comparison_memory": {
            "core": {"certainty": 0.51},
        },
    }


def _cold_session(
    payload: dict[str, object],
    *,
    last_output: dict[str, object] | None = None,
) -> session.ChainSessionV1:
    """Build a cold session for one input."""

    return session.create_cold_session(
        payload=payload,
        episode_id="episode-1",
        owner_identity="owner-1",
        last_output=last_output,
        ttl_seconds=60,
    )


def test_session_reattaches_exactly_and_cold_rebuilds_without_attempt_reset() -> None:
    """A matching cycle reattaches; an immutable mismatch cold-rebuilds."""

    willingness = _willingness()
    cold = _payload(cycle_index=0, willingness=willingness)
    created = _cold_session(cold, last_output=_output(cold))
    assert created.expected_cycle_index == 1
    assert created.schema_version == session.SESSION_SCHEMA
    assert created.expected_willingness_digest == session.canonical_json_digest(
        willingness
    )

    new_handle = "e2"
    reattached = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[
            dict(created.accepted_evidence[0]),
            _resolver_row(new_handle),
        ],
        willingness=willingness,
    )
    assert session.reattach_or_rebuild(
        session=created,
        payload=reattached,
    ).reattached is True

    duplicate_handle = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[
            dict(created.accepted_evidence[0]),
            _resolver_row("e1"),
        ],
        willingness=willingness,
    )
    duplicate_decision = session.reattach_or_rebuild(
        session=created,
        payload=duplicate_handle,
    )
    assert duplicate_decision.reattached is False
    assert duplicate_decision.divergent_field == "evidence_handle"

    diverged = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[
            dict(created.accepted_evidence[0]),
            _resolver_row("e2"),
        ],
        willingness=willingness,
    )
    diverged["private_continuity_context"] = "changed"
    diverged_decision = session.reattach_or_rebuild(
        session=created,
        payload=diverged,
    )
    assert diverged_decision.reattached is False
    assert diverged_decision.divergent_field == "private_continuity_context"


def test_session_digest_classifies_every_input_field_and_rejects_each_unapproved_mutation() -> None:
    """Immutable projection has every immutable field as presence plus value."""

    payload = _payload(cycle_index=0)
    digest_material = {
        field_name: {"present": field_name in payload, "value": payload.get(field_name)}
        for field_name in session._IMMUTABLE_FIELDS
    }
    digest_material["original_evidence_digest"] = (
        session.canonical_json_digest(payload["evidence"])
    )

    assert session.build_immutable_digest(payload) == (
        session.canonical_json_digest(digest_material)
    )
    assert session._IMMUTABLE_FIELDS == (
        "schema_version",
        "episode",
        "state_scope",
        "character_constraints",
        "character_identity_context",
        "character_operational_context",
        "relationship_context",
        "direct_facts",
        "available_actions",
        "available_resolver_capabilities",
        "runtime_capability_limits",
        "current_turn_relational_willingness",
        "scene_context",
        "private_continuity_context",
        "past_dialog_cognition_context",
        "group_engagement_action_context",
    )


def test_session_accepts_prior_replacement_and_rejects_other_mutable_state() -> None:
    """Only the prior validated replacement state may reattach."""

    cold = _payload(cycle_index=0)
    created = _cold_session(cold)
    created.attempt_ledger = {"serial_appraisal": 2}
    created.reanchor_used = True
    replacement = {
        "state_scope": "user",
        "updated_at": "2026-08-20T00:01:00Z",
    }
    advanced = session.advance_session_after_output(
        session=created,
        payload=cold,
        output={
            "state_update": {
                "expected_previous_state": cold["mutable_state"],
                "replacement_state": replacement,
            }
        },
    )
    assert advanced.last_output is not None
    assert advanced.last_output["state_update"]["replacement_state"] == replacement
    assert advanced.expected_mutable_state_digest == session.canonical_json_digest(
        replacement
    )
    assert advanced.attempt_ledger == {"serial_appraisal": 2}
    assert advanced.reanchor_used is True

    wrong_state = {
        "state_scope": "user",
        "updated_at": "2026-08-20T00:02:00Z",
    }
    diverged = _payload(
        cycle_index=1,
        mutable_state=wrong_state,
        evidence=[
            dict(created.accepted_evidence[0]),
            _resolver_row("e2"),
        ],
    )
    assert session.reattach_or_rebuild(
        session=advanced,
        payload=diverged,
    ).reattached is False


def test_session_cycle_index_accepts_zero_one_two_and_rejects_repeated_skipped_or_out_of_order() -> None:
    """The stored index is the next admissible input and never offsets twice."""

    cold = _payload(cycle_index=0)
    created = _cold_session(cold)
    assert created.expected_cycle_index == 1

    replacement = {
        "state_scope": "user",
        "updated_at": "2026-08-20T00:01:00Z",
    }
    advanced = session.advance_session_after_output(
        session=created,
        payload=cold,
        output={
            "state_update": {
                "expected_previous_state": cold["mutable_state"],
                "replacement_state": replacement,
            }
        },
    )
    assert advanced.expected_cycle_index == 1

    for bad_index in (0, 2):
        diverged = _payload(
            cycle_index=bad_index,
            mutable_state=replacement,
            evidence=[
                dict(created.accepted_evidence[0]),
                _resolver_row("e2"),
            ],
        )
        decision = session.reattach_or_rebuild(
            session=advanced,
            payload=diverged,
        )
        assert decision.reattached is False
        assert decision.divergent_field == "resolver_cycle_index"


def test_cold_session_canonicalizes_unsorted_state_and_rejects_semantic_change() -> None:
    """Cold CAS accepts V2-equivalent order and rejects changed content."""

    unsorted_state = _multi_entity_state()
    payload = _payload(cycle_index=0, mutable_state=unsorted_state)
    replacement = _multi_entity_state(reverse=True)
    replacement["updated_at"] = "2026-08-20T00:01:00Z"
    output = {
        "state_update": build_state_update(unsorted_state, replacement),
    }
    canonical_initial = build_state_update(
        unsorted_state,
        unsorted_state,
    )["expected_previous_state"]

    created = _cold_session(payload)
    created_with_output = session.create_cold_session(
        payload=payload,
        episode_id="episode-1",
        owner_identity="owner-1",
        last_output=output,
        ttl_seconds=60,
    )
    assert created.expected_mutable_state_digest == session.canonical_json_digest(
        canonical_initial
    )
    assert created_with_output.last_output == output
    assert created_with_output.expected_mutable_state_digest == (
        session.canonical_json_digest(
            output["state_update"]["replacement_state"]
        )
    )

    changed_payload = copy.deepcopy(payload)
    changed_state = changed_payload["mutable_state"]
    assert isinstance(changed_state, dict)
    changed_state["goals"][0]["description"] = "Different goal"
    with pytest.raises(
        session.SessionContractError,
        match="expected_previous_state",
    ):
        session.create_cold_session(
            payload=changed_payload,
            episode_id="episode-1",
            owner_identity="owner-1",
            last_output=output,
            ttl_seconds=60,
        )


def test_advance_session_canonicalizes_unsorted_state_and_rejects_semantic_change() -> None:
    """Advance CAS uses canonical order while retaining semantic equality."""

    unsorted_state = _multi_entity_state()
    payload = _payload(cycle_index=0, mutable_state=unsorted_state)
    created = _cold_session(payload)
    replacement = _multi_entity_state(reverse=True)
    replacement["updated_at"] = "2026-08-20T00:01:00Z"
    output = {
        "state_update": build_state_update(unsorted_state, replacement),
    }

    advanced = session.advance_session_after_output(
        session=created,
        payload=payload,
        output=output,
    )
    assert advanced.last_output == output
    assert advanced.expected_mutable_state_digest == (
        session.canonical_json_digest(
            output["state_update"]["replacement_state"]
        )
    )

    changed_payload = copy.deepcopy(payload)
    changed_state = changed_payload["mutable_state"]
    assert isinstance(changed_state, dict)
    changed_state["goals"][0]["description"] = "Different goal"
    with pytest.raises(
        session.SessionContractError,
        match="incoming mutable_state",
    ):
        session.advance_session_after_output(
            session=created,
            payload=changed_payload,
            output=output,
        )


def test_session_carrier_has_exact_schema_and_requires_explicit_positive_ttl() -> None:
    """The process-local carrier exposes only the sealed fields and TTL input."""

    field_names = tuple(field.name for field in dataclasses.fields(session.ChainSessionV1))
    assert field_names == (
        "schema_version",
        "session_key_digest",
        "episode_id_digest",
        "scope",
        "immutable_input_digest",
        "original_evidence_digest",
        "expected_mutable_state_digest",
        "expected_willingness_digest",
        "expected_cycle_index",
        "accepted_messages",
        "accepted_products",
        "accepted_evidence",
        "current_roster",
        "attempt_ledger",
        "token_ledger",
        "last_cycle_delta_digest",
        "reanchor_used",
        "last_output",
        "created_monotonic",
        "last_used_monotonic",
        "expires_monotonic",
        "owner_token",
    )
    ttl_parameter = inspect.signature(session.create_cold_session).parameters[
        "ttl_seconds"
    ]
    assert ttl_parameter.default is inspect.Parameter.empty
    payload = _payload(cycle_index=0)
    for ttl in (0, -1, math.inf):
        with pytest.raises(session.SessionContractError, match="TTL"):
            session.create_cold_session(
                payload=payload,
                episode_id="episode-1",
                owner_identity="owner-1",
                ttl_seconds=ttl,
            )


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        {"finite": 1.25},
        {"nan": float("nan")},
        {"infinite": float("inf")},
        {"tuple": (1, 2)},
        {"set": {1, 2}},
    ],
)
def test_canonical_json_digest_rejects_non_json_values_before_hashing(
    value: object,
) -> None:
    """The digest primitive admits only finite JSON-native values."""

    with pytest.raises(session.SessionContractError, match="canonical JSON"):
        session.canonical_json_digest(value)


def test_canonical_json_digest_is_compact_utf8_sorted_lowercase_sha256() -> None:
    """Canonical digest settings remain byte-stable and lowercase."""

    value = {"z": "value", "a": [1, True, None]}
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    assert session.canonical_json_digest(value) == expected
    assert session.canonical_json_digest(value).islower()


def test_identity_projection_accepts_finite_scalars_without_float_collisions() -> None:
    """Validated identity scalars survive session reattachment distinctly."""

    identity_context = _identity_context_with_finite_scalars()
    cold = _payload(cycle_index=0)
    cold["character_identity_context"] = identity_context
    created = _cold_session(cold, last_output=_output(cold))
    recurrence = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), _resolver_row()],
    )
    recurrence["character_identity_context"] = copy.deepcopy(
        identity_context
    )
    assert session.reattach_or_rebuild(
        session=created,
        payload=recurrence,
    ).reattached is True

    literal_marker_context = copy.deepcopy(identity_context)
    moral_core = literal_marker_context["moral_identity"]["core"]
    assert isinstance(moral_core, dict)
    moral_core["coherence"] = {
        "__canonical_float__": "0x1.8000000000000p-1",
    }
    literal_marker_payload = _payload(cycle_index=0)
    literal_marker_payload["character_identity_context"] = (
        literal_marker_context
    )
    assert session.build_immutable_digest(cold) != (
        session.build_immutable_digest(literal_marker_payload)
    )

    changed_identity = copy.deepcopy(identity_context)
    changed_core = changed_identity["moral_identity"]["core"]
    assert isinstance(changed_core, dict)
    changed_core["coherence"] = 0.76
    changed_payload = dict(recurrence)
    changed_payload["character_identity_context"] = changed_identity
    assert session.reattach_or_rebuild(
        session=created,
        payload=changed_payload,
    ).divergent_field == "character_identity_context"


def test_identity_projection_rejects_nonfinite_and_nonidentity_floats() -> None:
    """Only validated finite identity scalars receive projection encoding."""

    nonfinite_identity = _payload(cycle_index=0)
    nonfinite_identity["character_identity_context"] = (
        _identity_context_with_finite_scalars()
    )
    nonfinite_core = nonfinite_identity["character_identity_context"][
        "moral_identity"
    ]["core"]
    assert isinstance(nonfinite_core, dict)
    nonfinite_core["coherence"] = float("nan")
    with pytest.raises(
        session.SessionContractError,
        match="non-finite",
    ):
        session.build_immutable_digest(nonfinite_identity)

    nonidentity_float = _payload(cycle_index=0)
    nonidentity_float["character_constraints"] = {"threshold": 0.5}
    with pytest.raises(
        session.SessionContractError,
        match="floating-point",
    ):
        session.build_immutable_digest(nonidentity_float)


def test_reattachment_without_authoritative_last_output_fails_closed() -> None:
    """A direct cold carrier remains constructible but cannot reattach safely."""

    created = _cold_session(_payload(cycle_index=0))
    payload = _payload(
        cycle_index=1,
        mutable_state={
            "state_scope": "user",
            "updated_at": "2026-08-20T00:00:00Z",
        },
        evidence=[dict(created.accepted_evidence[0]), _resolver_row()],
    )
    decision = session.reattach_or_rebuild(
        session=created,
        payload=payload,
    )
    assert decision.reattached is False
    assert decision.divergent_field == "last_output"


def test_immutable_optional_presence_difference_reports_exact_field() -> None:
    """Optional-field absence and presence are distinct exact divergences."""

    cold = _payload(cycle_index=0)
    cold["past_dialog_cognition_context"] = "accepted context"
    created = _cold_session(cold, last_output=_output(cold))
    recurrence = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), _resolver_row()],
    )
    recurrence["past_dialog_cognition_context"] = "accepted context"
    recurrence.pop("past_dialog_cognition_context")
    decision = session.reattach_or_rebuild(
        session=created,
        payload=recurrence,
    )
    assert decision.reattached is False
    assert decision.divergent_field == "past_dialog_cognition_context"


def test_evidence_prefix_source_id_and_next_canonical_handle_are_sealed() -> None:
    """Only one novel resolver observation may follow the canonical prefix."""

    cold = _payload(cycle_index=0)
    created = _cold_session(cold, last_output=_output(cold))
    replacement = _replacement_from(created)

    def decide(new_row: dict[str, object]) -> session.ReattachmentDecision:
        return session.reattach_or_rebuild(
            session=created,
            payload=_payload(
                cycle_index=1,
                mutable_state=replacement,
                evidence=[dict(created.accepted_evidence[0]), new_row],
            ),
        )

    mutated_prefix = dict(created.accepted_evidence[0])
    mutated_prefix["evidence_handle"] = "changed"
    mutated_decision = session.reattach_or_rebuild(
        session=created,
        payload=_payload(
            cycle_index=1,
            mutable_state=replacement,
            evidence=[mutated_prefix, _resolver_row()],
        ),
    )
    assert mutated_decision.divergent_field == "evidence_prefix"
    assert decide(_resolver_row("e3")).divergent_field == "evidence_handle"
    assert decide(_resolver_row("e2", source_id="s1")).divergent_field == (
        "evidence_source_id"
    )
    assert decide(_resolver_row("e2", source_id="")).divergent_field == (
        "evidence_source_id"
    )
    too_many = _payload(
        cycle_index=1,
        mutable_state=replacement,
        evidence=[
            dict(created.accepted_evidence[0]),
            _resolver_row(),
            _resolver_row("e3"),
        ],
    )
    assert session.reattach_or_rebuild(
        session=created,
        payload=too_many,
    ).divergent_field == "evidence_append_count"


def test_goal_progress_presence_and_canonical_value_bind_to_last_output() -> None:
    """Goal progress must preserve both explicit presence and canonical value."""

    cold = _payload(cycle_index=0)
    progress = _goal_progress()
    output = _output(
        cold,
        goal_progress=progress,
        include_goal_progress=True,
    )
    created = _cold_session(cold, last_output=output)
    base = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), _resolver_row()],
    )
    base["resolver_goal_progress"] = dict(progress)
    assert session.reattach_or_rebuild(
        session=created,
        payload=base,
    ).reattached is True

    absent = dict(base)
    absent.pop("resolver_goal_progress")
    assert session.reattach_or_rebuild(
        session=created,
        payload=absent,
    ).divergent_field == "resolver_goal_progress"

    changed = dict(base)
    changed_progress = dict(progress)
    changed_progress["current_focus"] = "Different accepted focus"
    changed["resolver_goal_progress"] = changed_progress
    assert session.reattach_or_rebuild(
        session=created,
        payload=changed,
    ).divergent_field == "resolver_goal_progress"


def test_relational_willingness_presence_and_digest_remain_immutable() -> None:
    """The recurrence carrier keeps relational willingness presence and digest."""

    willingness = _willingness()
    cold = _payload(cycle_index=0, willingness=willingness)
    created = _cold_session(cold, last_output=_output(cold))
    recurrence = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), _resolver_row()],
    )
    missing_decision = session.reattach_or_rebuild(
        session=created,
        payload=recurrence,
    )
    assert missing_decision.reattached is False
    assert missing_decision.divergent_field == (
        "current_turn_relational_willingness"
    )

    changed = dict(recurrence)
    changed["current_turn_relational_willingness"] = dict(willingness)
    changed_carrier = changed["current_turn_relational_willingness"]
    assert isinstance(changed_carrier, dict)
    changed_decision = dict(changed_carrier["decision"])
    changed_decision["reason"] = "different evidence"
    changed_carrier["decision"] = changed_decision
    assert session.reattach_or_rebuild(
        session=created,
        payload=changed,
    ).divergent_field == "current_turn_relational_willingness"


def test_dependency_binds_selected_request_goal_continuation_and_observation() -> None:
    """Required evidence depends on the prior request and appended observation."""

    cold = _payload(cycle_index=0)
    request = _resolver_request()
    output = _output(cold, resolver_requests=[request])
    created = _cold_session(cold, last_output=output)
    observation_id = "observation-1"
    continuation = _continuation_ref()
    new_row = _resolver_row("e2", source_id=observation_id)
    dependency = {
        "schema_version": "required_resolver_evidence_dependency.v1",
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": observation_id,
        "prompt_safe_observation_handle": "resolver_observation_0_1",
        "capability_kind": "task_resolution_request",
        "state": "complete",
        "evidence_handles": ["resolver-evidence-1"],
        "remaining_needs": [],
        "goal_continuation_ref": continuation,
    }
    recurrence = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), new_row],
    )
    recurrence["required_resolver_evidence_dependency"] = dependency
    assert session.reattach_or_rebuild(
        session=created,
        payload=recurrence,
    ).reattached is True

    bad_observation = dict(recurrence)
    bad_observation_dependency = dict(dependency)
    bad_observation_dependency["observation_id"] = "other-observation"
    bad_observation["required_resolver_evidence_dependency"] = (
        bad_observation_dependency
    )
    assert session.reattach_or_rebuild(
        session=created,
        payload=bad_observation,
    ).divergent_field == "required_resolver_evidence_dependency"

    bad_request = dict(recurrence)
    bad_request_dependency = dict(dependency)
    bad_request_dependency["accepted_request_handle"] = "resolver_request_0_2"
    bad_request["required_resolver_evidence_dependency"] = bad_request_dependency
    assert session.reattach_or_rebuild(
        session=created,
        payload=bad_request,
    ).divergent_field == "required_resolver_evidence_dependency"


def test_pending_resume_validator_and_explicit_row_bindings_are_enforced() -> None:
    """Pending resume references remain bound to the appended observation row."""

    cold = _payload(cycle_index=0)
    created = _cold_session(cold, last_output=_output(cold))
    resume_id = "resume-1"
    pending = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": resume_id,
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
        "source_message_id": "message-1",
        "prompt_safe_original_goal": "Need one detail",
        "prompt_safe_question": "Which detail?",
        "prompt_safe_approval_summary": "",
        "created_at_utc": "2026-08-20T00:00:00Z",
        "expires_at_utc": "2026-08-21T00:00:00Z",
        "observation_id": "observation-1",
    }
    new_row = _resolver_row("e2", source_id="observation-1")
    new_row["pending_resume_id"] = resume_id
    recurrence = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), new_row],
    )
    recurrence["pending_resolver_resume"] = pending
    assert session.reattach_or_rebuild(
        session=created,
        payload=recurrence,
    ).reattached is True

    bad_pending = dict(recurrence)
    bad_pending_value = dict(pending)
    bad_pending_value["observation_id"] = "other-observation"
    bad_pending["pending_resolver_resume"] = bad_pending_value
    assert session.reattach_or_rebuild(
        session=created,
        payload=bad_pending,
    ).divergent_field == "pending_resolver_resume"

    bad_row = dict(recurrence)
    bad_row_value = dict(new_row)
    bad_row_value["pending_resume_id"] = "other-resume"
    bad_row["evidence"] = [dict(created.accepted_evidence[0]), bad_row_value]
    assert session.reattach_or_rebuild(
        session=created,
        payload=bad_row,
    ).divergent_field == "pending_resolver_resume"


def test_build_cycle_delta_keeps_explicit_presence_and_hardened_digest() -> None:
    """Cycle delta includes only the sealed fields with explicit presence."""

    cold = _payload(cycle_index=0)
    created = _cold_session(cold, last_output=_output(cold))
    recurrence = _payload(
        cycle_index=1,
        mutable_state=_replacement_from(created),
        evidence=[dict(created.accepted_evidence[0]), _resolver_row()],
    )
    recurrence["resolver_context"] = "updated resolver context"
    expected_projection: dict[str, object] = {
        "new_evidence": {
            "present": True,
            "value": recurrence["evidence"][-1],
        },
    }
    expected_projection.update({
        field_name: {
            "present": field_name in recurrence,
            "value": recurrence.get(field_name),
        }
        for field_name in session._CYCLE_FIELDS
    })
    assert session.build_cycle_delta(
        session=created,
        payload=recurrence,
    ) == session.canonical_json_digest(expected_projection)
