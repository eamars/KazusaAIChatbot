"""Deterministic tests for the Cognition V3 chain-session registry."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v3 import session


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
    handle: str = "resolver-observation-alpha",
) -> dict[str, object]:
    """Build one resolver-observation row for session matching."""

    return {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": "resolver_observation",
            "source_id": f"source-{handle}",
        },
    }


def _cold_session(
    payload: dict[str, object],
) -> session.ChainSessionV1:
    """Build a cold session for one input."""

    return session.create_cold_session(
        payload=payload,
        episode_id="episode-1",
        owner_identity="owner-1",
    )


def test_session_reattaches_exactly_and_cold_rebuilds_without_attempt_reset() -> None:
    """A matching cycle reattaches; an immutable mismatch cold-rebuilds."""

    willingness = {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": "episode-1",
        "branch_id": "ordinary_response",
        "decision": {"stance": "not_applicable"},
    }
    cold = _payload(cycle_index=0, willingness=willingness)
    created = _cold_session(cold)
    assert created.expected_cycle_index == 1
    assert created.expected_relational_willingness == willingness

    new_handle = "resolver-observation-alpha"
    reattached = _payload(
        cycle_index=1,
        mutable_state=created.expected_mutable_state,
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
        mutable_state=created.expected_mutable_state,
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
        mutable_state=created.expected_mutable_state,
        evidence=[
            dict(created.accepted_evidence[0]),
            _resolver_row("e2"),
        ],
        willingness=willingness,
    )
    diverged["private_continuity_context"] = "changed"
    assert session.reattach_or_rebuild(
        session=created,
        payload=diverged,
    ).reattached is False


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
    assert advanced.expected_mutable_state == replacement
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
