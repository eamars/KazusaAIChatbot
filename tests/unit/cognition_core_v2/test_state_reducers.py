"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import state_reducers
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    validate_cognition_state,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.state_reducers"
EXPECTED_SYMBOLS = ["apply_relationship_maintenance", "apply_state_update"]
_TIMESTAMP = "2026-08-18T00:00:00Z"


def test_state_reducers_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def _state() -> dict[str, object]:
    """Build one validated user state for maintenance tests."""

    return validate_cognition_state(
        build_acquaintance_user_state(
            global_user_id="reducer-maintenance-user",
            updated_at=_TIMESTAMP,
        )
    )


def _receipt(
    *,
    target_path: str = "relationship.trust",
    requested_delta: int = 4,
    applied_delta: int = 4,
) -> dict[str, object]:
    """Build one authoritative accepted relationship receipt."""

    return {
        "target_path": target_path,
        "relationship_axis": target_path.split(".", maxsplit=1)[1],
        "requested_delta": requested_delta,
        "applied_delta": applied_delta,
        "previous_value": 0,
        "next_value": applied_delta,
        "evidence_refs": [],
        "duplicate_disposition": "unique",
    }


def test_relationship_familiarity_reinforces_once_per_utc_interaction_date() -> None:
    """Advance familiarity once when a new interaction date is accepted."""

    updated = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert updated["relationship"]["familiarity"] == 11


def test_relationship_familiarity_applies_same_day_bonus_once() -> None:
    """Allow one same-day evidence bonus in addition to date reinforcement."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="second",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[_receipt()],
    )
    replay = state_reducers.apply_relationship_maintenance(
        updated,
        source_episode_id="third",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[_receipt()],
    )

    assert updated["relationship"]["familiarity"] == 12
    assert replay["relationship"]["familiarity"] == 12


def test_relationship_familiarity_accepts_clamped_zero_delta_bonus() -> None:
    """Count an accepted boundary-clamped receipt as relationship evidence."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="clamped",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[
            _receipt(requested_delta=4, applied_delta=0),
        ],
    )

    assert updated["relationship"]["familiarity"] == 12


def test_relationship_familiarity_crosses_utc_date_boundary() -> None:
    """Advance familiarity again after the canonical UTC date changes."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="next-day",
        interaction_date_utc="2026-08-19",
        elapsed_seconds=0,
    )

    assert updated["relationship"]["familiarity"] == 12
    assert updated["relationship"]["relationship_maintenance"][
        "last_interaction_date_utc"
    ] == "2026-08-19"


def test_relationship_familiarity_is_idempotent_for_replayed_source() -> None:
    """Replaying the same source leaves maintenance unchanged."""

    state = _state()
    first = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="same-source",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    replay = state_reducers.apply_relationship_maintenance(
        first,
        source_episode_id="same-source",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert replay == first


def test_relationship_duplicate_source_after_intervening_episode_is_ignored() -> None:
    """Keep an active-date ledger so an old duplicate cannot reinforce."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    state = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="second",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    replay = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert replay == state


def test_relationship_maintenance_ignores_older_source_after_newer_episode() -> None:
    """Ignore stale source dates after a newer interaction is recorded."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="newer",
        interaction_date_utc="2026-08-19",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="older",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[_receipt()],
    )

    assert updated == state


def test_relationship_maintenance_rejects_source_ledger_overflow() -> None:
    """Fail closed instead of evicting active-date source identities."""

    state = _state()
    maintenance = state["relationship"]["relationship_maintenance"]
    maintenance["last_interaction_date_utc"] = "2026-08-18"
    maintenance["processed_source_ids"] = [
        f"episode:source-{index}" for index in range(256)
    ]

    with pytest.raises(CognitionStateError, match="source ledger"):
        state_reducers.apply_relationship_maintenance(
            state,
            source_episode_id="overflow",
            interaction_date_utc="2026-08-18",
            elapsed_seconds=0,
        )


def test_relationship_salience_decays_before_downstream_derivation() -> None:
    """Apply elapsed salience decay before accepted delta reinforcement."""

    state = _state()
    state["relationship"]["salience"] = 50

    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="salience",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=3600,
    )

    assert updated["relationship"]["salience"] == 46


def test_relationship_salience_uses_four_points_per_hour() -> None:
    """Keep the policy's four-point hourly salience decay exact."""

    state = _state()
    state["relationship"]["salience"] = 100

    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="two-hours",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=2 * 3600,
    )

    assert updated["relationship"]["salience"] == 92


def test_final_goal_reconciliation_removes_stale_connection_goal() -> None:
    """Remove an active connection goal after final salience crosses below 40."""

    state = _state()
    state["relationship"].update({
        "attachment": 60,
        "desired_closeness": 80,
        "perceived_closeness": 0,
        "salience": 40,
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "episode:goal-reconciliation",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "The episode carries relationship evidence.",
        }],
    })
    with_goal = state_reducers.create_deterministic_goals(
        state,
        evidence=[],
        updated_at=_TIMESTAMP,
    )
    assert any(
        goal["goal_kind"] == "relationship_connection"
        for goal in with_goal["goals"]
    )

    with_goal["relationship"]["salience"] = 39
    reconciled = state_reducers.create_deterministic_goals(
        with_goal,
        evidence=[],
        updated_at=_TIMESTAMP,
        reconcile_salience_gated_goals=True,
    )

    assert not any(
        goal["goal_kind"] == "relationship_connection"
        and goal["status"] in {"pursuing", "blocked"}
        for goal in reconciled["goals"]
    )


def test_relationship_salience_reinforces_from_strongest_unique_delta() -> None:
    """Use only the strongest unique accepted relationship delta."""

    state = _state()
    state["relationship"]["salience"] = 10

    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="strongest",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[
            _receipt(target_path="relationship.trust", applied_delta=3),
            _receipt(
                target_path="relationship.attachment",
                requested_delta=-8,
                applied_delta=-8,
            ),
        ],
    )

    assert updated["relationship"]["salience"] == 18


def test_cumulative_trial_reductions_do_not_multiply_relationship_maintenance() -> None:
    """A final accepted receipt set applies maintenance exactly once."""

    updated = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="cumulative",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[
            _receipt(target_path="relationship.trust", applied_delta=4),
            _receipt(target_path="relationship.attachment", applied_delta=9),
        ],
    )

    assert updated["relationship"]["familiarity"] == 12
    assert updated["relationship"]["salience"] == 9
