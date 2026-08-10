"""Focused contract tests for the native character operational projection."""

from __future__ import annotations

from copy import deepcopy

from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    MAX_CHARACTER_OPERATIONAL_CONTEXT_CHARS,
    canonical_digest,
    fit_character_operational_context,
    serialized_character_count,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_character_operational_state,
    project_relationship_axis,
    project_relationship_context,
    select_character_operational_context,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_character_elapsed_decay,
)


NOW = "2026-08-02T00:00:00Z"
LATER = "2026-08-02T06:00:00Z"


def _event(event_id: str, *, outcome_impact: int = 0) -> dict[str, object]:
    """Build one source-free operational event for projection tests."""

    return {
        "entity_id": event_id,
        "description": "operational event",
        "salience": 80,
        "role_refs": [],
        "evidence_refs": [
            {
                "source_kind": "episode",
                "source_id": "episode-redacted",
                "occurred_at": NOW,
                "semantic_summary": "closed operational event",
            }
        ],
        "created_at": NOW,
        "updated_at": NOW,
        "status": "active",
        "outcome_impact": outcome_impact,
        "responsibility": 70,
        "intentionality": 70,
        "harm": 70,
        "unfairness": 70,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 80,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
    }


def _activation() -> dict[str, object]:
    """Build one valid active activation rooted in the closed event."""

    root = {
        "scope": "character",
        "kind": "event",
        "entity_id": "event:pressure",
    }
    return {
        "activation_id": "emotion:anger",
        "emotion_id": "anger",
        "primary_root": root,
        "root_refs": [root],
        "phase": "active",
        "score": 72,
        "peak_score": 72,
        "trend": "stable",
        "cause_status": "active",
        "started_at": NOW,
        "updated_at": NOW,
        "last_reinforced_at": NOW,
    }


def test_elapsed_decay_is_pure_and_uses_one_effective_base() -> None:
    """Decay returns a copy and leaves persisted state untouched."""

    state = build_character_production_state(updated_at=NOW)
    state["active_events"] = [_event("event:pressure")]
    state["affect_activations"] = [_activation()]

    faded = apply_character_elapsed_decay(state, elapsed_seconds=6 * 3600)

    assert faded is not state
    assert state["updated_at"] == NOW
    assert faded["updated_at"] == NOW
    assert faded["affect_activations"][0]["score"] < 72


def test_full_view_and_consumer_context_have_distinct_digests_and_caps() -> None:
    """The console view is complete while model context is role-selected."""

    state = build_character_production_state(updated_at=NOW)
    state["active_events"] = [_event("event:pressure")]
    state["affect_activations"] = [_activation()]

    view = project_character_operational_state(state, effective_at=LATER)
    context = select_character_operational_context(
        view,
        consumer_role="goal",
    )

    assert view["source_updated_at"] == NOW
    assert view["effective_at"] == LATER
    assert view["source_digest"]
    assert view["view_digest"]
    assert view["source_digest"] != view["view_digest"]
    assert len(view["affect"]) <= 21
    assert len(view["pressures"]) <= 8
    assert context["consumer_role"] == "goal"
    assert len(context["affect"]) <= 3
    assert len(context["pressures"]) <= 4
    serialized = repr(context)
    assert "episode-redacted" not in serialized
    assert "channel" not in serialized


def test_character_context_fit_accounts_for_final_digest() -> None:
    """The final digest-bearing packet remains within its complete cap."""

    affect_row = {
        "emotion_id": "emotion",
        "intensity": "high",
        "phase": "active",
        "trend": "stable",
        "root_kind": "event",
        "cause_class": "general_activation",
        "freshness": "f" * 300,
    }
    pressure_row = {
        "kind": "event",
        "salience": "high",
        "lifecycle": "active",
        "cause_class": "general_activation",
        "freshness": "p" * 300,
    }
    state_view = {
        "schema_version": "character_operational_state_view.v1",
        "source_updated_at": NOW,
        "effective_at": LATER,
        "view_digest": "v" * 64,
        "affect": [deepcopy(affect_row) for _ in range(3)],
        "pressures": [deepcopy(pressure_row) for _ in range(4)],
    }
    original = deepcopy(state_view)

    context = select_character_operational_context(
        state_view,
        consumer_role="goal",
    )
    context_body = {
        key: value
        for key, value in context.items()
        if key != "context_digest"
    }

    assert serialized_character_count(context) <= (
        MAX_CHARACTER_OPERATIONAL_CONTEXT_CHARS
    )
    assert context["context_digest"] == canonical_digest(context_body)
    assert state_view == original


def test_character_context_refit_preserves_a_valid_digest() -> None:
    """A consumer no-op fit keeps an already valid context digest stable."""

    state = build_character_production_state(updated_at=NOW)
    view = project_character_operational_state(state, effective_at=LATER)
    context = select_character_operational_context(
        view,
        consumer_role="goal",
    )

    refit = fit_character_operational_context(context)

    assert refit.payload == context
    assert refit.trimmed_fields == ()
    assert refit.dropped_rows == ()


def test_relationship_projection_keeps_causes_separate_from_character_view() -> None:
    """Relationship axes and causes remain user-scoped and source-safe."""

    user_state = build_acquaintance_user_state(
        global_user_id="user-a",
        updated_at=NOW,
    )
    user_state["relationship"].update({
        "attachment": 70,
        "care": 65,
        "positive_regard": 60,
        "trust": 55,
        "salience": 80,
        "updated_at": NOW,
    })
    projected = project_relationship_context(
        user_state,
        effective_at=LATER,
    )

    assert projected["relationship_id"] == "relationship:user:user-a"
    assert projected["axes"]["attachment"] == 70
    assert len(projected["causal_context"]) <= 2
    assert len(projected["affect"]) <= 2
    assert "user-a" not in repr(projected["causal_context"])


def test_projection_digests_change_when_native_state_changes() -> None:
    """Digest fields identify a changed redacted native view."""

    state = build_character_production_state(updated_at=NOW)
    state["active_events"] = [_event("event:pressure")]
    state["affect_activations"] = [_activation()]
    changed = deepcopy(state)
    changed["drives"]["safety"]["pressure"] = 80

    first = project_character_operational_state(state, effective_at=LATER)
    second = project_character_operational_state(changed, effective_at=LATER)

    assert first["view_digest"] != second["view_digest"]


def test_relationship_axis_prompt_projection_is_domain_specific() -> None:
    """Zero trust and zero boundary history retain different meanings."""

    trust = project_relationship_axis("trust", 0)
    boundary_safety = project_relationship_axis("boundary_safety", 0)

    assert trust != boundary_safety
    assert "中性或混合" not in trust
    assert "中性或混合" not in boundary_safety
