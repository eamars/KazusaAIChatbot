"""Deterministic tests for Cognition V2 operational-context fitting."""

from __future__ import annotations

from copy import deepcopy
import json

from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    fit_relationship_operational_context,
    serialized_character_count,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)


NOW = "2026-08-06T00:00:00Z"


def _relationship_context(
    *,
    first_summary_length: int,
    second_summary_length: int,
) -> dict[str, object]:
    """Build one relationship packet with controlled summary lengths."""

    state = build_acquaintance_user_state(
        global_user_id="context-limit-user",
        updated_at=NOW,
    )
    relationship = state["relationship"]
    summaries = [
        "a" * first_summary_length,
        "b" * second_summary_length,
    ]
    return {
        "schema_version": "relationship_operational_context.v1",
        "relationship_id": relationship["relationship_id"],
        "axes": {
            field_name: relationship[field_name]
            for field_name in (
                "familiarity",
                "positive_regard",
                "trust",
                "attachment",
                "desired_closeness",
                "perceived_closeness",
                "care",
                "boundary_safety",
                "exclusivity",
                "unresolved_injury",
                "salience",
            )
        },
        "causal_context": [
            {
                "entity_kind": "event",
                "semantic_summary": summaries[0],
                "salience": "极高",
                "lifecycle": "active",
                "freshness": "即时",
            },
            {
                "entity_kind": "event",
                "semantic_summary": summaries[1],
                "salience": "高",
                "lifecycle": "active",
                "freshness": "即时",
            },
        ],
        "affect": [],
        "relationship_freshness": "即时",
        "evidence_freshness": "无证据",
    }


def test_relationship_incident_shape_fits_and_preserves_required_fields() -> None:
    """The 914-character incident shape fits without changing identity data."""

    context = _relationship_context(
        first_summary_length=160,
        second_summary_length=155,
    )
    assert serialized_character_count(context) == 914
    original = deepcopy(context)

    result = fit_relationship_operational_context(context)

    assert result.original_size == 914
    assert result.final_size <= 900
    assert result.payload["relationship_id"] == original["relationship_id"]
    assert result.payload["axes"] == original["axes"]
    assert result.payload["relationship_freshness"] == (
        original["relationship_freshness"]
    )
    assert result.payload["evidence_freshness"] == (
        original["evidence_freshness"]
    )
    assert result.trimmed_fields == (
        "causal_context[1].semantic_summary",
    )
    assert result.fallback_level == 1
    assert context == original


def test_relationship_fit_handles_exact_limit_and_one_character_overflow() -> None:
    """The canonical serializer distinguishes the 900 and 901 boundaries."""

    minimal = _relationship_context(
        first_summary_length=1,
        second_summary_length=1,
    )
    minimal_size = serialized_character_count(minimal)

    exact = _relationship_context(
        first_summary_length=160,
        second_summary_length=900 - minimal_size - 158,
    )
    one_over = _relationship_context(
        first_summary_length=160,
        second_summary_length=901 - minimal_size - 158,
    )
    assert serialized_character_count(exact) == 900
    assert serialized_character_count(one_over) == 901

    exact_result = fit_relationship_operational_context(exact)
    over_result = fit_relationship_operational_context(one_over)

    assert exact_result.final_size == 900
    assert exact_result.trimmed_fields == ()
    assert over_result.original_size == 901
    assert over_result.final_size == 900
    assert over_result.trimmed_fields
    assert over_result.fallback_level == 1


def test_relationship_fit_drops_rows_in_stable_priority_order() -> None:
    """Causal rows are reduced before lower-priority affect rows."""

    context = _relationship_context(
        first_summary_length=80,
        second_summary_length=80,
    )
    context["affect"] = [
        {
            "emotion_id": "emotion-a",
            "intensity": "high",
            "phase": "active",
            "trend": "stable",
            "freshness": "f" * 300,
        },
        {
            "emotion_id": "emotion-b",
            "intensity": "high",
            "phase": "active",
            "trend": "stable",
            "freshness": "g" * 300,
        },
    ]
    original = deepcopy(context)

    result = fit_relationship_operational_context(context)

    assert result.final_size <= 900
    assert result.dropped_rows
    assert result.dropped_rows[0] == "causal_context"
    assert context == original


def test_serialized_character_count_uses_decoded_unicode_characters() -> None:
    """Size accounting follows compact JSON characters rather than bytes."""

    value = {"text": "汉字e\u0301😀"}
    expected = len(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))

    assert serialized_character_count(value) == expected
