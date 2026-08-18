"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py."""

from __future__ import annotations

from importlib import import_module

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    apply_semantic_deltas,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.transition_guards"
EXPECTED_SYMBOLS = ["apply_semantic_deltas", "compare_event"]
_TIMESTAMP = "2026-08-18T00:00:00Z"


def _state(global_user_id: str) -> dict[str, object]:
    """Build a state carrying the delta evidence source used below."""

    state = validate_cognition_state(
        build_acquaintance_user_state(
            global_user_id=global_user_id,
            updated_at=_TIMESTAMP,
        )
    )
    state["relationship"]["evidence_refs"] = [{
        "source_kind": "episode",
        "source_id": "source:episode",
        "occurred_at": _TIMESTAMP,
        "semantic_summary": "A bounded transition source.",
    }]
    return state


def test_transition_guards_exposes_owned_contract() -> None:
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


def test_apply_semantic_deltas_returns_authoritative_receipts() -> None:
    """Return the exact accepted delta receipt contract."""

    state = _state("transition-receipt-user")
    result = apply_semantic_deltas(
        state,
        [{
            "target_path": "relationship.trust",
            "delta": 10,
            "evidence_handles": ["source:episode"],
            "reason": "A current episode supports a trust change.",
        }],
    )

    assert set(result) == {
        "updated_state",
        "accepted_delta_receipts",
        "rejected_delta_receipts",
    }
    assert result["accepted_delta_receipts"][0] == {
        "target_path": "relationship.trust",
        "relationship_axis": "trust",
        "requested_delta": 10,
        "applied_delta": 10,
        "previous_value": 0,
        "next_value": 10,
        "evidence_refs": state["relationship"]["evidence_refs"],
        "duplicate_disposition": "unique",
    }


def test_duplicate_delta_is_rejected_from_receipts() -> None:
    """Reject duplicate target paths deterministically and without mutation."""

    state = _state("transition-duplicate-user")
    result = apply_semantic_deltas(
        state,
        [
            {
                "target_path": "relationship.trust",
                "delta": 3,
                "evidence_handles": ["source:episode"],
                "reason": "first",
            },
            {
                "target_path": "relationship.trust",
                "delta": 4,
                "evidence_handles": ["source:episode"],
                "reason": "duplicate",
            },
        ],
    )

    assert result["updated_state"]["relationship"]["trust"] == 0
    assert result["accepted_delta_receipts"] == []
    assert result["rejected_delta_receipts"] == [{
        "target_path": "relationship.trust",
        "disposition": "duplicate_target",
    }]


def test_relationship_receipt_records_clamped_applied_delta() -> None:
    """Record the bounded value change rather than the requested delta."""

    state = _state("transition-clamp-user")
    state["relationship"]["trust"] = 95
    result = apply_semantic_deltas(
        state,
        [{
            "target_path": "relationship.trust",
            "delta": 10,
            "evidence_handles": ["source:episode"],
            "reason": "A bounded trust increase.",
        }],
    )

    receipt = result["accepted_delta_receipts"][0]
    assert receipt["requested_delta"] == 10
    assert receipt["applied_delta"] == 5
    assert receipt["previous_value"] == 95
    assert receipt["next_value"] == 100
