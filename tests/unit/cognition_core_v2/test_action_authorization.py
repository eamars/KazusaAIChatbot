"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    derive_action_route,
    validate_authorization_decisions,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.action_authorization"
EXPECTED_SYMBOLS = ["derive_action_route"]


def test_action_authorization_exposes_owned_contract() -> None:
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


def test_public_authorization_validator_requires_exact_boolean_coverage() -> None:
    """The public validator preserves exact candidate and boolean checks."""

    decisions = validate_authorization_decisions(
        {"decisions": {"c1": True, "c2": False}},
        candidate_handles=["c1", "c2"],
    )

    assert decisions == {"c1": True, "c2": False}

    with pytest.raises(
        ValueError,
        match="action authorization decision must be boolean",
    ):
        validate_authorization_decisions(
            {"decisions": {"c1": 1, "c2": False}},
            candidate_handles=["c1", "c2"],
        )


def test_group_self_cognition_response_decision_derives_speech_or_silence() -> None:
    """Targetless think-only routing follows the semantic response decision."""

    episode = {
        "trigger_source": "self_cognition",
        "output_mode": "think_only",
        "target_scope": {
            "channel_type": "group",
            "current_global_user_id": "",
            "current_platform_user_id": "",
        },
    }
    silent_response = {
        "decision": "stay_silent",
    }
    proposal_response = {
        "decision": "propose_visible_reply",
    }

    assert derive_action_route(
        episode=episode,
        primary_bid={"evidence_handles": ["e1"]},
        action_requests=[],
        resolver_requests=[],
        self_cognition_response=silent_response,
    ) == "silence"
    assert derive_action_route(
        episode=episode,
        primary_bid={"evidence_handles": ["e1"]},
        action_requests=[],
        resolver_requests=[],
        self_cognition_response=proposal_response,
    ) == "speech"
    assert derive_action_route(
        episode=episode,
        primary_bid={"evidence_handles": ["e1"]},
        action_requests=[{"action_kind": "unrelated_capability"}],
        resolver_requests=[],
        self_cognition_response=proposal_response,
    ) == "speech"
