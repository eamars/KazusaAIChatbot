"""Deterministic ownership test for src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py."""

from __future__ import annotations

from importlib import import_module

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY

MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition"
EXPECTED_SYMBOLS = ["build_cognition_input_from_global_state"]


def test_persona_supervisor2_cognition_exposes_owned_contract() -> None:
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


def test_group_self_cognition_proposal_materializes_existing_speak_surface(
    monkeypatch,
) -> None:
    """A speech intention uses the existing canonical speak materializer."""

    captured: list[list[dict[str, object]]] = []

    def materialize(
        requests: list[dict[str, object]],
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        del state
        captured.append(requests)
        return [{"kind": request["capability"]} for request in requests]

    monkeypatch.setattr(
        cognition_module,
        "materialize_semantic_action_requests",
        materialize,
    )
    output = {
        "action_requests": [],
        "intention": {
            "route": "speech",
            "intention": "intervene in the current group scene",
            "target_roles": [],
            "reason": "The current scene supports a bounded intervention.",
        },
        "admitted_bid": {"evidence_handles": ["e1"]},
    }

    action_specs = cognition_module._materialize_v2_action_requests(
        output,
        {},
    )

    assert len(captured) == 2
    assert action_specs == [{"kind": SPEAK_CAPABILITY}]
    assert captured[1][0]["capability"] == SPEAK_CAPABILITY
    assert captured[1][0]["evidence_handles"] == ["e1"]
