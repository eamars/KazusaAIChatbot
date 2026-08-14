"""Deterministic ownership test for src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py."""

from __future__ import annotations

from importlib import import_module

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY

MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition"
EXPECTED_SYMBOLS = [
    "build_cognition_input_from_global_state",
    "build_scene_context_from_global_state",
]


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


def test_promoted_reflection_preserves_source_updated_at() -> None:
    """Promoted rows retain their valid source time and omit invalid rows."""

    evidence = cognition_module._promoted_reflection_evidence(
        {
            'promoted_lore': [{
                'memory_name': 'world context',
                'content': 'the setting remains stable',
                'updated_at': '2026-07-29T23:00:00Z',
            }],
            'promoted_self_guidance': [{
                'memory_name': 'tactic hint',
                'content': 'verify the current scene first',
                'updated_at': 'not-a-timestamp',
            }],
        },
        '2026-07-30T00:00:00Z',
    )

    assert len(evidence) == 1
    assert evidence[0]['evidence_ref']['occurred_at'] == (
        '2026-07-29T23:00:00Z'
    )


def test_promoted_self_guidance_is_goal_only_conditional_context() -> None:
    """Self-guidance carries conditional authority and no current fact role."""

    evidence = cognition_module._promoted_reflection_evidence(
        {
            'promoted_self_guidance': [{
                'memory_name': 'tactic hint',
                'content': 'verify the current scene first',
                'updated_at': '2026-07-29T23:00:00Z',
            }],
        },
        '2026-07-30T00:00:00Z',
    )

    assert evidence[0]['authority'] == 'conditional_character_guidance'
    assert evidence[0]['evidence_ref']['source_id'] == (
        'promoted-reflection:self_guidance:1'
    )
