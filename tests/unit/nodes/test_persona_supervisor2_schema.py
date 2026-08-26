"""Deterministic ownership test for src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py."""

from __future__ import annotations

from importlib import import_module
from typing import get_type_hints

MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_schema"
EXPECTED_SYMBOLS = ["GlobalPersonaState"]


def test_persona_supervisor2_schema_exposes_owned_contract() -> None:
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
    for state_name in ("GlobalPersonaState", "CognitionState"):
        state_type = getattr(module, state_name)
        state_hints = get_type_hints(state_type)
        assert state_hints["shared_memory_prewarm_outcome"]
        assert state_hints["public_group_scene_context"]
        assert state_hints["public_group_scene_projection_status"]
        assert state_hints["public_group_scene_projection_reason"]
