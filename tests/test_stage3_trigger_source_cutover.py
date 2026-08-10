"""Focused tests for the five grounded cognitive episode sources."""

from __future__ import annotations

import importlib
from typing import get_args


EXPECTED_TRIGGER_SOURCES = {
    "user_message",
    "internal_thought",
    "self_cognition",
    "scheduled_tick",
    "tool_result",
}


def test_trigger_source_literal_contains_only_five_grounded_sources() -> None:
    """The public trigger vocabulary must match grounded runtime events."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_episode",
    )

    assert set(get_args(module.TriggerSource)) == EXPECTED_TRIGGER_SOURCES


def test_trigger_source_registry_is_complete() -> None:
    """Every grounded source must have one registered owner and policy."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_episode",
    )
    registry = module.build_trigger_source_registry()

    assert set(registry) == EXPECTED_TRIGGER_SOURCES
    for source_name, source_spec in registry.items():
        assert source_spec["source_kind"] == source_name
        assert source_spec["owner"]
        assert source_spec["entrypoint"]
        assert source_spec["allowed_continuation_depth"] == 1


def test_public_episode_builders_cover_all_grounded_sources() -> None:
    """Each source must enter the same public episode contract."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_episode",
    )
    builder_names = {
        "build_user_message_episode",
        "build_internal_thought_episode",
        "build_self_cognition_episode",
        "build_scheduled_tick_episode",
        "build_tool_result_episode",
    }

    assert all(callable(getattr(module, name)) for name in builder_names)
