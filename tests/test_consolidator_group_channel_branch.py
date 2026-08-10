"""Tests for group-channel consolidation write separation."""

from __future__ import annotations

import inspect

from kazusa_ai_chatbot.consolidation import group_channel


def test_group_channel_module_does_not_call_user_profile_helpers() -> None:
    """Group-channel persistence must not use user-only DB helpers."""

    source_text = inspect.getsource(group_channel)

    forbidden_helpers = (
        "update_relationship_state",
        "update_semantic_relationship_projection",
        "update_user_memory_units_from_state",
    )
    for helper_name in forbidden_helpers:
        assert helper_name not in source_text
