"""Tests for persona_supervisor2_schema.py — GlobalPersonaState structure."""

from __future__ import annotations

import typing
import pytest

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState, CognitionState


class TestGlobalPersonaState:
    def test_is_typed_dict(self):
        assert issubclass(GlobalPersonaState, dict)

    def test_has_character_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "character_profile" in hints

    def test_has_input_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        input_fields = [
            "timestamp", "user_input", "platform", "platform_user_id",
            "global_user_id", "user_name", "user_profile", "platform_bot_id",
            "chat_history_wide", "chat_history_recent", "indirect_speech_context", "channel_topic",
        ]
        for field in input_fields:
            assert field in hints, f"Missing input field: {field}"

    def test_has_debug_modes_field(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "debug_modes" in hints, "Missing field: debug_modes"

    def test_has_stage_output_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        # Stage 0
        assert "decontexualized_input" in hints
        # Stage 1
        assert "research_facts" in hints
        assert "research_metadata" in hints
        # Stage 2
        assert "internal_monologue" in hints
        assert "action_directives" in hints
        # Stage 3
        assert "final_dialog" in hints
        # Stage 4
        assert "mood" in hints
        assert "global_vibe" in hints
        assert "reflection_summary" in hints
        assert "affinity_delta" in hints

    def test_has_consolidation_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "diary_entry" in hints
        assert "last_relationship_insight" in hints
        assert "new_facts" in hints
        assert "future_promises" in hints


class TestCognitionState:
    def test_has_core_cognition_fields(self):
        hints = typing.get_type_hints(CognitionState)
        required = [
            "character_profile",
            "user_input",
            "chat_history_recent",
            "indirect_speech_context",
            "decontexualized_input",
            "research_facts",
            "internal_monologue",
            "action_directives",
        ]
        for field in required:
            assert field in hints, f"Missing cognition field: {field}"
