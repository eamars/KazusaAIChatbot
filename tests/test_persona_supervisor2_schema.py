"""Tests for persona_supervisor2_schema.py — GlobalPersonaState structure."""

from __future__ import annotations

import typing

from kazusa_ai_chatbot.consolidation.schema import (
    normalize_subjective_appraisals,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState


class TestGlobalPersonaState:
    def test_is_typed_dict(self):
        assert issubclass(GlobalPersonaState, dict)

    def test_has_character_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "character_profile" in hints

    def test_has_input_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        input_fields = [
            "storage_timestamp_utc", "local_time_context", "user_input",
            "platform", "platform_user_id",
            "global_user_id", "user_name", "user_profile", "platform_bot_id",
            "chat_history_wide", "chat_history_recent", "reply_context", "indirect_speech_context", "channel_topic",
            "prompt_message_context",
        ]
        for field in input_fields:
            assert field in hints, f"Missing input field: {field}"

    def test_has_debug_modes_field(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "debug_modes" in hints, "Missing field: debug_modes"

    def test_has_stage_output_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        # Decontextualizer
        assert "decontextualized_input" in hints
        assert "referents" in hints
        # RAG
        assert "rag_result" in hints
        # Cognition
        assert "internal_monologue" in hints
        assert "cognition_core_output" in hints
        assert "text_surface_output_v2" in hints
        assert "action_specs" in hints
        # Dialog
        assert "final_dialog" in hints
        assert "target_addressed_user_ids" in hints
        assert "target_broadcast" in hints
        # Native consolidation boundary
        assert "action_results" in hints
        assert "episode_trace" in hints
        assert "memory_lifecycle_context" in hints

    def test_has_consolidation_fields(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "new_facts" in hints
        assert "future_promises" in hints

    def test_has_cognitive_episode_field(self):
        hints = typing.get_type_hints(GlobalPersonaState)
        assert "cognitive_episode" in hints, "Missing field: cognitive_episode"


def test_normalize_subjective_appraisals_accepts_string_and_string_list() -> None:
    """Subjective appraisal boundary accepts native string payloads."""

    assert normalize_subjective_appraisals("  one appraisal  ") == ["one appraisal"]
    assert normalize_subjective_appraisals([" first ", "", "second"]) == ["first", "second"]


def test_normalize_subjective_appraisals_does_not_stringify_container_items() -> None:
    """Subjective appraisal boundary must not turn dict/list items into repr text."""

    assert normalize_subjective_appraisals([
        {"entry": "do not stringify"},
        ["do not stringify"],
        " keep me ",
    ]) == ["keep me"]
