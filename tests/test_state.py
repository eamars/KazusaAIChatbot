"""Tests for state.py — DiscordProcessState TypedDict structure."""

from __future__ import annotations

import typing

from kazusa_ai_chatbot.state import DiscordProcessState


class TestDiscordProcessState:
    def test_is_typed_dict(self):
        assert issubclass(DiscordProcessState, dict)

    def test_has_required_input_fields(self):
        hints = typing.get_type_hints(DiscordProcessState)
        required_fields = [
            "timestamp", "user_name", "user_id", "user_input", "user_profile",
            "bot_id", "bot_name", "character_profile", "character_state",
            "channel_id", "channel_name", "chat_history",
        ]
        for field in required_fields:
            assert field in hints, f"Missing field: {field}"

    def test_has_relevance_output_fields(self):
        hints = typing.get_type_hints(DiscordProcessState)
        relevance_fields = [
            "should_respond", "reason_to_respond", "use_reply_feature",
            "channel_topic", "user_topic",
        ]
        for field in relevance_fields:
            assert field in hints, f"Missing relevance field: {field}"

    def test_has_persona_supervisor_output_fields(self):
        hints = typing.get_type_hints(DiscordProcessState)
        output_fields = ["final_dialog", "future_promises"]
        for field in output_fields:
            assert field in hints, f"Missing output field: {field}"

    def test_can_instantiate(self):
        state: DiscordProcessState = {
            "timestamp": "2024-01-01T00:00:00Z",
            "user_name": "TestUser",
            "user_id": "123",
            "user_input": "Hello",
            "user_profile": {},
            "bot_id": 456,
            "bot_name": "Bot",
            "character_profile": {},
            "character_state": {},
            "channel_id": "789",
            "channel_name": "test",
            "chat_history": [],
            "should_respond": True,
            "reason_to_respond": "",
            "use_reply_feature": False,
            "channel_topic": "",
            "user_topic": "",
            "final_dialog": [],
            "future_promises": [],
        }
        assert state["user_name"] == "TestUser"
        assert state["should_respond"] is True
