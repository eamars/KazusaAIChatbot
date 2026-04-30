"""Tests for state.py — IMProcessState TypedDict structure."""

from __future__ import annotations

import typing

from kazusa_ai_chatbot.state import IMProcessState, DebugModes, keep_false


class TestIMProcessState:
    def test_is_typed_dict(self):
        assert issubclass(IMProcessState, dict)

    def test_has_required_input_fields(self):
        hints = typing.get_type_hints(IMProcessState)
        required_fields = [
            "timestamp", "platform", "platform_message_id", "platform_user_id", "global_user_id",
            "user_name", "user_input", "message_envelope", "user_multimedia_input", "user_profile",
            "platform_bot_id", "character_name", "character_profile",
            "platform_channel_id", "channel_name", "chat_history_wide", "chat_history_recent", "reply_context",
        ]
        for field in required_fields:
            assert field in hints, f"Missing field: {field}"

    def test_has_relevance_output_fields(self):
        hints = typing.get_type_hints(IMProcessState)
        relevance_fields = [
            "should_respond", "reason_to_respond", "use_reply_feature",
            "channel_topic", "indirect_speech_context",
        ]
        for field in relevance_fields:
            assert field in hints, f"Missing relevance field: {field}"

    def test_has_debug_modes_field(self):
        hints = typing.get_type_hints(IMProcessState)
        assert "debug_modes" in hints, "Missing field: debug_modes"

    def test_has_persona_supervisor_output_fields(self):
        hints = typing.get_type_hints(IMProcessState)
        output_fields = [
            "final_dialog",
            "future_promises",
            "target_addressed_user_ids",
            "target_broadcast",
        ]
        for field in output_fields:
            assert field in hints, f"Missing output field: {field}"

    def test_has_message_envelope_field(self):
        hints = typing.get_type_hints(IMProcessState)
        assert "message_envelope" in hints, "Missing field: message_envelope"

    def test_can_instantiate(self):
        state: IMProcessState = {
            "timestamp": "2024-01-01T00:00:00Z",
            "platform": "discord",
            "platform_message_id": "message-123",
            "platform_user_id": "123",
            "global_user_id": "uuid-123",
            "user_name": "TestUser",
            "user_input": "Hello",
            "message_envelope": {
                "body_text": "Hello",
                "raw_wire_text": "Hello",
                "mentions": [],
                "attachments": [],
                "addressed_to_global_user_ids": [],
                "broadcast": True,
            },
            "user_multimedia_input": [],
            "user_profile": {},
            "platform_bot_id": "456",
            "character_name": "Character",
            "character_profile": {},
            "platform_channel_id": "789",
            "channel_name": "test",
            "chat_history_wide": [],
            "chat_history_recent": [],
            "reply_context": {},
            "should_respond": True,
            "reason_to_respond": "",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
            "debug_modes": {},
            "final_dialog": [],
            "future_promises": [],
        }
        assert state["user_name"] == "TestUser"
        assert state["should_respond"] is True


class TestDebugModes:
    def test_debug_modes_all_false(self):
        modes: DebugModes = {"listen_only": False, "think_only": False, "no_remember": False}
        assert modes["listen_only"] is False

    def test_debug_modes_partial(self):
        modes: DebugModes = {"listen_only": True}
        assert modes["listen_only"] is True
        assert modes.get("think_only") is None

    def test_debug_modes_compound(self):
        modes: DebugModes = {"listen_only": True, "no_remember": True}
        assert modes["listen_only"] is True
        assert modes["no_remember"] is True


class TestMonotonicLatches:
    def test_keep_false_preserves_false(self):
        assert keep_false(True, False) is False
        assert keep_false(False, True) is False
        assert keep_false(None, True) is True
