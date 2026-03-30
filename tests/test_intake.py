"""Tests for Stage 1 — Message Intake."""

from nodes.intake import intake


def test_intake_strips_mention_markup(base_state):
    base_state["message_text"] = "<@12345> Hello Zara"
    result = intake(base_state)
    assert result["message_text"] == "Hello Zara"
    assert result["should_respond"] is True


def test_intake_strips_nickname_mention(base_state):
    base_state["message_text"] = "<@!67890> What happened?"
    result = intake(base_state)
    assert result["message_text"] == "What happened?"


def test_intake_multiple_mentions(base_state):
    base_state["message_text"] = "<@111> <@!222> Tell me a story"
    result = intake(base_state)
    assert result["message_text"] == "Tell me a story"


def test_intake_empty_after_stripping(base_state):
    base_state["message_text"] = "<@12345>"
    result = intake(base_state)
    assert result["message_text"] == ""
    assert result["should_respond"] is False


def test_intake_preserves_existing_fields(base_state):
    result = intake(base_state)
    assert result["user_id"] == "user_123"
    assert result["channel_id"] == "chan_456"
    assert result["guild_id"] == "guild_789"


def test_intake_sets_timestamp_if_missing(base_state):
    del base_state["timestamp"]
    result = intake(base_state)
    assert "timestamp" in result
    assert result["timestamp"]  # non-empty


def test_intake_preserves_existing_timestamp(base_state):
    result = intake(base_state)
    assert result["timestamp"] == "2026-03-30T20:00:00Z"


def test_intake_plain_text_passthrough(base_state):
    base_state["message_text"] = "Just a normal message"
    result = intake(base_state)
    assert result["message_text"] == "Just a normal message"
    assert result["should_respond"] is True
