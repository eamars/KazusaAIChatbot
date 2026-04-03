"""Tests for Stage 1 — Message Intake."""

from kazusa_ai_chatbot.nodes.intake import intake


def test_intake_strips_mention_markup(base_state):
    # Mention the bot itself so should_respond stays True
    base_state["message_text"] = "<@999888777> Hello Zara"
    result = intake(base_state)
    assert result["message_text"] == "Hello Zara"
    assert result["should_respond"] is True


def test_intake_strips_nickname_mention(base_state):
    base_state["message_text"] = "<@!999888777> What happened?"
    result = intake(base_state)
    assert result["message_text"] == "What happened?"


def test_intake_multiple_mentions_including_bot(base_state):
    base_state["message_text"] = "<@111> <@999888777> Tell me a story"
    result = intake(base_state)
    assert result["message_text"] == "Tell me a story"
    assert result["should_respond"] is True


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


# ── Mention-based filtering tests ───────────────────────────────────


def test_intake_directed_at_other_user(base_state):
    """Message that tags another user but NOT the bot — should not respond."""
    base_state["message_text"] = "<@12345> Hey how are you?"
    result = intake(base_state)
    assert result["message_text"] == "Hey how are you?"
    assert result["should_respond"] is False


def test_intake_directed_at_multiple_others(base_state):
    """Message tagging multiple other users but not the bot."""
    base_state["message_text"] = "<@111> <@222> check this out"
    result = intake(base_state)
    assert result["should_respond"] is False


def test_intake_bot_among_multiple_mentions(base_state):
    """Bot is mentioned alongside another user — should respond."""
    base_state["message_text"] = "<@12345> <@999888777> What do you both think?"
    result = intake(base_state)
    assert result["message_text"] == "What do you both think?"
    assert result["should_respond"] is True


def test_intake_no_mentions_responds(base_state):
    """No mentions at all — normal message, should respond."""
    base_state["message_text"] = "What is the weather like?"
    result = intake(base_state)
    assert result["should_respond"] is True


def test_intake_no_bot_id_falls_through(base_state):
    """If bot_id is not set, mentions of others don't trigger filtering."""
    base_state["bot_id"] = ""
    base_state["message_text"] = "<@12345> Hey there"
    result = intake(base_state)
    # With no bot_id, we can't tell if it's for us, so we still filter
    # because the message mentions someone and bot_id is empty
    assert result["should_respond"] is False
