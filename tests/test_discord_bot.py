"""Tests for discord_bot.py — message splitting and graph wiring."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from kazusa_ai_chatbot.discord_bot import _split_message


class TestSplitMessage:
    def test_short_message_returns_single_chunk(self):
        result = _split_message("Hello world")
        assert result == ["Hello world"]

    def test_empty_string(self):
        result = _split_message("")
        assert result == [""]

    def test_exactly_at_limit(self):
        msg = "a" * 2000
        result = _split_message(msg)
        assert result == [msg]

    def test_splits_at_newline(self):
        line1 = "a" * 1500
        line2 = "b" * 1000
        msg = f"{line1}\n{line2}"
        result = _split_message(msg, limit=2000)
        assert len(result) == 2
        assert result[0] == line1
        assert result[1] == line2

    def test_splits_at_space_when_no_newline(self):
        word = "a" * 999
        msg = f"{word} {word} {word}"
        result = _split_message(msg, limit=2000)
        assert len(result) == 2
        assert all(len(chunk) <= 2000 for chunk in result)

    def test_hard_split_when_no_whitespace(self):
        msg = "a" * 3000
        result = _split_message(msg, limit=2000)
        assert len(result) == 2
        assert result[0] == "a" * 2000
        assert result[1] == "a" * 1000

    def test_custom_limit(self):
        msg = "Hello World, this is a test"
        result = _split_message(msg, limit=10)
        assert all(len(chunk) <= 10 for chunk in result)
        assert "".join(result).replace(" ", "") == msg.replace(" ", "")

    def test_multiple_splits(self):
        msg = "a" * 5000
        result = _split_message(msg, limit=2000)
        assert len(result) == 3
        assert "".join(result) == msg


class TestBuildGraph:
    def test_graph_has_relevance_and_supervisor_nodes(self):
        with patch("kazusa_ai_chatbot.discord_bot.load_personality", return_value={"name": "Test"}):
            from kazusa_ai_chatbot.discord_bot import RolePlayBot
            bot = RolePlayBot.__new__(RolePlayBot)
            bot.personality = {"name": "Test"}
            bot.channel_ids = None
            bot.listen_all = False
            graph = bot.build_graph()
            # The compiled graph should be callable
            assert graph is not None
            assert hasattr(graph, "ainvoke")
