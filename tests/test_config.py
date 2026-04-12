"""Tests for config.py — environment variable loading and constants."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestAffinityConstants:
    def test_affinity_default_within_bounds(self):
        from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, AFFINITY_MIN, AFFINITY_MAX
        assert AFFINITY_MIN <= AFFINITY_DEFAULT <= AFFINITY_MAX

    def test_affinity_min_less_than_max(self):
        from kazusa_ai_chatbot.config import AFFINITY_MIN, AFFINITY_MAX
        assert AFFINITY_MIN < AFFINITY_MAX

    def test_affinity_min_is_zero(self):
        from kazusa_ai_chatbot.config import AFFINITY_MIN
        assert AFFINITY_MIN == 0

    def test_affinity_max_is_1000(self):
        from kazusa_ai_chatbot.config import AFFINITY_MAX
        assert AFFINITY_MAX == 1000


class TestBreakpoints:
    def test_increment_breakpoints_sorted_by_threshold(self):
        from kazusa_ai_chatbot.config import AFFINITY_INCREMENT_BREAKPOINTS
        thresholds = [bp[0] for bp in AFFINITY_INCREMENT_BREAKPOINTS]
        assert thresholds == sorted(thresholds)

    def test_decrement_breakpoints_sorted_by_threshold(self):
        from kazusa_ai_chatbot.config import AFFINITY_DECREMENT_BREAKPOINTS
        thresholds = [bp[0] for bp in AFFINITY_DECREMENT_BREAKPOINTS]
        assert thresholds == sorted(thresholds)

    def test_increment_breakpoints_all_positive_scales(self):
        from kazusa_ai_chatbot.config import AFFINITY_INCREMENT_BREAKPOINTS
        for _, scale in AFFINITY_INCREMENT_BREAKPOINTS:
            assert scale > 0

    def test_decrement_breakpoints_all_positive_scales(self):
        from kazusa_ai_chatbot.config import AFFINITY_DECREMENT_BREAKPOINTS
        for _, scale in AFFINITY_DECREMENT_BREAKPOINTS:
            assert scale > 0


class TestRetryLimits:
    def test_retry_limits_are_positive(self):
        from kazusa_ai_chatbot.config import (
            MAX_PERSONA_SUPERVISOR_STAGE1_RETRY,
            MAX_MEMORY_RETRIEVER_AGENT_RETRY,
            MAX_WEB_SEARCH_AGENT_RETRY,
            MAX_DIALOG_AGENT_RETRY,
            MAX_FACT_HARVESTER_RETRY,
        )
        assert MAX_PERSONA_SUPERVISOR_STAGE1_RETRY > 0
        assert MAX_MEMORY_RETRIEVER_AGENT_RETRY > 0
        assert MAX_WEB_SEARCH_AGENT_RETRY > 0
        assert MAX_DIALOG_AGENT_RETRY > 0
        assert MAX_FACT_HARVESTER_RETRY > 0


class TestTokenBudget:
    def test_token_budget_has_expected_keys(self):
        from kazusa_ai_chatbot.config import TOKEN_BUDGET
        expected = {"system_personality", "character_state", "user_memory",
                    "conversation_history", "current_message"}
        assert set(TOKEN_BUDGET.keys()) == expected

    def test_token_budget_values_positive(self):
        from kazusa_ai_chatbot.config import TOKEN_BUDGET
        for key, value in TOKEN_BUDGET.items():
            assert value > 0, f"TOKEN_BUDGET[{key}] should be positive"


class TestMcpServersDefault:
    def test_mcp_servers_is_dict(self):
        from kazusa_ai_chatbot.config import MCP_SERVERS
        assert isinstance(MCP_SERVERS, dict)
